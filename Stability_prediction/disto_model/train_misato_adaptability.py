from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import torch
import torch.nn as nn
from dataset import ProteinGraphDataset, ProteinGraphDatasetTorch, protein_collate_fn  # The dataset class we wrote earlier
from model import RGAT, RGCN, Custom_EGNN, RGAT_bs, RGCN_bs  # The GNN model we defined
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR
import math
import yaml
import os
import argparse
from datetime import datetime
import numpy as np
import random
from focal_loss import FocalLoss    
from functools import partial

# =====================
# 2. Training Loop
# =====================

class Trainer:
    def __init__(self, config):
        self.config = config
        self.loss_fn = nn.MSELoss()

    def compute_loss(self, pred, target, writer, epoch):
        loss = self.loss_fn(pred.view(-1,), target)
        writer.add_scalar("Loss/MSE", loss, epoch + 1)
        return loss


    def train_loop_RGAT(self, model, loader, optimizer, device, writer, epoch):
        model.train()
        total_loss = 0
        loader_iter = tqdm(loader, desc="Training", leave=False)
        for batch in loader_iter:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            pred = model(batch.x.float(), batch.edge_index, batch.edge_type, batch.edge_attr, batch.batch)
            
            loss = self.compute_loss(pred, batch.y, writer, epoch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            loader_iter.set_postfix({"loss": total_loss / ((loader_iter.n + 1) * batch.num_graphs)})
            
        return total_loss / len(loader.dataset)


    def eval_loop_RGAT(self, model, loader, device, writer, epoch):
        model.eval()
        total_loss = 0
        loader_iter = tqdm(loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in loader_iter:
                batch = batch.to(device)
                pred = model(batch.x.float(), batch.edge_index, batch.edge_type, batch.edge_attr, batch.batch)
                loss = self.compute_loss(pred, batch.y, writer, epoch)
                total_loss += loss.item() * batch.num_graphs
                loader_iter.set_postfix({"val_loss": total_loss / ((loader_iter.n + 1) * batch.num_graphs)})
        return total_loss / len(loader.dataset)


    def train_loop_RGCN(self, model, loader, optimizer, loss_fn, device):
        model.train()
        total_loss = 0
        loader_iter = tqdm(loader, desc="Training", leave=False)
        for batch in loader_iter:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            pred = model(batch.x.float(), batch.edge_index, batch.edge_type, batch.batch)
            
            loss = loss_fn(pred.view(-1), batch.y.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            loader_iter.set_postfix({"loss": total_loss / ((loader_iter.n + 1) * batch.num_graphs)})
            
        return total_loss / len(loader.dataset)

    def eval_loop_RGCN(self, model, loader, loss_fn, device):
        model.eval()
        total_loss = 0
        loader_iter = tqdm(loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in loader_iter:
                batch = batch.to(device)
                pred = model(batch.x.float(), batch.edge_index, batch.edge_type, batch.batch)
                loss = loss_fn(pred.view(-1), batch.y.float())
                total_loss += loss.item() * batch.num_graphs
                loader_iter.set_postfix({"val_loss": total_loss / ((loader_iter.n + 1) * batch.num_graphs)})
        return total_loss / len(loader.dataset)

# =====================
# 3. Main Script
# =====================
def main():

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multiple GPUs

    g = torch.Generator()
    g.manual_seed(seed)

    parser = argparse.ArgumentParser(description="Example script")
    parser.add_argument("config", help="config file to use")

    args = parser.parse_args()
    args.config

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Access training parameters
    data_path = config["training"]["data_path"]
    epochs = config["training"]["epochs"]
    warmup_epochs = epochs // 50
    initial_lr = config["training"]["initial_lr"]
    eta_min = config["training"]["eta_min"]
    batch_size = config["training"]["batch_size"]
    device = config["training"]["device"]
    depth = config["model"]["depth"]  # Maximum sequence length for padding
    weight_decay = config["training"]["weight_decay"]   
    L_max = config["model"]["L_max"]  # Maximum sequence length for padding

    # Create timestamp (e.g., 2025-09-22_14-30-05)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

    # Build directory name
    log_dir = f"runs/{timestamp}"

    # Create writer
    writer = SummaryWriter(log_dir=log_dir)

    # Save config to another file
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    gt_data_path = os.path.join(os.path.dirname(data_path), 'adaptability')

    if config["model"]["type"] == "EGNN":
        train_dataset = ProteinGraphDatasetTorch(root=os.path.join(data_path, "train"), root_gt=gt_data_path)
        val_dataset = ProteinGraphDatasetTorch(root=os.path.join(data_path, "val"), root_gt=gt_data_path)

        # DataLoaders
        #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #val_loader = DataLoader(val_dataset, batch_size=batch_size)

        collate_fn = partial(protein_collate_fn, L_max=L_max)

        # DataLoaders
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True, generator=g)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False, generator=g)

        feats, coors, edges, mask, y = train_dataset[0]
        model = Custom_EGNN(dim=feats.shape[-1], depth=depth, num_positions=L_max, edge_dim=edges.shape[-1]).to(device)
    else:
        # Load train and validation datasets separately
        train_dataset = ProteinGraphDataset(root=os.path.join(data_path, "train"), root_gt=gt_data_path)
        val_dataset = ProteinGraphDataset(root=os.path.join(data_path, "val"), root_gt=gt_data_path)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, generator=g)

    # Model
    node_feat_dim = train_dataset[0].x.shape[1]
    edge_feat_dim = train_dataset[0].edge_attr.shape[1]  # your edge feature dimension
    num_relations = train_dataset[0].edge_type.max().item() + 1

    if config["model"]["type"] == "RGCN":
        model = RGCN_bs(
            in_channels=node_feat_dim,
            hidden_channels=config["model"]["hidden_channels"],
            out_channels=2 if config["training"]["loss"] == "CrossEntropyLoss" else 1,
            num_relations=num_relations,
            edge_dim=edge_feat_dim,
            depth=depth,
            dropout=config["model"]["dropout"]
        ).to(device)
    elif config["model"]["type"] == "RGAT":
        model = RGAT_bs(
            in_channels=node_feat_dim,
            hidden_channels=config["model"]["hidden_channels"],
            out_channels=2 if config["training"]["loss"] == "CrossEntropyLoss" else 1,
            num_relations=num_relations,
            edge_dim=edge_feat_dim,
            depth=depth,
            dropout=config["model"]["dropout"]
        ).to(device)


    #if config["model"]["type"] == "RGCN":
    #    model = RGCN(
    #        in_channels=node_feat_dim, 
    #        hidden_channels=config["model"]["hidden_channels"], 
    #        out_channels=1, 
    #        num_relations=num_relations,
    #        num_layers=config["model"]["layers"]
    #    ).to(device)
    #elif config["model"]["type"] == "RGAT":
    #    model = RGAT(
    #        in_channels=node_feat_dim, 
    #        hidden_channels=config["model"]["hidden_channels"], 
    #        out_channels=1, 
    #        num_relations=num_relations,
    #        edge_dim=edge_feat_dim
    #    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warm-up
            return float(epoch + 1) / warmup_epochs
        else:
            # Cosine decay after warm-up
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return eta_min / initial_lr + 0.5 * (1 - eta_min / initial_lr) * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    Trainer_instance = Trainer(config)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        if config["model"]["type"] == "RGCN":
            train_loss = Trainer_instance.train_loop_RGCN(model, train_loader, optimizer, device, writer, epoch)
            val_loss = Trainer_instance.eval_loop_RGCN(model, val_loader, device, writer, epoch)
        elif config["model"]["type"] == "RGAT":
            train_loss = Trainer_instance.train_loop_RGAT(model, train_loader, optimizer, device, writer, epoch)
            val_loss = Trainer_instance.eval_loop_RGAT(model, val_loader, device, writer, epoch)
        elif config["model"]["type"] == "EGNN":
            train_loss = Trainer_instance.train_loop_EGNN(model, train_loader, optimizer, device, writer, epoch)
            val_loss = Trainer_instance.eval_loop_EGNN(model, val_loader, device, writer, epoch)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        device = next(model.parameters()).device
        max_memory_allocated = torch.cuda.max_memory_allocated(device=device)
        print("Max GPU Memory allocated:", max_memory_allocated / 10e8, "Gb")

        # Step the scheduler
        scheduler.step()

        # Log to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch+1)
        writer.add_scalar("Loss/Val", val_loss, epoch+1)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch+1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            print(f"Saved new best model at epoch {epoch}")
    

    #if config["model"]["type"] == "RGCN":
    #    for epoch in range(epochs):
    #        train_loss = train_loop_RGCN(model, train_loader, optimizer, loss_fn, device)
    #        val_loss = eval_loop_RGCN(model, val_loader, loss_fn, device)
    #        #print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
#
    #        # Step the scheduler
    #        scheduler.step()
#
    #        # Log to TensorBoard
    #        writer.add_scalar("Loss/Train", train_loss, epoch+1)
    #        writer.add_scalar("Loss/Val", val_loss, epoch+1)
    #        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch+1)
    #elif config["model"]["type"] == "RGAT":
    #    for epoch in range(epochs):
    #        train_loss = train_loop_RGAT(model, train_loader, optimizer, loss_fn, device)
    #        val_loss = eval_loop_RGAT(model, val_loader, loss_fn, device)
    #        #print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
#
    #        # Step the scheduler
    #        scheduler.step()
#
    #        # Log to TensorBoard
    #        writer.add_scalar("Loss/Train", train_loss, epoch+1)
    #        writer.add_scalar("Loss/Val", val_loss, epoch+1)
    #        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch+1)

if __name__ == "__main__":
    main()