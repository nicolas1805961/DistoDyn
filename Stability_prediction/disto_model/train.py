from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import torch
import torch.nn as nn
from dataset import ProteinGraphDataset, ProteinGraphDatasetTorch, protein_collate_fn  # The dataset class we wrote earlier
from model import RGAT, RGCN, Custom_EGNN  # The GNN model we defined
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR
import math
import yaml
from functools import partial

# =====================
# 2. Training Loop
# =====================

def train_loop_EGNN(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    loader_iter = tqdm(loader, desc="Training", leave=False)
    for batch in loader_iter:
        # batch is a tuple: feats, coors, edges, mask, y
        feats, coors, edges, mask, y = batch

        # send each tensor to device
        feats = feats.to(device)
        coors = coors.to(device)
        edges = edges.to(device)
        mask = mask.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        pred = model(feats, coors, edges, mask, adj_mat=None)
        
        loss = loss_fn(pred.view(-1), y.float().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * feats.size(0)
        loader_iter.set_postfix({"loss": total_loss / ((loader_iter.n + 1) * feats.size(0))})
        
    return total_loss / len(loader.dataset)

def eval_loop_EGNN(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    loader_iter = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in loader_iter:
            # batch is a tuple: feats, coors, edges, mask, y
            feats, coors, edges, mask, y = batch

            # send each tensor to device
            feats = feats.to(device)
            coors = coors.to(device)
            edges = edges.to(device)
            mask = mask.to(device)
            y = y.to(device)

            pred = model(feats, coors, edges, mask, adj_mat=None)
            loss = loss_fn(pred.view(-1), y.float().view(-1))
            total_loss += loss.item() * feats.size(0)
            loader_iter.set_postfix({"val_loss": total_loss / ((loader_iter.n + 1) * feats.size(0))})
    return total_loss / len(loader.dataset)


# =====================
# 3. Main Script
# =====================
def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Access training parameters
    epochs = config["training"]["epochs"]
    warmup_epochs = config["training"]["warmup_epochs"]
    initial_lr = config["training"]["initial_lr"]
    eta_min = config["training"]["eta_min"]
    batch_size = config["training"]["batch_size"]
    device = config["training"]["device"]
    L_max = config["data"]["L_max"]  # Maximum sequence length for padding
    depth = config["model"]["depth"]  # Maximum sequence length for padding

    # Load train and validation datasets separately
    #train_dataset = ProteinGraphDataset(root="split/train")
    #val_dataset = ProteinGraphDataset(root="split/val")

    # Load train and validation datasets separately
    train_dataset = ProteinGraphDatasetTorch(root="split/train")
    val_dataset = ProteinGraphDatasetTorch(root="split/val")

    # DataLoaders
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size)

    collate_fn = partial(protein_collate_fn, L_max=L_max)

    # DataLoaders
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/protein_gnn_experiment1")

    # Model
    #node_feat_dim = train_dataset[0].x.shape[1]
    #edge_feat_dim = train_dataset[0].edge_attr.shape[1]  # your edge feature dimension
    #num_relations = train_dataset[0].edge_type.max().item() + 1

    feats, coors, edges, mask, y = train_dataset[0]

    model = Custom_EGNN(dim=feats.shape[-1], depth=depth, num_positions=L_max, edge_dim=edges.shape[-1]).to(device)


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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()  # Change to BCEWithLogitsLoss for binary classification

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warm-up
            return float(epoch + 1) / warmup_epochs
        else:
            # Cosine decay after warm-up
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return eta_min / initial_lr + 0.5 * (1 - eta_min / initial_lr) * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


    for epoch in range(epochs):
        train_loss = train_loop_EGNN(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_loop_EGNN(model, val_loader, loss_fn, device)
        #print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Step the scheduler
        scheduler.step()

        # Log to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch+1)
        writer.add_scalar("Loss/Val", val_loss, epoch+1)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch+1)
    

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