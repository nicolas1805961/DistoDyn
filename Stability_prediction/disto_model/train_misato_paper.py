from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import torch
import torch.nn as nn
from dataset import ProteinGraphDataset, ProteinGraphDatasetTorch, protein_collate_fn  # The dataset class we wrote earlier
from model_paper import RGCNNodeClassifier
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
from pytorch_lightning import Trainer as lightningTrainer
from pytorch_lightning.loggers import TensorBoardLogger
from callback import get_callbacks

# =====================
# 2. Training Loop
# =====================

class Trainer:
    def __init__(self, config):
        self.config = config
        device = config["training"]["device"]
        p = 0.8194692836899553

        if self.config["training"]["loss"] == "CrossEntropyLoss":
            class_weight = torch.tensor([1 - p, p], dtype=torch.float).to(device)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)

            if self.config["training"]["focal_loss"]:
                self.focal_loss = FocalLoss(
                    gamma=2, alpha=[1 - p, p], task_type="multi-class", num_classes=2
                )
            else:
                self.focal_loss = None

        else:  # BCE case
            class_weight = torch.tensor([8.194692836899553], dtype=torch.float).to(device)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight)

            if self.config["training"]["focal_loss"]:
                self.focal_loss = FocalLoss(gamma=2, alpha=0.25, task_type="binary")
            else:
                self.focal_loss = None

    def compute_loss_ce(self, pred, target, writer, epoch):
        loss = self.loss_fn(pred, target.long())  # CrossEntropyLoss expects Long targets
        writer.add_scalar("Loss/cross_entropy", loss, epoch + 1)
        if self.focal_loss is not None:
            fl_loss = self.focal_loss(pred, target.long())
            writer.add_scalar("Loss/focal_loss", fl_loss, epoch + 1)
            loss = loss + fl_loss
        return loss

    def compute_loss_bce(self, pred, target, writer, epoch):
        loss = self.loss_fn(pred.view(-1,), target.float())  # BCE expects Float targets
        writer.add_scalar("Loss/bce", loss, epoch + 1)
        if self.focal_loss is not None:
            fl_loss = self.focal_loss(pred.view(-1,), target.float())
            writer.add_scalar("Loss/focal_loss", fl_loss, epoch + 1)
            loss = loss + fl_loss
        return loss


    def train_loop_RGCN(self, model, loader, optimizer, device):
        model.train()
        total_loss = 0
        loader_iter = tqdm(loader, desc="Training", leave=False)
        for batch in loader_iter:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            loader_iter.set_postfix({"loss": total_loss / ((loader_iter.n + 1) * batch.num_graphs)})
            
        return total_loss / len(loader.dataset)

    def eval_loop_RGCN(self, model, loader, device):
        model.eval()
        total_loss = 0
        loader_iter = tqdm(loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in loader_iter:
                batch = batch.to(device)
                loss = model.validation_step(batch)
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
    warmup_epochs = epochs // 100
    initial_lr = config["training"]["initial_lr"]
    eta_min = config["training"]["eta_min"]
    batch_size = config["training"]["batch_size"]
    device = config["training"]["device"]

    # Create timestamp (e.g., 2025-09-22_14-30-05)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

    # Build directory name
    log_dir = f"runs/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Save config to another file
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    gt_data_path = os.path.join(os.path.dirname(data_path), 'binding_sites')

    train_dataset = ProteinGraphDataset(root=os.path.join(data_path, "train"), root_gt=gt_data_path)
    val_dataset = ProteinGraphDataset(root=os.path.join(data_path, "val"), root_gt=gt_data_path)

        # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, generator=g)

        # Model
    node_feat_dim = train_dataset[0].x.shape[1]

    model = RGCNNodeClassifier(
            in_dim=node_feat_dim,
            lr=initial_lr,
            use_class_weights=True,
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

    #scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    #Trainer_instance = Trainer(config)
    callbacks = get_callbacks(config, log_dir)

    # Create the TensorBoard logger instance properly
    logger = TensorBoardLogger(
        save_dir=log_dir,        # directory for TensorBoard logs
        name="",         # unique subfolder for this run
        default_hp_metric=False # optional, avoids "hp_metric" spam
    )

    trainer = lightningTrainer(
        callbacks=callbacks,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=1,
        val_check_interval=1.0,
        accumulate_grad_batches=4,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=0,
    )
    trainer.fit(model, train_loader, val_loader)

    #best_val_loss = float("inf")
#
    #for epoch in range(epochs):
    #    train_loss = Trainer_instance.train_loop_RGCN(model, train_loader, optimizer, device)
    #    val_loss = Trainer_instance.eval_loop_RGCN(model, val_loader, device)
    #    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
#
    #    device = next(model.parameters()).device
    #    max_memory_allocated = torch.cuda.max_memory_allocated(device=device)
    #    print("Max GPU Memory allocated:", max_memory_allocated / 10e8, "Gb")
#
    #    # Step the scheduler
    #    scheduler.step()
#
    #    # Log to TensorBoard
    #    writer.add_scalar("Loss/Train", train_loss, epoch+1)
    #    writer.add_scalar("Loss/Val", val_loss, epoch+1)
    #    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch+1)
#
    #    if val_loss < best_val_loss:
    #        best_val_loss = val_loss
    #        torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
    #        print(f"Saved new best model at epoch {epoch}")
    

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