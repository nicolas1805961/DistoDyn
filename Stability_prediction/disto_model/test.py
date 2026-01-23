# =====================
# 4. Testing Loop
# =====================

import torch
from tqdm import tqdm
import os
from dataset import ProteinGraphDataset
from torch_geometric.loader import DataLoader
import yaml
from model import RGAT_bs, RGCN_bs, RGCNNodeClassifier, REGNNNodeClassifier  # The GNN model we defined
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import argparse
import numpy as np



def test_loop_regnn(model, loader, device, config):
    """
    Evaluate model on a test dataset.
    
    Args:
        model: trained model
        loader: DataLoader for test dataset
        device: 'cuda' or 'cpu'
        loss_fn: optional, e.g., nn.MSELoss() or nn.BCEWithLogitsLoss()
        
    Returns:
        preds: list of model predictions
        targets: list of true labels (if available, else None)
        test_loss: average loss (if loss_fn provided, else None)
    """
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0
    
    loader_iter = tqdm(loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for batch in loader_iter:
            batch = batch.to(device)

            out = model(batch.x.float(), batch.pos.float(), batch.edge_index, batch.edge_type, batch.edge_attr, batch.batch)
            
            if config["training"]["loss"] == "CrossEntropyLoss":
                pred = out.argmax(dim=1)
            else:
                pred = torch.sigmoid(out) >= 0.5
            
            preds.append(pred.cpu())
            targets.append(batch.y.cpu())
    
    # Concatenate
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # Compute metrics
    y_true = targets.numpy()
    y_pred = preds.numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    return preds, targets, metrics




def test_loop_rgat(model, loader, device, config):
    """
    Evaluate model on a test dataset.
    
    Args:
        model: trained model
        loader: DataLoader for test dataset
        device: 'cuda' or 'cpu'
        loss_fn: optional, e.g., nn.MSELoss() or nn.BCEWithLogitsLoss()
        
    Returns:
        preds: list of model predictions
        targets: list of true labels (if available, else None)
        test_loss: average loss (if loss_fn provided, else None)
    """
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0
    
    loader_iter = tqdm(loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for batch in loader_iter:
            batch = batch.to(device)

            out = model(batch.x.float(), batch.edge_index, batch.edge_type, batch.edge_attr, batch.batch)
            
            if config["training"]["loss"] == "CrossEntropyLoss":
                pred = out.argmax(dim=1)
            else:
                pred = torch.sigmoid(out) >= 0.5
            
            preds.append(pred.cpu())
            targets.append(batch.y.cpu())
    
    # Concatenate
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # Compute metrics
    y_true = targets.numpy()
    y_pred = preds.numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    return preds, targets, metrics



def test_loop_rgcn(model, loader, device, config):
    """
    Evaluate model on a test dataset.
    
    Args:
        model: trained model
        loader: DataLoader for test dataset
        device: 'cuda' or 'cpu'
        loss_fn: optional, e.g., nn.MSELoss() or nn.BCEWithLogitsLoss()
        
    Returns:
        preds: list of model predictions
        targets: list of true labels (if available, else None)
        test_loss: average loss (if loss_fn provided, else None)
    """
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0
    
    loader_iter = tqdm(loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for batch in loader_iter:
            batch = batch.to(device)

            out = model(batch.x.float(), batch.edge_index, batch.edge_type, batch.batch)

            if config["training"]["loss"] == "CrossEntropyLoss":
                pred = out.argmax(dim=1)
            else:
                pred = torch.sigmoid(out) >= 0.5
            
            preds.append(pred.cpu())
            targets.append(batch.y.cpu())
    
    # Concatenate
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # Compute metrics
    y_true = targets.numpy()
    y_pred = preds.numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    return preds, targets, metrics




def test_loop_rgcn_paper(model, loader, device):
    """
    Evaluate model on a test dataset.
    
    Args:
        model: trained model
        loader: DataLoader for test dataset
        device: 'cuda' or 'cpu'
        loss_fn: optional, e.g., nn.MSELoss() or nn.BCEWithLogitsLoss()
        
    Returns:
        preds: list of model predictions
        targets: list of true labels (if available, else None)
        test_loss: average loss (if loss_fn provided, else None)
    """
    model.eval()
    preds = []
    targets = []
    #total_loss = 0.0
    #all_metrics = []
    
    loader_iter = tqdm(loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for batch in loader_iter:
            batch = batch.to(device)

            #out = model(batch.x.float(), batch.edge_index, batch.edge_type)
            out = model(batch)
            
            # Predicted class per node (0 or 1)
            pred = out.argmax(dim=1)
            
            preds.append(pred.cpu())
            targets.append(batch.y.cpu())

            #curent_metrics = {
            #            "accuracy": accuracy_score(batch.y.cpu().numpy(), pred.cpu().numpy()),
            #            "precision": precision_score(batch.y.cpu().numpy(), pred.cpu().numpy(), zero_division=0),
            #            "recall": recall_score(batch.y.cpu().numpy(), pred.cpu().numpy(), zero_division=0),
            #            "f1": f1_score(batch.y.cpu().numpy(), pred.cpu().numpy(), zero_division=0),
            #        }
            #
            #all_metrics.append(curent_metrics)
    
    # Concatenate
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # Compute metrics
    y_true = targets.numpy()
    y_pred = preds.numpy()


    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    return preds, targets, metrics




parser = argparse.ArgumentParser(description="Evaluate a trained model")
parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pt or .pth)")
args = parser.parse_args()

model_path = args.model_path

with open(os.path.join(model_path, 'config.yaml'), "r") as f:
    config = yaml.safe_load(f)

# Access training parameters
epochs = config["training"]["epochs"]
warmup_epochs = epochs // 50
initial_lr = config["training"]["initial_lr"]
eta_min = config["training"]["eta_min"]
device = config["training"]["device"]
depth = config["model"]["depth"]  # Maximum sequence length for padding
weight_decay = config["training"]["weight_decay"]  


#if 'boltz_parsed' in config["training"]["data_path"]:
#    test_root_path = 'test_non_experimental'
#else:
test_root_path = 'test_experimental'

if 'distance_rbf' in config["training"]["data_path"]:
    print("Using distance_rbf_distogram_full dataset")
    data_path = os.path.join(test_root_path, 'test_distance_rbf_distogram_full')
elif 'distance_distogram_correlation' in config["training"]["data_path"]:
    print("Using distance_distogram_correlation dataset")
    data_path = os.path.join(test_root_path, 'test_distance_distogram_correlation')
elif 'distance_distogram_full' in config["training"]["data_path"]:
    print("Using distance_distogram_full dataset")
    data_path = os.path.join(test_root_path, 'test_distance_distogram_full')
elif 'distance_distogram_random_edge' in config["training"]["data_path"]:
    print("Using distance_distogram_random_edge dataset")
    data_path = os.path.join(test_root_path, 'test_distance_distogram_random_edge')
elif 'distances_correlation' in config["training"]["data_path"]:
    print("Using distances_correlation dataset")
    data_path = os.path.join(test_root_path, 'test_distance_correlation')
elif 'distance_distogram' in config["training"]["data_path"]:
    print("Using distance_distogram dataset")
    data_path = os.path.join(test_root_path, 'test_distance_distogram')
elif 'distogram_full' in config["training"]["data_path"]:
    print("Using distogram_full dataset")
    data_path = os.path.join(test_root_path, 'test_distogram_full')
elif 'correlation_full' in config["training"]["data_path"]:
    print("Using correlation_full dataset")
    data_path = os.path.join(test_root_path, 'test_correlation_full')
elif 'correlation' in config["training"]["data_path"]:
    print("Using correlation dataset")
    data_path = os.path.join(test_root_path, 'test_correlation')
elif 'distance' in config["training"]["data_path"]:
    print("Using distance dataset")
    data_path = os.path.join(test_root_path, 'test_distance')
elif 'distogram' in config["training"]["data_path"]:
    print("Using distogram dataset")
    nb = config["training"]["data_path"].split('_')[-1]
    data_path = os.path.join(test_root_path, 'test_distogram')

# --- Load test dataset ---
test_dataset = ProteinGraphDataset(root=os.path.join(data_path, "test"), root_gt='binding_sites')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model
node_feat_dim = test_dataset[0].x.shape[1]
edge_feat_dim = test_dataset[0].edge_attr.shape[1]  # your edge feature dimension
num_relations = test_dataset[0].edge_type.max().item() + 1

#model = RGCNNodeClassifier(
#            in_dim=node_feat_dim,
#            lr=initial_lr,
#            use_class_weights=True,
#        ).to(device)

if config["model"]["type"] == "RGCN":
    #model = RGCNNodeClassifier(
    #        in_channels=node_feat_dim,
    #        hidden_channels=config["model"]["hidden_channels"],
    #        out_channels=2 if config["training"]["loss"] == "CrossEntropyLoss" else 1,
    #        num_relations=num_relations,
    #        depth=depth,
    #        dropout=config["model"]["dropout"]
    #    ).to(device)
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
elif config["model"]["type"] == "EGNN":
    model = REGNNNodeClassifier(
            in_channels=node_feat_dim,
            hidden_channels=config["model"]["hidden_channels"],
            out_channels=2 if config["training"]["loss"] == "CrossEntropyLoss" else 1,
            num_relations=num_relations,
            edge_dim=edge_feat_dim,
            depth=depth,
            dropout=config["model"]["dropout"]
        ).to(device)

# --- Load best model ---
model.load_state_dict(torch.load(os.path.join(model_path, "best_model.pth")))

# Load the checkpoint into the Lightning module
#model = RGCNNodeClassifier.load_from_checkpoint(
#    os.path.join(model_path, 'checkpoints', 'best.ckpt')
#).to(device)

if config["model"]["type"] == "RGCN" or config["model"]["type"] == "RGCN_paper":
    preds, targets, metrics = test_loop_rgcn(model, test_loader, device, config)
    #preds, targets, metrics = test_loop_rgcn_paper(model, test_loader, device)
elif config["model"]["type"] == "RGAT":
    preds, targets, metrics = test_loop_rgat(model, test_loader, device, config)
elif config["model"]["type"] == "EGNN":
    preds, targets, metrics = test_loop_regnn(model, test_loader, device, config)
print(metrics)
# Save to JSON file
with open(os.path.join(model_path, 'metrics.json'), "w") as f:
    json.dump(metrics, f, indent=4)