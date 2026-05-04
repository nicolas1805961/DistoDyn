# =====================
# 4. Testing Loop
# =====================

import torch
from tqdm import tqdm
import os
from dataset import ProteinGraphDataset
from torch_geometric.loader import DataLoader
import yaml
from model import RGAT_affinity, RGCN_affinity, New_RGCN_complex_affinity, RGCN_affinity_no_residual, RGCN_affinity_bn, RGAT_affinity_no_residual, RGAT_complex_affinity, REGNN_complex_affinity  # The GNN model we defined
import json
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

def test_loop_rgat(model, loader, device):
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
    
    loader_iter = tqdm(loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for batch in loader_iter:
            batch = batch.to(device)
            if "edge_type" not in batch:
                batch.edge_type = torch.zeros(batch.edge_index.size(1), dtype=torch.long, device=device)

            out = model(batch.x.float(), batch.edge_index, batch.edge_type, batch.edge_attr, batch.batch)
            
            preds.append(out.cpu())
            targets.append(batch.y.cpu())
    
    # Concatenate
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # Compute metrics
    y_true = targets.numpy()
    y_pred = preds.numpy()

    # Compute regression metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)

    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "PearsonR": float(pearson_r),
        "SpearmanR": float(spearman_r)
    }
    
    return preds, targets, metrics




def test_loop_EGNN(model, loader, device):
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
    
    loader_iter = tqdm(loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for batch in loader_iter:
            batch = batch.to(device)
            if "edge_type" not in batch:
                batch.edge_type = torch.zeros(batch.edge_index.size(1), dtype=torch.long, device=device)

            out = model(batch.x.float(), batch.pos.float(), batch.edge_index, batch.edge_type, batch.edge_attr, batch.batch)
            
            preds.append(out.cpu())
            targets.append(batch.y.cpu())
    
    # Concatenate
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # Compute metrics
    y_true = targets.numpy()
    y_pred = preds.numpy()

    # Compute regression metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)

    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "PearsonR": float(pearson_r),
        "SpearmanR": float(spearman_r)
    }
    
    return preds, targets, metrics



def test_loop_rgcn(model, loader, device):
    """
    Evaluate model on a test dataset (regression setting).

    Args:
        model: trained model
        loader: DataLoader for test dataset
        device: 'cuda' or 'cpu'
        config: optional config object (not used here)
        
    Returns:
        preds: torch.Tensor of predictions
        targets: torch.Tensor of true labels
        metrics: dict with RMSE, MAE, Pearson R, and Spearman R
    """
    model.eval()
    preds = []
    targets = []
    
    loader_iter = tqdm(loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for batch in loader_iter:
            batch = batch.to(device)
            if "edge_type" not in batch:
                batch.edge_type = torch.zeros(batch.edge_index.size(1), dtype=torch.long, device=device)

            out = model(batch.x.float(), batch.edge_index, batch.edge_type, batch.batch)
            
            preds.append(out.cpu())
            targets.append(batch.y.cpu())
            #print(np.sqrt(mean_squared_error(targets, preds)))
    
    # Concatenate
    preds = torch.cat(preds, dim=0).squeeze()
    targets = torch.cat(targets, dim=0).squeeze()

    # Convert to numpy
    y_true = targets.numpy()
    y_pred = preds.numpy()

    # Compute regression metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)

    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "PearsonR": float(pearson_r),
        "SpearmanR": float(spearman_r)
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

if 'distance_rbf' in config["training"]["data_path"]:
    print("Using test_distance_rbf_distogram_full_new dataset")
    data_path = r'test_distance_rbf_distogram_full_new'
elif 'distance_distance_random' in config["training"]["data_path"]:
    print("Using test_distance_distance_random dataset")
    data_path = r'test_distance_distance_random'
elif 'rewired_5000_distogram' in config["training"]["data_path"]:
    print("Using test_distance_rewired_5000_distogram dataset")
    data_path = r'test_distance_rewired_5000_distogram'
elif 'rewired_1000_distogram' in config["training"]["data_path"]:
    print("Using test_distance_rewired_1000_distogram dataset")
    data_path = r'test_distance_rewired_1000_distogram'
elif 'rewired_100_distogram' in config["training"]["data_path"]:
    print("Using test_distance_rewired_100_distogram dataset")
    data_path = r'test_distance_rewired_100_distogram'
elif 'rewired_10_distogram' in config["training"]["data_path"]:
    print("Using test_distance_rewired_10_distogram dataset")
    data_path = r'test_distance_rewired_10_distogram'
elif 'rewired_5000' in config["training"]["data_path"]:
    print("Using test_distance_rewired_5000 dataset")
    data_path = r'test_distance_rewired_5000'
elif 'rewired_1000' in config["training"]["data_path"]:
    print("Using test_distance_rewired_1000 dataset")
    data_path = r'test_distance_rewired_1000'
elif 'rewired_100' in config["training"]["data_path"]:
    print("Using test_distance_rewired_100 dataset")
    data_path = r'test_distance_rewired_100'
elif 'rewired_10' in config["training"]["data_path"]:
    print("Using test_distance_rewired_10 dataset")
    data_path = r'test_distance_rewired_10'
elif 'distance_distogram_new_correlation' in config["training"]["data_path"]:
    print("Using test_distance_distogram_new_correlation dataset")
    data_path = r'test_distance_distogram_new_correlation'
elif 'distance_distogram_full_correlation' in config["training"]["data_path"]:
    print("Using test_distance_distogram_full_correlation dataset")
    data_path = r'test_distance_distogram_full_correlation'
elif 'distance_distogram_new_new_new' in config["training"]["data_path"]:
    print("Using test_distance_distogram_new_new_new dataset")
    data_path = r'test_distance_distogram_new_new_new'
elif 'distance_distogram_new_new' in config["training"]["data_path"]:
    print("Using test_distance_distogram_new_new dataset")
    data_path = r'test_distance_distogram_new_new'
elif 'distance_distogram_full_new' in config["training"]["data_path"]:
    print("Using test_distance_distogram_full_new dataset")
    data_path = r'test_distance_distogram_full_new'
elif 'distance_distogram_new' in config["training"]["data_path"]:
    print("Using test_distance_distogram_new dataset")
    data_path = r'test_distance_distogram_new'
elif 'distogram_new_new_new' in config["training"]["data_path"]:
    print("Using test_distogram_new_new_new dataset")
    data_path = r'test_distogram_new_new_new'
elif 'distogram_new_new' in config["training"]["data_path"]:
    print("Using test_distogram_new_new dataset")
    data_path = r'test_distogram_new_new'
elif 'distogram_new' in config["training"]["data_path"]:
    print("Using test_distogram_new dataset")
    data_path = r'test_distogram_new'
elif 'distance_distogram_random_edge' in config["training"]["data_path"]:
    print("Using test_distance_distogram_random_edge dataset")
    data_path = r'test_distance_distogram_random_edge'
elif 'distances_correlation' in config["training"]["data_path"]:
    print("Using test_distance_correlation dataset")
    data_path = r'test_distance_correlation'
elif 'distance_distogram' in config["training"]["data_path"]:
    print("Using test_distance_distogram dataset")
    data_path = r'test_distance_distogram'
elif 'correlation_full' in config["training"]["data_path"]:
    print("Using test_correlation_full dataset")
    data_path = r'test_correlation_full'
elif 'distance' in config["training"]["data_path"]:
    print("Using test_distance dataset")
    data_path = r'test_distance'
elif 'correlation' in config["training"]["data_path"]:
    print("Using test_correlation dataset")
    data_path = r'test_correlation'
elif 'distogram' in config["training"]["data_path"]:
    print("Using test_distogram dataset")
    data_path = r'test_distogram'

# --- Load test dataset ---
test_dataset = ProteinGraphDataset(root=os.path.join(data_path, "test"), root_gt='affinity_gt_2')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model
data_example = test_dataset[0]
node_feat_dim = data_example.x.shape[1]
if "edge_type" not in data_example:
    num_relations = 1
else:
    num_relations = data_example.edge_type.max().item() + 1
if data_example.edge_attr is not None:
    edge_feat_dim = data_example.edge_attr.shape[1]  # your edge feature dimension

print(num_relations)

#model = RGCNNodeClassifier(
#            in_dim=node_feat_dim,
#            lr=initial_lr,
#            use_class_weights=True,
#        ).to(device)

if config["model"]["type"] == "RGCN":
    #model = RGCN_affinity_bn(
    #            in_channels=node_feat_dim,
    #            hidden_channels=config["model"]["hidden_channels"],
    #            num_relations=num_relations,
    #            depth=depth,
    #            dropout=config["model"]["dropout"]
    #        ).to(device)
    #model = RGCN_affinity_no_residual(
    #            in_channels=node_feat_dim,
    #            hidden_channels=config["model"]["hidden_channels"],
    #            num_relations=num_relations,
    #            depth=depth,
    #            dropout=config["model"]["dropout"]
    #        ).to(device)
    model = New_RGCN_complex_affinity(
                in_channels=node_feat_dim,
                hidden_channels=config["model"]["hidden_channels"],
                num_relations=num_relations,
                depth=depth,
                dropout=config["model"]["dropout"]
            ).to(device)
    #model = RGCN_affinity(
    #        in_channels=node_feat_dim,
    #        hidden_channels=config["model"]["hidden_channels"],
    #        num_relations=num_relations,
    #        depth=depth,
    #        dropout=config["model"]["dropout"]
    #    ).to(device)
elif config["model"]["type"] == "RGAT":
    model = RGAT_complex_affinity(
            in_channels=node_feat_dim,
            hidden_channels=config["model"]["hidden_channels"],
            num_relations=num_relations,
            edge_dim=edge_feat_dim,
            depth=depth,
            dropout=config["model"]["dropout"]
        ).to(device)
elif config["model"]["type"] == "EGNN":
        model = REGNN_complex_affinity(
            in_channels=node_feat_dim,
            hidden_channels=config["model"]["hidden_channels"],
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
    preds, targets, metrics = test_loop_rgcn(model, test_loader, device)
    #preds, targets, metrics = test_loop_rgcn_paper(model, test_loader, device)
elif config["model"]["type"] == "RGAT":
    preds, targets, metrics = test_loop_rgat(model, test_loader, device)
elif config["model"]["type"] == "EGNN":
    preds, targets, metrics = test_loop_EGNN(model, test_loader, device)
print(metrics)
# Save to JSON file
with open(os.path.join(model_path, 'metrics.json'), "w") as f:
    json.dump(metrics, f, indent=4)