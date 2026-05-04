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
import matplotlib.pyplot as plt




def visualize_attention_adj(edge_index, attn_weights, num_nodes, edge_type=None, ax=None, ax_mask=None, title="Attention Weights"):

    # Separate mask plot
    if edge_type is not None and ax_mask is not None:
        mask_matrix_distance = np.zeros((num_nodes, num_nodes))
        mask = edge_type == 0
        srcs = edge_index[0, mask]
        dsts = edge_index[1, mask]
        mask_matrix_distance[srcs, dsts] = 1.0

        mask_matrix_distogram = np.zeros((num_nodes, num_nodes))
        mask = edge_type == 1
        srcs = edge_index[0, mask]
        dsts = edge_index[1, mask]
        mask_matrix_distogram[srcs, dsts] = 1.0

    mask_type1 = (edge_type == 1)

    edge_index_filtered = edge_index[:, mask_type1]
    attn_weights_filtered = attn_weights[mask_type1]
    
    if attn_weights_filtered.ndim == 2:
        attn_weights_filtered = attn_weights_filtered.mean(axis=1)
    attn_weights_filtered = attn_weights_filtered.squeeze()

    attn_matrix_disto = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index_filtered.shape[1]):
        src = int(edge_index_filtered[0, i])
        dst = int(edge_index_filtered[1, i])
        attn_matrix_disto[src, dst] = attn_weights_filtered[i]
    
    mask_type0 = (edge_type == 0)

    edge_index_filtered = edge_index[:, mask_type0]
    attn_weights_filtered = attn_weights[mask_type0]
    
    if attn_weights_filtered.ndim == 2:
        attn_weights_filtered = attn_weights_filtered.mean(axis=1)
    attn_weights_filtered = attn_weights_filtered.squeeze()

    attn_matrix_distance = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index_filtered.shape[1]):
        src = int(edge_index_filtered[0, i])
        dst = int(edge_index_filtered[1, i])
        attn_matrix_distance[src, dst] = attn_weights_filtered[i]

    if ax is None:
        fig, ax = plt.subplots()

    final_mask = (mask_matrix_distogram - mask_matrix_distance)
    final_mask[final_mask < 0] = 0

    # Compute global max across both attention matrices
    global_max = max(attn_matrix_disto.max(), attn_matrix_distance.max())
    global_min = 0.0  # usually attention is non-negative

    # Plot distogram
    mesh = ax.pcolormesh(attn_matrix_disto, cmap="viridis",
                     vmin=global_min, vmax=global_max)
    plt.colorbar(mesh, ax=ax)
    ax.invert_yaxis()
    ax.set_title(f"{title} - distogram edge type")

    mesh2 = ax_mask.pcolormesh(attn_matrix_distance, cmap="viridis",
                           vmin=global_min, vmax=global_max)
    plt.colorbar(mesh2, ax=ax_mask)
    ax_mask.invert_yaxis()
    ax_mask.set_title(f"{title} - distance edge type")



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

            if "edge_type" not in batch:
                batch.edge_type = torch.zeros(batch.edge_index.size(1), dtype=torch.long, device=device)

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




def test_loop_rgat(model, loader, device, config, attention=False):
    model.eval()
    preds = []
    targets = []

    # --- NEW: storage for attention ---
    attentions = []
    edge_types_list = []
    num_nodes_list = []

    loader_iter = tqdm(loader, desc="Testing", leave=False)

    with torch.no_grad():
        for batch in loader_iter:
            batch = batch.to(device)

            if "edge_type" not in batch:
                batch.edge_type = torch.zeros(batch.edge_index.size(1), dtype=torch.long, device=device)

            if attention:
                out, attn = model(
                    batch.x.float(),
                    batch.edge_index,
                    batch.edge_type,
                    batch.edge_attr,
                    batch.batch
                )
                attentions.append(attn)  # list of layers
                edge_types_list.append(batch.edge_type.cpu())
                num_nodes_list.append(batch.num_nodes)
            else:
                out = model(
                    batch.x.float(),
                    batch.edge_index,
                    batch.edge_type,
                    batch.edge_attr,
                    batch.batch
                )

            # Predictions
            if config["training"]["loss"] == "CrossEntropyLoss":
                pred = out.argmax(dim=1)
            else:
                pred = (torch.sigmoid(out) >= 0.5)

            preds.append(pred.cpu())
            targets.append(batch.y.cpu())

    # Concatenate
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # Convert to numpy
    y_true = targets.numpy()
    y_pred = preds.numpy()

    if attention:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns

        for i, sample_nb in enumerate(range(410, 414)):
            ax = axes[0, i]
            ax_mask = axes[1, i]

            payload = attentions[sample_nb]

            attn_weights = torch.stack([aw for _, aw in payload], dim=0).mean(dim=0)

            aw = attentions[sample_nb][2]  # last GAT layer
            edge_index, _ = aw
            edge_index = edge_index.cpu().numpy()
            attn_weights = attn_weights.detach().cpu().numpy()

            # Average multi-head attention if needed
            if attn_weights.ndim > 1:
                attn_weights = attn_weights.mean(axis=1)

            num_nodes = num_nodes_list[sample_nb]
            edge_type = edge_types_list[sample_nb].numpy()

            # Filter only edge_type == 1

            # --- Remove top 10 edges ---
            #if len(attn_weights) > 10:
            #    top_idx = np.argsort(attn_weights)[-1000:]
            #    mask = np.ones(len(attn_weights), dtype=bool)
            #    mask[top_idx] = False
            #    edge_index = edge_index[:, mask]
            #    attn_weights = attn_weights[mask]
            #    edge_type = edge_type[mask]  # safe now

            visualize_attention_adj(
                edge_index,
                attn_weights,
                num_nodes,
                edge_type=edge_type,
                ax=ax,
                ax_mask=ax_mask,
                title=f"Sample {sample_nb}"
            )

        plt.tight_layout()
        plt.savefig("attention_grid_2.png", dpi=300, bbox_inches="tight")
        plt.show()

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

            if "edge_type" not in batch:
                batch.edge_type = torch.zeros(batch.edge_index.size(1), dtype=torch.long, device=device)

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
parser.add_argument("--attention", action="store_true", help="visualize attention")
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
folder = 'boltz2'

if folder == 'boltz1':
    if 'distance_distogram_0001' in config["training"]["data_path"]:
        print("Using distance_distogram_0001 dataset from boltz1")
        data_path = os.path.join(test_root_path, 'Boltz1','test_distance_distogram_0001')
    elif 'distance_distogram_00001' in config["training"]["data_path"]:
        print("Using distance_distogram_00001 dataset from boltz1")
        data_path = os.path.join(test_root_path, 'Boltz1','test_distance_distogram_00001')
    elif 'distance_rbf_distogram_full_0001' in config["training"]["data_path"]:
        print("Using distance_rbf_distogram_full_0001 dataset from boltz1")
        data_path = os.path.join(test_root_path, 'Boltz1','test_distance_rbf_distogram_full_0001')
else:
    if 'distance_rbf' in config["training"]["data_path"]:
        print("Using distance_rbf_distogram_full dataset")
        data_path = os.path.join(test_root_path, 'test_distance_rbf_distogram_full')
    elif 'distance_distance_random' in config["training"]["data_path"]:
        print("Using distance_distance_random dataset")
        data_path = os.path.join(test_root_path, 'test_distance_distance_random')
    elif '_md' in config["training"]["data_path"]:
        print("Using distance_distogram_md dataset")
        data_path = os.path.join(test_root_path, 'test_distance_distogram_md')
    elif '5000_2_distogram' in config["training"]["data_path"]:
        print("Using distance_rewired_5000_distogram dataset")
        data_path = os.path.join(test_root_path, 'test_distance_rewired_5000_distogram')
    elif '10000_2_distogram' in config["training"]["data_path"]:
        print("Using distance_rewired_10000_distogram dataset")
        data_path = os.path.join(test_root_path, 'test_distance_rewired_10000_distogram')
    elif '1000_2_distogram' in config["training"]["data_path"]:
        print("Using distance_rewired_1000_distogram dataset")
        data_path = os.path.join(test_root_path, 'test_distance_rewired_1000_distogram')
    elif '100_2_distogram' in config["training"]["data_path"]:
        print("Using distance_rewired_100_distogram dataset")
        data_path = os.path.join(test_root_path, 'test_distance_rewired_100_distogram')
    elif '10_2_distogram' in config["training"]["data_path"]:
        print("Using distance_rewired_10_2_distogram dataset")
        data_path = os.path.join(test_root_path, 'test_distance_rewired_10_distogram')
    elif 'rewired_5000' in config["training"]["data_path"]:
        print("Using distances_rewired_5000 dataset")
        data_path = os.path.join(test_root_path, 'test_distance_rewired_5000')
    elif 'rewired_10000' in config["training"]["data_path"]:
        print("Using distances_rewired_10000 dataset")
        data_path = os.path.join(test_root_path, 'test_distance_rewired_10000')
    elif 'rewired_1000' in config["training"]["data_path"]:
        print("Using distances_rewired_1000 dataset")
        data_path = os.path.join(test_root_path, 'test_distance_rewired_1000')
    elif 'rewired_100' in config["training"]["data_path"]:
        print("Using distances_rewired_100 dataset")
        data_path = os.path.join(test_root_path, 'test_distance_rewired_100')
    elif 'rewired_10' in config["training"]["data_path"]:
        print("Using distances_rewired_10 dataset")
        data_path = os.path.join(test_root_path, 'test_distance_rewired_10')
    elif 'distance_distogram_full_adj' in config["training"]["data_path"]:
        print("Using distance_distogram_full_adj dataset")
        data_path = os.path.join(test_root_path, 'test_distance_distogram_full_adj')
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
data_example = test_dataset[0]
node_feat_dim = data_example.x.shape[1]
if "edge_type" not in data_example:
    num_relations = 1
else:
    num_relations = data_example.edge_type.max().item() + 1
print(num_relations)
if data_example.edge_attr is not None:
    edge_feat_dim = data_example.edge_attr.shape[1] 
    print(edge_feat_dim) # your edge feature dimension

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
            dropout=config["model"]["dropout"],
            return_attention=args.attention,
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
    preds, targets, metrics = test_loop_rgat(model, test_loader, device, config, attention=args.attention)
elif config["model"]["type"] == "EGNN":
    preds, targets, metrics = test_loop_regnn(model, test_loader, device, config)
print(metrics)
# Save to JSON file
with open(os.path.join(model_path, 'metrics.json'), "w") as f:
    json.dump(metrics, f, indent=4)