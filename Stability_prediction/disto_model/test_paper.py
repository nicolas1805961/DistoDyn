# =====================
# 4. Testing Loop
# =====================

import torch
from tqdm import tqdm
import os
from dataset import ProteinGraphDataset
from torch_geometric.loader import DataLoader
import yaml
from model import RGAT_bs, RGCN_bs  # The GNN model we defined
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import argparse
from pytorch_lightning import Trainer as lightningTrainer
from pytorch_lightning.loggers import TensorBoardLogger
from callback import get_callbacks
from model_paper import RGCNNodeClassifier


def test_loop_rgcn(model, loader, device):
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

            out = model(batch.x.float(), batch.edge_index, batch.edge_type)
            
            # Predicted class per node (0 or 1)
            #pred = out.argmax(dim=-1)
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

if 'distances_correlation' in config["training"]["data_path"]:
    data_path = 'test_distance_correlation'
elif 'distance_distogram' in config["training"]["data_path"]:
    data_path = 'test_distance_distogram'
elif 'correlation' in config["training"]["data_path"]:
    data_path = 'test_correlation'
elif 'distance' in config["training"]["data_path"]:
    data_path = 'test_distance'
elif 'distogram' in config["training"]["data_path"]:
    nb = config["training"]["data_path"].split('_')[-1]
    data_path = os.path.join('test_distogram', 'test_' + nb)

# --- Load test dataset ---
test_dataset = ProteinGraphDataset(root=os.path.join(data_path, "test"), root_gt='binding_sites')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model
node_feat_dim = test_dataset[0].x.shape[1]
edge_feat_dim = test_dataset[0].edge_attr.shape[1]  # your edge feature dimension
num_relations = test_dataset[0].edge_type.max().item() + 1

model = RGCNNodeClassifier(
            in_dim=node_feat_dim,
            lr=initial_lr,
            use_class_weights=True,
        ).to(device)

callbacks = get_callbacks(config, model_path)
# Instantiate and start the Trainer

logger = TensorBoardLogger(
        save_dir=model_path,        # directory for TensorBoard logs
        name="",         # unique subfolder for this run
        default_hp_metric=False # optional, avoids "hp_metric" spam
    )

trainer = lightningTrainer(
    callbacks=callbacks,
    max_epochs=epochs,
    logger=logger,
    #        log_every_n_steps=config.trainer.log_steps,
    #        val_check_interval=config.trainer.val_interval,
    #        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
    accelerator="gpu",
    devices=1,
)

predictions = trainer.test(
    model=model, ckpt_path=os.path.join(model_path, 'checkpoints', 'best.ckpt'), dataloaders=test_loader
)
#pickle.dump(predictions, open("data/output/predictions_" + config.name + ".pickle", "wb"))
# return config, predictions

#print(metrics)
## Save to JSON file
#with open(os.path.join(model_path, 'metrics.json'), "w") as f:
#    json.dump(metrics, f, indent=4)