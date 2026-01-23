import torch
import os
from tqdm import tqdm
import pandas as pd

# CSV with labels
df = pd.read_csv('pandora.csv')
labels = dict(zip(df['new_name'], df['dG_ML']))

# Full path to root folder containing train/val/test
root_dir = "/pasteur/appa/scratch/nportal/boltz/stability_prediction/pt_folder_distogram"
splits = ["train", "val", "test"]

# Output folder: one level above root_dir
parent_dir = os.path.dirname(root_dir)
output_dir = os.path.join(parent_dir, "packed_distogram")
os.makedirs(output_dir, exist_ok=True)

for split in splits:
    split_dir = os.path.join(root_dir, split)
    if not os.path.exists(split_dir):
        print(f"Skipping missing split: {split}")
        continue

    all_graphs = []  # will hold all graphs for this split

    # List all .pt files in the split folder
    pt_files = sorted([f for f in os.listdir(split_dir) if f.endswith(".pt") and f.startswith("protein_")])
    print(f"[{split}] Found {len(pt_files)} files")

    for f in tqdm(pt_files, desc=f"Packing {split}"):
        path = os.path.join(split_dir, f)
        # Load the graph
        graph = torch.load(path, weights_only=False)
        full_name = os.path.splitext(f)[0]  # e.g., protein_0_12
        all_graphs.append((full_name, graph))

    # Save one big file per split
    out_file = os.path.join(output_dir, f"{split}_all_graphs.pt")
    torch.save(all_graphs, out_file)
    print(f"[{split}] Saved {len(all_graphs)} graphs to {out_file}")
