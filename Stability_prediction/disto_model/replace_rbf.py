import os
import torch
import pickle
from tqdm import tqdm

def rbf_expand(distances, num_kernels=64, d_min=0.0, d_max=1.0):
    """
    Standard RBF expansion for normalized distances.
    """
    if distances.dim() == 1:
        distances = distances.unsqueeze(-1)

    device = distances.device
    centers = torch.linspace(d_min, d_max, num_kernels, device=device)
    delta = centers[1] - centers[0]
    gamma = 1.0 / (delta ** 2)

    return torch.exp(-gamma * (distances - centers) ** 2)


def fix_edge_features(pt_root, pkl_root, num_kernels=64):
    """
    Fix RBF edge features for all .pt files in train/val/test subfolders,
    using .pkl files from a single flat folder.
    """
    subfolders = ["train", "val", "test"]

    for sub in subfolders:
        pt_folder = os.path.join(pt_root, sub)

        if not os.path.exists(pt_folder):
            print(f"Skipping {sub}: folder not found")
            continue

        pt_files = [f for f in os.listdir(pt_folder) if f.endswith(".pt")]

        for fname in tqdm(pt_files, desc=f"Fixing {sub} graphs"):
            pt_path = os.path.join(pt_folder, fname)
            pkl_path = os.path.join(pkl_root, fname.replace(".pt", ".pkl"))

            if not os.path.exists(pkl_path):
                print(f"Warning: {fname} has no corresponding .pkl file, skipping")
                continue

            # Load PyG Data
            data = torch.load(pt_path, map_location="cpu", weights_only=False)

            # Load distance matrix from corresponding .pkl
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
                dmat_distance = torch.from_numpy(pkl_data['matrix'])

            # Mask edges where edge_type == 0
            mask = (data.edge_type == 0)
            if not mask.any():
                continue  # nothing to fix

            # Get edge indices for these edges
            src, dst = data.edge_index[:, mask]

            # Compute normalized distances for these edges
            dists = dmat_distance[src, dst]       # raw distances
            dists = 1.0 - dists / 10.0            # normalize
            dists = dists[:, None]                # shape (num_edges, 1)

            # Recompute RBF features
            new_features = rbf_expand(dists, num_kernels=num_kernels, d_max=1.0)

            # Ensure dtype and device match existing edge_attr
            new_features = new_features.to(dtype=data.edge_attr.dtype, device=data.edge_attr.device)

            # Replace edge_attr for these edges
            data.edge_attr[mask] = new_features

            # Overwrite the .pt file
            torch.save(data, pt_path)

    print("Done. All subfolders processed.")


# --- Example usage ---
#pt_root = 'temp_dummy'  # your .pt folder root with train/val/test
#pkl_root = 'distance_dir'  # flat folder containing all .pkl files


# Example usage
pt_root = '/pasteur/appa/scratch/nportal/MISATO/Binding_site/pt_folder_distance_rbf_distogram_full_random'
pkl_root = '/pasteur/appa/scratch/nportal/MISATO/distances'
fix_edge_features(pt_root, pkl_root, num_kernels=64)
