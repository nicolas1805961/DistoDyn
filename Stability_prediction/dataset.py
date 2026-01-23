import os
import torch
from torch_geometric.data import Dataset, Data
from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import copy


import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import os
import copy
import random

class ProteinGraphDataset(Dataset):
    """
    Loads a single packed protein graph file containing all mutation graphs.
    Each item corresponds to one mutation graph with its correct label from the CSV.
    """

    def __init__(self, packed_file, labels_csv='pandora.csv', transform=None, pre_transform=None):
        super().__init__()
        self.packed_file = packed_file
        self.transform = transform
        self.pre_transform = pre_transform

        # Load CSV labels
        self.df = pd.read_csv(labels_csv)
        self.labels = dict(zip(self.df['new_name'], self.df['dG_ML']))

        # Load all graphs from the packed file
        # Each element is a tuple: (full_name, graph)
        self.graph_list = torch.load(packed_file, weights_only=False)
        print(f"Loaded {len(self.graph_list)} graphs from {packed_file}")

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        full_name, graph_data = copy.deepcopy(self.graph_list[idx])
        #print(f"Processing graph: {full_name}")

        # Assign the correct label
        if full_name not in self.labels:
            raise ValueError(f"Label for {full_name} not found in CSV")
        graph_data.y = torch.tensor([self.labels[full_name]], dtype=torch.float32)
        #print(f"Assigned label: {graph_data.y.item()}")

        # Apply optional transforms
        if self.transform:
            graph_data = self.transform(graph_data)

        # Ensure it's a PyG Data object
        if not isinstance(graph_data, Data):
            raise ValueError(f"Selected item {full_name} is not a PyG Data object")

        return graph_data





#class ProteinGraphDataset(Dataset):
#    """
#    Loads preprocessed protein graphs (.pt) into PyTorch Geometric format.
#
#    Example:
#    >>> dataset = ProteinGraphDataset(root="processed_graphs")
#    >>> data = dataset[0]
#    >>> print(data)
#    Data(x=[N, F], edge_index=[2, E], edge_attr=[E, 1], sequence='...', pdb_id='...')
#    """
#
#    def __init__(self, root, transform=None, pre_transform=None):
#        super().__init__(root, transform, pre_transform)
#
#        # List files
#        pt_files = [f for f in os.listdir(root) if f.endswith(".pt")]
#        self.df = pd.read_csv('pandora.csv')
#
#        # Strip extensions to compare
#        pt_basenames = {os.path.splitext(f)[0] for f in pt_files}
#
#        # Rebuild full paths with correct extensions
#        self.pt_files = [os.path.join(root, f"{name}.pt") for name in pt_basenames]
#        self.labels = dict(zip(self.df['new_name'], self.df['dG_ML']))
#
#        self.graphs = {}
#        for path in self.pt_files:
#            name = os.path.basename(path).split('.')[0]
#            self.graphs[name] = torch.load(path, weights_only=False)
#
#        print(len(self.pt_files))
#
#    def len(self):
#        return len(self.pt_files)
#
#    def get(self, idx):
#        # Load the .pt file
#        protein_name = os.path.basename(self.pt_files[idx]).split('.')[0]
#        graph_data = copy.deepcopy(self.graphs[protein_name])
#        graph_data.y = torch.tensor([self.labels[protein_name]], dtype=torch.float32)
#
#        # Make sure it's a Data object (it should be from our preprocessing)
#        if not isinstance(graph_data, Data):
#            raise ValueError(f"File {self.pt_files[idx]} does not contain a PyG Data object.")
#
#        return graph_data
    



class ProteinGraphDatasetTorch(TorchDataset):
    """
    Custom PyTorch Dataset that loads preprocessed protein graphs (.pt)
    into tensors suitable for EGNN input.

    Each .pt file should contain a dict or Data object with keys:
        - x: [L, 20]  (residue features)
        - coords: [L, 3]  (Cα coordinates)
        - edge_attr: [L, L, 64]  (distogram features)
        - y: [1] (label, e.g. ΔG)
    """

    def __init__(self, root):
        super().__init__()
        self.pt_files = sorted([
            os.path.join(root, f) for f in os.listdir(root)
            if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        graph_data = torch.load(self.pt_files[idx], weights_only=False)

        # Extract fields
        x = graph_data.x.float()                # [L, 20]
        coords = graph_data.coords.float()      # [L, 3]
        edges = graph_data.edge_attr.float()    # [L, L, 64]
        y = graph_data.y.float()                # [1]

        # Add batch dimension
        feats = x.unsqueeze(0)      # [1, L, 20]
        coors = coords.unsqueeze(0) # [1, L, 3]
        edges = edges.unsqueeze(0)  # [1, L, L, 64]
        mask = torch.ones(1, x.size(0), dtype=torch.bool)  # [1, L]

        return feats, coors, edges, mask, y




def protein_collate_fn(batch, L_max):
    feats_list, coors_list, edges_list, mask_list, y_list = zip(*batch)

    # Remove the first dimension (already batch=1 in dataset)
    feats_list = [f.squeeze(0) for f in feats_list]    # [L, d]
    coors_list = [c.squeeze(0) for c in coors_list]    # [L, 3]
    edges_list = [e.squeeze(0) for e in edges_list]    # [L, L, edge_dim]
    mask_list = [m.squeeze(0) for m in mask_list]      # [L]

    B = len(feats_list)
    d = feats_list[0].shape[1]
    feats_padded = torch.zeros(B, L_max, d)
    coors_padded = torch.zeros(B, L_max, 3)

    mask_padded = torch.zeros(B, L_max, dtype=torch.bool)

    for i in range(B):
        L = feats_list[i].shape[0]
        feats_padded[i, :L] = feats_list[i]
        coors_padded[i, :L] = coors_list[i]
        mask_padded[i, :L] = mask_list[i]

    # Pad edges
    edge_dim = edges_list[0].shape[-1]
    edges_padded = torch.zeros(B, L_max, L_max, edge_dim)
    for i in range(B):
        L = edges_list[i].shape[0]
        edges_padded[i, :L, :L, :] = edges_list[i]

    # Stack labels
    y = torch.stack(y_list, dim=0)

    return feats_padded, coors_padded, edges_padded, mask_padded, y
