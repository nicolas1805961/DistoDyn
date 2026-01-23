import os
import pickle
import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

# Directories
subfolders_dir = "subfolder_dir"
files_dir = "distances"

parser = PDBParser(QUIET=True)
distance_threshold = 0.1  # Angstroms, adjust if needed

for subfolder in os.listdir(subfolders_dir):
    subfolder_path = os.path.join(subfolders_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    pdb_id = subfolder  # subfolder name
    pdb_filename = f"{pdb_id}_pocket.pdb"
    pdb_path = os.path.join(subfolder_path, pdb_filename)
    
    if not os.path.exists(pdb_path):
        print(f"PDB file not found: {pdb_path}")
        continue

    # === Load .pkl file ===
    pkl_path = os.path.join(files_dir, f"{pdb_id}.pkl")
    if not os.path.exists(pkl_path):
        print(f".pkl file not found: {pkl_path}")
        continue

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    coords = data['coords']       # shape (N, 3)
    matrix = data['matrix']       # shape (N, N)

    # === Parse PDB file and extract Cα coordinates ===
    structure = parser.get_structure(pdb_id, pdb_path)
    pdb_ca_coords = []

    for model in structure:
        for chain in model:
            for res in chain:
                if 'CA' in res:
                    pdb_ca_coords.append(res['CA'].get_coord())

    pdb_ca_coords = np.array(pdb_ca_coords)

    # === Match PDB Cα to .pkl coords using nearest neighbor search ===
    tree = cKDTree(coords)
    distances, indices = tree.query(pdb_ca_coords, k=1)
    print(pdb_ca_coords)
    print(coords)

    # Keep only matches within the distance threshold
    matched_indices = [i for i, d in zip(indices, distances) if d < distance_threshold]

    filtered_coords = coords[matched_indices]
    filtered_matrix = matrix[np.ix_(matched_indices, matched_indices)]

    filtered_data = {
        'coords': filtered_coords,
        'matrix': filtered_matrix,
        'indices': matched_indices
    }

    print(f"{pdb_id}: {len(matched_indices)} residues matched in binding pocket")
    # Optionally save filtered .pkl
    # with open(os.path.join(subfolder_path, f"{pdb_id}_filtered.pkl"), "wb") as f:
    #     pickle.dump(filtered_data, f)
