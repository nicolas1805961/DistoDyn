import h5py
import numpy as np
import pickle
import os
import argparse
from src.data.processing.h5_to_pdb import get_maps, get_atom_name, get_entries, update_residue_indices, insert_TERS

# -----------------------------
# Kabsch / alignment not needed
# -----------------------------

# -----------------------------
# Parse single pdb_id argument
# -----------------------------
parser = argparse.ArgumentParser(description="Extract Cα adaptability per residue for a single PDB ID")
parser.add_argument("-pdb", "--pdb_id", type=str, required=True, help="PDB ID to process")
args = parser.parse_args()
pdb_id = args.pdb_id

print('Processing single PDB ID:', pdb_id)

# Load atom / residue maps
residue_Map, typeMap, nameMap = get_maps("src/data/processing/Maps/")

# Open HDF5 file
h5_file_path = "/pasteur/appa/scratch/nportal/MISATO/MD_preprocessed.hdf5"
#h5_file_path = r"data\MD\h5_files\tiny_md_out.hdf5"
f = h5py.File(h5_file_path, "r")

# Read entries for PDB
_, atoms_type, atoms_number, atoms_residue, molecules_begin_atom_index = get_entries(pdb_id, f, 0)
adaptability = f[pdb_id]['feature_atoms_adaptability'][:]  # shape (n_atoms,)

# Initialize per-chain Cα storage
ca_scores_list = {}
ca_scores_list[0] = []
ca_res_ids_list = {}
ca_res_ids_list[0] = []

lines = []
prev_lines = []
chain_nb = 0

residue_number = 1
residue_atom_index = 0

for i in range(len(atoms_type)):
    residue_atom_index += 1
    type_string = typeMap[atoms_type[i]]
    residue_name = residue_Map[atoms_residue[i]]
    atom_name = get_atom_name(i, atoms_number, residue_atom_index, residue_name, type_string, nameMap)

    residue_number, residue_atom_index = update_residue_indices(residue_number, i, type_string, atoms_type, atoms_residue, residue_name, residue_atom_index,residue_Map, typeMap)

    residue_number, residue_atom_index, lines = insert_TERS(
        i, molecules_begin_atom_index, residue_number, residue_atom_index, lines
    )

    # Detect new chain
    if prev_lines.count('TER') != lines.count('TER'):
        chain_nb += 1
        ca_scores_list[chain_nb] = []
        ca_res_ids_list[chain_nb] = []

    # Collect Cα adaptability
    if "CA" in atom_name:
        ca_scores_list[chain_nb].append(adaptability[i])
        ca_res_ids_list[chain_nb].append(residue_number)

    prev_lines = list(lines)

print(len(ca_scores_list), "chains found in PDB.")

# Concatenate all chains into a single 1D vector
ca_adapt_vector = []
for k in sorted(ca_scores_list.keys()):
    if len(ca_scores_list[k]) > 30 and k < len(ca_scores_list)-1:  # filter short chains and 
        print(f" Including chain {k} with {len(ca_scores_list[k])} CA atoms.")
        ca_adapt_vector.extend(ca_scores_list[k])

bs_path = "/pasteur/appa/scratch/nportal/MISATO/binding_sites"

with open(os.path.join(bs_path, pdb_id + '.pkl'), "rb") as f:
    bs_vector = pickle.load(f)

assert len(bs_vector) == len(ca_adapt_vector), \
    f"Mismatch in number of residues between binding site vector ({len(bs_vector)}) and Cα adaptability vector ({len(ca_adapt_vector)}) for PDB ID {pdb_id}."

ca_adapt_vector = np.array(ca_adapt_vector, dtype=float)
print(f"Extracted Cα adaptability vector of length {len(ca_adapt_vector)}")

# Read a text file and store each line in a list
with open("test_MD.txt", "r", encoding="utf-8") as f:
    test_lines = f.readlines()
test_lines = [line.strip() for line in test_lines]

with open("val_MD.txt", "r", encoding="utf-8") as f:
    val_lines = f.readlines()
val_lines = [line.strip() for line in val_lines]

with open("train_MD.txt", "r", encoding="utf-8") as f:
    train_lines = f.readlines()
train_lines = [line.strip() for line in train_lines]

output_dir = "/pasteur/appa/scratch/nportal/MISATO/adaptability"

if pdb_id in test_lines:
    save_path = os.path.join(output_dir, "test", pdb_id + ".npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
elif pdb_id in val_lines:
    save_path = os.path.join(output_dir, "val", pdb_id + ".npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
elif pdb_id in train_lines:
    save_path = os.path.join(output_dir, "train", pdb_id + ".npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save to pickle
os.makedirs(output_dir, exist_ok=True)
np.save(save_path, ca_adapt_vector)
#with open(save_path, "wb") as pf:
#    pickle.dump(ca_adapt_vector, pf)

print(f"Saved Cα adaptability vector to {save_path}")
