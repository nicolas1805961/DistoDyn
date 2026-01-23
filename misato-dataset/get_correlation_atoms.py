import h5py
import numpy as np
import pickle
import os
import argparse
from src.data.processing.h5_to_pdb import get_maps, get_atom_name, get_entries, update_residue_indices, insert_TERS
import pickle
import pytraj as pt
#import matplotlib.pyplot as plt
import pandas as pd
import re
import glob
import logging


def list_subfolders(path):
    return [
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]


def load_pdbbind_index(file_path):
    records = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # skip comments and empty lines
            if not line or line.startswith("#") or line.startswith("="):
                continue

            # Example line:
            # 2r58  2.00  2007   2.00  Kd=10mM       // 2r58.pdf (MLY)
            parts = line.split()
            pdb_id = parts[0]
            resolution = float(parts[1])
            year = int(parts[2])
            neg_log_k = float(parts[3])
            kdki_str = parts[4]

            # extract Kd/Ki numeric value and unit
            m = re.match(r'(K[di])=([\d\.]+)([munp]?M)', kdki_str, re.IGNORECASE)
            if m:
                kind, value, unit = m.groups()
            else:
                kind, value, unit = None, None, None

            # extract ligand name (inside parentheses)
            ligand_match = re.search(r'\(([^)]+)\)', line)
            ligand = ligand_match.group(1) if ligand_match else None

            records.append({
                "pdb_id": pdb_id,
                "resolution": resolution,
                "year": year,
                "pKd_pKi": neg_log_k,
                "affinity_raw": kdki_str,
                "affinity_type": kind,
                "affinity_value": float(value) if value else None,
                "affinity_unit": unit,
                "ligand": ligand
            })

    return pd.DataFrame(records)


atomic_numbers_Map = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',19:'K',20:'Ca',34:'Se',35:'Br',53:'I'}

def compute_distance_matrix_all_chains(chains):
    """
    Concatenate all chains and compute a single distance matrix.
    """
    coords = []
    residues = []
    for chain in chains:
        residues.extend([res for res, _ in chain])
        coords.extend([c for _, c in chain])

    coords = np.array(coords)
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = np.linalg.norm(coords - coords[i], axis=1)
    return residues, dist_matrix, coords



def get_binding_mask_from_coords(protein_coords, ligand_coords, cutoff=10.0):
    """
    Returns a binary mask (shape: N1,) marking protein atoms within `cutoff` Å 
    of any ligand atom.

    Args:
        protein_coords (np.ndarray): shape (N1, 3) array of protein atom coordinates.
        ligand_coords (np.ndarray): shape (N2, 3) array of ligand atom coordinates.
        cutoff (float): distance threshold in Ångströms (default: 10.0).

    Returns:
        np.ndarray: binary mask of shape (N1,), with 1 where distance <= cutoff.
    """
    # Compute pairwise distances efficiently
    diff = protein_coords[:, None, :] - ligand_coords[None, :, :]
    dists = np.linalg.norm(diff, axis=2)  # shape (N1, N2)

    # Find protein atoms close to any ligand atom
    min_dists = np.min(dists, axis=1)  # shape (N1,)
    mask = (min_dists <= cutoff).astype(int)

    return mask, min_dists



def compute_distance_matrix(coords):
    """
    Compute the pairwise Euclidean distance matrix for a set of 3D coordinates.

    Args:
        coords (np.ndarray): Array of shape (N, 3)

    Returns:
        np.ndarray: Distance matrix of shape (N, N)
    """
    # Ensure input is a numpy array
    coords = np.asarray(coords)

    # Compute pairwise squared differences efficiently
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    return dist_matrix


def parse_sequences_from_pdb(pdb_file):
    sequences = []
    ca_coords_all = []  # list to store CA coordinates per chain

    current_residues = []
    current_ca_coords = []
    seen_residues = set()

    with open(pdb_file, "r") as f:
        for line in f:
            record = line[:6].strip()

            if record == "ATOM":
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                resseq  = line[22:26].strip()
                icode   = line[26].strip()
                resid   = (resseq, icode)

                # Add residue once per chain segment
                if resid not in seen_residues:
                    seen_residues.add(resid)
                    if resname in three_to_one:
                        current_residues.append(three_to_one[resname])
                    else:
                        current_residues.append('X')  # unknown/non-standard aa

                # Save CA coordinates
                if atom_name == "CA":
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    current_ca_coords.append([x, y, z])

            elif record == "TER":
                if current_residues:
                    sequences.append("".join(current_residues))
                    ca_coords_all.append(np.array(current_ca_coords))
                    current_residues = []
                    current_ca_coords = []
                    seen_residues = set()

        # catch last chain if no TER at end
        if current_residues:
            sequences.append("".join(current_residues))
            ca_coords_all.append(np.array(current_ca_coords))

    return sequences, ca_coords_all



# -----------------------------
# Parse single pdb_id argument
# -----------------------------
parser = argparse.ArgumentParser(description="Process a single PDB ID and generate Boltz2 YAML")
parser.add_argument("-pdb", "--pdb_id", type=str, required=True, help="PDB ID to process")
args = parser.parse_args()
pdb_id = args.pdb_id

print(' Processing single PDB ID: ', pdb_id)

logging.basicConfig(
    filename="error_log.txt",  # your log file
    filemode="a",                       # append mode (default)
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "CYX": "C", "HID": "H", "HIE": "H", "HIP": "H",
}


# Directory to store logs
#log_dir = "./logs"
#os.makedirs(log_dir, exist_ok=True)
#log_file = os.path.join(log_dir, f"processing_{pdb_id}.log")

#pdb_dir_downloaded = "pdb_dir_downloaded"
#os.makedirs(pdb_dir_downloaded, exist_ok=True)

## Option 1: Using logging module (recommended)
#logging.basicConfig(
#    filename=log_file,
#    filemode="w",  # overwrite each run
#    level=logging.INFO,
#    format="%(asctime)s - %(levelname)s - %(message)s"
#)

# Replace all print statements with logging.info
#logging.info(f"Processing single PDB ID: {pdb_id}")


# Open HDF5 file and get PDB IDs
h5_file_path = "data/MD/h5_files/tiny_md.hdf5"
#h5_file_path = "/pasteur/appa/scratch/nportal/MISATO/MD.hdf5"

f = h5py.File(h5_file_path)

h5_file_path_2 = "data/QM/h5_files/tiny_qm.hdf5"
#h5_file_path_2 = "/pasteur/appa/scratch/nportal/MISATO/QM.hdf5"

f_qm = h5py.File(h5_file_path_2)

# Initialize PDB downloader, parser, and polypeptide builder
#pdbl = PDBList()
#parser = PDBParser(QUIET=True)
#ppb = PPBuilder()

# Dictionary to store sequences
sequences = {}

#with open(r"src\data\processing\Maps\atoms_residue_map.pickle", "rb") as f2:
#    mapping_residue = pickle.load(f2)
#    #print(mapping_residue.keys())
#
#with open(r"src\data\processing\Maps\atoms_type_map.pickle", "rb") as f3:
#    mapping_atoms = pickle.load(f3)
#    #print(mapping_atoms.keys())


with open("src/data/processing/Maps/atoms_residue_map.pickle", "rb") as f2:
    mapping_residue = pickle.load(f2)
    #print(mapping_residue.keys())

with open("src/data/processing/Maps/atoms_type_map.pickle", "rb") as f3:
    mapping_atoms = pickle.load(f3)
    #print(mapping_atoms.keys())

#print(f[pdb_id].keys())
try:
    atoms_residue = f[pdb_id]['atoms_residue'][:]
except KeyError:
    logging.error(f"{pdb_id} not found in HDF5 file")
    exit(1)

atoms_type = f[pdb_id]['atoms_type']
trajectory_coordinates = f[pdb_id]['trajectory_coordinates']
atoms_number = f[pdb_id]['atoms_number']
atoms_list = [mapping_atoms[atom_id] for atom_id in atoms_type]
#print(atoms_type.shape)
#print(trajectory_coordinates.shape)
#print(atoms_list[0])

pdb_folder = 'pdb_dir'
pdb_path = os.path.join(pdb_folder, pdb_id + '_MD_frame0.pdb')
# trajectory_coordinates: shape (T, N_atoms, 3)
# pdb_file: path to the PDB of the first frame (your topology)
traj = pt.Trajectory(xyz=trajectory_coordinates, top=pdb_path)

# Align all frames to the first frame
pt.align(traj, ref=0)

# Get aligned coordinates back
trajectory_coordinates = traj.xyz.copy()  # shape (T, N_atoms, 3)


nb_ca = 0
lines = []
prev_lines = []

coords_list = {}   # collect coordinates of CA atoms
coords_list[0] = []
res_ids = {}  # keep track of which residue each CA belongs to
res_ids[0] = []
mask_list = {}   # collect coordinates of CA atoms
mask_list[0] = []

one_hot_data = {}   # collect coordinates of CA atoms
one_hot_data[0] = []

res_ids_chain = 0  # keep track of which residue each CA belongs to

atom_element = f[pdb_id]['atoms_element'][:]

_, atoms_type, atoms_number, atoms_residue, molecules_begin_atom_index = get_entries(pdb_id, f, 0)

seq_dict = {}
chain_nb = 0

residue_Map, typeMap, nameMap = get_maps("src/data/processing/Maps/")
residue_number = 1
residue_atom_index = 0
for i in range(len(atoms_type)):
    residue_atom_index +=1
    type_string = typeMap[atoms_type[i]]
    residue_name = residue_Map[atoms_residue[i]]
    atom_name = get_atom_name(i, atoms_number, residue_atom_index, residue_name, type_string, nameMap)

    x,y,z = trajectory_coordinates[0, i, 0],trajectory_coordinates[0, i, 1],trajectory_coordinates[0, i, 2]
    residue_number, residue_atom_index = update_residue_indices(residue_number, i, type_string, atoms_type, atoms_residue, residue_name, residue_atom_index,residue_Map, typeMap)
    residue_number, residue_atom_index, lines = insert_TERS(i, molecules_begin_atom_index, residue_number, residue_atom_index, lines)

    #prev_lines = list(lines)
    # Collect CA coordinates
    if atom_element[i] != 1:  # skip hydrogens
        coords_list[chain_nb].append(trajectory_coordinates[:, i, :])  # shape (T, 3)
        one_hot_data[chain_nb].append(np.array([atoms_type[i], atoms_residue[i]]))  # shape (T, 3)

        if 'CA' in atom_name:
            mask_list[chain_nb].append(1)
        else:
            mask_list[chain_nb].append(0)
    
    if prev_lines.count('TER') != lines.count('TER'):
        chain_nb += 1
        coords_list[chain_nb] = []
        one_hot_data[chain_nb] = []
        mask_list[chain_nb] = []
        prev_lines = list(lines)
    
    #print(atom_name)

coords_list_final = []
one_hot_final = []
disto_mask = []
ca_mask = []
ligand_mask = []
ligand_coords = []
for k, v in coords_list.items():
    one_hot_final.append(np.stack(one_hot_data[k]))  # shape (N, 2)
    chain_coords = np.stack(v, axis=1)
    coords_list_final.append(chain_coords)  # shape (T, N, 3)
    chain_ca_mask = np.array(mask_list[k]).astype(bool)  # shape (N,)
    aa_count = np.count_nonzero(chain_ca_mask)
    #print(f"Chain {k} has {len(v)} atoms.")
    if aa_count > 30 and k < len(coords_list)-1:  # only consider chains with >30 residues and ignore last chain (ligand)
        #print(f" Including chain {k} with {len(v)} atoms.")
        chain_disto_mask = np.ones(chain_coords.shape[1]).astype(bool)
    else:
        chain_disto_mask = np.zeros(chain_coords.shape[1]).astype(bool)
    if k == len(coords_list)-1:
        ligand_coords.append(chain_coords)
        ligand_mask.append(np.ones(chain_coords.shape[1]).astype(bool))
    else:
        ligand_mask.append(np.zeros(chain_coords.shape[1]).astype(bool))
    disto_mask.append(chain_disto_mask)  # shape (N,)
    ca_mask.append(chain_ca_mask)  # shape (N,)
# Convert list to array: shape (T, N, 3)
coords = np.concatenate(coords_list_final, axis=1) 
ligand_coords = np.concatenate(ligand_coords, axis=1) 
disto_mask = np.concatenate(disto_mask, axis=0)  # shape (N,)
ca_mask = np.concatenate(ca_mask, axis=0)  # shape (N,)
one_hot = np.concatenate(one_hot_final, axis=0)  # shape (N, 2)
ligand_mask = np.concatenate(ligand_mask, axis=0)  # shape (N,)

coords = np.round(coords, 3)
ligand_coords = np.round(ligand_coords, 3)

fused_mask = np.logical_and(disto_mask, ca_mask)

dist_matrix = compute_distance_matrix(coords[0])

binding_mask, binding_dist = get_binding_mask_from_coords(coords[0], ligand_coords[0], cutoff=10.0)
binding_mask = binding_mask.astype(bool)
fused_mask_filtered = fused_mask[binding_mask]
#displacements = aligned_coords - aligned_coords[0]

# --- Compute displacements relative to mean position per atom ---
mean_coords = coords.mean(axis=0)   # shape (N_atoms, 3)
displacements = coords - mean_coords  # shape (T, N_atoms, 3)

# --- Normalize displacements ---
norms = np.linalg.norm(displacements, axis=2, keepdims=True)
norms[norms == 0] = 1e-8
disp_norm = displacements / norms

# --- Compute correlation matrix ---
C = np.einsum('tia,tja->ij', disp_norm, disp_norm) / disp_norm.shape[0]

#binding_site_path = "/pasteur/appa/scratch/nportal/MISATO/binding_sites"
binding_site_path = "binding_site"
# search in train, test, val folders
file_list = glob.glob(os.path.join(binding_site_path, "*", pdb_id + ".npy"))

if not file_list:
    raise FileNotFoundError(f"{pdb_id}.npy not found in any folder")

bs_vector = np.load(file_list[0]).astype(bool)

#distogram_folder_path = "/pasteur/appa/scratch/nportal/MISATO/inference"
distogram_folder_path = "distogram"
# Open the file in binary read mode
with open(os.path.join(distogram_folder_path, f'boltz_results_{pdb_id}', 'predictions', f'{pdb_id}', f'distogram_{pdb_id}_model_0.pkl'), 'rb') as f:
    data = pickle.load(f)['distogram']
    softmax = data['softmax']
    bin_edges = data['bin_edges']

binding_dist_atoms = dist_matrix[binding_mask][:, binding_mask]
one_hot = one_hot[binding_mask]
if pdb_id == "1FO0":
    print(one_hot.shape)

noHindices_lig = np.where(f_qm[pdb_id]["atom_properties"]["atom_names"][()] != b"1")[0]
ligand_charge = f_qm[pdb_id]["atom_properties"]["atom_properties_values"][:, 7][noHindices_lig]

charges_prot = np.zeros(np.count_nonzero(ligand_mask[binding_mask] == 0))
charges = np.concatenate((charges_prot, ligand_charge))

assert charges.shape[0] == one_hot.shape[0], f"Charges shape {charges.shape} does not match one_hot shape {one_hot.shape}"

for i in range(len(one_hot[fused_mask_filtered])):
    assert typeMap[one_hot[fused_mask_filtered][i, 0]] == 'CX', f"Atom type mismatch at index {i}: {one_hot[fused_mask_filtered][i, 0]}"

binding_corr_atoms = C[binding_mask][:, binding_mask]

binding_distogram = softmax[bs_vector][:, bs_vector]

distogram = np.full((binding_corr_atoms.shape[0], binding_corr_atoms.shape[1], softmax.shape[-1]), np.nan)
distogram[np.ix_(fused_mask_filtered, fused_mask_filtered)] = binding_distogram
count = binding_dist_atoms.shape[0] * binding_dist_atoms.shape[1] - np.isnan(distogram.sum(-1)).sum()
assert count == np.sum(fused_mask_filtered) * np.sum(fused_mask_filtered)

corr = np.full((binding_corr_atoms.shape[0], binding_corr_atoms.shape[1]), np.nan)
corr[np.ix_(fused_mask_filtered, fused_mask_filtered)] = binding_corr_atoms[np.ix_(fused_mask_filtered, fused_mask_filtered)]
count = binding_dist_atoms.shape[0] * binding_dist_atoms.shape[1] - np.isnan(corr).sum()
assert count == np.sum(fused_mask_filtered) * np.sum(fused_mask_filtered)

df = load_pdbbind_index("INDEX_refined_data.2020")
affinity = df.loc[df["pdb_id"] == pdb_id.lower(), "pKd_pKi"].values[0]

test_folder = "coreset"
train_folder = "refined-set"

test_files = list_subfolders(test_folder)
train_files = list_subfolders(train_folder)



#fig, ax = plt.subplots(1, 2)
#ax[0].imshow(C, cmap='bwr', vmin=-1, vmax=1)
#ax[1].imshow(np.abs(C) > 0.3, cmap='grey')
#plt.show()

# Write YAML
output_dir = "./affinity_data"
#output_dir = "/pasteur/appa/scratch/nportal/MISATO/affinity_data"
subset = "train"

if pdb_id.lower() in test_files:
    subset = "test"
elif pdb_id.lower() in train_files:
    with open("val_MD.txt", "r", encoding="utf-8") as f:
        val_lines = f.readlines()
    val_lines = [line.strip() for line in val_lines]

    with open("train_MD.txt", "r", encoding="utf-8") as f:
        train_lines = f.readlines()
    train_lines = [line.strip() for line in train_lines]

    if pdb_id in train_lines:
        subset = "train"
    elif pdb_id in val_lines:
        subset = "val"
    

output_path = os.path.join(output_dir, subset)
os.makedirs(output_path, exist_ok=True)

to_save = {
    "correlation": corr,
    "distance": binding_dist_atoms,
    'distogram': distogram,
    'bin_edges': bin_edges,
    "coords": coords[0][binding_mask],
    "one_hot": one_hot,
    "charges": charges
}

# Save C to a .pkl file
with open(os.path.join(output_path, f"{pdb_id}.pkl"), "wb") as f:
    pickle.dump(to_save, f)

#print(C.shape)

# --- Threshold to define correlated edges ---
#edges_ca = np.argwhere(np.abs(C) > 0.3)
