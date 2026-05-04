import h5py
import numpy as np
import pickle
import os
import argparse
from src.data.processing.h5_to_pdb import get_maps, get_atom_name, get_entries, update_residue_indices, insert_TERS
import pickle
import pytraj as pt
#import matplotlib.pyplot as plt


atomic_numbers_Map = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',19:'K',20:'Ca',34:'Se',35:'Br',53:'I'}


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
#h5_file_path = "data/MD/h5_files/tiny_md.hdf5"
h5_file_path = "/pasteur/appa/scratch/nportal/MISATO/MD.hdf5"

f = h5py.File(h5_file_path)

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
atoms_residue = f[pdb_id]['atoms_residue']
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

ca_coords_list = {}   # collect coordinates of CA atoms
ca_coords_list[0] = []
ca_res_ids = {}  # keep track of which residue each CA belongs to
ca_res_ids[0] = []

ca_res_ids_chain = 0  # keep track of which residue each CA belongs to

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
    #print(type_string)
    atom_name = get_atom_name(i, atoms_number, residue_atom_index, residue_name, type_string, nameMap)

    x,y,z = trajectory_coordinates[0, i, 0],trajectory_coordinates[0, i, 1],trajectory_coordinates[0, i, 2]
    residue_number, residue_atom_index = update_residue_indices(residue_number, i, type_string, atoms_type, atoms_residue, residue_name, residue_atom_index,residue_Map, typeMap)
    residue_number, residue_atom_index, lines = insert_TERS(i, molecules_begin_atom_index, residue_number, residue_atom_index, lines)

    if prev_lines.count('TER') != lines.count('TER'):
        chain_nb += 1
        ca_coords_list[chain_nb] = []
        ca_res_ids[chain_nb] = []
    
    # Collect CA coordinates
    if "CA" in atom_name:
        ca_coords_list[chain_nb].append(trajectory_coordinates[:, i, :])  # shape (T, 3)
        ca_res_ids[chain_nb].append(residue_number)
    
    prev_lines = list(lines)
    #print(atom_name)

ca_coords_list_final = []
for k, v in ca_coords_list.items():
    print(f"Chain {k} has {len(v)} CA atoms.")
    if len(v) > 30 and k < len(ca_coords_list)-1:  # only consider chains with >30 residues and ignore last chain (ligand)
        print(f" Including chain {k} with {len(v)} CA atoms.")
        ca_coords_list_final.append(np.stack(v, axis=1))  # shape (T, N_CA, 3)
# Convert list to array: shape (T, N_CA, 3)
ca_coords = np.concatenate(ca_coords_list_final, axis=1) 

# ca_coords: shape (T, N_CA, 3)
T, N_CA, _ = ca_coords.shape

# Preallocate distance array: shape (T, N_CA, N_CA)
distances = np.zeros((T, N_CA, N_CA))

for t in range(T):
    # compute pairwise distance matrix for frame t
    diff = ca_coords[t, :, np.newaxis, :] - ca_coords[t, np.newaxis, :, :]  # shape (N_CA, N_CA, 3)
    distances[t] = np.linalg.norm(diff, axis=2)  # shape (N_CA, N_CA)

bin_edges = np.arange(0, 21, 1)  # 0–20 Å, 1 Å bins
n_bins = len(bin_edges) - 1

# Initialize distogram: shape (N_CA, N_CA, n_bins)
distogram = np.zeros((N_CA, N_CA, n_bins))

for i in range(N_CA):
    for j in range(N_CA):
        hist, _ = np.histogram(distances[:, i, j], bins=bin_edges, density=True)
        distogram[i, j] = hist  # probability distribution

print(distogram.shape)  # should be (N_CA, N_CA, n_bins)

#displacements = aligned_coords - aligned_coords[0]

# --- Compute displacements relative to mean position per atom ---
mean_coords = ca_coords.mean(axis=0)   # shape (N_atoms, 3)
displacements = ca_coords - mean_coords  # shape (T, N_atoms, 3)

# --- Normalize displacements ---
norms = np.linalg.norm(displacements, axis=2, keepdims=True)
norms[norms == 0] = 1e-8
disp_norm = displacements / norms

# --- Compute correlation matrix ---
C = np.einsum('tia,tja->ij', disp_norm, disp_norm) / disp_norm.shape[0]

#fig, ax = plt.subplots(1, 2)
#ax[0].imshow(C, cmap='bwr', vmin=-1, vmax=1)
#ax[1].imshow(np.abs(C) > 0.3, cmap='grey')
#plt.show()

# Write YAML
#output_dir = "./correlations"
output_dir = "/pasteur/appa/scratch/nportal/MISATO/correlations_2"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{pdb_id}.pkl")

to_save = {
    "distogram": distogram,
    "bin_edges": bin_edges,
    "matrix": C,
    "coords": ca_coords[0],
}

# Save C to a .pkl file
#with open(output_path, "wb") as f:
#    pickle.dump(to_save, f)
