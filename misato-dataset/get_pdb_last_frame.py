import h5py
from Bio.PDB import PDBList, PDBParser, PPBuilder
import numpy as np
import pickle
import yaml  # for writing YAML files
import os
import subprocess
from tqdm import tqdm
import logging
import argparse
from src.data.processing.h5_to_pdb import get_maps, get_atom_name, get_entries, update_residue_indices, insert_TERS


atomic_numbers_Map = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',19:'K',20:'Ca',34:'Se',35:'Br',53:'I'}


def parse_sequences_from_pdb(pdb_file):
    sequences = []
    ca_coords_all = []       # list to store CA coordinates per chain
    atom_counts = []         # list to store number of atoms per chain

    current_residues = []
    current_ca_coords = []
    seen_residues = set()
    current_atom_count = 0

    with open(pdb_file, "r") as f:
        for line in f:
            record = line[:6].strip()

            if record == "ATOM":
                current_atom_count += 1
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
                    atom_counts.append(current_atom_count)
                    # Reset for next chain
                    current_residues = []
                    current_ca_coords = []
                    seen_residues = set()
                    current_atom_count = 0

        # catch last chain if no TER at end
        if current_residues:
            sequences.append("".join(current_residues))
            ca_coords_all.append(np.array(current_ca_coords))
            atom_counts.append(current_atom_count)

    return sequences, ca_coords_all, atom_counts



# --- Kabsch algorithm ---
def kabsch(P, Q):
    """
    Computes optimal rotation matrix R to align P -> Q using Kabsch algorithm.
    P, Q: (N, 3) arrays
    Returns: rotation matrix R (3x3), centroid of P, centroid of Q
    """
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid
    C = P_centered.T @ Q_centered
    V, S, Wt = np.linalg.svd(C)
    d = np.linalg.det(V @ Wt)
    D = np.diag([1, 1, d])
    R = V @ D @ Wt
    return R, P_centroid, Q_centroid

def align_frame(frame_coords, ref_coords):
    R, P_centroid, Q_centroid = kabsch(frame_coords, ref_coords)
    aligned = (frame_coords - P_centroid) @ R + Q_centroid
    return aligned



# -----------------------------
# Parse single pdb_id argument
# -----------------------------
parser = argparse.ArgumentParser(description="Process a single PDB ID and generate Boltz2 YAML")
parser.add_argument("-pdb", "--pdb_id", type=str, required=True, help="PDB ID to process")
args = parser.parse_args()
pdb_id = args.pdb_id

print(f"Processing single PDB ID: {pdb_id}")


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

pdb_dir = "pdb_dir_last_frame"
os.makedirs(pdb_dir, exist_ok=True)

#cmd = [
#    "python",
#    r"src\data\processing\h5_to_pdb.py",
#    "-s", pdb_id,
#    "-f", "99",
#    "-dMD", r"data\MD\h5_files\tiny_md.hdf5",
#    "-mdir", r"src\\data\\processing\\Maps\\",
#    "-o", "pdb_dir_last_frame",
#]

cmd = [
    "python3",
    "src/data/processing/h5_to_pdb.py",
    "-s", pdb_id,
    "-f", "99",
    "-dMD", "/pasteur/appa/scratch/nportal/MISATO/MD.hdf5",
    "-mdir", "src/data/processing/Maps/",
    "-o", "pdb_dir_last_frame",
]

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stderr)