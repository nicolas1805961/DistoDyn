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

#h5_file_path_2 = "data/QM/h5_files/tiny_qm.hdf5"
h5_file_path_2 = "/pasteur/appa/scratch/nportal/MISATO/QM.hdf5"
f_qm = h5py.File(h5_file_path_2)

sequences = {}

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

#ligand_indices = np.where(atoms_residue[:]==0)[0]
#print(ligand_indices)
#print(len(atoms_residue))
#print(atoms_type.shape)
#print(trajectory_coordinates.shape)
#print(atoms_list[0])

##print("Number of atoms:")
##print(len(atoms_residue))
#sequences = {}
#for idx, i in enumerate(range(len(f[pdb_id]['molecules_begin_atom_index']))):
#    #print(f"molecule {idx} begins at atom index:")
#    #print(f[pdb_id]['molecules_begin_atom_index'][i])
#    sequences[idx] = []
#
## atoms_residue: array of residue indices (like [0,0,0,1,1,2,2,...])
## atoms_type: array of atom types for each atom
## mapping_atoms: dict mapping atom_id → atom name
## mapping_residue: dict mapping residue_id → residue name
#
## Find the indices where a new residue starts
#residue_change = np.concatenate(([True], np.diff(atoms_residue) != 0))
#
## Get the boundaries (start and end) of each residue
#residue_starts = np.where(residue_change)[0]
#residue_ends = np.append(residue_starts[1:], len(atoms_residue))
#
#molecule_starts = np.array(f[pdb_id]['molecules_begin_atom_index'])
#
## Loop over residues
#for idx, (start, end) in enumerate(zip(residue_starts, residue_ends)):
#    # Which molecule does this residue belong to?
#    sequence_idx = np.searchsorted(molecule_starts, start, side="right") - 1
#    residue_atoms = atoms_residue[start:end]
#    atom_type_residue = atoms_type[start:end]
#
#    residue_name3 = mapping_residue[residue_atoms[-1]]   # e.g. "GLN"
#    residue_name1 = three_to_one.get(residue_name3, "X") # "Q", default "X" if unknown
#    if residue_name1 == "X":
#        print(f"Unknown residue {residue_name3}")
#        if residue_name3 != 'MOL':
#            exit(1)
#    
#    atoms_list = [mapping_atoms[atom_id] for atom_id in atom_type_residue]
#    cx_count = atoms_list.count("CX")
#    #if cx_count > 1:
#    #    print(f"Multiple CX atoms in residue {three_to_one.get(mapping_residue[residue_atoms[-1]], 'X')}, residue starts at index {start}, molecule index {sequence_idx}")
#    #    print(atoms_list)
#
#    if residue_name3 == 'MOL':
#        for atom in atoms_list:
#            sequences[sequence_idx].append(atom)
#    else:
#        for i in range(cx_count):
#            sequences[sequence_idx].append(residue_name1)

    #for residue_id, atom_id in zip(residue_atoms, atom_type_residue):
    #    print(mapping_residue[residue_id])
    #    print(mapping_atoms[atom_id])
    #print(atoms_list)
    #if 'CA' not in atoms_list and 'CX' not in atoms_list:
    #    print(f"No CA in residue {mapping_residue[residue_atoms[-1]]}, residue starts at index {start}, molecule index {sequence_idx}")
    #    print(atoms_list)
    #    continue
    #else:
    #    print(f"CA found in residue {mapping_residue[residue_atoms[-1]]}, residue starts at index {start}, molecule index {sequence_idx}")

# Convert to strings
#for k in sequences.keys():
#    if isinstance(sequences[k], list):
#        sequences[k] = ''.join(sequences[k])
#
#unique_seqs = set(sequences.values())
#
#if len(unique_seqs) > 1:
#    print("Different sequences detected!")
#    for k, v in sequences.items():
#        print(f"Sequence {k}: {v}")



# Command as a list
#cmd = [
#    "python",
#    r"src\data\processing\h5_to_pdb.py",
#    "-s", pdb_id,
#    "-dMD", r"data\MD\h5_files\tiny_md.hdf5",
#    "-mdir", r"src\\data\\processing\\Maps\\",
#    "-o", "pdb_dir",
#]

cmd = [
    "python3",
    "src/data/processing/h5_to_pdb.py",
    "-s", pdb_id,
    "-dMD", "/pasteur/appa/scratch/nportal/MISATO/MD.hdf5",
    "-mdir", "src/data/processing/Maps/",
    "-o", "pdb_dir_last_frame",
]

#cmd = [
#    "python3",
#    "src/data/processing/h5_to_pdb.py",
#    "-s", pdb_id,
#    "-dMD", "MD.hdf5",
#    "-mdir", "src/data/processing/Maps/",
#    "-o", "pdb_dir",
#]

# Run the command
result = subprocess.run(cmd, capture_output=True, text=True)


# Initialize parser and peptide builder
parser = PDBParser(QUIET=True)
ppb = PPBuilder()
#
## Path to your PDB file
pdb_file_path = os.path.join(pdb_dir, pdb_id + "_MD_frame0.pdb")

h5_to_pdb_sequences, ca_oords, atom_count_list = parse_sequences_from_pdb(pdb_file_path)

noHindices_lig = np.where(f_qm[pdb_id]["atom_properties"]["atom_names"][()] != b"1")[0]
ligand_atom_charge = f_qm[pdb_id]["atom_properties"]["atom_properties_values"][:, 7]
ligand_charge = ligand_atom_charge[noHindices_lig]
#logging.info('Method 1)')
#for i, s in enumerate(h5_to_pdb_sequences, 1):
#    logging.info(f"molecule {i}: {s}")

#structure = parser.get_structure(pdb_id, pdb_file_path)


#pdbl = PDBList(verbose=False)
#
## Download PDB file in PDB format
#pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir_downloaded, file_format='pdb', overwrite=True)
#
## Parse the structure
#structure = parser.get_structure(pdb_id, pdb_file)
#
## Build sequences from all chains
#seqs = []
#for model in structure:
#    for chain in model:
#        for pp in ppb.build_peptides(chain):
#            seq = str(pp.get_sequence())
#            #print(len(seq))
#            seqs.append(seq)
#
## Flag to check if any match exists
#match_found = False
#
#for i, s in enumerate(seqs):
#    l = min(len(s), len(h5_to_pdb_sequences[0]))
#    if s[:l] == h5_to_pdb_sequences[0][:l]:
#        match_found = True
#        #logging.info(f"Sequence match found for {pdb_id}, chain {i}")
#        break  # stop at the first match
#
#if not match_found:
#    print(f"No sequences in seqs match h5_to_pdb_sequences[0] for {pdb_id}")
#    print("h5_to_pdb_sequences[0]:", h5_to_pdb_sequences[0])
#    for i, s in enumerate(seqs):
#        print(f"seqs[{i}] (length {len(s)}): {s}")
#    raise ValueError(f"No sequences in seqs match h5_to_pdb_sequences[0] for {pdb_id}")

# Generate Boltz2 YAML
yaml_data = {"sequences": [], "constraints": [], "templates": [], "properties": []}
to_save = False

for seq_idx, seq in enumerate(h5_to_pdb_sequences):

    print(f"Processing sequence {seq_idx} with length {len(seq)} and atom count {atom_count_list[seq_idx]}")

    if atom_count_list[seq_idx] == len(ligand_atom_charge):
        if seq_idx < len(h5_to_pdb_sequences) - 1 and len(h5_to_pdb_sequences) > 1 and atom_count_list[-1] != len(ligand_atom_charge):
            print(f"Skipping sequence {seq_idx} as it matches ligand length")
            to_save = True
            continue

    if not 'X' in seq and len(seq) > 30:  # skip sequences with unknown residues or too short

        entry = {
            "protein": {
                "id": str(seq_idx),
                "sequence": seq,
                "msa": "",       # optional, can provide path to .a3m
                "cyclic": False  # set True if cyclic
            }
        }

        yaml_data["sequences"].append(entry)

if to_save and len(yaml_data["sequences"]) > 0:

    # Constraints, templates, properties (optional)
    yaml_data["constraints"] = []
    yaml_data["templates"] = []
    yaml_data["properties"] = []

    # Write YAML
    output_dir = "./boltz_inputs_yaml_2"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pdb_id}.yaml")
    with open(output_path, "w") as f_yaml:
        yaml.dump(yaml_data, f_yaml, sort_keys=False)

    print(f"Written Boltz YAML for {pdb_id} -> {output_path}")
    #logging.info('#'*100)

    # Output
    output_dir_fasta = "./boltz_inputs_fasta_2"
    os.makedirs(output_dir_fasta, exist_ok=True)
    fasta_path = os.path.join(output_dir_fasta, f"{pdb_id}.fasta")

    with open(fasta_path, "w") as f:
        # Write protein sequences
        for idx, seq in enumerate(h5_to_pdb_sequences):
            if atom_count_list[idx] == len(ligand_atom_charge):
                continue
            #if idx == len(h5_to_pdb_sequences) - 1:
            #    continue
            if not 'X' in seq and len(seq) > 30:  # skip sequences with unknown residues or too short
                chain_id = chr(ord("A") + idx)  # A, B, C...
                msa_path = ""  # or "empty" if you want single sequence mode
                f.write(f">{chain_id}|protein|{msa_path}\n")
                f.write(seq + "\n")
