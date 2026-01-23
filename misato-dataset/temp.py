import h5py
import numpy as np
import pickle

three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "CYX": "C", "HID": "H", "HIE": "H", "HIP": "H",
}

h5_file_path = "/pasteur/appa/scratch/nportal/MISATO/MD.hdf5"

f = h5py.File(h5_file_path, 'r')

sequences = {}
for idx, i in enumerate(range(len(f['1QAW']['molecules_begin_atom_index']))):
    print(f"molecule {idx} begins at atom index:")
    print(f['1QAW']['molecules_begin_atom_index'][i])
    sequences[idx] = []

with open("src/data/processing/Maps/atoms_residue_map.pickle", "rb") as f2:
    mapping_residue = pickle.load(f2)
    #print(mapping_residue.keys())

with open("src/data/processing/Maps/atoms_type_map.pickle", "rb") as f3:
    mapping_atoms = pickle.load(f3)
    #print(mapping_atoms.keys())

# atoms_residue: array of residue indices (like [0,0,0,1,1,2,2,...])
# atoms_type: array of atom types for each atom
# mapping_atoms: dict mapping atom_id → atom name
# mapping_residue: dict mapping residue_id → residue name

atoms_residue = f['1QAW']['atoms_residue']
atoms_type = f['1QAW']['atoms_type']

# Find the indices where a new residue starts
residue_change = np.concatenate(([True], np.diff(atoms_residue) != 0))

# Get the boundaries (start and end) of each residue
residue_starts = np.where(residue_change)[0]
residue_ends = np.append(residue_starts[1:], len(atoms_residue))

molecule_starts = np.array(f['1QAW']['molecules_begin_atom_index'])

for idx, (start, end) in enumerate(zip(residue_starts, residue_ends)):
    # Which molecule does this residue belong to?
    sequence_idx = np.searchsorted(molecule_starts, start, side="right") - 1
    residue_atoms = atoms_residue[start:end]
    atom_type_residue = atoms_type[start:end]

    residue_name3 = mapping_residue[residue_atoms[-1]]   # e.g. "GLN"
    print(start)
    print(end)
    print(residue_name3)
    residue_name1 = three_to_one.get(residue_name3, "X") # "Q", default "X" if unknown
    if residue_name1 == "X":
        print(f"Unknown residue {residue_name3}")
        if residue_name3 != 'MOL':
            exit(1)
    
    atoms_list = [mapping_atoms[atom_id] for atom_id in atom_type_residue]
    cx_count = atoms_list.count("CX")
    #if cx_count > 1:
    #    print(f"Multiple CX atoms in residue {three_to_one.get(mapping_residue[residue_atoms[-1]], 'X')}, residue starts at index {start}, molecule index {sequence_idx}")
    #    print(atoms_list)

    if residue_name3 == 'MOL':
        for atom in atoms_list:
            sequences[sequence_idx].append(atom)
    else:
        for i in range(cx_count):
            sequences[sequence_idx].append(residue_name1)