import os
import pickle
import numpy as np
from Bio.PDB import PDBParser

# Paths to the folders
folder1 = "binding_sites"  # .npy files
folder2 = "distances"  # .pkl files
folder3 = "correlations"  # .pkl files
folder4 = "distograms"  # .pkl files
folder5 = "pdb_dir"  # .pkl files

def get_last_chain_atoms(pdb_file):
    """
    Returns a list of non-hydrogen atoms (atom_name, res_name, res_id, x, y, z)
    from the last chain in a PDB file (chains separated by TER).
    """
    atoms = []
    current_chain_atoms = []

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                res_id = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                # Skip hydrogen atoms
                if atom_name.startswith("H"):
                    continue

                current_chain_atoms.append((atom_name, res_name, res_id, x, y, z))

            elif line.startswith("TER"):
                # When TER is encountered, save current chain
                if current_chain_atoms:
                    atoms = current_chain_atoms  # this will keep the last finished chain
                    current_chain_atoms = []

    # After finishing file, check if last chain has atoms (no TER at the end)
    if current_chain_atoms:
        atoms = current_chain_atoms

    return atoms

# Initialize PDB parser
parser = PDBParser(QUIET=True)

# Iterate over all .npy files in folder1
for filename in os.listdir(folder1):
    if filename.endswith(".npy"):
        filename_no_ext = os.path.basename(filename).split('.')[0]
        # Construct file paths
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename.replace(".npy", ".pkl"))
        file3_path = os.path.join(folder3, filename.replace(".npy", ".pkl"))
        file4_path = os.path.join(folder4, f'boltz_results_{filename_no_ext}', 'predictions', f'{filename_no_ext}', f'distogram_{filename_no_ext}_model_0.pkl')
        file5_path = os.path.join(folder5, f'{filename_no_ext}_MD_frame0.pdb')

        # Check if all files exist
        if all(os.path.isfile(p) for p in [file1_path, file2_path, file3_path, file4_path, file5_path]):
            print(f"Processing {filename}")

            # Load files
            data1 = np.load(file1_path)
            with open(file2_path, "rb") as f2:
                data2 = pickle.load(f2)
                print(data2.keys())
            with open(file3_path, "rb") as f3:
                data3 = pickle.load(f3)
                print(data3.keys())
            with open(file4_path, "rb") as f4:
                data4 = pickle.load(f4)
                print(data4['distogram'].keys())

            # Load pdb and extract non-hydrogen atoms in the last chain
            last_chain_atoms = get_last_chain_atoms(file5_path)

            # Example: print number of atoms
            print(f"Loaded {len(last_chain_atoms)} non-hydrogen atoms from last chain")

            # Do something with the data
            print(type(data1), type(data2), type(data3), type(data4))
        else:
            print(f"Skipping {filename}: file missing in one of the folders")
