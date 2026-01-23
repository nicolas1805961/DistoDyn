import os
import shutil

# Path to the file containing PDB IDs
missing_file = "missing_files.txt"

# Parent directory containing the folders
parent_dir = "/pasteur/appa/scratch/nportal/MISATO/inference"

# Read PDB IDs from file
with open(missing_file, 'r') as f:
    pdb_ids = [line.strip() for line in f if line.strip()]

# Loop over each PDB ID and delete the corresponding folder
for pdb_id in pdb_ids:
    folder_path = os.path.join(parent_dir, 'boltz_results_' + pdb_id)
    if os.path.isdir(folder_path):
        print(f"Deleting folder: {folder_path}")
        shutil.rmtree(folder_path)
    else:
        print(f"Folder does not exist: {folder_path}")
