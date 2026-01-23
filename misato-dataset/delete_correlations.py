import os

# Path to the file containing PDB IDs
missing_file = "missing_files.txt"

# Parent directory containing the files
parent_dir = "/pasteur/appa/scratch/nportal/MISATO/correlations"

# Read PDB IDs from file
with open(missing_file, 'r') as f:
    pdb_ids = [line.strip() for line in f if line.strip()]

# Loop over each PDB ID and delete the corresponding file
for pdb_id in pdb_ids:
    file_path = os.path.join(parent_dir, pdb_id + '.pkl')
    if os.path.isfile(file_path):
        print(f"Deleting file: {file_path}")
        os.remove(file_path)
    else:
        print(f"File does not exist: {file_path}")
