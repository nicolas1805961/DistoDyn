import os
from tqdm import tqdm  # progress bar

# Path to the parent directory containing all boltz_results_protein_* folders
parent_dir = "/pasteur/appa/scratch/nportal/boltz/stability_prediction/results_mutated/"

# Path to log file
log_file = "cif_check.log"

# Collect all relevant folders first
folders = [
    folder_name for folder_name in os.listdir(parent_dir)
    if os.path.isdir(os.path.join(parent_dir, folder_name)) and folder_name.startswith("boltz_results_protein_")
]

with open(log_file, "w") as f:
    for folder_name in tqdm(folders, desc="Checking .cif files", unit="folder"):
        folder_path = os.path.join(parent_dir, folder_name)
        
        try:
            parts = folder_name.split("_")
            protein_num = parts[3]
            mutation_num = parts[4]
        except IndexError:
            f.write(f"Skipping folder with unexpected name format: {folder_name}\n")
            continue
        
        # Path to the predictions folder
        pred_folder = os.path.join(folder_path, "predictions", f"protein_{protein_num}_{mutation_num}")
        
        # Path to the cif file
        cif_file = os.path.join(pred_folder, f"protein_{protein_num}_{mutation_num}_model_0.cif")
        
        # Log existence
        if os.path.isfile(cif_file):
            f.write(f"Found: {cif_file}\n")
        else:
            f.write(f"Missing: {cif_file}\n")

print(f"Log saved to {log_file}")
