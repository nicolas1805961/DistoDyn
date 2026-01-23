import os
import shutil

# === SETTINGS ===
source_dir = r"/pasteur/appa/scratch/nportal/boltz/stability_prediction/results_mutated"
destination_dir = r"moved"
substring = "protein_0_"  # the text to look for in subfolder names

# Make sure destination exists
os.makedirs(destination_dir, exist_ok=True)

# Loop over items in the source directory
for item in os.listdir(source_dir):
    item_path = os.path.join(source_dir, item)
    
    # Check if it's a directory and contains the substring
    if os.path.isdir(item_path) and substring in item:
        dest_path = os.path.join(destination_dir, item)
        print(f"Copying {item_path} -> {dest_path}")
        shutil.copytree(item_path, dest_path, dirs_exist_ok=True)

print("Done.")
