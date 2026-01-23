import os
import shutil

# === SETTINGS ===
source_dir = r"/pasteur/appa/homes/nportal/boltz/data/Sequences/fasta_sequences_boltz_mutated"
destination_dir = r"moved_fasta"
substring = "protein_0_"  # the text to look for in file names

# Make sure destination exists
os.makedirs(destination_dir, exist_ok=True)

# Loop over items in the source directory
for item in os.listdir(source_dir):
    item_path = os.path.join(source_dir, item)

    # Check if it's a file, contains the substring, and ends with .fasta
    if os.path.isfile(item_path) and substring in item and item.endswith(".fasta"):
        dest_path = os.path.join(destination_dir, item)
        print(f"Copying {item_path} -> {dest_path}")
        shutil.copy2(item_path, dest_path)  # copy file with metadata

print("Done.")
