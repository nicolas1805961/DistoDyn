import os
import pickle
import shutil
import logging
import argparse

# --- CONFIG ---
pkl_path = "split_dict.pkl"  # path to your pickle file
output_dir = "/pasteur/appa/scratch/nportal/boltz/stability_prediction"
log_file = "copy_folders.log"
# --------------

# Setup argument parser
parser = argparse.ArgumentParser(description="Copy a single Boltz results folder to train/test/validation split.")
parser.add_argument("input_path", type=str, help="Path to the folder to copy")
args = parser.parse_args()
src_folder = args.input_path

# Setup logging
logging.basicConfig(
    filename=log_file,
    filemode="a",  # overwrite each run; use "a" to append
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logging.info("Starting folder copy process.")

# Load the pickle file
with open(pkl_path, "rb") as f:
    splits_dict = pickle.load(f)
logging.info(f"Loaded pickle file: {pkl_path}")

# Make sure train/test/validation folders exist
for split in ["train", "test", "validation"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
logging.info(f"Ensured output split folders exist in {output_dir}")

# Extract folder name
folder = os.path.basename(src_folder.rstrip("/"))

if not folder.startswith("boltz_results_protein_"):
    logging.error(f"Folder {folder} does not start with 'boltz_results_protein_' — skipping")
    exit(1)

# Parse protein number and mutation number
try:
    parts = folder.split("_")
    if len(parts) == 5:
        protein_num = int(parts[3])
        mutation_num = int(parts[4])
    else:
        logging.error(f"Unexpected folder format: {folder}")
        exit(1)
except ValueError:
    logging.error(f"Failed to parse protein/mutation numbers from {folder}")
    exit(1)

# Lookup split from pkl dict
key = f'protein_{protein_num}'

if key not in splits_dict:
    logging.error(f"Protein {protein_num} not in pickle — skipping {folder}")
    exit(1)

splits_list = splits_dict[key]  # use the string key

if mutation_num >= len(splits_list):
    logging.error(f"Mutation index {mutation_num} out of range for protein {protein_num} — skipping {folder}")
    exit(1)

split_folder = splits_list[mutation_num]

if split_folder not in ["train", "test", "validation"]:
    logging.error(f"Invalid split '{split_folder}' for {folder} — skipping")
    exit(1)

# Copy the folder
dest_path = os.path.join(output_dir, split_folder, folder)
logging.info(f"Copying {src_folder} → {dest_path}")
try:
    shutil.copytree(src_folder, dest_path, dirs_exist_ok=True)
except Exception as e:
    logging.error(f"Failed to copy {folder}: {e}")

logging.info("Folder copy process completed.")
