import os
import pickle
import numpy as np

# Path to your main folder
main_folder = "/pasteur/appa/scratch/nportal/MISATO/Affinity/affinity_data"

# Path to save extracted affinity files
save_folder = "/pasteur/appa/scratch/nportal/MISATO/Affinity/affinity_gt_2"

# Ensure the save folder exists
os.makedirs(save_folder, exist_ok=True)

for split in ["train", "test", "val"]:
    subfolder = os.path.join(main_folder, split)
    if not os.path.exists(subfolder):
        print(f"⚠️ Subfolder '{split}' not found, skipping...")
        continue

    # Create corresponding subfolder in save_folder
    split_save_folder = os.path.join(save_folder, split)
    os.makedirs(split_save_folder, exist_ok=True)

    print(f"\n📂 Processing folder: {subfolder}")

    for filename in os.listdir(subfolder):
        if filename.endswith(".pkl"):
            filepath = os.path.join(subfolder, filename)
            print(f"Opening {filepath} ...")

            # Load the pickle file
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                aff = data['affinity']

            # Save `aff` as a .npy file
            base_name = os.path.splitext(filename)[0]  # remove .pkl extension
            save_path = os.path.join(split_save_folder, base_name + ".npy")
            np.save(save_path, aff)

            print(f"  → Saved affinity to {save_path}")

print("\n✅ Done extracting and saving all affinities as .npy files.")
