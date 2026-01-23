import os
import shutil
import random

# === CONFIGURATION ===
source_dir = "pt_folder"  # folder containing .pt files
dest_dir = "split"        # will contain train/ and val/ folders
train_ratio = 0.8         

# === SETUP ===
train_dir = os.path.join(dest_dir, "train")
val_dir = os.path.join(dest_dir, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# === GET ALL .PT FILES ===
pt_files = [f for f in os.listdir(source_dir) if f.endswith(".pt")]

# Shuffle to randomize the split
random.shuffle(pt_files)

# Split into train and val
split_idx = int(len(pt_files) * train_ratio)
train_files = pt_files[:split_idx]
val_files = pt_files[split_idx:]

# === COPY FILES ===
def copy_files(files, destination):
    for f in files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(destination, f)
        if not os.path.exists(dst):
            print(f"Copying {f} -> {destination}")
            shutil.copy2(src, dst)
        else:
            print(f"Skipping {f}, already exists in {destination}")

copy_files(train_files, train_dir)
copy_files(val_files, val_dir)

print(f"\nDone! {len(train_files)} train files, {len(val_files)} val files.")
