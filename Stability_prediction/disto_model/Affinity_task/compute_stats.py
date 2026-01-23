import os
import numpy as np
from tqdm import tqdm

save_folder = "/pasteur/appa/scratch/nportal/MISATO/Affinity/affinity_gt_2"
splits = ["train", "val"]

all_affinities = []

for split in splits:
    split_folder = os.path.join(save_folder, split)
    if not os.path.exists(split_folder):
        print(f"⚠️ Folder '{split}' not found, skipping...")
        continue

    for filename in tqdm(os.listdir(split_folder)):
        if filename.endswith(".npy"):
            filepath = os.path.join(split_folder, filename)
            aff = np.load(filepath)
            # ensure it's 1D
            aff = np.atleast_1d(aff)
            all_affinities.append(aff)

# concatenate all
all_affinities = np.concatenate(all_affinities)

mean_aff = np.mean(all_affinities)
std_aff = np.std(all_affinities)

print(f"Mean affinity: {mean_aff:.4f}")
print(f"Std affinity: {std_aff:.4f}")
