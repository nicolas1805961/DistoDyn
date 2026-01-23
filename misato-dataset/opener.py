import numpy as np
import os
import pickle

# Path to your .npy file
file_path = "/pasteur/appa/scratch/nportal/MISATO/Affinity/binding_sites"

filename = '1FO0'

if not filename.endswith(".npy"):
    filename += ".npy"

for split in ["train", "val", "test"]:
    path = os.path.join(file_path, split, filename)
    if os.path.exists(path):
        print(f"✅ Found {filename} in {split}/")
        data = np.load(path, allow_pickle=True)
        print(data.shape)




file_path_2 = "/pasteur/appa/scratch/nportal/MISATO/binding_sites"

filename = '1FO0'

if not filename.endswith(".npy"):
    filename += ".npy"

for split in ["train", "val", "test"]:
    path = os.path.join(file_path_2, split, filename)
    if os.path.exists(path):
        print(f"✅ Found {filename} in {split}/")
        data = np.load(path, allow_pickle=True)
        print(data.shape)



file_path_3 = "/pasteur/appa/scratch/nportal/MISATO/inference_merged/boltz_results_1FO0/predictions/1FO0/distogram_1FO0_model_0.pkl"

with open(file_path_3, "rb") as f:  # "rb" = read binary
    data = pickle.load(f)
    print(data['distogram']['softmax'].shape)

file_path_4 = "/pasteur/appa/scratch/nportal/MISATO/inference/boltz_results_1FO0/predictions/1FO0/distogram_1FO0_model_0.pkl"

with open(file_path_4, "rb") as f:  # "rb" = read binary
    data = pickle.load(f)
    print(data['distogram']['softmax'].shape)


file_path_5 = "/pasteur/appa/scratch/nportal/MISATO/inference_2/boltz_results_1FO0/predictions/1FO0/distogram_1FO0_model_0.pkl"

with open(file_path_5, "rb") as f:  # "rb" = read binary
    data = pickle.load(f)
    print(data['distogram']['softmax'].shape)
