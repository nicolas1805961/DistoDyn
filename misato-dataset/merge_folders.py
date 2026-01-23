import os
import shutil
from tqdm import tqdm

def merge_folders(folder1, folder2, output_folder):
    """
    Merge two folders into `output_folder`.
    - If a subfolder or file exists in both, keep the version from folder2.
    - Otherwise, copy from folder1 or folder2 as appropriate.
    - Shows tqdm progress bar.
    """

    os.makedirs(output_folder, exist_ok=True)

    # Copy everything from folder1 first
    all_files1 = []
    for root, dirs, files in os.walk(folder1):
        for file in files:
            all_files1.append(os.path.join(root, file))

    for src_file in tqdm(all_files1, desc="Copying folder1"):
        rel_path = os.path.relpath(src_file, folder1)
        dst_file = os.path.join(output_folder, rel_path)
        dst_dir = os.path.dirname(dst_file)
        os.makedirs(dst_dir, exist_ok=True)
        if not os.path.exists(dst_file):  # don't overwrite, folder2 has priority
            shutil.copy2(src_file, dst_file)

    # Copy everything from folder2, overwriting existing folders/files
    all_files2 = []
    for root, dirs, files in os.walk(folder2):
        for file in files:
            all_files2.append(os.path.join(root, file))

    for src_file in tqdm(all_files2, desc="Copying folder2"):
        rel_path = os.path.relpath(src_file, folder2)
        dst_file = os.path.join(output_folder, rel_path)
        dst_dir = os.path.dirname(dst_file)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_file, dst_file)  # overwrite from folder2

    print(f"✅ Merged folders into: {output_folder}")


# Example usage:
merge_folders(
    "/pasteur/appa/scratch/nportal/MISATO/inference",
    "/pasteur/appa/scratch/nportal/MISATO/inference_2",
    "/pasteur/appa/scratch/nportal/MISATO/inference_merged"
)