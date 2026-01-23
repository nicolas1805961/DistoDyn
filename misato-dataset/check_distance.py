import os

# Path to the folder containing your files
folder_path = "slurm_OUT_distances"

# Loop over all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Only process files (skip subfolders)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if "Saved distance matrix" in content:
                print(f"[OK] '{filename}' contains the text.")
            else:
                print(f"[MISSING] '{filename}' does NOT contain the text.")
