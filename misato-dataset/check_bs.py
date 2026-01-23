import os

# Path to the folder containing your files
folder_path = "slurm_OUT_binding_site"

# File to store missing filenames
missing_log = "missing_files.txt"

# Open the log file in append mode
with open(missing_log, 'a', encoding='utf-8') as log_file:
    # Loop over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Only process files (skip subfolders)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if "Binding vector saved to" in content:
                    print(f"[OK] '{filename}' contains the text.")
                else:
                    print(f"[MISSING] '{filename}' does NOT contain the text.")
                    log_file.write(os.path.basename(filename).split('.')[0] + "\n")
