import os
from glob import glob

# Paths
distogram_folder = "/pasteur/appa/scratch/nportal/MISATO/inference"
file_folder = "/pasteur/appa/scratch/nportal/MISATO/binding_sites"
output_file = "missing_disto.txt"

# Collect distogram file names (basenames without extension)
distogram_files = [os.path.basename(f).split('_')[1]
                   for f in glob(os.path.join(distogram_folder, "**", "distogram*.pkl"), recursive=True)
                   if os.path.isfile(f)]

print(distogram_files)

# Collect file basenames in the other folder
file_basenames = {os.path.splitext(f)[0] for f in os.listdir(file_folder)
                  if os.path.isfile(os.path.join(file_folder, f))}

# Check which distogram files do NOT have a matching file
missing = [d for d in file_basenames if d not in distogram_files]

# Write results
with open(output_file, "w") as f:
    for m in missing:
        f.write(m + "\n")

print(f"Found {len(missing)} distogram files without matching files → saved to {output_file}")
