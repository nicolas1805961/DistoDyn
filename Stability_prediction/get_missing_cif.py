from pathlib import Path

# Folder containing the CIF files
folder = Path("/pasteur/appa/scratch/nportal/boltz/stability_prediction/all_cif_files")
output_file = Path("missing_files.txt")

# List all CIF files in the folder
files = sorted(folder.glob("protein_*.cif"))

# Extract indices from filenames
indices = []
for f in files:
    stem = f.stem  # e.g., "protein_0"
    try:
        idx = int(stem.split("_")[1])
        indices.append(idx)
    except (IndexError, ValueError):
        continue

if not indices:
    raise ValueError("No valid protein files found in folder.")

# Find missing indices
min_idx = min(indices)
max_idx = max(indices)
all_indices = set(range(min_idx, max_idx + 1))
missing = sorted(all_indices - set(indices))

# Write missing filenames to output file
with open(output_file, "w") as f:
    for idx in missing:
        f.write(f"protein_{idx}.cif\n")

print(f"Missing files written to {output_file}")
