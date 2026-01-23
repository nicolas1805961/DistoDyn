import os

# === Configuration ===
data_dir = "/pasteur/appa/scratch/nportal/MISATO/refined-set"
files_dir = "/pasteur/appa/scratch/nportal/MISATO/distances"
file_extension = ".pkl"  # change as needed, or set to "" if no extension

# === Logic ===
# Collect subfolder names (case-insensitive)
subfolders = [
    name for name in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, name))
]
subfolders_lower = {name.lower() for name in subfolders}

# Collect file base names (case-insensitive)
files = [
    os.path.splitext(f)[0]
    for f in os.listdir(files_dir)
    if os.path.isfile(os.path.join(files_dir, f))
]
files_lower = {name.lower() for name in files}

# Compare sets
missing = [sub for sub in subfolders if sub.lower() not in files_lower]

# === Results ===
print(f"Total subfolders: {len(subfolders)}")
print(f"Total files: {len(files)}")
print(f"Missing matches: {len(missing)}")

if missing:
    print("\nSubfolders with no matching file (case-insensitive):")
    for m in missing:
        print(f"  - {m}{file_extension}")
else:
    print("\n✅ All subfolders have matching files (case-insensitive).")
