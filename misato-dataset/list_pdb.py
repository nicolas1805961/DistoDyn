import h5py

#h5_file_path = "/pasteur/appa/scratch/nportal/MISATO/MD.hdf5"
h5_file_path = "MD.hdf5"

with h5py.File(h5_file_path, "r") as f:
    pdb_ids = list(f.keys())

# Write to file
output_file = "pdb_ids.txt"
with open(output_file, "w") as f_out:
    for pdb_id in pdb_ids:
        f_out.write(pdb_id + "\n")

print(f"Wrote {len(pdb_ids)} PDB IDs to {output_file}")
