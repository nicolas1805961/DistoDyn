import os
import pickle
import pandas as pd
from Bio import SeqIO

# Directory containing FASTA files
fasta_dir = "/pasteur/appa/homes/nportal/boltz/data/Sequences/fasta_sequences_boltz_mutated"

# CSV file
csv_path = "pandora.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Dictionary to store mapping
name_to_fasta = {}

# Iterate over FASTA files
for fasta_file in os.listdir(fasta_dir):
    if fasta_file.endswith(".fasta") or fasta_file.endswith(".fa"):
        fasta_path = os.path.join(fasta_dir, fasta_file)

        # Read the sequence (assuming one record per FASTA)
        seq_record = next(SeqIO.parse(fasta_path, "fasta"))
        sequence = str(seq_record.seq)

        # Find matching row in CSV
        match_row = df[df["aa_seq"] == sequence]

        if not match_row.empty:
            csv_name = match_row["name"].values[0]
            name_to_fasta[csv_name] = os.path.basename(fasta_path)
        else:
            # Optional: handle unmatched sequences
            name_to_fasta[None] = os.path.basename(fasta_path)

# Print dictionary to check
print("Mapping from CSV name to FASTA file:")
for k, v in name_to_fasta.items():
    print(f"{k}: {v}")

# Save dictionary as pickle
with open("name_to_fasta.pkl", "wb") as f:
    pickle.dump(name_to_fasta, f)

print(f"\nSaved mapping for {len(name_to_fasta)} FASTA files to name_to_fasta.pkl")
