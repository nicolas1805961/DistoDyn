import os
import pickle
from Bio import SeqIO

# === CONFIGURATION ===
fasta_dir = "moved_fasta"   # directory with fasta files
rec_file = "dG_wtname_clean_nan.rec"     # .rec file
output_pkl = "fasta_dGmean_mapping.pkl"  # output pickle

# === PARSE .rec FILE INTO RECORDS ===
def parse_rec(filepath):
    records = []
    with open(filepath, "r") as f:
        block = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "--":
                if block:
                    records.append(block)
                    block = {}
            elif "=" in line:
                key, value = line.split("=", 1)
                block[key.strip()] = value.strip()
        if block:  # catch last record if no trailing "--"
            records.append(block)
    return records

rec_records = parse_rec(rec_file)
print(f"Loaded {len(rec_records)} records from {rec_file}")

# Build lookup: sequence -> dGmean
seq_to_dgmean = {}
for rec in rec_records:
    if "aa_seq" in rec and "dGmean" in rec:
        seq_to_dgmean[rec["aa_seq"]] = float(rec["dGmean"])

# === SCAN FASTA FILES AND BUILD MAPPING ===
fasta_dg_mapping = {}

for fasta_file in os.listdir(fasta_dir):
    if fasta_file.endswith(".fasta") or fasta_file.endswith(".fa"):
        fasta_path = os.path.join(fasta_dir, fasta_file)

        # Each FASTA has only one sequence
        record = next(SeqIO.parse(fasta_path, "fasta"))
        seq = str(record.seq)

        base_name = os.path.splitext(fasta_file)[0]

        if seq in seq_to_dgmean:
            fasta_dg_mapping[base_name] = seq_to_dgmean[seq]
            print(f"[MATCH] {base_name} -> dGmean={seq_to_dgmean[seq]}")
        else:
            print(f"[NO MATCH] {base_name}")

# === SAVE MAPPING AS PICKLE ===
with open(output_pkl, "wb") as f:
    pickle.dump(fasta_dg_mapping, f)

print(f"Saved mapping for {len(fasta_dg_mapping)} sequences to {output_pkl}")
