import os
import json
import argparse
import re

# 3-letter to 1-letter amino acid mapping
aa_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z', 'XLE': 'J', 'UNK': 'X'
}

def extract_chain_id_from_filename(filename):
    match = re.search(r"__(\w)", filename)
    if not match:
        raise ValueError(f"Could not extract chain ID from filename: {filename}")
    return match.group(1)

def extract_sequence_for_chain(pdb_path):
    sequence = []
    seen_residues = set()

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                res_name = line[17:20].strip()
                res_seq = int(line[22:26].strip())
                res_uid = (res_seq)
                if res_uid not in seen_residues:
                    seen_residues.add(res_uid)
                    sequence.append(aa_dict.get(res_name, 'X'))

    return ''.join(sequence)

def generate_json_from_pdb(pdb_path, output_dir):
    base_filename = os.path.basename(pdb_path)
    pdb_name = os.path.splitext(base_filename)[0]

    try:
        chain_id = extract_chain_id_from_filename(pdb_name)
    except ValueError as e:
        print(e)
        return

    sequence = extract_sequence_for_chain(pdb_path)
    if not sequence:
        print(f"❌ No sequence found for chain {chain_id} in {pdb_name}")
        return

    json_data = {
        "name": pdb_name,
        "sequences": [
            {
                "protein": {
                    "id": [chain_id],
                    "sequence": sequence
                }
            }
        ],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1
    }

    output_path = os.path.join(output_dir, f"{pdb_name}.json")
    with open(output_path, "w") as json_file:
        json.dump(json_data, json_file, indent=2)
    print(f"✅ Wrote {output_path}")

def process_pdb_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdb"):
            pdb_path = os.path.join(input_folder, filename)
            generate_json_from_pdb(pdb_path, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDB files to JSON format for AlphaFold3.")
    parser.add_argument("-i", "--input", required=True, help="Path to folder containing PDB files.")
    parser.add_argument("-o", "--output", required=True, help="Path to folder to save JSON files.")

    args = parser.parse_args()
    process_pdb_folder(args.input, args.output)
