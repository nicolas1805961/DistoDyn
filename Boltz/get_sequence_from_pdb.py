import os
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

def extract_sequence_for_chain(pdb_path):
    sequence = []
    seen_residues = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                res_name = line[17:20].strip()
                chain_id = line[21]
                res_seq = int(line[22:26].strip())
                res_id = (chain_id, res_seq)
                if res_id not in seen_residues:
                    seen_residues.add(res_id)
                    sequence.append(aa_dict.get(res_name, 'X'))
    return ''.join(sequence)

def extract_chain_id_from_filename(filename):
    match = re.search(r"__(\w)", filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract chain ID from filename: {filename}")

def pdb_to_fasta_with_chain_from_filename(folder_path, output_folder=None):
    if output_folder is None:
        output_folder = folder_path

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdb"):
            pdb_path = os.path.join(folder_path, filename)
            base_name = os.path.splitext(filename)[0]
            try:
                chain_id = extract_chain_id_from_filename(base_name)
            except ValueError as e:
                print(e)
                continue

            sequence = extract_sequence_for_chain(pdb_path)
            if not sequence:
                print(f"No sequence found for chain {chain_id} in {filename}")
                continue

            fasta_path = os.path.join(output_folder, f"{base_name}.fasta")
            with open(fasta_path, "w") as fasta_file:
                fasta_file.write(f">{chain_id}|protein\n{sequence}\n")
            print(f"Wrote {fasta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 1D sequences from PDB files and save as FASTA.")
    parser.add_argument("-i", "--input", required=True, help="Path to the folder containing PDB files.")
    parser.add_argument("-o", "--output", default=None, help="Folder to save FASTA files (default: same as input).")

    args = parser.parse_args()
    pdb_to_fasta_with_chain_from_filename(args.input, args.output)
