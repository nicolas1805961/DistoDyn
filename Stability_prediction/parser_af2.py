import os
from tqdm import tqdm
from collections import defaultdict
import pickle

def parse_rec_file(file_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with open(file_path, 'r') as f:
        content = f.read()

    protein_blocks = content.strip().split('--')
    seen_sequences = {}  # maps sequence to assigned filename
    mut_dict = defaultdict(list)
    unique_counter = 0

    for block in tqdm(protein_blocks, desc="Parsing proteins"):
        if not block.strip():
            continue

        lines = block.strip().split('\n')
        data = {}
        for line in lines:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                data[key.strip()] = value.strip()

        sequence = data.get('aa_seq_wt')
        mut_type = data.get('mut_type')

        if not sequence or not mut_type:
            continue  # skip if missing info or mut_type is "wt"

        if sequence not in seen_sequences:
            name = f'protein_{unique_counter}'
            fasta_path = os.path.join(output_folder, f'{name}.fasta')
            with open(fasta_path, 'w') as fasta_file:
                fasta_file.write(f'>{name}\n{sequence}\n')
            seen_sequences[sequence] = name
            unique_counter += 1

        name = seen_sequences[sequence]
        mut_dict[name].append(mut_type)

    return dict(mut_dict)


# Example usage:
if __name__ == "__main__":
    rec_file_path = 'dG_wtname_clean_nan.rec'
    output_dir = 'fasta_sequences_boltz'
    mut_types_dict = parse_rec_file(rec_file_path, output_dir)

    for file_name, mut_types in mut_types_dict.items():
        print(f"{file_name}: {mut_types}")

    with open("mut_types_dict.pkl", "wb") as f:
        pickle.dump(mut_types_dict, f)
