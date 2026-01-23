import os
import pickle
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import argparse


def modify_amino_acid_a3m(file_path, output_path, csv_path, replace_map=None, insert_map=None, delete_positions=None):
    with open(file_path, "r") as f:
        lines = f.readlines()

    if not lines or not lines[0].startswith('>'):
        raise ValueError("Invalid A3M format. The first line must start with '>'.")

    new_lines = []
    inserted_positions = []

    seq_lines = []
    current_header = ""
    sequences = []

    # Parse sequences and headers
    for line in lines:
        if line.startswith(">"):
            if seq_lines:
                sequences.append(("".join(seq_lines), current_header))
                seq_lines = []
            current_header = line.rstrip()
        else:
            seq_lines.append(line.strip())
    if seq_lines:
        sequences.append(("".join(seq_lines), current_header))

    if not sequences:
        raise ValueError("No sequences found in A3M file.")

    query_seq, query_header = sequences[0]
    query_chars = list(query_seq)
    #print(query_chars)

    # Ungapped query sequence for position reference (uppercase + '-')
    ungapped_query = [c for c in query_chars if c.isupper() or c == '-']

    # Deletion
    if delete_positions:
        for pos, expected_aa in sorted(delete_positions.items(), reverse=True):
            if pos < 1 or pos > len(ungapped_query):
                raise ValueError(f"Delete position {pos} out of range.")
            if ungapped_query[pos - 1] != expected_aa:
                raise ValueError(f"Expected {expected_aa} at position {pos}, found {ungapped_query[pos - 1]}")

            count = 0
            for i, c in enumerate(query_chars):
                if c.isupper():
                    count += 1
                if count == pos:
                    # Remove the amino acid entirely
                    del query_chars[i]
                    # Also remove same column from all other sequences
                    for idx in range(1, len(sequences)):
                        seq_list = list(sequences[idx][0])
                        del seq_list[i]
                        sequences[idx] = ("".join(seq_list), sequences[idx][1])
                    break

    # Insertion (query insertion is uppercase; others get '-' at same positions)
    if insert_map:
        for pos, aa in sorted(insert_map.items(), reverse=True):
            if not aa.isalpha() or not aa.isupper():
                raise ValueError("Insertions must be uppercase letters.")
            if pos < 1 or pos > len(ungapped_query) + 1:
                raise ValueError(f"Insertion position {pos} out of range.")
            count = 0
            for i, c in enumerate(query_chars):
                if c.isupper():
                    count += 1
                if count == pos:
                    query_chars.insert(i, aa.upper())
                    inserted_positions.append(i)
                    break
            else:
                query_chars.append(aa.upper())
                inserted_positions.append(len(query_chars) - 1)

    # Replacement
    if replace_map:
        for pos, (expected, new_aa) in replace_map.items():
            if pos < 1 or pos > len(ungapped_query):
                raise ValueError(f"Replacement position {pos} out of range.")
            if not new_aa.isalpha() or not new_aa.isupper():
                raise ValueError("Replacement amino acids must be uppercase letters.")
            count = 0
            for i, c in enumerate(query_chars):
                if c.isupper():
                    count += 1
                if count == pos:
                    if query_chars[i] != expected:
                        raise ValueError(f"Expected {expected} at position {pos}, found {query_chars[i]}")
                    query_chars[i] = new_aa
                    break
    
    query_header = query_header + f'{csv_path}'
    new_lines.append(query_header + "\n")
    new_lines.append("".join(query_chars) + "\n")

    # Apply insertions (gaps) to other sequences only if insert_map is active
    for seq, header in sequences[1:]:
        seq_chars = list(seq)

        if insert_map:
            for idx in sorted(inserted_positions):
                if idx <= len(seq_chars):
                    seq_chars.insert(idx, '-')
                else:
                    seq_chars.append('-')

        # For deletions or replacements, other sequences remain unchanged

        new_lines.append(header + "\n")
        new_lines.append("".join(seq_chars) + "\n")

    with open(output_path, "w") as f:
        f.writelines(new_lines)




def modify_amino_acid_csv_fixed_key(file_path, output_path,
                                    replace_map=None,
                                    insert_map=None,
                                    delete_positions=None):
    df = pd.read_csv(file_path)

    if 'sequence' not in df.columns or 'key' not in df.columns:
        raise ValueError("CSV must contain 'sequence' and 'key' columns.")

    df_group = df[df['key'] == -1].copy()
    if df_group.empty:
        raise ValueError("No sequences found with key = -1")

    query_idx = df_group.index[0]
    query_chars = list(df.loc[query_idx, 'sequence'])

    def ungapped(chars):
        return [c for c in chars if c.isupper()]

    def ugpos_to_aln_idx(chars, pos_1based):
        count = 0
        for i, c in enumerate(chars):
            if c.isupper():
                count += 1
            if count == pos_1based:
                return i
        raise ValueError(f"Ungapped position {pos_1based} out of range.")

    # --- Deletion ---
    if delete_positions:
        for pos, expected_aa in sorted(delete_positions.items()):
            aln_i = ugpos_to_aln_idx(query_chars, pos)
            if query_chars[aln_i] != expected_aa:
                raise ValueError(f"Expected {expected_aa} at pos {pos}, found {query_chars[aln_i]}")
            
            # Remove from target sequence
            deleted_aa = query_chars.pop(aln_i)
            df.loc[query_idx, 'sequence'] = ''.join(query_chars)
            
            # Update other sequences in the group
            for idx in df_group.index:
                if idx == query_idx:
                    continue
                chars = list(df.loc[idx, 'sequence'])
                # Replace the character at the deletion position with lowercase deleted AA
                chars[aln_i] = deleted_aa.lower()
                df.loc[idx, 'sequence'] = ''.join(chars)

    # --- Insertion ---
    elif insert_map:
        for pos, aa in sorted(insert_map.items(), reverse=True):
            if not aa.isupper():
                raise ValueError("Inserted amino acid must be uppercase.")
            aln_i = ugpos_to_aln_idx(query_chars, pos) if pos <= len(ungapped(query_chars)) else len(query_chars)
            query_chars.insert(aln_i, aa)
            df.loc[query_idx, 'sequence'] = ''.join(query_chars)
            for idx in df_group.index:
                if idx != query_idx:
                    chars = list(df.loc[idx, 'sequence'])
                    chars.insert(aln_i, '-')
                    df.loc[idx, 'sequence'] = ''.join(chars)

    # --- Replacement ---
    elif replace_map:
        for pos, (expected_aa, new_aa) in sorted(replace_map.items()):
            aln_i = ugpos_to_aln_idx(query_chars, pos)
            if query_chars[aln_i] != expected_aa:
                raise ValueError(f"Expected {expected_aa} at pos {pos}, found {query_chars[aln_i]}")
            query_chars[aln_i] = new_aa
        df.loc[query_idx, 'sequence'] = ''.join(query_chars)

    # --- No change ---
    else:
        pass  # MSA stays the same

    df.to_csv(output_path, index=False)




#input_path = fake_parent_boltz\boltz_results_protein_163
#input_path_fasta_dir = 'fasta_sequences_boltz'
#output_path = 'temp_boltz'
#output_path_fasta = 'temp_fasta_boltz'

# Hard-coded paths
output_path = '/pasteur/appa/scratch/nportal/boltz/stability_prediction/boltz2_sp_mutated'
output_path_fasta = '/pasteur/appa/homes/nportal/boltz/data/Sequences/fasta_sequences_boltz_mutated'
input_path_fasta_dir = '/pasteur/appa/homes/nportal/boltz/data/Sequences/fasta_sequences_boltz'
mut_types_dict_path = "mut_types_dict.pkl"

def main():
    parser = argparse.ArgumentParser(description="Mutate protein MSA and FASTA")
    parser.add_argument("--input_path", required=True, help="Path to folder containing the 'msa' folder")
    args = parser.parse_args()

    directory = Path(output_path_fasta)
    directory.mkdir(parents=True, exist_ok=True)

    with open(mut_types_dict_path, "rb") as f:
        loaded_dict = pickle.load(f)

    protein_name = '_'.join(Path(args.input_path).name.split('_')[2:])
    print(protein_name)

    replace_list = loaded_dict[protein_name]

    print({i: j for i, j in enumerate(replace_list)})

    msa_dir = os.path.join(args.input_path, 'msa')
    path1 = os.path.join(msa_dir, f'{protein_name}_0.csv')
    fasta_path_in = os.path.join(input_path_fasta_dir, protein_name + '.fasta')

    for idx, change in tqdm(enumerate(replace_list), total=len(replace_list), desc="Processing mutations"):

        out_msa_dir = os.path.join(output_path, Path(args.input_path).name + f'_{idx}', 'msas')

        out_path1 = os.path.join(out_msa_dir, f'{protein_name}_0.csv')
        fasta_path_out = os.path.join(output_path_fasta, protein_name + f'_{idx}' + '.fasta')

        insertion_map = None
        deletion_map = None
        replacement_map = None

        if 'ins' in change:
            insertion_map = {}
            position = int(change[4:])
            new_aa = change[3]
            insertion_map[position] = new_aa
        elif 'del' in change:
            deletion_map = {}
            position = int(change[4:])
            new_aa = change[3]
            deletion_map[position] = new_aa
        elif 'wt' not in change:
            if ':' in change:
                replacement_list = change.split(':')
            else:
                replacement_list = [change]
            replacement_map = {}
            for replacement in replacement_list:
                position = int(replacement[1:-1])
                new_aa = replacement[-1]
                prev_aa = replacement[0]
                replacement_map[position] = (prev_aa, new_aa)

        os.makedirs(out_msa_dir, exist_ok=True)

        if os.path.exists(path1):
            modify_amino_acid_csv_fixed_key(path1, out_path1, replace_map=replacement_map, insert_map=insertion_map, delete_positions=deletion_map)
        if os.path.exists(fasta_path_in):
            print(fasta_path_out)
            modify_amino_acid_a3m(fasta_path_in, fasta_path_out, out_path1, replace_map=replacement_map, insert_map=insertion_map, delete_positions=deletion_map)

if __name__ == "__main__":
    main()