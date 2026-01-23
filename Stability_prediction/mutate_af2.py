import os
import pickle
import re
import shutil
from tqdm import tqdm
import logging
from pathlib import Path

logging.basicConfig(
    filename='mutation_log',
    filemode='w',  # Append mode
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
)

def modify_amino_acid_sto(file_path, output_path, replace_map=None, insert_map=None, delete_positions=None):
    with open(file_path, "r") as f:
        lines = f.readlines()

    blocks = []
    current_block = []

    for line in lines:
        current_block.append(line)
        if line.startswith("#=GC RF"):
            blocks.append(current_block)
            current_block = []
    # if any leftover lines after last block
    if current_block:
        blocks.append(current_block)

    # Extract and concatenate the query sequence parts (first sequence line in each block that is not a comment)
    query_seq_parts = []
    query_headers = []
    query_line_indices = []  # track line indices within blocks for query lines

    for b_idx, block in enumerate(blocks):
        # Skip non-alignment blocks (like '#=GC RF' lines or empty lines)
        if len(block) == 1 and (block[0].strip() == "" or block[0].startswith("#=GC RF") or block[0].strip() == "//"):
            continue

        # find first sequence line (not starting with '#' or blank)
        for line_idx, line in enumerate(block):
            if line.startswith("#") or line.strip() == "":
                continue
            match = re.match(r"^(\S+)(\s+)([A-Za-z\-]+)$", line.rstrip())
            if match:
                name, spacing, seq = match.groups()
                assert len(seq) <= 200
                # Assuming first sequence line is the query line
                query_seq_parts.append(seq)
                query_headers.append((b_idx, line_idx))
                query_line_indices.append((b_idx, line_idx))
                break

    # Concatenate query sequence
    concatenated_query = "".join(query_seq_parts)
    query_chars = list(concatenated_query)

    # Strip gaps for position-based operations
    ungapped_query = [c for c in query_chars if c != '-']

    split_positions = [len(part) for part in query_seq_parts]
    # Apply modifications ONLY ONCE on the concatenated query sequence

    block_insertion_nb = 100000000
    block_insertion_idx = 1000000000

    # Deletion
    if delete_positions:
        #print('deletion')
        for pos, aa in sorted(delete_positions.items(), reverse=True):
            if pos < 1 or pos > len(ungapped_query):
                raise ValueError(f"Delete position {pos} out of range.")
            if ungapped_query[pos - 1] != aa:
                raise ValueError(f"Expected {aa} at position {pos}, found {ungapped_query[pos - 1]}")
            # find real index in query_chars
            count = 0
            for i, c in enumerate(query_chars):
                if c != '-':
                    count += 1
                if count == pos:
                    query_chars[i] = '-'
                    break

    # Insertion
    if insert_map:
        #print('insertion')
        for pos, aa in sorted(insert_map.items(), reverse=True):
            if pos < 1 or pos > len(ungapped_query) + 1:
                raise ValueError(f"Insertion position {pos} out of range.")
            # find real index in query_chars
            count = 0
            inserted = False
            for i, c in enumerate(query_chars):
                if c != '-':
                    count += 1
                if count == pos:
                    block_insertion_nb = i // 200
                    block_insertion_idx = i % 200
                    query_chars.insert(i, aa)
                    inserted = True
                    break
            if not inserted:
                #print('append')
                query_chars.append(aa)

    # Replacement
    if replace_map:
        #print('replacement')
        for pos, (expected, new_aa) in replace_map.items():
            if pos < 1 or pos > len(ungapped_query):
                raise ValueError(f"Replacement position {pos} out of range.")
            count = 0
            for i, c in enumerate(query_chars):
                if c != '-':
                    count += 1
                if count == pos:
                    if query_chars[i] != expected:
                        raise ValueError(f"Expected {expected} at position {pos}, found {query_chars[i]}")
                    query_chars[i] = new_aa
                    break

    modified_query = "".join(query_chars)

    # Extend the last block to cover all characters
    length_drift = len(modified_query) - sum(split_positions)
    if length_drift > 0:
        split_positions[-1] += length_drift

    # Re-split modified query into blocks, preserving original block lengths
    reconstructed_blocks_query = []
    start = 0
    for length in split_positions:
        reconstructed_blocks_query.append(modified_query[start:start + length])
        start += length

    ## Compute insertion columns from original insert_map
    #inserted_columns = []
    #if insert_map:
    #    # For each insertion position, find corresponding column index in concatenated_query
    #    for pos in insert_map:
    #        count = 0
    #        for idx, c in enumerate(query_chars):
    #            if c != '-':
    #                count += 1
    #            if count == pos:
    #                print(idx)
    #                inserted_columns.append(idx)
    #                break

    # Now rebuild blocks, replacing query sequences and adding gaps for insertions in other sequences
    new_lines = []

    count = [0] * len(blocks)
    for b_idx, block in enumerate(blocks):
        for line in block:
            if not (line.startswith('#') or line.strip() == "" or line.strip() == "//"):
                count[b_idx] += 1

    assert all([count[k] == count[k-1] for k in range(1, len(blocks) - 1)])

    pushed_letter = [''] * count[0]
    for b_idx, block in enumerate(blocks):
        if len(block) == 1 and (block[0].strip() == "" or block[0].startswith("#=GC RF") or block[0].strip() == "//"):
            # Just separator or empty line, keep as is
            new_lines.extend(block)
            continue

        #print(query_line_indices)
        block_copy = block.copy()
        # Replace query line with modified query segment
        _, q_line_idx = query_line_indices[b_idx]
        #print(q_line_idx)
        # Actually, the headers and line indices are parallel, so
        q_line = block[q_line_idx]
        #print(q_line)
        match = re.match(r"^(\S+)(\s+)([A-Za-z\-]+)$", q_line.rstrip())
        if not match:
            # Should not happen, but keep line unchanged
            new_lines.extend(block)
            continue

        name, spacing, _ = match.groups()
        # Replace query sequence in this block
        new_query_seq = reconstructed_blocks_query.pop(0)
        #print(new_query_seq)
        block_copy[q_line_idx] = f"{name}{spacing}{new_query_seq}\n"

        j = 0

        # Insert gaps in other sequences at insertion columns
        for i, line in enumerate(block_copy):
            if i == q_line_idx:
                continue  # skip query line
            if line.startswith("#") or line.strip() == "" or line.strip() == "//":
                continue
            match2 = re.match(r"^(\S+)(\s+)([A-Za-z\-]+)$", line.rstrip())
            if not match2:
                continue
            seq_name, seq_spacing, seq_str = match2.groups()
            seq_list = list(seq_str)
            if pushed_letter[j] != '':
                seq_list.insert(0, pushed_letter[j])
                if b_idx < len(blocks) - 2:
                    pushed_letter[j] = seq_list[-1]
                    seq_list = seq_list[:-1]
            else:
                if b_idx == block_insertion_nb:
                    seq_list.insert(block_insertion_idx, '-')
                    if b_idx < len(blocks) - 2:
                        pushed_letter[j] = seq_list[-1]
                        seq_list = seq_list[:-1]
                else:
                    pushed_letter[j] = ''
            j += 1
            block_copy[i] = f"{seq_name}{seq_spacing}{''.join(seq_list)}\n"

        assert j == count[0] - 1

        new_lines.extend(block_copy)

    with open(output_path, "w") as f:
        f.writelines(new_lines)

    logging.info(f"Modifications applied and written to '{output_path}'")




def modify_amino_acid_a3m(file_path, output_path, replace_map=None, insert_map=None, delete_positions=None):
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
                    query_chars[i] = '-'
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
    #print(query_chars)
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




input_path = '/pasteur/appa/scratch/nportal/af2_sp_results'
output_path = '/pasteur/appa/scratch/nportal/af2_sp_mutated' # also output path for alphafold2
input_path_fasta = '/pasteur/appa/homes/nportal/alphafold2/data/sequences/fasta_sequences'
output_path_fasta = '/pasteur/appa/homes/nportal/alphafold2/data/sequences/fasta_sequences_mutated' # input path for alphafold2

#input_path = 'fake_parent'
#input_path_fasta = 'fasta_sequences'
#output_path = 'temp'
#output_path_fasta = 'temp_fasta'

directory = Path(output_path_fasta)
directory.mkdir(parents=True, exist_ok=True)

with open("mut_types_dict.pkl", "rb") as f:
    loaded_dict = pickle.load(f)

for dir_name in tqdm(os.listdir(input_path)[:1]):
    logging.info(dir_name)

    replace_list = loaded_dict[dir_name]

    dir_path = os.path.join(input_path, dir_name)
    if not os.path.isdir(dir_path):
        continue  # skip non-directories

    msa_dir = os.path.join(dir_path, 'msas')

    for idx, change in enumerate(replace_list):

        out_msa_dir = os.path.join(output_path, dir_name + f'_{idx}', 'msas')

        path1 = os.path.join(msa_dir, 'uniref90_hits.sto')
        path2 = os.path.join(msa_dir, 'mgnify_hits.sto')
        path3 = os.path.join(msa_dir, 'bfd_uniref_hits.a3m')
        path4 = os.path.join(msa_dir, 'pdb_hits.hhr')
        fasta_path_in = os.path.join(input_path_fasta, dir_name + '.fasta')

        out_path1 = os.path.join(out_msa_dir, 'uniref90_hits.sto')
        out_path2 = os.path.join(out_msa_dir, 'mgnify_hits.sto')
        out_path3 = os.path.join(out_msa_dir, 'bfd_uniref_hits.a3m')
        out_path4 = os.path.join(out_msa_dir, 'pdb_hits.hhr')
        fasta_path_out = os.path.join(output_path_fasta, dir_name + f'_{idx}' + '.fasta')

        insertion_map=None
        deletion_map=None
        replacement_map=None

        #print(change)
        logging.info(change)

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
            modify_amino_acid_sto(path1, out_path1, replace_map=replacement_map, insert_map=insertion_map, delete_positions=deletion_map)
        if os.path.exists(path2):
            modify_amino_acid_sto(path2, out_path2, replace_map=replacement_map, insert_map=insertion_map, delete_positions=deletion_map)
        if os.path.exists(path3):
            modify_amino_acid_a3m(path3, out_path3, replace_map=replacement_map, insert_map=insertion_map, delete_positions=deletion_map)
        if os.path.exists(path4):
            shutil.copy(path4, out_path4)
        if os.path.exists(fasta_path_in):
            modify_amino_acid_a3m(fasta_path_in, fasta_path_out, replace_map=replacement_map, insert_map=insertion_map, delete_positions=deletion_map)