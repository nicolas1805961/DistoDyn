import os

input_dir = "AF2_input_paper"       # <- set this to your input FASTA directory
output_dir = "Boltz_input_paper"     # <- set this to your desired output directory
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".fasta"):
        fasta_path = os.path.join(input_dir, filename)
        with open(fasta_path, "r") as f:
            lines = f.read().strip().splitlines()

        if len(lines) >= 4:
            try:
                # Extract chain IDs and sequences
                chain_id1 = lines[0].strip().split()[0][1:]  # Removes '>'
                seq1 = lines[1].strip()

                chain_id2 = lines[2].strip().split()[0][1:]
                seq2 = lines[3].strip()

                if seq1 and seq2:
                    new_fasta_content = (
                        f">{chain_id1}|PROTEIN|\n{seq1}\n"
                        f">{chain_id2}|PROTEIN|\n{seq2}\n"
                    )

                    output_filename = os.path.splitext(filename)[0] + ".fasta"
                    output_path = os.path.join(output_dir, output_filename)

                    with open(output_path, "w") as f:
                        f.write(new_fasta_content)
                    print(f"[✓] Wrote: {output_path}")
                else:
                    print(f"[!] Skipping {filename}: one or both sequences are empty")
            except Exception as e:
                print(f"[!] Error processing {filename}: {e}")
        else:
            print(f"[!] Skipping {filename}: unexpected FASTA format")
