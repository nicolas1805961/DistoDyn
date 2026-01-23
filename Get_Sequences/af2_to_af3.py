import os
import json

input_dir = "benchmark_fastafiles"      # <- set this to your input FASTA directory
output_dir = "json_output"     # <- set this to your desired output directory
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".fasta"):
        fasta_path = os.path.join(input_dir, filename)
        with open(fasta_path, "r") as f:
            lines = f.read().strip().splitlines()

        if len(lines) >= 4:
            try:
                # Parse first sequence
                header1 = lines[0].strip().split()
                chain_id1 = header1[0][1:]  # Remove '>'
                seq1 = lines[1].strip()

                # Parse second sequence
                header2 = lines[2].strip().split()
                chain_id2 = header2[0][1:]
                seq2 = lines[3].strip()

                if seq1 and seq2:
                    row_id = os.path.splitext(filename)[0]

                    json_content = {
                        "name": row_id,
                        "modelSeeds": [1],
                        "sequences": [
                            {
                                "protein": {
                                    "id": chain_id1,
                                    "sequence": seq1
                                }
                            },
                            {
                                "protein": {
                                    "id": chain_id2,
                                    "sequence": seq2
                                }
                            }
                        ],
                        "dialect": "alphafold3",
                        "version": 1
                    }

                    json_filename = os.path.join(output_dir, f"{row_id}.json")
                    with open(json_filename, "w") as f:
                        json.dump(json_content, f, indent=2)
                    print(f"[✓] Wrote: {json_filename}")
                else:
                    print(f"[!] Skipping {filename}: one or both sequences are empty")

            except Exception as e:
                print(f"[!] Error processing {filename}: {e}")

        else:
            print(f"[!] Skipping {filename}: unexpected FASTA format")
