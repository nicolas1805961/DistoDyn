import os
import csv
import requests
from io import StringIO
from Bio.PDB import MMCIFParser, MMCIF2Dict, PDBIO
import numpy as np


def extract_chains_from_pdb(pdb_text, label_to_auth, auth_to_label):
    chain_ids_label = list(label_to_auth.keys())
    chain_ids_auth = list(label_to_auth.values())
    """Manually extract ATOM lines for specific chains from PDB text."""
    lines = pdb_text.splitlines()
    extracted = []
    for line in lines:
        if line.startswith("ATOM") and line[21] in chain_ids_auth:
            # Replace character at position 21 with new chain ID
            new_chain_id = auth_to_label[line[21]]
            # Build new line by concatenation
            new_line = line[:21] + new_chain_id + line[22:]
            extracted.append(new_line)


    #extracted = [line for line in lines if line.startswith("ATOM") and line[21] in chain_ids_auth]

    if not extracted:
        raise ValueError(f"❌ No ATOM lines found for chains {chain_ids_label}")

    extracted.append("END")
    return "\n".join(extracted)


def build_auth_to_label_map(cif_string):
    """
    Build a mapping from auth_asym_id → label_asym_id.

    Parameters:
        cif_string (str): mmCIF content as a string.

    Returns:
        dict: A dictionary mapping auth_asym_id → label_asym_id.
    """
    mmcif_dict = MMCIF2Dict.MMCIF2Dict(StringIO(cif_string))


    auth_ids = mmcif_dict['_atom_site.auth_asym_id']
    label_ids = mmcif_dict['_atom_site.label_asym_id']

    auth_to_label = {}

    for auth, label in zip(auth_ids, label_ids):
        if auth not in auth_to_label:
            auth_to_label[auth] = label

    return auth_to_label


def download_cif_and_extract_chains(csv_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    parser = MMCIFParser(QUIET=True)

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_id = row['id']
            pdb_id = row['pdb_id'].lower()

            # Extract chain letters from the id field (e.g. A, B)
            parts = full_id.split('--')
            label_chain_ids = [part.split('__')[1][0] for part in parts]

            url = f"https://files.rcsb.org/download/{pdb_id}.cif"
            pdb_out_path = os.path.join(output_folder, f"{full_id.lower()}.pdb")

            try:
                response = requests.get(url)
                response.raise_for_status()
                cif_string = response.text

                auth_to_label = build_auth_to_label_map(cif_string)
                # Invert the dict to get label → auth
                label_to_auth = {v: k for k, v in auth_to_label.items() if v in label_chain_ids}
                auth_ids_to_keep = [label_to_auth[c] for c in label_chain_ids if c in label_to_auth]

                # Parse mmCIF structure
                structure = parser.get_structure(pdb_id, StringIO(cif_string))

                # Write full structure to PDB in-memory
                pdb_buffer = StringIO()
                io = PDBIO()
                io.set_structure(structure)
                io.save(pdb_buffer)
                pdb_text = pdb_buffer.getvalue()

                # Manually extract only desired chains
                filtered_pdb = extract_chains_from_pdb(pdb_text, label_to_auth, auth_to_label)

                # Save final filtered PDB
                with open(pdb_out_path, 'w') as f_out:
                    f_out.write(filtered_pdb)

                print(f"✅ Extracted chains {','.join(label_chain_ids)} from {pdb_id}.cif to {full_id.lower()}.pdb")

            except Exception as e:
                print(f"❌ Failed to process {full_id}: {e}")


# Example usage
download_cif_and_extract_chains("best_clusters_cleaned.csv", "downloaded_chains")
