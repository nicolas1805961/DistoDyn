import os
import json
import pandas as pd
import requests
from io import StringIO
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
from Bio.Align import PairwiseAligner
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

def extract_chain_sequence(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_path)
    model = structure[0]

    chains = list(model.get_chains())
    if len(chains) != 2:
        raise ValueError(f"Expected 2 chains, found {len(chains)} in {pdb_path}")

    sequences = []
    chain_ids = []

    for chain in chains:
        resnames = []
        for res in chain:
            if is_aa(res, standard=True):
                resname = res.get_resname().strip().capitalize()
                resnames.append(resname)
        
        if not resnames:
            raise ValueError(f"No sequence found in chain {chain.id}")
        
        seq = seq1("".join(resnames), undef_code='X')  # convert using seq1
        sequences.append(seq)
        chain_ids.append(chain.id)

    return chain_ids, sequences

def fetch_uniprot_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Could not fetch UniProt sequence for {uniprot_id}")
    fasta_str = response.text
    seq_record = SeqIO.read(StringIO(fasta_str), "fasta")
    return str(seq_record.seq)

def align_and_extract(pdb_seq, uniprot_seq):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    #aligner.match_score = 1.0
    #aligner.mismatch_score = 0.0
    #aligner.open_gap_score = -1.0
    #aligner.extend_gap_score = -0.5

    best = next(iter(aligner.align(pdb_seq, uniprot_seq)))
    if not best:
        raise ValueError("No alignment found")

    #best = alignments[0]
    matched_seq = []

    for (start_pdb, end_pdb), (start_uni, end_uni) in zip(best.aligned[0], best.aligned[1]):
        for i, j in zip(range(start_pdb, end_pdb), range(start_uni, end_uni)):
            if pdb_seq[i] == uniprot_seq[j]:
                matched_seq.append(pdb_seq[i])

    return ''.join(matched_seq)

def main():
    input_csv = "best_clusters_cleaned.csv"
    output_dir = "fasta_sequences_boltz"
    pdb_dir = "downloaded_chains"

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    for index, row in df.iterrows():
        #try:
        pdb_id = row["pdb_id"]
        full_id = row["id"]
        #print(full_id)
        uniprot_ids = row["uniprotR_L"].split(";")

        # Chain IDs from ID string like: 3mi9__A1_P50750--3mi9__C1_P04608
        parts = full_id.split("--")
        chain_ids = [part.split("__")[1][0] for part in parts]

        pdb_path = os.path.join(pdb_dir, f"{full_id}.pdb")
        if not os.path.isfile(pdb_path):
            print(f"[✗] Missing PDB file: {pdb_path}")
            continue

        _, pdb_seqs = extract_chain_sequence(pdb_path)
        seq1_pdb, seq2_pdb = pdb_seqs

        seq1_uniprot = fetch_uniprot_sequence(uniprot_ids[0])
        seq2_uniprot = fetch_uniprot_sequence(uniprot_ids[1])

        seq1 = align_and_extract(seq1_pdb, seq1_uniprot)
        seq2 = align_and_extract(seq2_pdb, seq2_uniprot)

        if seq1 and seq2:
            fasta_content = (
                f">{chain_ids[0]}|PROTEIN|\n{seq1}\n"
                f">{chain_ids[1]}|PROTEIN|\n{seq2}\n"
            )
            fasta_filename = os.path.join(output_dir, f"{row['id']}.fasta")
            with open(fasta_filename, "w") as f:
                f.write(fasta_content)
            print(f"Wrote: {fasta_filename}")
        else:
            print(f"Skipping row {index} due to missing sequence(s)")

        #except Exception as e:
        #    print(f"[✗] Error in row {index} ({row['id']}): {e}")

if __name__ == "__main__":
    main()