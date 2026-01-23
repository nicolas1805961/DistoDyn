import os
import subprocess
from pathlib import Path

def run_mmseqs2_msa(
    query_fasta: str,
    mmseqs_db_paths: dict,
    output_dir: str,
    max_seqs: int = 1000,
    tmp_dir: str = "tmp"
):
    """
    Run MMseqs2 on a query against multiple databases and produce a combined A3M MSA.

    Args:
        query_fasta: path to input query FASTA
        mmseqs_db_paths: dict with {db_name: path_to_mmseqs_db}
        output_dir: folder to save final combined MSA
        max_seqs: maximum sequences to keep per database
        tmp_dir: temporary folder for MMseqs2 intermediate files

    Returns:
        combined_a3m_path: path to combined MSA file
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    combined_a3m = Path(output_dir) / "combined.a3m"

    a3m_files = []

    for db_name, db_path in mmseqs_db_paths.items():
        print(f"Running MMseqs2 search against {db_name}...")
        result_prefix = Path(tmp_dir) / f"{db_name}_res"

        # Run MMseqs2 easy-search
        subprocess.run([
            "mmseqs", "easy-search",
            query_fasta,
            db_path,
            str(result_prefix),
            tmp_dir,
            "--max-seqs", str(max_seqs),
            "-a"
        ], check=True)

        # Convert results to A3M
        a3m_path = Path(output_dir) / f"{db_name}.a3m"
        subprocess.run([
            "mmseqs", "result2msa",
            db_path, db_path,
            str(result_prefix),
            str(a3m_path)
        ], check=True)

        a3m_files.append(a3m_path)

    # Combine all A3Ms into one
    print("Combining MSAs...")
    with open(combined_a3m, "w") as outfile:
        for a3m_file in a3m_files:
            with open(a3m_file, "r") as infile:
                outfile.write(infile.read())

    print(f"Combined MSA saved to {combined_a3m}")
    return combined_a3m


if __name__ == "__main__":
    query = "query.fasta"
    dbs = {
        "uniref90": "/opt/gensoft/data/alphafold/2.3.2/uniref90/uniref90_db",
        "mgnify":  "/opt/gensoft/data/alphafold/2.3.2/mgnify/mgnify_db",
        "bfd":     "/opt/gensoft/data/alphafold/2.3.2/bfd/bfd_db",
        "uniref30":"/opt/gensoft/data/alphafold/2.3.2/uniref30/uniref30_db"
    }

    combined_msa_path = run_mmseqs2_msa(query, dbs, output_dir="msa_results")
