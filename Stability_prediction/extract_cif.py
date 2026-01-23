import os
import shutil
from glob import glob

def extract_cif_files(subfolder_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(subfolder_path):
        print(f"[ERROR] Subfolder does not exist: {subfolder_path}")
        return

    # Search for CIF files: predictions/** /*_model_0.cif
    pattern = os.path.join(subfolder_path, "predictions", "**", "*_model_0.cif")
    cif_files = glob(pattern, recursive=True)

    if len(cif_files) == 0:
        print(f"[WARN] No CIF files found in {subfolder_path}")
        return

    for cif_file in cif_files:
        filename = os.path.basename(cif_file)
        dest_path = os.path.join(output_dir, filename)

        shutil.copy(cif_file, dest_path)
        print(f"[OK] Copied {cif_file} → {dest_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract CIF files from a single Boltz results subfolder")
    parser.add_argument("subfolder", help="Path to the subfolder (e.g., boltz_results_protein_4_0)")
    parser.add_argument("out", help="Output directory to store CIF files")
    args = parser.parse_args()

    extract_cif_files(args.subfolder, args.out)
