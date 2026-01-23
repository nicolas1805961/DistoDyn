import os
import shutil
import argparse
from Bio.PDB import MMCIFParser, PDBIO

def convert_cif_to_pdb(cif_path, pdb_path):
    parser = MMCIFParser(QUIET=True)
    io = PDBIO()
    try:
        structure = parser.get_structure("converted", cif_path)
        io.set_structure(structure)
        io.save(pdb_path)
        print(f"✅ Converted: {cif_path} → {pdb_path}")
        return True
    except Exception as e:
        print(f"❌ Conversion failed for {cif_path}: {e}")
        return False

def convert_and_copy_files(src_root, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)

    # Only go through first-level subdirectories
    for subdir in os.listdir(src_root):
        subdir_path = os.path.join(src_root, subdir)
        if os.path.isdir(subdir_path):
            # Look for .cif files directly inside this subdir
            for file in os.listdir(subdir_path):
                if file.lower().endswith(".cif"):
                    src_file = os.path.join(subdir_path, file)
                    output_name = os.path.splitext(file)[0] + ".pdb"
                    dst_file = os.path.join(dst_folder, output_name)
                    convert_cif_to_pdb(src_file, dst_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .cif files in first-level subfolders to .pdb and copy to flat folder.")
    parser.add_argument("-s", "--src", required=True, help="Source root folder with subfolders.")
    parser.add_argument("-d", "--dst", required=True, help="Destination folder to store converted PDB files.")

    args = parser.parse_args()
    convert_and_copy_files(args.src, args.dst)
