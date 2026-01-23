import os
import shutil
import argparse
from Bio.PDB import MMCIFParser, PDBIO

def copy_files(src_root, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)

    # Only go through first-level subdirectories
    for subdir in os.listdir(src_root):
        subdir_path = os.path.join(src_root, subdir)
        if os.path.isdir(subdir_path):
            # Look for specific .pdb file inside this subdir
            file = 'ranked_0.pdb'
            src_file = os.path.join(subdir_path, file)

            if os.path.isfile(src_file):  # Only proceed if file exists
                output_name = subdir + ".pdb"
                dst_file = os.path.join(dst_folder, output_name)

                shutil.copyfile(src_file, dst_file)
                print(f"✅ Copied: {src_file} → {dst_file}")
            else:
                print(f"❌ File not found: {src_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .cif files in first-level subfolders to .pdb and copy to flat folder.")
    parser.add_argument("-s", "--src", required=True, help="Source root folder with subfolders.")
    parser.add_argument("-d", "--dst", required=True, help="Destination folder to store converted PDB files.")

    args = parser.parse_args()
    copy_files(args.src, args.dst)
