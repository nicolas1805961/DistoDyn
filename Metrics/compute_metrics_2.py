import os
import argparse
import csv
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data

from Bio.PDB import PDBParser, Superimposer
import numpy as np
import sys

import os
from Bio.PDB import MMCIFParser, PDBIO
from pymol import cmd


#def extract_ca_coordinates(pdb_file):
#    parser = PDBParser(QUIET=True)
#    structure = parser.get_structure("struct", pdb_file)
#
#    ca_atoms = []
#    for model in structure:
#        for chain in model:
#            for residue in chain:
#                if 'CA' in residue:
#                    ca_atoms.append(residue['CA'])
#        break  # use only first model
#    return ca_atoms

#def compute_rmsd95(pdb1_path, pdb2_path, num_iterations=5):
#    atoms1 = extract_ca_coordinates(pdb1_path)
#    atoms2 = extract_ca_coordinates(pdb2_path)
#
#    if len(atoms1) != len(atoms2):
#        raise ValueError(f"Cα atom count mismatch: {len(atoms1)} vs {len(atoms2)}")
#
#    coords1 = np.array([atom.get_coord() for atom in atoms1])
#    coords2 = np.array([atom.get_coord() for atom in atoms2])
#
#    indices = np.arange(len(coords1))
#
#    for i in range(num_iterations):
#        sup = Superimposer()
#        sup.set_atoms(coords1[indices], coords2[indices])
#        coords2_aligned = np.dot(coords2 - sup.rotran[1], sup.rotran[0].T)
#
#        diffs = np.linalg.norm(coords1 - coords2_aligned, axis=1)
#        sorted_idx = np.argsort(diffs)
#        keep_n = int(0.95 * len(coords1))
#        indices = sorted_idx[:keep_n]
#
#    final_diffs = coords1[indices] - coords2_aligned[indices]
#    rmsd95 = np.sqrt((final_diffs ** 2).sum() / len(indices))
#    return rmsd95
#
#

def find_one_pdb_file(folder_path):
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdb"):
            print(f"✅ Found PDB file: {fname}")
            return True
    
    print("❌ No PDB file found in the folder.")
    return False

def convert_cif_to_pdb_folder(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    parser = MMCIFParser(QUIET=True)
    io = PDBIO()

    for fname in os.listdir(folder_path):
        cif_path = os.path.join(folder_path, fname, fname + '_model.cif')
        if cif_path.endswith(".cif"):
            pdb_name = os.path.splitext(fname)[0] + ".pdb"
            pdb_path = os.path.join(output_folder, pdb_name)

            try:
                structure = parser.get_structure(pdb_name, cif_path)
                io.set_structure(structure)
                io.save(pdb_path)
                print(f"✅ Converted: {cif_path} → {pdb_path}")
            except Exception as e:
                print(f"❌ Error converting {cif_path}: {e}")


def compute_metrics(pdb1_path, pdb2_path):
    # Load structures and get first chain only
    s1 = get_structure(pdb1_path)
    s2 = get_structure(pdb2_path)

    chain1 = next(s1.get_chains())
    chain2 = next(s2.get_chains())

    coords1, seq1 = get_residue_data(chain1)
    coords2, seq2 = get_residue_data(chain2)

    result = tm_align(coords1, coords2, seq1, seq2)  # your existing call
    
    tm_score1 = result.tm_norm_chain1
    tm_score2 = result.tm_norm_chain2
    
    # Use PyMOL for RMSD
    cmd.reinitialize()
    cmd.load(pdb1_path, "ref")
    cmd.load(pdb2_path, "mobile")
    pymol_rmsd = cmd.align("mobile", "ref")[0]

    return tm_score1, tm_score2, pymol_rmsd, result.rmsd

def compare_pdb_folders(gt_folder, pred_folder, output_file="tm_results.csv"):
    gt_files = {f for f in os.listdir(gt_folder) if f.endswith('.pdb')}
    pred_files = {f for f in os.listdir(pred_folder) if f.endswith('.pdb')}
    print(gt_files)
    print(pred_files)
    common_files = sorted(gt_files & pred_files)

    if not common_files:
        print("No matching PDB file names found between folders.")
        return

    print(f"{'File':<30} {'RMSD':>10} {'RMSD_TM':>10} {'TM_gt':>10} {'TM_pred':>10}")
    print("-" * 60)

    results = []

    for fname in common_files:
        gt_path = os.path.join(gt_folder, fname)
        pred_path = os.path.join(pred_folder, fname)

        try:
            tm_gt, tm_pred, rmsd, rmsd_tm = compute_metrics(gt_path, pred_path)
            print(f"{fname:<30} {rmsd:10.3f} {rmsd_tm:10.3f} {tm_gt:10.3f} {tm_pred:10.3f}")
            results.append((fname, rmsd, rmsd_tm, tm_gt, tm_pred))
        except Exception as e:
            print(f"{fname:<30} ERROR: {e}")
            results.append((fname, "ERROR", "ERROR", "ERROR", "ERROR"))

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "RMSD", "RMSD_TM", "TM_score_gt", "TM_score_pred"])
        for row in results:
            writer.writerow(row)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare predicted and ground truth PDB files using tmtools.")
    parser.add_argument("-p", "--pred", required=True, help="Folder with predicted PDB files.")
    parser.add_argument("-o", "--out", default="tm_results.csv", help="Output CSV file to save results.")
    args = parser.parse_args()

    gt_folder = 'downloaded_chains'
    pred_folder = args.pred

    # Example usage:
    assert find_one_pdb_file(gt_folder)

    pred_folder_pdb = pred_folder.strip("/") + '_pdb'

    # Example usage:
    convert_cif_to_pdb_folder(pred_folder, pred_folder_pdb)
    compare_pdb_folders(gt_folder, pred_folder_pdb, args.out)
