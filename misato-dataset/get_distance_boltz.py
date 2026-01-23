#!/usr/bin/env python3
import sys
import os
import numpy as np
import pickle
from Bio.PDB import MMCIFParser
from glob import glob
import argparse

def parse_cif_chains(cif_file):
    """
    Parse a CIF file and extract CA atom coordinates for each chain.
    Returns a list of chains, each chain is a list of (res_seq, (x,y,z)) tuples.
    Only chains with more than 30 residues are kept.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)

    chains_list = []

    for model in structure:  # iterate over models (usually 1)
        for chain in model:
            current_chain = []
            for residue in chain:
                # skip HETATM / non-amino acid residues
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    res_seq = residue.get_id()[1]  # residue sequence number
                    x, y, z = ca_atom.get_coord()
                    current_chain.append((res_seq, (x, y, z)))
            if len(current_chain) > 30:
                chains_list.append(current_chain)

    return chains_list


def compute_distance_matrix_all_chains(chains):
    """
    Concatenate all chains and compute a single distance matrix.
    """
    coords = []
    residues = []
    for chain in chains:
        residues.extend([res for res, _ in chain])
        coords.extend([c for _, c in chain])

    coords = np.array(coords)
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = np.linalg.norm(coords - coords[i], axis=1)
    return residues, dist_matrix, coords

def parse_args():
    parser = argparse.ArgumentParser(description="Compute Boltz distance matrices.")
    parser.add_argument("pdb_file", type=str,
                        help="Name of the PDB folder (e.g. 1abc_XYZ)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pdb_file = args.pdb_file

    base = os.path.basename(pdb_file).split('_')[0]
    output_file = f"{base}.pkl"

    #pdb_folder = 'temp'
    pdb_folder = '/pasteur/appa/scratch/nportal/MISATO/inference'
    folder = os.path.join(pdb_folder,
                      f'boltz_results_{pdb_file}',
                      'predictions',
                      pdb_file)
    cif_files = glob(os.path.join(folder, '*.cif'))
    chains = parse_cif_chains(cif_files[0])

    residues, dist_matrix, coords = compute_distance_matrix_all_chains(chains)

    # Path to your pickle file
    #correlation_path = "correlations"
    correlation_path = "/pasteur/appa/scratch/nportal/MISATO/correlations"
    pkl_file = os.path.join(correlation_path, output_file)

    # Load the data
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
        print(data['matrix'].shape[0])
        print(dist_matrix.shape[0])
        assert data['matrix'].shape[0] == dist_matrix.shape[0], f"Mismatch in number of residues between distance and correlation matrices for pdb {base}."
        assert data['coords'].shape[0] == coords.shape[0], f"Mismatch in number of residues between distance and correlation matrices for pdb {base}."

    to_save = {
    "matrix": dist_matrix,
    "coords": coords,
    }

    # save everything as pkl
    #output_folder = 'distances'
    output_folder = '/pasteur/appa/scratch/nportal/MISATO/boltz_parsed/distances'
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, output_file), "wb") as f:
        pickle.dump(to_save, f)

    print(f"Saved distance matrix with {len(residues)} residues → {os.path.join(output_folder, output_file)}")
