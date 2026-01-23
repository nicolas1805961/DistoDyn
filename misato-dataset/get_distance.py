#!/usr/bin/env python3
import sys
import os
import numpy as np
import pickle

def parse_pdb_chains(pdb_file):
    """
    Parse a PDB file and extract CA atom coordinates for each chain block.
    Each chain is terminated with 'TER'.
    Returns a list of chains, each chain is a list of (res_seq, (x,y,z)) tuples.
    """
    chains = []
    current_chain = []

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_seq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current_chain.append((res_seq, (x, y, z)))
            elif line.startswith("TER"):
                if len(current_chain) > 30:
                    chains.append(current_chain)
                current_chain = []
        # catch last chain if no TER at end
    return chains


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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdb_dist_matrix.py input.pdb")
        sys.exit(1)

    pdb_file = sys.argv[1]
    base = os.path.basename(pdb_file).split('_')[0]
    output_file = f"{base}.pkl"

    pdb_folder = 'pdb_dir'
    chains = parse_pdb_chains(os.path.join(pdb_folder, pdb_file + '_MD_frame0.pdb'))

    residues, dist_matrix, coords = compute_distance_matrix_all_chains(chains)

    # Path to your pickle file
    #correlation_path = "correlations"
    correlation_path = "/pasteur/appa/scratch/nportal/MISATO/correlations"
    pkl_file = os.path.join(correlation_path, output_file)

    # Load the data
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
        assert data['matrix'].shape[0] == dist_matrix.shape[0], "Mismatch in number of residues between distance and correlation matrices."
        assert data['coords'].shape[0] == coords.shape[0], "Mismatch in number of residues between distance and correlation matrices."

    to_save = {
    "matrix": dist_matrix,
    "coords": coords,
    }

    # save everything as pkl
    #output_folder = 'distances'
    output_folder = '/pasteur/appa/scratch/nportal/MISATO/distances'
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, output_file), "wb") as f:
        pickle.dump(to_save, f)

    print(f"Saved distance matrix with {len(residues)} residues → {os.path.join(output_folder, output_file)}")
