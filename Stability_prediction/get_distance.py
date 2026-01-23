#!/usr/bin/env python3
import sys
import os
import numpy as np
import pickle
from glob import glob
import argparse
from pathlib import Path

def parse_cif_ca_coords_with_residues(cif_path):
    """
    Parses a .cif file and extracts Cα coordinates and residue IDs for each chain.

    Args:
        cif_path (str): Path to the .cif file.

    Returns:
        dict: Dictionary with chain IDs as keys. Each value is a list of tuples:
              (residue_name, residue_number, (x, y, z))
    """
    ca_data = {}
    with open(cif_path, 'r') as f:
        headers = []
        for line in f:
            line = line.strip()
            
            # Collect headers for the atom_site loop
            if line.startswith("loop_"):
                headers = []
                continue
            if line.startswith("_atom_site."):
                headers.append(line)
                continue
            
            # Process data lines once headers are collected
            if headers and line and not line.startswith("_"):
                tokens = line.split()
                if len(tokens) != len(headers):
                    # Skip malformed lines
                    continue
                
                data = dict(zip(headers, tokens))
                atom_type = data.get("_atom_site.label_atom_id")
                if atom_type != "CA":
                    continue
                
                chain_id = data.get("_atom_site.auth_asym_id")
                residue_name = data.get("_atom_site.label_comp_id")
                residue_number = data.get("_atom_site.auth_seq_id")
                x = float(data.get("_atom_site.Cartn_x", 0))
                y = float(data.get("_atom_site.Cartn_y", 0))
                z = float(data.get("_atom_site.Cartn_z", 0))
                
                if chain_id not in ca_data:
                    ca_data[chain_id] = []
                
                ca_data[chain_id].append((residue_name, residue_number, (x, y, z)))
    
    return ca_data


def compute_distance_matrix_all_chains(ca_data):
    """
    Concatenate all chains from parse_cif_ca_coords_with_residues
    and compute a single distance matrix.

    Args:
        ca_data (dict): Output from parse_cif_ca_coords_with_residues.
                        Keys are chain IDs, values are lists of
                        (residue_name, residue_number, (x, y, z)) tuples.

    Returns:
        residues (list): List of residue identifiers as "RESNUM_CHAIN".
        dist_matrix (np.ndarray): NxN distance matrix of Cα atoms.
        coords (np.ndarray): Nx3 array of coordinates.
    """
    coords = []
    residues = []
    for chain_id, chain in ca_data.items():
        for res_name, res_num, coord in chain:
            residues.append(f"{res_name}")
            coords.append(coord)

    coords = np.array(coords)
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = np.linalg.norm(coords - coords[i], axis=1)

    return residues, dist_matrix, coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single PDB ID and create pt distogram")
    parser.add_argument("-name", "--name", type=str, required=True, help="PDB ID to process")
    args = parser.parse_args()
    cif_path = args.name

    p = Path(cif_path)
    parts = p.parts
    protein_name = parts[-1]

    base = '_'.join(protein_name.split('_')[:3])
    output_file = f"{base}.pkl"

    chains = parse_cif_ca_coords_with_residues(cif_path)
    assert len(chains) == 1, "Expected only one chain per cif file"

    residues, dist_matrix, coords = compute_distance_matrix_all_chains(chains)
    assert len(residues) == dist_matrix.shape[0] == coords.shape[0]

    to_save = {
    "matrix": dist_matrix,
    "coords": coords,
    "residues": residues
    }

    # save everything as pkl
    #output_folder = 'distances'
    output_folder = '/pasteur/appa/scratch/nportal/boltz/stability_prediction/distances'
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, output_file), "wb") as f:
        pickle.dump(to_save, f)

    print(f"Saved distance matrix with {len(residues)} residues → {os.path.join(output_folder, output_file)}")
