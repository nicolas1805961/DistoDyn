#!/usr/bin/env python3
import sys
import os
import numpy as np
import pickle
from glob import glob
import argparse
from pathlib import Path
from Bio.PDB import MMCIFParser



# Simple amino acid mapping to integer types
AA_TO_INT = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19
}

def parse_cif_atoms_per_chain(cif_path):
    """
    Parse a mmCIF file and return atom-level info for each chain.
    Returns a list of dicts (one per chain):
        - amino_types: int array of residue types
        - atom_amino_id: maps each atom to its residue index
        - atom_names: list of atom names (as bytes)
        - atom_pos: Nx3 numpy array of coordinates
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("struct", cif_path)

    chains_data = []

    for model in structure:
        for chain in model:
            amino_types = []
            atom_amino_id = []
            atom_names = []
            atom_pos = []

            res_index = 0

            for residue in chain:
                if residue.id[0] != " ":  # skip hetero/water
                    continue

                res_name = residue.resname
                res_type = AA_TO_INT.get(res_name, -1)  # -1 if unknown
                amino_types.append(res_type)

                for atom in residue.get_atoms():
                    atom_names.append(atom.name.encode("utf-8"))
                    atom_pos.append(atom.coord)
                    atom_amino_id.append(res_index)

                res_index += 1

            chains_data.append({
                "chain_id": chain.id,
                "amino_types": np.array(amino_types, dtype=np.int32),
                "atom_amino_id": np.array(atom_amino_id, dtype=np.int32),
                "atom_names": np.array(atom_names),
                "atom_pos": np.array(atom_pos, dtype=np.float32)
            })

    return chains_data


def compute_distance_matrix_all_atoms(chains_data):
    """
    Compute a global pairwise distance matrix between ALL atoms
    from the output of parse_cif_atoms_single_chain().

    Args:
        chains_data (list of dict):
            Each dict has:
                - chain_id
                - atom_names
                - res_ids
                - res_names
                - atom_positions (N_atoms_chain, 3)

    Returns:
        dist_matrix (np.ndarray): (N_atoms_total, N_atoms_total)
            Full pairwise atom distance matrix.
    """

    coords = []

    # Collect all atom coordinates from all chains
    for chain in chains_data:
        positions = chain["atom_positions"]   # (N_chain_atoms, 3)
        coords.append(positions)

    # Concatenate across all chains
    coords = np.vstack(coords)   # (N_atoms_total, 3)

    # Compute full distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    return dist_matrix


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

    chains = parse_cif_atoms_per_chain(cif_path)
    assert len(chains) == 1, "Expected only one chain per cif file"

    #dist_matrix = compute_distance_matrix_all_atoms(chains)
    #assert len(residues) == dist_matrix.shape[0] == coords.shape[0]
#
    #chains[0][]
#
    #to_save = {
    #"matrix": dist_matrix,
    #"coords": coords,
    #"residues": residues
    #}

    # save everything as pkl
    #output_folder = 'distances'
    output_folder = '/pasteur/appa/scratch/nportal/boltz/stability_prediction/Pronet/distances'
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, output_file), "wb") as f:
        pickle.dump(chains[0], f)

    print(f"Saved distance matrix with {len(chains[0]['amino_types'])} residues → {os.path.join(output_folder, output_file)}")
