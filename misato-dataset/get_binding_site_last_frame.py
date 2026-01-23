import sys
import numpy as np
import os
import pickle
import h5py

# Standard 20 amino acids (could include HID/HIE/HIP and CYX if needed)
STANDARD_AAS = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
}

def parse_pdb_ter(pdb_file):
    chains = []
    atom_counts = []  # number of non-hydrogen atoms per chain

    current_chain = []
    current_resseq = None
    current_residue = None
    current_atom_count = 0

    with open(pdb_file) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                resseq = int(line[22:26])
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coord = np.array([x, y, z])

                # Start new residue if needed
                if resseq != current_resseq:
                    if current_residue:
                        current_chain.append(current_residue)
                    current_residue = {"resname": resname, "resseq": resseq, "atoms": {}}
                    current_resseq = resseq

                current_residue["atoms"][atom_name] = coord

                # Count only non-hydrogen atoms
                if not atom_name.startswith("H"):
                    current_atom_count += 1

            elif line.startswith("TER"):
                # End current residue and chain
                if current_residue:
                    current_chain.append(current_residue)
                    current_residue = None
                    current_resseq = None
                if current_chain:
                    chains.append(current_chain)
                    atom_counts.append(current_atom_count)
                    current_chain = []
                    current_atom_count = 0  # reset for next chain

    # Append last residue/chain if any
    if current_residue:
        current_chain.append(current_residue)
    if current_chain:
        chains.append(current_chain)
        atom_counts.append(current_atom_count)

    return chains, atom_counts



def get_binding_vector_last_chain_ligand(pdb_file, nb_atoms_ligand, cutoff=10.0, min_chain_length=30):
    """
    Returns a binary vector for residues in chains >= min_chain_length.
    The last chain in the PDB is treated as the ligand.
    """
    chains, atom_counts = parse_pdb_ter(pdb_file)
    print(len(chains), "chains found in PDB.")
    if len(chains) < 2:
        raise ValueError("PDB must contain at least one protein chain and one ligand chain.")

    
    ligand_indices = np.where(np.array(atom_counts) == nb_atoms_ligand)[0]

    # Last chain is the ligand
    ligand_chain = chains[ligand_indices[-1]]
    #print(len(ligand_chain), "residues in ligand chain.")
    #ligand_coords = []
    #for res in ligand_chain:
    #    for coord in res["atoms"].values():
    #        ligand_coords.append(coord)
    #ligand_coords = np.array(ligand_coords)
    #print(ligand_coords.shape)

    ligand_coords = []
    for res in ligand_chain:
        for atom_name, coord in res["atoms"].items():
            # Skip hydrogens (atom name starts with H)
            if atom_name.startswith("H"):
                continue
            ligand_coords.append(coord)
    ligand_coords = np.array(ligand_coords)

    # Prepare residues from all other chains (excluding ligand chain) that satisfy min_chain_length
    residues_all = []
    chain_index = 1
    for chain_idx, chain in enumerate(chains):
        if chain_idx == ligand_indices[-1]:
            continue
        # only protein residues (Cα present)
        aa_residues = [res for res in chain]
        if len(aa_residues) <= min_chain_length:
            continue
        chain_name = f"CHAIN_{chain_index}"
        chain_index += 1
        for res in aa_residues:
            residues_all.append((chain_name, res))

    # Initialize binding vector
    binding_vector = np.zeros(len(residues_all), dtype=int)

    print(len(residues_all), "residues considered for binding site determination.")
    print(len(ligand_coords), "residues considered for binding site determination.")

    # Fill in 1 for residues close to ligand
    for i, (chain_name, res) in enumerate(residues_all):
        ca_coord = res["atoms"]["CA"]
        dists = np.linalg.norm(ligand_coords - ca_coord, axis=1)
        if np.min(dists) <= cutoff:
            binding_vector[i] = 1

    return binding_vector

if __name__ == "__main__":
    pdb_dir = 'pdb_dir_last_frame'
    pdb_id = sys.argv[1]
    pdb_file = os.path.join(pdb_dir, pdb_id + '_MD_frame99.pdb')

    #h5_file_path_2 = "data/QM/h5_files/tiny_qm.hdf5"
    h5_file_path_2 = "/pasteur/appa/scratch/nportal/MISATO/QM.hdf5"

    f_qm = h5py.File(h5_file_path_2)

    noHindices_lig = np.where(f_qm[pdb_id]["atom_properties"]["atom_names"][()] != b"1")[0]
    nb_atoms_ligand = len(noHindices_lig)

    binding_vector = get_binding_vector_last_chain_ligand(pdb_file, nb_atoms_ligand, cutoff=10.0, min_chain_length=30)

    # Path for correlation matrix
    #correlation_path = "/pasteur/appa/scratch/nportal/MISATO/correlations"
    #correlation_path = "correlations"
    #pkl_corr_file = os.path.join(correlation_path, pdb_id + '.pkl')
#
    ## Load correlation matrix
    #with open(pkl_corr_file, "rb") as f:
    #    data = pickle.load(f)
    #    data = data['matrix']
    #    print("Correlation matrix residues:", data.shape[0])
    #    print("Binding vector residues:", binding_vector.shape[0])
    #    assert data.shape[0] == binding_vector.shape[0], \
    #        "Mismatch in number of residues between binding vector and correlation matrix."

    # Save binding vector as pickle
    binding_site_path = '/pasteur/appa/scratch/nportal/MISATO/Affinity/last_frame/binding_sites'
    os.makedirs(binding_site_path, exist_ok=True)

    with open("test_MD.txt", "r", encoding="utf-8") as f:
        test_lines = f.readlines()
    test_lines = [line.strip() for line in test_lines]

    with open("val_MD.txt", "r", encoding="utf-8") as f:
        val_lines = f.readlines()
    val_lines = [line.strip() for line in val_lines]

    with open("train_MD.txt", "r", encoding="utf-8") as f:
        train_lines = f.readlines()
    train_lines = [line.strip() for line in train_lines]

    if pdb_id in test_lines:
        save_path = os.path.join(binding_site_path, "test", pdb_id + ".npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    elif pdb_id in val_lines:
        save_path = os.path.join(binding_site_path, "val", pdb_id + ".npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    elif pdb_id in train_lines:
        save_path = os.path.join(binding_site_path, "train", pdb_id + ".npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.save(save_path, binding_vector)

    print(f"Binding vector saved to {save_path}")
    print("Binding vector (1 = binding site, 0 = non-binding site):")
    print(binding_vector)
    print("Number of residues considered:", len(binding_vector))
    print("Number of binding site residues:", np.sum(binding_vector))
