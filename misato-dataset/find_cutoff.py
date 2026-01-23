import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist

# Paths
ligand_sdf_path = r"subfolder_dir\1a1e\1a1e_ligand.sdf"
pocket_pdb_path = r"subfolder_dir\1a1e\1a1e_pocket.pdb"

# === Load ligand coordinates from SDF ===
supplier = Chem.SDMolSupplier(ligand_sdf_path)
ligand = supplier[0]
if ligand is None:
    raise ValueError("Failed to load ligand from SDF")

AllChem.EmbedMolecule(ligand)  # just in case 3D coords are missing
conf = ligand.GetConformer()
ligand_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(ligand.GetNumAtoms())])

# === Load pocket PDB protein atoms ===
parser = PDBParser(QUIET=True)
structure = parser.get_structure("pocket", pocket_pdb_path)
protein_coords = []
for model in structure:
    for chain in model:
        for res in chain:
            for atom in res:
                protein_coords.append(atom.get_coord())
protein_coords = np.array(protein_coords)

# === Compute minimum distance of each protein atom to any ligand atom ===
dist_matrix = cdist(protein_coords, ligand_coords)  # shape (num_protein_atoms, num_ligand_atoms)
min_distances = dist_matrix.min(axis=1)  # min distance for each protein atom to the ligand

# Maximum of these minimum distances gives approximate cutoff
approx_cutoff = min_distances.max()
print(f"Approximate distance cutoff used to define the pocket: {approx_cutoff:.2f} Å")

# Optional: histogram to see distribution
import matplotlib.pyplot as plt
plt.hist(min_distances, bins=30)
plt.xlabel("Distance to nearest ligand atom (Å)")
plt.ylabel("Number of protein atoms")
plt.title("Distribution of protein-ligand distances in the pocket")
plt.show()
