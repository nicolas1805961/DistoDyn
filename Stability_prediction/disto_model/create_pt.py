import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import MMCIFParser, is_aa, Polypeptide
from glob import glob
from Bio.SeqUtils import seq1
import os
import pickle
import re
import argparse
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import matplotlib.pyplot as plt


one_to_three = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL"
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load tokenizer + model only once
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

if device.type == "cpu":
    model.to(torch.float32)
model.eval()


def get_prot_t5_embeddings(sequence, tokenizer, model, device):
    # replace rare aa and add spaces
    seq_proc = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    
    ids = tokenizer(seq_proc, return_tensors="pt", add_special_tokens=True)
    input_ids = ids["input_ids"].to(device)

    # Convert IDs back to tokens
    #tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    #print(tokens)
    attention_mask = ids["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    embeddings = outputs.last_hidden_state.squeeze(0)  # (L_with_specials, 1024)
    length = attention_mask.sum().item()
    embeddings = embeddings[:length-1]  # remove special tokens ([CLS],[SEP]) 
    return embeddings.cpu()  # shape (L, 1024)


def sort_key(filename):
    # Extract numbers with regex
    match = re.match(r"protein_(\d+)_(\d+)", filename)
    if match:
        x, y = map(int, match.groups())
        return (x, y)
    else:
        return (float("inf"), float("inf"))  # put non-matching at end

def read_fasta(fasta_file):
    """Read a FASTA file and return the sequence."""
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
    return sequence

# Amino acid to index mapping for one-hot encoding
AA_LIST = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
    'T', 'V', 'W', 'Y'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def one_hot_encode_sequence(seq):
    """One-hot encode a protein sequence."""
    one_hot = np.zeros((len(seq), len(AA_LIST)), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            one_hot[i, AA_TO_IDX[aa]] = 1.0
    return torch.tensor(one_hot)

def get_sequence_and_coords(cif_file):
    """Parse CIF file and extract sequence and CA coordinates."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_file)

    sequence = []
    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True):
                    # Convert 3-letter residue name to 1-letter code
                    sequence.append(seq1(residue.get_resname()))
                    coords.append(residue['CA'].coord)
                    #if 'CA' in residue:
                    #    coords.append(residue['CA'].coord)

    return ''.join(sequence), torch.tensor(np.array(coords), dtype=torch.float32)

def get_edges(coords, d_threshold, edge_index=None):
    """Compute edges based on distance threshold."""
    if edge_index is None:
        dmat = torch.cdist(coords, coords)
        edge_index = torch.nonzero(dmat < d_threshold, as_tuple=False).T
        edge_features = 1.0 - dmat[tuple(edge_index)] / d_threshold
        edge_features = edge_features[:, None]  # (E, 1)
    else:
        dists = torch.norm(coords[edge_index[0]] - coords[edge_index[1]], p=2)
        edge_features = 1.0 - dists / d_threshold
        edge_features = edge_features[:, None]
    return edge_index, edge_features


def get_edges_for_rgcn(coords, distogram, bin_centers, d_threshold, method):
    """
    Build edge_index, edge_features, and edge_type for an R-GCN.

    Args:
        coords: (N, 3) tensor of coordinates
        distogram: (N, N, num_bins) predicted distance distributions
        bin_centers: (num_bins,) tensor of bin centers
        d_threshold: distance cutoff
        no_structure: if True, use expected distance from distogram instead of coords
        use_distogram: if True, create two relation types (distance + distogram); else only distance

    Returns:
        edge_index: (2, E) tensor of edges
        edge_features: (E, F) tensor (optional, can be None for R-GCN)
        edge_type: (E,) tensor of relation types (long)
    """
    if 'expected_distance' in method:
        assert 'contact_prob' not in method, "Choose only one method"
    elif 'contact_prob' in method:
        assert 'expected_distance' not in method, "Choose only one method"

    dists_disto = None
    dists = None

    dmat = torch.cdist(coords, coords)
    mask = dmat < d_threshold
    edge_index = torch.nonzero(mask, as_tuple=False).T
    dists = dmat[edge_index[0], edge_index[1]]
    dists = 1.0 - dists / d_threshold
    dists = dists[:, None]  # (E, 1)

    if 'expected_distance' in method:
        # Expected distance for each pair
        expected_d = (distogram * bin_centers[None, None, :]).sum(dim=-1)  # (N, N)
        dists_disto = expected_d[edge_index[0], edge_index[1]]
        dists_disto = 1.0 - dists_disto / d_threshold
        dists_disto = dists_disto[:, None]  # (E, 1)
    elif 'contact_prob' in method:
        contact_bins = bin_centers < d_threshold
        contact_prob = distogram[:, :, contact_bins].sum(dim=-1)  # probability of contact
        dists_disto = contact_prob[edge_index[0], edge_index[1]]
        dists_disto = dists_disto[:, None]  # (E, 1)

    num_edges = edge_index.shape[1]
    if dists_disto is not None:
        # Each pair has two relations: distance and distogram
        # edge_type 0 = distance, 1 = distogram
        # For RGCN, we repeat edges for each relation type
        edge_index = torch.cat([edge_index, edge_index], dim=1)  # (2, 2E)
        edge_type = torch.cat([torch.zeros(num_edges, dtype=torch.long),
                               torch.ones(num_edges, dtype=torch.long)], dim=0)  # (2E,)

        # Edge features: distance for type 0, distogram for type 1
        edge_features = torch.cat([dists, dists_disto], dim=0)  # (2E, F)
        # Note: you may also set edge_features=None if you only want RGCN to use edge_type
    else:
        # Single relation: distance
        edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
        edge_features = dists  # (E, 1)

    assert edge_index.shape[1] == edge_type.shape[0] == edge_features.shape[0], f"Shapes do not match: {edge_index.shape}, {edge_type.shape}, {edge_features.shape}"

    return edge_index, edge_features, edge_type



def get_edges_for_rgcn_no_distance(distogram, bin_centers, d_threshold, method):
    """
    Build edge_index, edge_features, and edge_type for an R-GCN.

    Args:
        coords: (N, 3) tensor of coordinates
        distogram: (N, N, num_bins) predicted distance distributions
        bin_centers: (num_bins,) tensor of bin centers
        d_threshold: distance cutoff
        no_structure: if True, use expected distance from distogram instead of coords
        use_distogram: if True, create two relation types (distance + distogram); else only distance

    Returns:
        edge_index: (2, E) tensor of edges
        edge_features: (E, F) tensor (optional, can be None for R-GCN)
        edge_type: (E,) tensor of relation types (long)
    """
    if 'expected_distance' in method:
        assert 'contact_prob' not in method, "Choose only one method"
    elif 'contact_prob' in method:
        assert 'expected_distance' not in method, "Choose only one method"

    if 'expected_distance' in method:
        # Expected distance for each pair
        expected_d = (distogram * bin_centers[None, None, :]).sum(dim=-1)  # (N, N)
        mask = expected_d < d_threshold   # (N, N)
        edge_index = torch.nonzero(mask, as_tuple=False).T  # shape (2, E)
        dists_disto = expected_d[edge_index[0], edge_index[1]]
        dists_disto = 1.0 - dists_disto / d_threshold
        dists_disto = dists_disto[:, None]  # (E, 1)
    elif 'contact_prob' in method:
        contact_bins = bin_centers < d_threshold
        contact_prob = distogram[:, :, contact_bins].sum(dim=-1)  # probability of contact
        mask = contact_prob > 0.5
        edge_index = torch.nonzero(mask, as_tuple=False).T  # shape (2, E)
        print(edge_index.shape)
        dists_disto = contact_prob[edge_index[0], edge_index[1]]
        dists_disto = dists_disto[:, None]  # (E, 1)
        
    edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
    edge_features = dists_disto

    assert edge_index.shape[1] == edge_type.shape[0] == edge_features.shape[0], f"Shapes do not match: {edge_index.shape}, {edge_type.shape}, {edge_features.shape}"

    return edge_index, edge_features, edge_type



def cif_to_graph(cif_file, fasta_file, dg_mapping, node_feature, method, distogram_file, distance_threshold=8.0):
    key_name = os.path.basename(fasta_file).split('.')[0]
    dGmean = dg_mapping[key_name]

    sequence, coords = get_sequence_and_coords(cif_file)
    fasta_sequence = read_fasta(fasta_file)
    assert sequence == fasta_sequence, "Sequence from CIF and FASTA files do not match."

    if node_feature == "prot_t5":
        # Option 1: Use ProtT5 embeddings directly
        x = get_prot_t5_embeddings(sequence, tokenizer, model, device)
    elif node_feature == "one_hot":
        # Option 2: One-hot encoding
        x = one_hot_encode_sequence(sequence)
    elif node_feature == "both":
        # Option 2: Concatenate one-hot and ProtT5 (dim = 20 + 1024)
        one_hot = one_hot_encode_sequence(sequence)
        prot_t5 = get_prot_t5_embeddings(sequence, tokenizer, model, device)
        x = torch.cat([one_hot, prot_t5], dim=1)
    
    
    if distogram_file is not None:
        with open(distogram_file, "rb") as f:
            data = pickle.load(f)
            distogram = torch.from_numpy(data['distogram']['softmax'])
            bin_edges = torch.from_numpy(data['distogram']['bin_edges'])
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            # Bin 0 center
            first_center = 0.5 * bin_edges[0:1]  # shape (1,)
            # Last bin center (approximate)
            delta = bin_edges[1] - bin_edges[0]  # uniform spacing
            last_center = bin_edges[-1:] + delta  # (1,)
            bin_centers = torch.cat([first_center, bin_centers, last_center])
    else:
        distogram = None
        bin_centers = None

    if 'distance' in method:
        edge_index, edge_attr, edge_type = get_edges_for_rgcn(coords, distogram, bin_centers, distance_threshold, method)
    else:
        edge_index, edge_attr, edge_type = get_edges_for_rgcn_no_distance(distogram, bin_centers, distance_threshold, method)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        sequence=sequence,
        edge_type=edge_type,
        y=torch.tensor([dGmean], dtype=torch.float)
    )



def build_adjacency(sequence, distogram, bin_edges, cutoff_matrix, use_std=False, k=1.0):
    n = len(sequence)
    A = np.zeros((n, n), dtype=int)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    # Bin 0 center
    first_center = 0.5 * bin_edges[0:1]  # shape (1,)
    # Last bin center (approximate)
    delta = bin_edges[1] - bin_edges[0]  # uniform spacing
    last_center = bin_edges[-1:] + delta  # (1,)
    bin_centers = torch.cat([first_center, bin_centers, last_center])
    expected_d = (distogram * bin_centers[None, None, :]).sum(dim=-1)  # (N, N)
    
    for i in range(n):
        for j in range(i+1, n):
            res_i, res_j = one_to_three[sequence[i]], one_to_three[sequence[j]]
            dist1 = expected_d[i, j].item()
            dist2 = expected_d[j, i].item()
            assert abs(dist1 - dist2) < 1e-3, f"Distances not symmetric: {dist1} vs {dist2}"
            pair = (res_i, res_j)
            if pair not in cutoff_matrix:
                pair = (res_j, res_i)  # symmetry
            
            if pair in cutoff_matrix:
                mean, std = cutoff_matrix[pair]
                cutoff = mean + (k * std if use_std else 0.0)
                if dist1 < cutoff:
                    A[i, j] = A[j, i] = 1
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(A, cmap='grey', interpolation='none')
    ax[1].imshow(expected_d, cmap='plasma', interpolation='none')
    plt.show()
    return A




def cif_to_graph_egnn(cif_file, fasta_file, dg_mapping, distogram_file, cutoff_matrix, distance_threshold=10):
    key_name = os.path.basename(fasta_file).split('.')[0]
    dGmean = dg_mapping[key_name]

    sequence, coords = get_sequence_and_coords(cif_file)
    fasta_sequence = read_fasta(fasta_file)
    assert sequence == fasta_sequence, "Sequence from CIF and FASTA files do not match."

    one_hot = one_hot_encode_sequence(sequence)
    prot_t5 = get_prot_t5_embeddings(sequence, tokenizer, model, device)

    dmat = torch.cdist(coords, coords)
    mask = dmat < distance_threshold
    dists = 1.0 - dmat / distance_threshold
    
    with open(distogram_file, "rb") as f:
        data = pickle.load(f)
        distogram = torch.from_numpy(data['distogram']['softmax'])
        bin_edges = torch.from_numpy(data['distogram']['bin_edges'])
        adjacency_matrix = build_adjacency(sequence, distogram, bin_edges, cutoff_matrix=cutoff_matrix, use_std=False, k=1.0)

    return Data(
        one_hot=one_hot,
        prot_t5=prot_t5,
        distogram=distogram,
        adjacency_distogram=adjacency_matrix,
        adjacency_distance=mask,
        distance=dists,
        coords=coords,
        y=torch.tensor([dGmean], dtype=torch.float)
    )



def build_graph_dataset(base_cif_folder, output_base_folder, fasta_folder, cutoff_matrix):
    """
    Process all CIF files found in any nested subfolder of base_cif_folder,
    preserving the relative folder structure in output_base_folder.
    """
    # Find all CIF files recursively
    cif_files = glob(os.path.join(base_cif_folder, "**", "*.cif"), recursive=True)
    if not cif_files:
        print(f"No CIF files found in {base_cif_folder}")
        return
    
    #if 'expected_distance' in method or 'contact_prob' in method:
    #    # Find all distogram files recursively
    #    distogram_files = glob(os.path.join(base_cif_folder, "**", "distogram*.pkl"), recursive=True)
    #    if not distogram_files:
    #        print(f"No distogram files found in {base_cif_folder}")
    #        return
    #    assert len(cif_files) == len(distogram_files), "Number of CIF files and distogram files must match."
    #    distogram_files = sorted(distogram_files, key=sort_key)  # Sort files by name for consistency
    #else:
    #    distogram_files = [None] * len(cif_files)

    distogram_files = glob(os.path.join(base_cif_folder, "**", "distogram*.pkl"), recursive=True)
    if not distogram_files:
        print(f"No distogram files found in {base_cif_folder}")
        return
    assert len(cif_files) == len(distogram_files), "Number of CIF files and distogram files must match."
    distogram_files = sorted(distogram_files, key=sort_key)  # Sort files by name for consistency
        
    
    fasta_files = glob(os.path.join(fasta_folder, "*.fasta"))
    if not fasta_files:
        print(f"No fasta files found in {fasta_folder}")
        return
    
    # Open and load the pickle file
    with open('fasta_dGmean_mapping.pkl', "rb") as f:  # 'rb' = read binary
        dg_mapping = pickle.load(f)
    
    #print(sorted([int(os.path.basename(x).split('_')[2]) for x in cif_files]))
    
    cif_files = sorted(cif_files, key=sort_key)  # Sort files by name for consistency
    fasta_files = sorted(fasta_files, key=sort_key)  # Sort files by name for consistency
    
    assert len(cif_files) == len(fasta_files), "Number of CIF files and FASTA files must match."
    

    for cif_file, fasta_file, distogram_file in tqdm(zip(cif_files, fasta_files, distogram_files), total=len(cif_files), desc="Processing CIF files"):
        #print(cif_file)
        #print(fasta_file)
        #print(f"Saved: {save_path}")
        graph = cif_to_graph(cif_file, fasta_file, dg_mapping, node_feature=node_feature, method=method, distogram_file=distogram_file, distance_threshold=distance_threshold)  # <-- your function to parse CIF
        #graph = cif_to_graph_egnn(cif_file, fasta_file, dg_mapping, distogram_file=distogram_file, cutoff_matrix=cutoff_matrix)  # <-- your function to parse CIF

        rel_path_no_ext = os.path.splitext(os.path.basename(cif_file))[0]  # remove .cif
        name = '_'.join(rel_path_no_ext.split('_')[:3])

        save_path = os.path.join(output_base_folder, name + ".pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(graph, save_path)
        #print(f"Saved: {save_path}")
            

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Build protein graph dataset.")
    #parser.add_argument("--node_feature", type=str, choices=["prot_t5", "one_hot", "both"], default="one_hot",
    #                    help="Node feature type: 'prot_t5' for ProtT5 embeddings, "
    #                         "'one_hot' for one-hot encoding, or 'both' for concatenation.")
#
    #args = parser.parse_args()

    cif_folder = "moved"         # contains subfolders of subfolders with cif files
    fasta_folder = "moved_fasta" # contains subfolders of subfolders with fasta files
    output_folder = "pt_folder"  # will mirror folder structure

    #method=['distance', 'contact_prob']  # options: "distance", "expected_distance", "contact_prob"

    cutoff_matrix = {
    ("GLY", "GLY"): (4.467,0.017),
    ("GLY", "ALA"): (5.201,0.269),
    ("GLY", "SER"): (5.51,0.153),
    ("GLY", "VAL"): (5.671,0.107),
    ("GLY", "CYS"): (5.777,0.129),
    ("GLY", "THR"): (5.619,0.12),
    ("GLY", "PRO"): (6.14,0.245),
    ("GLY", "ASP"): (6.135,0.193),
    ("GLY", "ASN"): (6.321,0.169),
    ("GLY", "ILE"): (6.413,0.179),
    ("GLY", "LEU"): (6.554,0.125),
    ("GLY", "GLU"): (7.036,0.249),
    ("GLY", "GLN"): (7.297,0.216),
    ("GLY", "MET"): (7.383,0.255),
    ("GLY", "HIS"): (7.472,0.206),
    ("GLY", "LYS"): (8.216,0.358),
    ("GLY", "PHE"): (7.966,0.219),
    ("GLY", "TYR"): (9.098,0.267),
    ("GLY", "ARG"): (9.166,0.334),
    ("GLY", "TRP"): (8.966,0.239),
    ("ALA", "ALA"): (5.381,0.262),
    ("ALA", "SER"): (5.829,0.291),
    ("ALA", "VAL"): (5.854,0.312),
    ("ALA", "CYS"): (6.057,0.394),
    ("ALA", "THR"): (5.982,0.378),
    ("ALA", "PRO"): (6.412,0.399),
    ("ALA", "ASP"): (6.388,0.289),
    ("ALA", "ASN"): (6.766,0.349),
    ("ALA", "ILE"): (6.587,0.214),
    ("ALA", "LEU"): (6.707,0.25),
    ("ALA", "GLU"): (7.124,0.34),
    ("ALA", "GLN"): (7.583,0.356),
    ("ALA", "MET"): (7.605,0.394),
    ("ALA", "HIS"): (7.591,0.38),
    ("ALA", "LYS"): (8.327,0.55),
    ("ALA", "PHE"): (8.162,0.26),
    ("ALA", "TYR"): (9.121,0.443),
    ("ALA", "ARG"): (9.365,0.485),
    ("ALA", "TRP"): (9.252,0.29),
    ("SER", "SER"): (6.19,0.292),
    ("SER", "VAL"): (6.567,0.205),
    ("SER", "CYS"): (6.59,0.24),
    ("SER", "THR"): (6.45,0.214),
    ("SER", "PRO"): (6.937,0.321),
    ("SER", "ASP"): (6.76,0.323),
    ("SER", "ASN"): (7.081,0.305),
    ("SER", "ILE"): (7.142,0.342),
    ("SER", "LEU"): (7.394,0.287),
    ("SER", "GLU"): (7.483,0.446),
    ("SER", "GLN"): (7.807,0.408),
    ("SER", "MET"): (8.01,0.369),
    ("SER", "HIS"): (8.051,0.435),
    ("SER", "LYS"): (8.792,0.445),
    ("SER", "PHE"): (8.694,0.394),
    ("SER", "TYR"): (9.594,0.467),
    ("SER", "ARG"): (9.753,0.483),
    ("SER", "TRP"): (9.77,0.497),
    ("VAL", "VAL"): (6.759,0.145),
    ("VAL", "CYS"): (6.941,0.173),
    ("VAL", "THR"): (6.791,0.138),
    ("VAL", "PRO"): (7.063,0.298),
    ("VAL", "ASP"): (6.972,0.287),
    ("VAL", "ASN"): (7.219,0.232),
    ("VAL", "ILE"): (7.441,0.242),
    ("VAL", "LEU"): (7.633,0.179),
    ("VAL", "GLU"): (7.404,0.51),
    ("VAL", "GLN"): (8.008,0.359),
    ("VAL", "MET"): (8.335,0.295),
    ("VAL", "HIS"): (8.179,0.383),
    ("VAL", "LYS"): (8.077,0.634),
    ("VAL", "PHE"): (9.057,0.246),
    ("VAL", "TYR"): (9.442,0.535),
    ("VAL", "ARG"): (9.513,0.514),
    ("VAL", "TRP"): (10.021,0.271),
    ("CYS", "CYS"): (6.426,0.178),
    ("CYS", "THR"): (6.801,0.181),
    ("CYS", "PRO"): (7.157,0.259),
    ("CYS", "ASP"): (6.985,0.299),
    ("CYS", "ASN"): (7.205,0.24),
    ("CYS", "ILE"): (7.476,0.295),
    ("CYS", "LEU"): (7.685,0.206),
    ("CYS", "GLU"): (7.449,0.538),
    ("CYS", "GLN"): (7.962,0.347),
    ("CYS", "MET"): (8.265,0.439),
    ("CYS", "HIS"): (8.422,0.203),
    ("CYS", "LYS"): (8.494,0.521),
    ("CYS", "PHE"): (9.026,0.286),
    ("CYS", "TYR"): (9.362,0.585),
    ("CYS", "ARG"): (9.46,0.491),
    ("CYS", "TRP"): (9.752,0.417),
    ("THR", "THR"): (6.676,0.188),
    ("THR", "PRO"): (7.062,0.32),
    ("THR", "ASP"): (6.971,0.307),
    ("THR", "ASN"): (7.159,0.262),
    ("THR", "ILE"): (7.442,0.259),
    ("THR", "LEU"): (7.642,0.19),
    ("THR", "GLU"): (7.628,0.409),
    ("THR", "GLN"): (8.055,0.378),
    ("THR", "MET"): (8.397,0.292),
    ("THR", "HIS"): (8.221,0.417),
    ("THR", "LYS"): (8.715,0.464),
    ("THR", "PHE"): (9.03,0.264),
    ("THR", "TYR"): (9.813,0.43),
    ("THR", "ARG"): (9.764,0.477),
    ("THR", "TRP"): (9.98,0.315),
    ("PRO", "PRO"): (7.288,0.339),
    ("PRO", "ASP"): (7.321,0.416),
    ("PRO", "ASN"): (7.497,0.334),
    ("PRO", "ILE"): (7.554,0.336),
    ("PRO", "LEU"): (7.751,0.317),
    ("PRO", "GLU"): (7.938,0.475),
    ("PRO", "GLN"): (8.308,0.41),
    ("PRO", "MET"): (8.247,0.388),
    ("PRO", "HIS"): (8.537,0.457),
    ("PRO", "LYS"): (9.198,0.55),
    ("PRO", "PHE"): (8.895,0.425),
    ("PRO", "TYR"): (9.965,0.506),
    ("PRO", "ARG"): (10.266,0.506),
    ("PRO", "TRP"): (9.719,0.462),
    ("ASP", "ASP"): (8.001,0.392),
    ("ASP", "ASN"): (7.672,0.337),
    ("ASP", "ILE"): (7.472,0.341),
    ("ASP", "LEU"): (7.696,0.348),
    ("ASP", "GLU"): (8.945,0.354),
    ("ASP", "GLN"): (8.601,0.357),
    ("ASP", "MET"): (8.401,0.361),
    ("ASP", "HIS"): (8.634,0.325),
    ("ASP", "LYS"): (9.306,0.343),
    ("ASP", "PHE"): (9.111,0.351),
    ("ASP", "TYR"): (9.979,0.676),
    ("ASP", "ARG"): (10.123,0.327),
    ("ASP", "TRP"): (9.867,0.475),
    ("ASN", "ASN"): (7.682,0.249),
    ("ASN", "ILE"): (7.631,0.341),
    ("ASN", "LEU"): (7.889,0.279),
    ("ASN", "GLU"): (8.485,0.423),
    ("ASN", "GLN"): (8.502,0.373),
    ("ASN", "MET"): (8.55,0.31),
    ("ASN", "HIS"): (8.672,0.289),
    ("ASN", "LYS"): (9.319,0.398),
    ("ASN", "PHE"): (9.168,0.393),
    ("ASN", "TYR"): (10.039,0.586),
    ("ASN", "ARG"): (10.135,0.372),
    ("ASN", "TRP"): (9.976,0.458),
    ("ILE", "ILE"): (8.096,0.321),
    ("ILE", "LEU"): (8.342,0.261),
    ("ILE", "GLU"): (7.949,0.453),
    ("ILE", "GLN"): (8.302,0.406),
    ("ILE", "MET"): (8.874,0.327),
    ("ILE", "HIS"): (8.523,0.379),
    ("ILE", "LYS"): (8.329,0.582),
    ("ILE", "PHE"): (9.602,0.347),
    ("ILE", "TYR"): (9.719,0.589),
    ("ILE", "ARG"): (9.746,0.557),
    ("ILE", "TRP"): (10.47,0.397),
    ("LEU", "LEU"): (8.522,0.198),
    ("LEU", "GLU"): (8.077,0.475),
    ("LEU", "GLN"): (8.48,0.411),
    ("LEU", "MET"): (9.122,0.318),
    ("LEU", "HIS"): (8.676,0.401),
    ("LEU", "LYS"): (8.479,0.591),
    ("LEU", "PHE"): (9.9,0.26),
    ("LEU", "TYR"): (9.889,0.611),
    ("LEU", "ARG"): (9.852,0.578),
    ("LEU", "TRP"): (10.707,0.331),
    ("GLU", "GLU"): (9.863,0.389),
    ("GLU", "GLN"): (9.328,0.45),
    ("GLU", "MET"): (8.87,0.511),
    ("GLU", "HIS"): (9.454,0.443),
    ("GLU", "LYS"): (9.842,0.434),
    ("GLU", "PHE"): (9.403,0.512),
    ("GLU", "TYR"): (10.544,0.469),
    ("GLU", "ARG"): (10.713,0.363),
    ("GLU", "TRP"): (10.303,0.493),
    ("GLN", "GLN"): (9.074,0.436),
    ("GLN", "MET"): (9.102,0.498),
    ("GLN", "HIS"): (9.391,0.401),
    ("GLN", "LYS"): (9.667,0.521),
    ("GLN", "PHE"): (9.506,0.451),
    ("GLN", "TYR"): (10.534,0.547),
    ("GLN", "ARG"): (10.61,0.535),
    ("GLN", "TRP"): (10.429,0.49),
    ("MET", "MET"): (9.53,0.457),
    ("MET", "HIS"): (9.396,0.342),
    ("MET", "LYS"): (9.096,0.611),
    ("MET", "PHE"): (10.253,0.377),
    ("MET", "TYR"): (10.4,0.661),
    ("MET", "ARG"): (10.25,0.641),
    ("MET", "TRP"): (11.11,0.397),
    ("HIS", "HIS"): (10.606,0.333),
    ("HIS", "LYS"): (9.582,0.714),
    ("HIS", "PHE"): (9.602,0.542),
    ("HIS", "TYR"): (10.843,0.554),
    ("HIS", "ARG"): (10.879,0.595),
    ("HIS", "TRP"): (10.661,0.458),
    ("LYS", "LYS"): (10.662,0.738),
    ("LYS", "PHE"): (9.344,0.441),
    ("LYS", "TYR"): (10.627,0.704),
    ("LYS", "ARG"): (11.322,0.648),
    ("LYS", "TRP"): (10.136,0.47),
    ("PHE", "PHE"): (10.903,0.46),
    ("PHE", "TYR"): (10.999,0.767),
    ("PHE", "ARG"): (10.577,0.738),
    ("PHE", "TRP"): (11.758,0.447),
    ("TYR", "TYR"): (11.536,0.855),
    ("TYR", "ARG"): (11.615,0.822),
    ("TYR", "TRP"): (11.807,0.684),
    ("ARG", "ARG"): (12.05,0.704),
    ("ARG", "TRP"): (11.355,0.889),
    ("TRP", "TRP"): (12.806,0.473)}
    
    build_graph_dataset(
        cif_folder, 
        output_folder, 
        fasta_folder,
        cutoff_matrix=cutoff_matrix
    )

