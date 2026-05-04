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


def normalized(feature):
    feature = feature - feature.mean()
    feature = feature / feature.std()
    return feature


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



def cif_to_graph(pkl_file, d_threshold_corr, d_threshold_dist):

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
        dmat = torch.from_numpy(data['distance'])
        charges = normalized(torch.from_numpy(data['charges'])) # (N,)
        one_hot = data['one_hot'] # (N, 2)
        correlation = torch.from_numpy(data['correlation_all']) # (N,)
        node_type = torch.from_numpy(data['node_type']) # (N,)
        pos = torch.from_numpy(data['coords']) # (N, 3)
        sequence = data['sequence'] # (N,)
        assert len(pos) == len(one_hot)
    
    with open('atom_classes.pickle', "rb") as f:
        one_hot_data = pickle.load(f)
        my_max = max(list(one_hot_data.values())) + 1

    class_indices = []

    for p in one_hot:
        key = tuple(p.tolist())
        if key in one_hot_data:
            class_indices.append(one_hot_data[key])
        else:
            raise KeyError(f"Missing key in one_hot_data: {key}")

    #class_indices = torch.tensor([one_hot_data.get(tuple(p.tolist()), 0) for p in one_hot])
    x = torch.nn.functional.one_hot(torch.tensor(class_indices), num_classes=my_max).float()
    x = torch.cat([x, charges.unsqueeze(1), node_type.unsqueeze(1)], dim=1)

    mask_distance = dmat < d_threshold_dist
    edge_index_distance = torch.nonzero(mask_distance, as_tuple=False).T
    dists = dmat[edge_index_distance[0], edge_index_distance[1]]
    dists = 1.0 - dists / d_threshold_dist
    dists = dists[:, None]  # (E, 1)
    edge_type_distance = torch.zeros(edge_index_distance.shape[1], dtype=torch.long)

    mask_corr = torch.abs(correlation) > d_threshold_corr
    edge_index_corr = torch.nonzero(mask_corr, as_tuple=False).T
    corr = correlation[edge_index_corr[0], edge_index_corr[1]]
    corr = corr[:, None]  # (E, 1)
    edge_type_corr = torch.ones(edge_index_corr.shape[1], dtype=torch.long)

    #fig, ax = plt.subplots(1, 2)
    #ax[0].imshow(mask_distance, cmap='grey')
    #ax[1].imshow(mask_corr, cmap='grey')
    #plt.show()

    edge_index = torch.cat([edge_index_distance, edge_index_corr], dim=1)
    edge_type  = torch.cat([edge_type_distance, edge_type_corr], dim=0)  # (E_d + E_c,)
    edge_features  = torch.cat([dists, corr], dim=0).to(torch.float32)        # (E_d + E_c, 1)

    assert edge_index.shape[1] == edge_type.shape[0] == edge_features.shape[0], f"Shapes do not match: {edge_index.shape}, {edge_type.shape}, {edge_features.shape}"

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_features,
        sequence=sequence,
        edge_type=edge_type,
    )



def build_graph_dataset(base_folder, output_base_folder, d_threshold_corr, d_threshold_dist):
    """
    Process all CIF files found in any nested subfolder of base_cif_folder,
    preserving the relative folder structure in output_base_folder.
    """
    # Find all CIF files recursively
    pkl_files = glob(os.path.join(base_folder, "*", "*.pkl"))
    if not pkl_files:
        print(f"No PKL files found in {base_folder}")
        return
    
    pkl_files = sorted(pkl_files, key=lambda f: os.path.basename(f))  # Sort files by name for consistency
    
    for pkl_file in tqdm(pkl_files, total=len(pkl_files), desc="Processing files"):

        print(f"Processing: {os.path.basename(pkl_file)}")
        #print(cif_file)
        #print(fasta_file)
        #print(f"Saved: {save_path}")
        graph = cif_to_graph(pkl_file, d_threshold_corr=d_threshold_corr, d_threshold_dist=d_threshold_dist)  # <-- your function to parse CIF
        #graph = cif_to_graph_egnn(cif_file, fasta_file, dg_mapping, distogram_file=distogram_file, cutoff_matrix=cutoff_matrix)  # <-- your function to parse CIF

        name = os.path.splitext(os.path.basename(pkl_file))[0]  # remove .pkl

        if 'test' in pkl_file:
            save_path = os.path.join(output_base_folder, "test", name + ".pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        elif 'val' in pkl_file:
            save_path = os.path.join(output_base_folder, "val", name + ".pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        elif 'train' in pkl_file:
            save_path = os.path.join(output_base_folder, "train", name + ".pt")
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

    d_threshold_corr = 0.6
    d_threshold_dist = 4.5

    #base_folder = 'affinity_data'
    #output_base_folder = 'pt_folder_distance_distogram'

    base_folder = '/pasteur/appa/scratch/nportal/MISATO/Affinity/affinity_data'
    output_base_folder = '/pasteur/appa/scratch/nportal/MISATO/Affinity/pt_folder_distances_correlation_2'

    
    build_graph_dataset(
        base_folder, output_base_folder, d_threshold_corr, d_threshold_dist
    )

