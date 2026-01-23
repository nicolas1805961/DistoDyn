import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
from scipy.ndimage import zoom
import numpy as np
from matplotlib_venn import venn2, venn3

def edge_index_to_adj(edge_index, num_nodes):
    # Create an empty adjacency matrix
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    # Fill with 1s at edge positions
    src = edge_index[0]
    dst = edge_index[1]
    adj[src, dst] = 1.0
    
    return adj

def load_triplets(folder1, folder2, cmap, norm):
    files = sorted([f for f in os.listdir(folder1) if f.endswith(".pt")])
    nb_nodes = 0

    matrices = []

    for fname in tqdm(files):
        #print(f"Processing file: {fname}")
        f1 = os.path.join(folder1, fname)
        f2 = os.path.join(folder2, fname)

        if not (os.path.exists(f1) and os.path.exists(f2)):
            print(f"Skipping {fname}: missing in one of the folders.")
            continue

        # Important fix: weights_only must be False
        g1 = torch.load(f1, map_location="cpu", weights_only=False)
        g2 = torch.load(f2, map_location="cpu", weights_only=False)

        num_nodes = g1.x.size(0)
        nb_nodes += num_nodes
        adj1 = edge_index_to_adj(g1.edge_index, num_nodes)
        adj2 = edge_index_to_adj(g2.edge_index, num_nodes)
        
        res = adj1 * 2 + adj2

        '''3 --> both edges present
           2 --> only in g1
           1 --> only in g2
           0 --> no edges'''

        zoom_factors = (392 / res.shape[0], 392 / res.shape[1])
#
        resized_matrix = zoom(res, zoom_factors, order=1)
        matrices.append(resized_matrix)

        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(adj1, cmap='grey')
        #ax[1].imshow(adj2, cmap='grey')
        #ax[2].imshow(adj1 * 2 + adj2, cmap=cmap, norm=norm)
        #plt.show()

        #print(f"Loaded triplet: {fname}")

    matrix = np.stack(matrices, axis=0)

    m1 = np.mean(matrix == 3, axis=0)
    m2 = np.mean(matrix == 2, axis=0)
    m3 = np.mean(matrix == 1, axis=0)
    m4 = np.mean(matrix == 0, axis=0)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    # List of matrices
    matrices = [m1, m2, m3, m4]

    for i in range(4):
        im = ax[i].imshow(matrices[i], cmap='hot')
        fig.colorbar(im, ax=ax[i])  # add a colorbar for this subplot

    plt.tight_layout()

    plt.savefig("average_image.png", dpi=300, bbox_inches="tight")  # PNG format

    plt.show()

    return nb_nodes


if __name__ == "__main__":
    folder1 = r"pt_folder_distogram_00001\train"
    folder2 = r"pt_folder_correlations\train"

    cmap = ListedColormap(["white", "red", "blue", "black"])

    # Boundaries: one more than number of categories
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]

    norm = BoundaryNorm(bounds, cmap.N)

    data = load_triplets(folder1, folder2, cmap=cmap, norm=norm)
    print(f"\nLoaded {len(data)} triplets.")
