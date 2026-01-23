import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import numpy as np
from matplotlib_venn import venn3


def edge_index_to_adj(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.uint8)

    src, dst = edge_index
    adj[src, dst] = 1
    adj[dst, src] = 1 

    return adj


def load_triplets(folder1, folder2, folder3, cmap, norm):
    files = sorted([f for f in os.listdir(folder1) if f.endswith(".pt")])

    only_A = only_B = only_C = 0
    intersection_A_B = intersection_A_C = intersection_B_C = 0
    intersection_A_B_C = 0
    count_none = 0
    total_pairs_all = 0
    n_graphs = 0

    for fname in tqdm(files):
        f1 = os.path.join(folder1, fname)
        f2 = os.path.join(folder2, fname)
        f3 = os.path.join(folder3, fname)

        if not (os.path.exists(f1) and os.path.exists(f2) and os.path.exists(f3)):
            print(f"Skipping {fname}: missing file.")
            continue

        g1 = torch.load(f1, map_location="cpu", weights_only=False)
        g2 = torch.load(f2, map_location="cpu", weights_only=False)
        g3 = torch.load(f3, map_location="cpu", weights_only=False)

        num_nodes = g1.num_nodes
        n_graphs += 1

        adj1 = edge_index_to_adj(g1.edge_index, num_nodes)
        adj2 = edge_index_to_adj(g2.edge_index, num_nodes)
        adj3 = edge_index_to_adj(g3.edge_index, num_nodes)

        mask = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1).bool()

        a1 = adj1[mask]
        a2 = adj2[mask]
        a3 = adj3[mask]

        only_A += np.count_nonzero((a1 == 1) & (a2 == 0) & (a3 == 0))
        only_B += np.count_nonzero((a1 == 0) & (a2 == 1) & (a3 == 0))
        only_C += np.count_nonzero((a1 == 0) & (a2 == 0) & (a3 == 1))

        intersection_A_B += np.count_nonzero((a1 == 1) & (a2 == 1) & (a3 == 0))
        intersection_A_C += np.count_nonzero((a1 == 1) & (a2 == 0) & (a3 == 1))
        intersection_B_C += np.count_nonzero((a1 == 0) & (a2 == 1) & (a3 == 1))

        intersection_A_B_C += np.count_nonzero((a1 == 1) & (a2 == 1) & (a3 == 1))
        count_none += np.count_nonzero((a1 == 0) & (a2 == 0) & (a3 == 0))

        total_pairs_all += num_nodes * (num_nodes - 1) // 2

    counts = [
        only_A,
        only_B,
        intersection_A_B,
        only_C,
        intersection_A_C,
        intersection_B_C,
        intersection_A_B_C,
    ]

    total = total_pairs_all
    counts_pct = [x / total * 100 for x in counts]

    print("Counts:", counts)
    print(f"Total unordered node pairs: {total}")

    v = venn3(
        subsets=counts,
        set_labels=("Distogram", "Distance", "Correlation"),
    )

    ids = ["100", "010", "110", "001", "101", "011", "111"]
    for idx, pct in zip(ids, counts_pct):
        label = v.get_label_by_id(idx)
        if label:
            label.set_text(f"{pct:.1f}%")

    plt.savefig("venn_diagram.png", dpi=300, bbox_inches="tight")
    plt.show()

    return n_graphs


if __name__ == "__main__":
    folder1 = r"pt_folder_distogram_00001\train"
    folder2 = r"pt_folder_distances\train"
    folder3 = r"pt_folder_correlations\train"

    cmap = ListedColormap(["white", "red", "blue", "black"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    n = load_triplets(folder1, folder2, folder3, cmap, norm)
    print(f"\nLoaded {n} triplets.")
