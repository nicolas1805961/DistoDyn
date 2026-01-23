import os
import torch

def neighbor_percentage(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    src, dst = edge_index
    edges = torch.stack([torch.min(src, dst), torch.max(src, dst)], dim=1)
    unique_edges = torch.unique(edges, dim=0)

    num_neighbor_pairs = unique_edges.size(0)
    total_pairs = num_nodes * (num_nodes - 1) / 2

    percentage = 100.0 * num_neighbor_pairs / total_pairs
    return percentage, num_neighbor_pairs, total_pairs


def process_folder(folder_path):
    results = []

    for fname in os.listdir(folder_path):
        if fname.endswith(".pt"):
            path = os.path.join(folder_path, fname)
            data = torch.load(path, weights_only=False)

            pct, num_edges, total_pairs = neighbor_percentage(data)

            results.append(pct)
            print(
                f"{fname}: "
                f"{pct:.4f}% "
                f"({num_edges}/{int(total_pairs)} neighbor pairs)"
            )

    if results:
        avg_pct = sum(results) / len(results)
        print(f"\nAverage neighbor percentage: {avg_pct:.4f}%")

    return results


if __name__ == "__main__":
    folder = r"pt_folder_distances\train"
    process_folder(folder)
