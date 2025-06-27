import numpy as np
import torch
import random
from tqdm import tqdm
import os
import sys

# Parameters
dataset_dir = sys.argv[1]
output_file = os.path.join(dataset_dir, "edge_pred_data.npz")
num_neg_samples = None  # set to None to match positives, or define manually

# Load existing edge data
edge_data = np.load(os.path.join(dataset_dir, "edge_index_filtered.npz"))
edge_index = edge_data['edge_index']  # shape (2, N)
sources = edge_index[0]
targets = edge_index[1]
positive_edges = set(zip(sources, targets))

# Load node list to get total nodes
node_data = np.load(os.path.join(dataset_dir, "node_features.npz"))
num_nodes = node_data['embedding'].shape[0]

# Construct positive edge_index and labels
pos_edge_index = np.array(list(positive_edges)).T  # shape (2, N)
pos_labels = np.ones(pos_edge_index.shape[1], dtype=np.float32)

# Sample negative edges
if num_neg_samples is None:
    num_neg_samples = pos_edge_index.shape[1]

neg_edges = set()
attempts = 0
max_attempts = num_neg_samples * 10
print("Sampling negative edges...")
while len(neg_edges) < num_neg_samples and attempts < max_attempts:
    u = random.randint(0, num_nodes - 1)
    v = random.randint(0, num_nodes - 1)
    if (u, v) not in positive_edges and (u, v) not in neg_edges and u != v:
        neg_edges.add((u, v))
    attempts += 1

neg_edge_index = np.array(list(neg_edges)).T
neg_labels = np.zeros(neg_edge_index.shape[1], dtype=np.float32)

# Combine and shuffle
edge_index = np.concatenate([pos_edge_index, neg_edge_index], axis=1)
labels = np.concatenate([pos_labels, neg_labels], axis=0)

# Shuffle both edge_index and labels together
indices = np.arange(edge_index.shape[1])
np.random.shuffle(indices)
edge_index = edge_index[:, indices]
labels = labels[indices]

# Save
np.savez(output_file, edge_index=edge_index, label=labels)
print(f"✅ Saved edge prediction dataset to {output_file}")
print(f"Total edges: {edge_index.shape[1]}, Positive: {pos_labels.shape[0]}, Negative: {neg_labels.shape[0]}")
