import numpy as np
from scipy import sparse
import sys

directory = sys.argv[1]

adj_matrix_file = f"{directory}/adj_matrix.npy"
sparse_file = f"{directory}/adj_matrix_sparse.npz"

# Load the directed adjacency matrix
adj_matrix = np.load(adj_matrix_file)

# Print shape and basic info
print(type(adj_matrix))
print(f"Shape: {adj_matrix.shape}")
print(f"Number of non-zero edges: {np.count_nonzero(adj_matrix)}")



adj_matrix_sparse = sparse.csr_matrix(adj_matrix)

# Confirm sparsity
print(type(adj_matrix_sparse))
print(f"Sparse matrix shape: {adj_matrix_sparse.shape}")
print(f"Non-zero entries: {adj_matrix_sparse.nnz}")

sparse.save_npz(sparse_file, adj_matrix_sparse)
