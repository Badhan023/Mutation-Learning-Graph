#!/bin/bash

echo "Running baseline models"
hostname
start=$(date +%s%3N)  # Start time in milliseconds

# Dataset directory passed as first argument
echo "Input directory: $1"
dir="$1"

# PCA-reduced node features
python3 pca_node_features.py "$dir"

# Extract valid indices
python3 extract_valid_indices.py "$dir"

# GCN
python3 GCN.py "$dir"

# GraphSAGE
python3 graphSAGE_node_pred.py "$dir"
python3 edge_index_label.py "$dir"
python3 graphSAGE_edge_pred.py "$dir"

# GAT
python3 GAT_node_pred.py "$dir"
python3 GAT_edge_pred.py "$dir"

# VGAE
python3 VGAE_edge_pred.py "$dir"

# GGNN
python3 GGNN_node_pred.py "$dir"
python3 GGNN_edge_pred.py "$dir"

# MLP
python3 MLP.py "$dir"

end=$(date +%s%3N)
elapsed=$((end - start))
echo "Elapsed time: $elapsed milliseconds"
