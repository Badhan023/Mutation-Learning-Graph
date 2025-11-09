#!/bin/bash
#add your slurm spec here

echo $1
dir="$1"   #directory

#node feature with pca
python3 scripts/pca_node_features.py "$dir"

#extract valid indices
python3 scripts/extract_valid_indices.py "$dir"

#GCN
python3 scripts/GCN.py "$dir"

#graphSAGE node prediction
python3 scripts/graphSAGE_node_pred.py "$dir"

python3 scripts/edge_index_label.py "$dir"

#graphSAGE edge prediction
python3 scripts/graphSAGE_edge_pred.py "$dir"

#GAT node prediction
python3 scripts/GAT_node_pred.py "$dir"

#GAT edge prediction
python3 scripts/GAT_edge_pred.py "$dir"

#VGAE edge prediction
python3 scripts/VGAE_edge_pred.py "$dir"

#GGNN node prediction
python3 scripts/GGNN_node_pred.py "$dir"

#GGNN edge prediction
python3 scripts/GGNN_edge_pred.py "$dir"

#MLP node prediction
python3 scripts/MLP.py "$dir"

#MLP edge prediction
python3 scripts/MLP_edge_pred.py "$dir"
