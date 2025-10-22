#!/bin/bash
#SBATCH --job-name=habijabi
#SBATCH --account=gpce
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --mem=32G           # Request 32 GB of memory
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL   # Send t notification at the start and end of the job
#SBATCH --mail-user=badhan@vt.edu   # Send email notification to this address
#SBATCH --output=/home/badhan/quasi/slurm_out/%j.out



echo $1
dir="$1"   #directory

#node feature with pca
python3 pca_node_features.py "$dir"

#extract valid indices
python3 extract_valid_indices.py "$dir"

#GCN
python3 GCN.py "$dir"

#graphSAGE node prediction
python3 graphSAGE_node_pred.py "$dir"

python3 edge_index_label.py "$dir"

#graphSAGE edge prediction
python3 graphSAGE_edge_pred.py "$dir"

#GAT node prediction
python3 GAT_node_pred.py "$dir"

#GAT edge prediction
python3 GAT_edge_pred.py "$dir"

#VGAE edge prediction
python3 VGAE_edge_pred.py "$dir"

#GGNN node prediction
python3 GGNN_node_pred.py "$dir"

#GGNN edge prediction
python3 GGNN_edge_pred.py "$dir"

#MLP node prediction
python3 MLP.py "$dir"

#MLP edge prediction
python3 MLP_edge_pred.py "$dir"

end=$(date +%s%3N)  # End time in milliseconds
elapsed=$((end - start))

echo "Elapsed time: $elapsed milliseconds"

echo "job $SLURM_JOB_ID has ended on node"