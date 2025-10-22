#!/bin/bash
#add slurm specification here
echo "job $SLURM_JOB_ID has started on node" 


conda activate mlg

# Check if the GPU is available and print its details
#echo "-----------------------------------------------------------------------------------------------------"
#python -c "import torch; print('PyTorch:', torch.__version__)"
#python -c "import torchvision; print('TorchVision:', torchvision.__version__)"
#python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
#echo "-----------------------------------------------------------------------------------------------------"

echo $1     
refSeq="$1"      #refseq
echo $2
dir="$2"   #directory

conda activate mlg

#lineage to label
#python3 lineages/lineage_to_label.py

#sort by date
python3 scripts/sort_by_date.py "$dir"/sequences.fasta "$dir"/sorted_sequences.fasta

#mafft
mafft --6merpair --thread -1 --keeplength --addfragments "$dir"/sorted_sequences.fasta "$refSeq" > "$dir"/aligned_sequences.fasta
rm "$dir"/sorted_sequences.fasta

#truncate coding region
python3 scripts/truncate.py "$dir"/aligned_sequences.fasta "$dir"/truncated_sequences.fasta
rm "$dir"/aligned_sequences.fasta

#filter truncated sequences with N
python3 scripts/filter_fasta.py "$dir"/truncated_sequences.fasta "$dir"/filtered_sequences.fasta

#unique sequences
python3 scripts/unique.py "$dir"/filtered_sequences.fasta "$dir"/unique_sequences.fasta
rm "$dir"/truncated_sequences.fasta

#mutation positions
python3 scripts/mutation_positions.py "$dir"

#process metadata to get lineage json
python3 scripts/process_metadata.py "$dir"

#convert date to datetime
python3 scripts/date_to_datetime.py "$dir"/date.json "$dir"/datetime.json

#edit distance
python3 scripts/editdistance.py "$dir"/mutation.json "$dir"/editdistance_matrix.csv

#viral mutation network
python3 scripts/ancestor_joining.py "$dir"

#hypothetical variants' sequences reconstruction
python3 scripts/hypothetical_genomes.py "$refSeq" "$dir"

#gap removal of hypothetical sequences for pangolin
python3 scripts/gap_remove.py "$dir"/hypothetical_alignments.fasta "$dir"/hypothetical_sequences.fasta

#update the unique_sequences.fasta file
python3 scripts/update_fasta.py "$dir"/unique_sequences.fasta "$dir"/updated_unique_sequences.fasta
rm "$dir"/unique_sequences.fasta

#combine the original and hypothetical sequences
cat "$dir"/updated_unique_sequences.fasta "$dir"/hypothetical_alignments.fasta > "$dir"/combined_sequences.fasta
rm "$dir"/updated_unique_sequences.fasta
rm "$dir"/hypothetical_alignments.fasta

#depth count from updated_mutation.json
python3 scripts/depth_from_mutation.py "$dir"/updated_mutation.json "$dir"/depth.json

# Run your Python script
python3 scripts/dnabert2_embeddings.py "$dir"/combined_sequences.fasta "$dir"/dnabert2_embeddings.npy 4

#map embedding to ids
python3 scripts/map_embeddings_to_ids.py "$dir"

#mutation list to vector
python3 scripts/mutation_to_vector.py "$dir"

#node feature
python3 scripts/node_feature.py "$dir"

#convert to sparse adj matrix and delete the adj_matrix.npy
python3 scripts/convert_to_sparse.py "$dir"

#edge feature
python3 scripts/edge_feature.py "$dir"
