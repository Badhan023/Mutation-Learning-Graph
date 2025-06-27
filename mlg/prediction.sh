#!/bin/bash

echo "Pipeline started on:"
hostname
start=$(date +%s%3N)  # Start time in milliseconds

echo "Reference genome: $1"
refSeq="$1"

echo "Working directory: $2"
dir="$2"

# Uncomment to regenerate lineage2label.json
# python3 lineages/lineage_to_label.py --input lineages/lineage_notes.txt --output scripts/lineage2label.json

#sort by date and output two files: one with refseq and sone without(original)
python3 sort_by_date.py "$dir"/sequences.fasta "$dir"/sorted_sequences.fasta

#mafft
mafft --6merpair --thread -1 --keeplength --addfragments "$dir"/sorted_sequences.fasta "$refSeq" > "$dir"/aligned_sequences.fasta
rm "$dir"/sorted_sequences.fasta

mid=$(date +%s%3N)  # End time in milliseconds
elapsed=$((mid - start))

echo "Elapsed time: $elapsed milliseconds"

#truncate coding region
python3 truncate.py "$dir"/aligned_sequences.fasta "$dir"/truncated_sequences.fasta
rm "$dir"/aligned_sequences.fasta

#unique sequences
python3 unique.py "$dir"/truncated_sequences.fasta "$dir"/unique_sequences.fasta
rm "$dir"/truncated_sequences.fasta

#mutation positions
python3 mutation_positions.py "$dir"

#process metadata to get lineage json
python3 process_metadata.py "$dir"

#convert date to datetime
python3 date_to_datetime.py "$dir"/date.json "$dir"/datetime.json

#edit distance
python3 editdistance.py "$dir"/mutation.json "$dir"/editdistance_matrix.csv

#viral mutation network
python3 ancestor_joining.py "$dir" 

#hypothetical variants' sequences reconstruction
python3 hypothetical_genomes.py "$refSeq" "$dir"

#gap removal of hypothetical sequences for pangolin
python3 gap_remove.py "$dir"/hypothetical_alignments.fasta "$dir"/hypothetical_sequences.fasta

#run pangolin on the hypothetical sequences
pangolin "$dir"/hypothetical_sequences.fasta --outfile "$dir"/pangolin_results.csv
rm "$dir"/hypothetical_sequences.fasta

#update lineage json with hypothetical sequences
python3 update_lineage.py "$dir"

#update the unique_sequences.fasta file
python3 update_fasta.py "$dir"/unique_sequences.fasta "$dir"/updated_unique_sequences.fasta
rm "$dir"/unique_sequences.fasta

#combine the original and hypothetical sequences
cat "$dir"/updated_unique_sequences.fasta "$dir"/hypothetical_alignments.fasta > "$dir"/combined_sequences.fasta
rm "$dir"/updated_unique_sequences.fasta
rm "$dir"/hypothetical_alignments.fasta

#depth count from updated_mutation.json
python3 depth_from_mutation.py  "$dir"/updated_mutation.json  "$dir"/depth.json

# Run your Python script
python3 dnabert2_embeddings.py "$dir"/combined_sequences.fasta "$dir"/dnabert2_embeddings.npy 4
#python3 quick_test.py

#map embedding to ids
python3 map_embeddings_to_ids.py "$dir"

#mutation list to vector
python3 mutation_to_vector.py "$dir"

#node feature
python3 node_feature.py "$dir"

#convert to sparse adj matrix and delete the adj_matrix.npy
python3 convert_to_sparse.py "$dir"

#edge feature
python3 edge_feature.py "$dir"

end=$(date +%s%3N)  # End time in milliseconds
elapsed=$((end - start))

echo "Elapsed time: $elapsed milliseconds"
