from Bio import SeqIO
import sys

input_file = sys.argv[1]        #this is the sorted truncated alignment fasta file
output_file = sys.argv[2]       #this is the output file with the unique sequences only

#set to store unique sequences
seen_sequences = set()

records = list(SeqIO.parse(input_file,"fasta"))

#separate the reference sequence (first one) and the rest
reference_sequence = records[0]
sequences_to_check = records[1:]

#list to hold unique sequences
unique_records = [reference_sequence]       #start with the reference sequence

for record in sequences_to_check:
    sequence_str = str(record.seq)
    if sequence_str not in seen_sequences:
        seen_sequences.add(sequence_str)    #mark as seen
        unique_records.append(record)

#write the unique sequences to the output_file
SeqIO.write(unique_records, output_file, "fasta")

print ("count of unique sequences : "+ str(len(unique_records)-1))