from Bio import SeqIO
import sys

input_file = sys.argv[1]            #input aligned fasta file
output_file = sys.argv[2]           #output fasta with truncated sequences
#start = sys.argv[3]             #start position
#end = sys.argv[4]               #end position
start = 266
end = 29674

with open(output_file, 'w') as output_handle:
    for record in SeqIO.parse(input_file, 'fasta'):
        truncated_seq = record.seq[start-1:end]
        record.seq = truncated_seq
        SeqIO.write(record, output_handle, 'fasta')
        
print ('Truncation is done!')