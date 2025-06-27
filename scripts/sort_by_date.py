from Bio import SeqIO
from datetime import datetime
import sys

inputfile = sys.argv[1]             #all_sequence.fasta file
outputfile = sys.argv[2]            #sorted all_sequence.fasta file

def extract_date(record_id):
    date_string = record_id.split('|')[2]
    return datetime.strptime(date_string, "%Y-%m-%d")

#read the records
records = list(SeqIO.parse(inputfile,"fasta"))

#separate the reference sequence (first one) and the rest
#reference_sequence = records[0]
#sequences_to_sort = records[1:]

#sort the sequences by date
sorted_sequence = sorted(records, key=lambda rec: extract_date(rec.id), reverse=True)

#combine the reference sequence with the sorted sequences
#final_sequences = [reference_sequence] + sorted_sequence

#write the final sequences to a new output file
SeqIO.write(sorted_sequence, outputfile,"fasta")

print("Sequences sorted by date!")