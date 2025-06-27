from Bio import AlignIO
import sys
import numpy as np
import json
from datetime import datetime
from sequence_class import Sequence
import pandas as pd

directory = sys.argv[1]  # directory of the dataset

alignment_file = f"{directory}/unique_sequences.fasta"        #sorted, truncated, unique alignment file
matrix_output = f"{directory}/similarity_matrix.csv"         #similarity matrix output
#json_position = f"{directory}/position.json"           #json output that holds the mutation position dictionary
json_information = f"{directory}/mutation.json"          #json output that holds the mutation information dictionary
json_date = f"{directory}/date.json"         #json output that holds the date information dictionary

#load the alignment
alignment = AlignIO.read(alignment_file, "fasta")
class_list = []

#wild-type sequence is the first sequence in the alignment file
wild_type = alignment[0].seq
wild_type_id = alignment[0].id

mutation_dict = {}          #dictionary for all the information
#mutation_pos = {}           #dictionary for just the mutation location
date_dict = {}              #to save the earliest collection date per genome
#loop through each sequence in the alignment and compare to wild-type
for record in alignment[1:]:
    seq = record.seq
    seq_id = record.id
    name = seq_id.split('|')[1]
    date_string = seq_id.split('|')[2]     
    parsed_date = datetime.strptime(date_string, "%Y-%m-%d").date()
    mutations = []          #will be saved in this format: (position, current)
    #mutation_pos[name]=[]
    mutation_dict[name]=[]
    date_dict[name] = date_string
    
    for i, (ref_nt, nt) in enumerate(zip(wild_type, seq)):
        if ref_nt != nt:
            mutations.append((i, nt))
    
    mutation_dict[name] = mutations    
    variant = Sequence(id=name, date=parsed_date, seq=seq, mutation=mutations)      #creating instance
    class_list.append(variant)
        
#save the json files of mutation_dict and date_dict
with open(json_information, 'w') as info_file:
    json.dump(mutation_dict, info_file)

with open(json_date, "w") as date_file:
    json.dump(date_dict, date_file)

#making sure not to get any duplicates
unique_dict = {}
seen_values = set()

for key, value in mutation_dict.items():
    value_as_tuple = tuple(map(tuple, value))
    if value_as_tuple not in seen_values:
        seen_values.add(value_as_tuple)     #mark as seen
        unique_dict[key] = value
        

mutation_dict = unique_dict

#creating similarity matrix
matrix = np.zeros((len(mutation_dict), len(mutation_dict)))

i=0

for key1, value1 in mutation_dict.items():
    j=0
    for key2, value2 in mutation_dict.items():
        if key1 == key2:
            matrix[i,j] = 0
        else:
            common = list(set(value1).intersection(value2))
            matrix[i,j] = len(common)
            matrix[j,i] = len(common)
        j=j+1
    i=i+1    


# Extract keys into a list
labels = list(mutation_dict.keys())
# Convert to a DataFrame and save
df = pd.DataFrame(matrix, columns=labels, index=labels)
df = df.astype(int)  # Ensures all columns are integers
df.to_csv(matrix_output, index=True)

print ("Mutation similarity matrix saved!")