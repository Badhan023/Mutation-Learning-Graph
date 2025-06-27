import sys
import json
import numpy as np
import pandas as pd

mutation_json = sys.argv[1]  # mutation json file
editdistance_file = sys.argv[2]  # output edit distance matrix file

with open(mutation_json, "r") as file:
    mutation_dict = json.load(file)       #pairs converted to lists while saving through json
    
editdistance_matrix = np.zeros((len(mutation_dict), len(mutation_dict)), dtype=int)

keys = list(mutation_dict.keys())

for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        key_i = keys[i]
        key_j = keys[j]
        
        mut_i = mutation_dict[key_i]
        mut_j = mutation_dict[key_j]

        # Convert to dictionaries
        dict_i = {pos: base for pos, base in mut_i}
        dict_j = {pos: base for pos, base in mut_j}

        # Union of all mutated positions
        all_positions = set(dict_i.keys()).union(dict_j.keys())

        # Count mismatches
        distance = 0
        for pos in all_positions:
            base_a = dict_i.get(pos)
            base_b = dict_j.get(pos)
            if base_a != base_b:
                distance += 1
        editdistance_matrix[i][j] = distance
        editdistance_matrix[j][i] = distance

# Extract keys into a list
labels = list(mutation_dict.keys())
# Convert to a DataFrame and save
df = pd.DataFrame(editdistance_matrix, columns=labels, index=labels)
df.to_csv(editdistance_file, index=True)

print ("Edit distance matrix saved!")