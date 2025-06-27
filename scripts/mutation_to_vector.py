import json
import numpy as np
import sys

directory = sys.argv[1]  # directory of the dataset

UPDATED_MUTATION_JSON = f"{directory}/updated_mutation.json"  # mutation.json file
MUTATION_VECTOR_JSON = f"{directory}/mutation_vector.json"  # mutation_vector.json file

# Constants
SEQ_LEN = 29674 - 266 + 1  # Adjusted alignment-based sequence length
BASE_TO_INT = {'A': 0, 'T': 1, 'C': 2, 'G': 3, '-': 4, 'a': 0, 't': 1, 'c': 2, 'g': 3}

# Load mutations
with open(UPDATED_MUTATION_JSON, "r") as f:
    mutation_data = json.load(f)

mutation_vectors = {}

for vid, mutations in mutation_data.items():
    vector = np.zeros(SEQ_LEN, dtype=int)  # Default 0 means reference base
    for mut in mutations:
        pos, base = mut
        if base in BASE_TO_INT:
            vector[pos] = BASE_TO_INT[base] + 1  # 1 to 4 for mutated base
    mutation_vectors[vid] = vector.tolist()

# Save to mutation_vector.json
with open(MUTATION_VECTOR_JSON, "w") as f:
    json.dump(mutation_vectors, f)

print("mutation_vector written successfully.")
