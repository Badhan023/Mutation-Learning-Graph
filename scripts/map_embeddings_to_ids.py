import json
import numpy as np
import sys

directory = sys.argv[1]  # directory of the dataset

EMBEDDING_FILE = f"{directory}/dnabert2_embeddings.npy"        #dnabert2_embedding.npy file
UPDATED_MUTATION_FILE = f"{directory}/updated_mutation.json"     #update_mutation.json file for the ids
EMBEDDING_JSON = f"{directory}/embedding.json"        #output embedding.json file


# Load embeddings and mutation IDs
embeddings = np.load(EMBEDDING_FILE)
with open(UPDATED_MUTATION_FILE, "r") as f:
    mutations = json.load(f)

# Ensure matching lengths
variant_ids = list(mutations.keys())
if len(variant_ids) != embeddings.shape[0]:
    raise ValueError(f"Mismatch: {len(variant_ids)} IDs vs {embeddings.shape[0]} embeddings")

# Create dictionary
embedding_dict = {
    variant_id: embedding.tolist()
    for variant_id, embedding in zip(variant_ids, embeddings)
}

# Save to JSON
with open(EMBEDDING_JSON, "w") as f:
    json.dump(embedding_dict, f)

print(f"Saved embeddings for {len(embedding_dict)} variants to {EMBEDDING_JSON}")
print("embedding size:", len(embedding_dict))
print("mutation size:", len(mutations))
