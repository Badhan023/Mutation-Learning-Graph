from collections import Counter
import numpy as np
import json
import sys
from Bio import SeqIO
from scipy.sparse import load_npz



directory = sys.argv[1]  # Directory of the dataset

# === File paths ===
ADJ_MATRIX_FILE = f"{directory}/adj_matrix_sparse.npz"
EDIT_DIST_MATRIX = f"{directory}/updated_editdistance_matrix.npy"
MUTATION_SIM_MATRIX = f"{directory}/updated_similarity_matrix.npy"
MUTATION_FILE = f"{directory}/updated_mutation.json"
REFSEQ_CDS_FILE = "refseq_cds.fasta"
OUTPUT_FILE = f"{directory}/edge_features.npz"
MUTATION_SPECTRUM = f"{directory}/mutation_spectrum.txt"
EDGELIST_JSON = f"{directory}/edgelist.json"

# === Load node IDs and mutation dict ===
with open(MUTATION_FILE) as f:
    mutation_data = json.load(f)
node_list = list(mutation_data.keys())
node_to_index = {node: idx for idx, node in enumerate(node_list)}

# === Load refseq coding region ===
record = SeqIO.read(REFSEQ_CDS_FILE, "fasta")

# === Load adjacency and matrices ===
adj_matrix_sparse = load_npz(ADJ_MATRIX_FILE).tocoo()
edit_matrix = np.load(EDIT_DIST_MATRIX)
sim_matrix = np.load(MUTATION_SIM_MATRIX)

# === Initialize containers ===
sources = []
targets = []
mutation_counts = []
edit_distances = []
mutation_similarities = []
reverse_flags = []
edgelist = {}

# === Iterate over directed edges ===
for i, j in zip(adj_matrix_sparse.row, adj_matrix_sparse.col):
    source = node_list[i]
    target = node_list[j]
    mut_i = set(tuple(m) for m in mutation_data[source])
    mut_j = set(tuple(m) for m in mutation_data[target])
    edgelist[(source, target)] = []
    reverse = int(len(mut_i - mut_j) > 0)  # if reverse mutation
    if reverse == 1:
        mut_diff = mut_i - mut_j  # reverse mutation
        for pair in mut_diff:
            exists = any(x[0] == pair[0] for x in mut_j)
            if exists:
                prev = next((x[1] for x in mut_j if x[0] == pair[0]), None)
            else:
                prev = record.seq[pair[0]]
            edgelist[(source, target)].append((pair[1], pair[0], prev.lower()))
    else:
        mut_diff = mut_j - mut_i  # forward mutation
        for pair in mut_diff:
            exists = any(x[0] == pair[0] for x in mut_i)
            if exists:
                prev = next((x[1] for x in mut_i if x[0] == pair[0]), None)
            else:
                prev = record.seq[pair[0]]
            edgelist[(source, target)].append((prev.lower(), pair[0], pair[1]))
    sources.append(i)
    targets.append(j)
    mutation_counts.append(len(mut_diff))
    edit_distances.append(edit_matrix[i, j])
    mutation_similarities.append(sim_matrix[i, j])
    reverse_flags.append(reverse)

# === Save as NPZ ===
np.savez_compressed(
    OUTPUT_FILE,
    source=np.array(sources),
    target=np.array(targets),
    mutation_count=np.array(mutation_counts),
    edit_distance=np.array(edit_distances),
    mutation_similarity=np.array(mutation_similarities),
    reverse=np.array(reverse_flags)
)

print(f"✅ Edge features saved to: {OUTPUT_FILE}")
# Optional: print stats
data = np.load(OUTPUT_FILE, allow_pickle=True)
print("── Saved NPZ contents ──")
for k in data.files:
    arr = data[k]
    print(f"{k:18s}: shape={arr.shape}, dtype={arr.dtype}")
print("────────────────────────")

# Define all 12 possible SNP types
snp_types = [
    "A>T", "A>C", "A>G",
    "T>A", "T>C", "T>G",
    "C>A", "C>T", "C>G",
    "G>A", "G>T", "G>C"
]

# Initialize a counter for each SNP type
snp_counter = Counter({snp: 0 for snp in snp_types})

# Loop through the edge list and count SNPs
for (source, target), mutations in edgelist.items():
    for mut in mutations:
        from_base, _, to_base = mut  # we don't care about the position
        snp = f"{from_base.upper()}>{to_base.upper()}"
        if snp in snp_counter:
            snp_counter[snp] += 1

# Calculate total SNPs
total_snps = sum(snp_counter.values())

# Prepare the output lines with percentages
output_lines = []
for snp in snp_types:
    count = snp_counter[snp]
    percent = (count / total_snps * 100) if total_snps > 0 else 0.0
    output_lines.append(f"{snp}: {count} ({percent:.2f}%)")

# Save to file
with open(MUTATION_SPECTRUM, "w") as f:
    for line in output_lines:
        f.write(line + "\n")

# Optionally print results
for line in output_lines:
    print(line)

# === Save edgelist as JSON ===
with open(EDGELIST_JSON, "w") as f:
    json.dump({f"{k[0]}->{k[1]}": v for k, v in edgelist.items()}, f)