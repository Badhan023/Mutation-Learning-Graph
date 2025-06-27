import pandas as pd
import json
import sys

directory = sys.argv[1]  # directory of the dataset

metadata_file = f"{directory}/metadata.tsv"  # input metadata TSV file
mutation_json = f"{directory}/mutation.json"  # input mutation dictionary file
lineage_json = f"{directory}/lineage.json"  # output lineage JSON file

# Load mutation dictionary
with open(mutation_json) as f:
    mutation_dict = json.load(f)

# Load metadata TSV
metadata = pd.read_csv(metadata_file, sep='\t', dtype=str)

# Filter rows based on mutation_dict keys (Accession IDs)
filtered_metadata = metadata[metadata['Accession ID'].isin(mutation_dict.keys())].copy()

# Clean lineage names (remove anything after the first space)
filtered_metadata['Lineage'] = filtered_metadata['Lineage'].apply(lambda x: x.split()[0] if isinstance(x, str) else x)
# Create the lineage.json dictionary
lineage_dict = dict(zip(filtered_metadata['Accession ID'], filtered_metadata['Lineage']))

# Save to lineage.json
with open(lineage_json, 'w') as f:
    json.dump(lineage_dict, f)

print(f"✅ Saved lineage.json with {len(lineage_dict)} entries.")
