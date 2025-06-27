import sys
import pandas as pd
import json  # assuming mutations are in a JSON dict

directory = sys.argv[1]  # directory of the dataset

lineage_json_file = f"{directory}/lineage.json"  # input lineage json file
pangolin_result = f"{directory}/pangolin_results.csv"  # pangolin result file
updated_lineage_json = f"{directory}/updated_lineage.json"  # output lineage json file

df = pd.read_csv(pangolin_result)

# Load JSON file into a dictionary
with open(lineage_json_file, "r") as f:
    updated_lineage_dict = json.load(f)

for _, row in df.iterrows():
    key = row['taxon'].split()[0]
    value = row['lineage']
    updated_lineage_dict[key] = value

# Save the dictionary to a JSON file
with open(updated_lineage_json, "w") as f:
    json.dump(updated_lineage_dict, f)
    
print ("lineages updated.")