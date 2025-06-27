import json
import argparse

def generate_lineage_labels(input_txt, output_json):
    lineage_set = set()

    with open(input_txt, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            lineage = parts[0]
            if lineage.startswith("*"):
                continue
            lineage_set.add(lineage)

    sorted_lineages = sorted(lineage_set)
    lineage2label = {lineage: idx for idx, lineage in enumerate(sorted_lineages)}

    with open(output_json, 'w') as out:
        json.dump(lineage2label, out, indent=2)

    print(f"✅ Saved {len(lineage2label)} encoded lineages to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pangolin lineage notes into label JSON.")
    parser.add_argument("--input", default="lineages/lineage_notes.txt", help="Path to lineage_notes.txt")
    parser.add_argument("--output", default="scripts/lineage2label.json", help="Output path for lineage2label.json")
    args = parser.parse_args()

    generate_lineage_labels(args.input, args.output)

'''
import json

def generate_lineage_labels(input_txt, output_json):
    lineage_set = set()

    with open(input_txt, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            lineage = parts[0]
            if lineage.startswith("*"):
                continue
            lineage_set.add(lineage)

    sorted_lineages = sorted(lineage_set)
    lineage2label = {lineage: idx for idx, lineage in enumerate(sorted_lineages)}

    with open(output_json, 'w') as out:
        json.dump(lineage2label, out, indent=2)

    print(f"Saved {len(lineage2label)} encoded lineages to {output_json}")

# Example usage
generate_lineage_labels("lineage_notes.txt", "lineage2label.json")
'''