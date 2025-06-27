#!/usr/bin/env python3
"""
Create *depth.json* that stores, for every variant ID in **updated_mutation.json**,
the number of mutations it carries (i.e. the length of the list that
mutation.json stores for that ID).

Usage
-----
python3 depth_from_mutations.py  updated_mutation.json  depth.json
"""

import json
import sys
from pathlib import Path


def main(mutation_file: str, depth_file: str) -> None:
    # --- load ---
    with open(mutation_file, encoding="utf‑8") as f:
        mutations = json.load(f)

    # --- count ---
    depth = {nid: len(m_list) for nid, m_list in mutations.items()}
    
    print(f"Number of depth: {len(depth)}")
    print(f"Number of mutations: {len(mutations)}")

    # --- save ---
    with open(depth_file, "w", encoding="utf‑8") as f:
        json.dump(depth, f, indent=2)



if __name__ == "__main__":
    if len(sys.argv) != 3:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog}  mutation.json  depth.json")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
