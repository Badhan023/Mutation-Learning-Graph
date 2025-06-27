#!/usr/bin/env python3
"""
node_feature.py

Build node‑feature matrix and save as compressed NPZ.

Inputs expected in <dataset_dir>:
    embedding.json              – ID ➜ 768-dim embedding (list[float])
    mutation_vector.json        – ID ➜ positional mutation vector (list[int])
    depth.json                  – ID ➜ # mutations from Wuhan reference (int)
    datetime.json               – ID ➜ days-since-ref or timestamp (int)
    lineage.json                – ID ➜ Pango lineage (str)
    lineage2label.json          – lineage ➜ numeric label (int)

Usage:
    python node_feature.py <dataset_dir>
"""

import json
import sys
from pathlib import Path
import numpy as np

# --------------------------------------------------------------------------- #
#                             Load all inputs                                 #
# --------------------------------------------------------------------------- #
if len(sys.argv) != 2:
    print(f"Usage: {Path(sys.argv[0]).name} <dataset_dir>")
    sys.exit(1)

dset_dir = Path(sys.argv[1]).resolve()

embedding_file        = dset_dir / "embedding.json"
mutation_vector_file  = dset_dir / "mutation_vector.json"
depth_file            = dset_dir / "depth.json"
datetime_file         = dset_dir / "datetime.json"
lineage_file          = dset_dir / "lineage.json"  # updated
lineage_label_file    = Path("lineage2label.json")
output_npz            = dset_dir / "node_features.npz"

# ---------- JSON loads ----------
with embedding_file.open() as f:
    embeddings = json.load(f)
with mutation_vector_file.open() as f:
    mutation_vectors = json.load(f)
with depth_file.open() as f:
    depths = json.load(f)
with datetime_file.open() as f:
    dates = json.load(f)
with lineage_file.open() as f:
    lineages = json.load(f)
with lineage_label_file.open() as f:
    lineage2label = json.load(f)

print("Size of embeddings:", len(embeddings))
print("Size of mutation vectors:", len(mutation_vectors))
print("Size of depths:", len(depths))
print("Size of dates:", len(dates))
print("Size of lineages:", len(lineages))

# --------------------------------------------------------------------------- #
#                            Assemble feature arrays                          #
# --------------------------------------------------------------------------- #
node_ids = sorted(embeddings.keys())

# Prepare lists
embed_lst, mutvec_lst, depth_lst = [], [], []
date_lst, mask_lst, lineage_lst = [], [], []
is_hypothetical_lst = []

for nid in node_ids:
    embed_lst.append(embeddings[nid])
    mutvec_lst.append(mutation_vectors.get(nid, []))
    depth_lst.append(depths.get(nid, -1))

    is_hypothetical = nid not in dates or nid not in lineages
    is_hypothetical_lst.append(1 if is_hypothetical else 0)

    if nid in dates:
        date_lst.append(dates[nid])
        mask_lst.append(1)
    else:
        date_lst.append(-1)
        mask_lst.append(0)

    if is_hypothetical:
        lineage_lst.append(-1)
    else:
        lineage = lineages[nid]
        lineage_lst.append(lineage2label.get(lineage, -1))

print(lineage_lst.count(-1))
missing_labels = {lineage for lineage in lineages.values() if lineage2label.get(lineage, -1) == -1}
if missing_labels:
    print(f"Warning: {len(missing_labels)} unmapped lineages found:\n", missing_labels)

# --------------------------------------------------------------------------- #
#                             Convert to NumPy arrays                         #
# --------------------------------------------------------------------------- #
embedding_arr       = np.asarray(embed_lst, dtype=np.float32)
mutation_arr        = np.asarray(mutvec_lst, dtype=np.int8)
depth_arr           = np.asarray(depth_lst, dtype=np.int16)
date_arr            = np.asarray(date_lst, dtype=np.int32)
mask_arr            = np.asarray(mask_lst, dtype=np.int8)
lineage_arr         = np.asarray(lineage_lst, dtype=np.int32)
is_hypothetical_arr = np.asarray(is_hypothetical_lst, dtype=np.int8)

# --------------------------------------------------------------------------- #
#                             Save as .npz                                    #
# --------------------------------------------------------------------------- #
np.savez_compressed(
    output_npz,
    node_ids=np.array(node_ids),
    embedding=embedding_arr,
    mutation_vector=mutation_arr,
    depth=depth_arr,
    date=date_arr,
    date_mask=mask_arr,
    lineage_label=lineage_arr,
    is_hypothetical=is_hypothetical_arr,
)

print(f"✅ Node features saved ➜ {output_npz}")

# Optional: print stats
data = np.load(output_npz, allow_pickle=True)
print("── Saved NPZ contents ──")
for k in data.files:
    arr = data[k]
    print(f"{k:18s}: shape={arr.shape}, dtype={arr.dtype}")
print("────────────────────────")


'''
#!/usr/bin/env python3


import json
import sys
from pathlib import Path
import numpy as np

# --------------------------------------------------------------------------- #
#                             Load all inputs                                 #
# --------------------------------------------------------------------------- #
if len(sys.argv) != 2:
    print(f"Usage: {Path(sys.argv[0]).name} <dataset_dir>")
    sys.exit(1)

dset_dir = Path(sys.argv[1]).resolve()

embedding_file        = dset_dir / "embedding.json"
mutation_vector_file  = dset_dir / "mutation_vector.json"
depth_file            = dset_dir / "depth.json"
datetime_file         = dset_dir / "datetime.json"
lineage_file          = dset_dir / "updated_lineage.json"
lineage_label_file    = Path("lineage2label.json")
output_npz            = dset_dir / "node_features.npz"

# ---------- JSON loads ----------
with embedding_file.open() as f:
    embeddings = json.load(f)
with mutation_vector_file.open() as f:
    mutation_vectors = json.load(f)
with depth_file.open() as f:
    depths = json.load(f)
with datetime_file.open() as f:
    dates = json.load(f)
with lineage_file.open() as f:
    lineages = json.load(f)
with lineage_label_file.open() as f:
    lineage2label = json.load(f)
    
print("Size of embeddings:", len(embeddings))
print("Size of mutation vectors:", len(mutation_vectors))
print("Size of depths:", len(depths))
print("Size of dates:", len(dates))
print("Size of lineages:", len(lineages))

# --------------------------------------------------------------------------- #
#                            Assemble feature arrays                          #
# --------------------------------------------------------------------------- #
node_ids = sorted(embeddings.keys())

# Prepare lists
embed_lst, mutvec_lst, depth_lst = [], [], []
date_lst, mask_lst, lineage_lst = [], [], []
is_hypothetical_lst = []

for nid in node_ids:
    embed_lst.append(embeddings[nid])
    mutvec_lst.append(mutation_vectors.get(nid, []))
    depth_lst.append(depths.get(nid, -1))

    if nid in dates:
        date_lst.append(dates[nid])
        mask_lst.append(1)
        is_hypothetical_lst.append(0)
    else:
        date_lst.append(-1)
        mask_lst.append(0)
        is_hypothetical_lst.append(1)

    lineage = lineages.get(nid, "unknown")
    lineage_lst.append(lineage2label.get(lineage, -1))
print(lineage_lst.count(-1))
missing_labels = {lineage for lineage in lineages.values() if lineage2label.get(lineage, -1) == -1}
if missing_labels:
    print(f"⚠️ Warning: {len(missing_labels)} unmapped lineages found:\n", missing_labels)


# --------------------------------------------------------------------------- #
#                             Convert to NumPy arrays                         #
# --------------------------------------------------------------------------- #
embedding_arr       = np.asarray(embed_lst, dtype=np.float32)
mutation_arr        = np.asarray(mutvec_lst, dtype=np.int8)
depth_arr           = np.asarray(depth_lst, dtype=np.int16)
date_arr            = np.asarray(date_lst, dtype=np.int32)
mask_arr            = np.asarray(mask_lst, dtype=np.int8)
lineage_arr         = np.asarray(lineage_lst, dtype=np.int32)
is_hypothetical_arr = np.asarray(is_hypothetical_lst, dtype=np.int8)

# --------------------------------------------------------------------------- #
#                             Save as .npz                                    #
# --------------------------------------------------------------------------- #
np.savez_compressed(
    output_npz,
    node_ids=np.array(node_ids),
    embedding=embedding_arr,
    mutation_vector=mutation_arr,
    depth=depth_arr,
    date=date_arr,
    date_mask=mask_arr,
    lineage_label=lineage_arr,
    is_hypothetical=is_hypothetical_arr,
)

print(f"✅ Node features saved ➜ {output_npz}")

# Optional: print stats
data = np.load(output_npz, allow_pickle=True)
print("── Saved NPZ contents ──")
for k in data.files:
    arr = data[k]
    print(f"{k:18s}: shape={arr.shape}, dtype={arr.dtype}")
print("────────────────────────")
'''