import sys
import numpy as np
from collections import Counter

#===================Inputs=====================================
directory = sys.argv[1]
node_feature_file = f"{directory}/node_features_pca.npz"

#==================Outputs=====================================
valid_indices_file = f"{directory}/valid_indices.npy"

#===================Load lineage labels========================
node_data = np.load(node_feature_file)
y = node_data['lineage_label']

#===================Filter rare classes========================
min_count = 3
counts = Counter(y)
keep_classes = {cls for cls, count in counts.items() if count >= min_count}
valid_indices = np.array([i for i, label in enumerate(y) if label in keep_classes])

#===================Save valid indices=========================
np.save(valid_indices_file, valid_indices)
print(f"✅ Saved valid indices to {valid_indices_file} with {len(valid_indices)} entries.")
