import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#===================Inputs=====================================
directory = sys.argv[1]
node_feature_file = f"{directory}/node_features.npz"
node_feature_w_pca = f"{directory}/node_features_pca.npz"

#===================Load raw node features=====================
node_data = np.load(node_feature_file)

embedding = node_data['embedding']
mutation_vector = node_data['mutation_vector'].astype(np.float32)
depth = node_data['depth'][:, None].astype(np.float32)
date = node_data['date'][:, None].astype(np.float32)
date_mask = node_data['date_mask']
is_hypothetical = node_data['is_hypothetical'][:, None].astype(np.float32)

# Normalize depth and date
depth = StandardScaler().fit_transform(depth)
mean_date = np.mean(date[date_mask])
date[~date_mask] = mean_date
date = StandardScaler().fit_transform(date)

# Apply PCA to mutation vector with adaptive number of components
n_samples, n_features = mutation_vector.shape
n_components = min(100, n_samples, n_features)
pca = PCA(n_components=n_components)
mutation_pca = pca.fit_transform(mutation_vector)

# Save reduced features
np.savez(
    node_feature_w_pca,
    node_ids=node_data['node_ids'],
    embedding=embedding,
    mutation_vector_pca=mutation_pca,
    depth=depth,
    date=date,
    is_hypothetical=is_hypothetical,
    lineage_label=node_data['lineage_label']
)
