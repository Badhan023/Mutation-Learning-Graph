import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import os

# ---------------------------
# Directory & config
# ---------------------------
dataset_dir = sys.argv[1]
output_file = os.path.join(dataset_dir, "gat_edge_results.txt")
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# Load node data
# ---------------------------
node_data = np.load(f"{dataset_dir}/node_features_pca.npz")
valid_indices = np.load(f"{dataset_dir}/valid_indices.npy")
embedding = node_data['embedding'][valid_indices]
mutation_pca = node_data['mutation_vector_pca'][valid_indices]
depth = node_data['depth'][valid_indices].reshape(-1, 1)
x = np.concatenate([embedding, mutation_pca, depth], axis=1)
x_tensor = torch.tensor(x, dtype=torch.float, device=device)

# ---------------------------
# Load candidate edges & labels
# ---------------------------
edge_data = np.load(f"{dataset_dir}/edge_pred_data.npz", allow_pickle=True)
orig_edge_index = edge_data['edge_index']          # shape [2, E_all] (original IDs)
edge_labels_np = edge_data['label'].astype(int)    # shape [E_all]

# ---------------------------
# Load edge-level features
# ---------------------------
edge_attr_data = np.load(f"{dataset_dir}/edge_features.npz")
edge_features_np = np.stack([
    edge_attr_data['mutation_count'],
    edge_attr_data['edit_distance'],
    edge_attr_data['mutation_similarity'],
    edge_attr_data['reverse']
], axis=1)  # expected shape [E_all, 4]

# ---------------------------
# Filter to valid nodes (both endpoints in valid_indices)
# ---------------------------
mask_src = np.isin(orig_edge_index[0], valid_indices)
mask_dst = np.isin(orig_edge_index[1], valid_indices)
mask_edge = mask_src & mask_dst

# --- Prefer clean boolean masking when arrays are aligned ---
# If lengths mismatch (common in some datasets), fall back to your
# original "index then truncate" path to stay minimally invasive.
filtered_edges = orig_edge_index[:, mask_edge]
filtered_labels = edge_labels_np[mask_edge]

if edge_features_np.shape[0] == orig_edge_index.shape[1]:
    # aligned -> safe boolean mask
    filtered_edge_feats = edge_features_np[mask_edge]
else:
    # fall back to original indexing strategy (minimal change)
    edge_mask_indices = np.where(mask_edge)[0]
    edge_mask_indices = edge_mask_indices[edge_mask_indices < edge_features_np.shape[0]]
    filtered_edge_feats = edge_features_np[edge_mask_indices]
    # keep all three in sync if lengths still differ
    min_len = min(filtered_labels.shape[0], filtered_edge_feats.shape[0], filtered_edges.shape[1])
    filtered_labels = filtered_labels[:min_len]
    filtered_edges = filtered_edges[:, :min_len]
    filtered_edge_feats = filtered_edge_feats[:min_len]

# ---------------------------
# Remap node IDs to compact [0..N_valid-1]
# ---------------------------
old_to_new = {int(old): int(new) for new, old in enumerate(valid_indices)}
src_remapped = np.array([old_to_new[int(u)] for u in filtered_edges[0]], dtype=np.int64)
dst_remapped = np.array([old_to_new[int(v)] for v in filtered_edges[1]], dtype=np.int64)
edge_index_np = np.vstack([src_remapped, dst_remapped])

# ---------------------------
# Torch tensors
# ---------------------------
edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
edge_labels_full = torch.tensor(filtered_labels.astype(np.float32), dtype=torch.float32, device=device)
edge_features_full = torch.tensor(filtered_edge_feats, dtype=torch.float32, device=device)

# ---------------------------
# Stratified 60/20/20 split (once)
# ---------------------------
labels_np = filtered_labels
E = labels_np.shape[0]
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
trainval_idx, test_idx_np = next(sss1.split(np.zeros(E), labels_np))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=43)  # 0.25 of 0.8 -> 0.2
train_idx_sub, val_idx_sub = next(sss2.split(np.zeros(len(trainval_idx)), labels_np[trainval_idx]))
train_idx_np = trainval_idx[train_idx_sub]
val_idx_np   = trainval_idx[val_idx_sub]

train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=device)
val_idx   = torch.tensor(val_idx_np,   dtype=torch.long, device=device)
test_idx  = torch.tensor(test_idx_np,  dtype=torch.long, device=device)

# ---------------------------
# Model (return logits; no sigmoid here)
# ---------------------------
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_feature_dim):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.6)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 2 + edge_feature_dim, 128),  # heads=4, concat -> node dim=hidden*4; src+dst => *2
            nn.ReLU(),
            nn.Linear(128, 1)  # logits
        )

    def forward(self, x, edge_index, edge_features):
        x = self.gat1(x, edge_index)         # [N, hidden*heads] with concat=True default
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        src, dst = edge_index
        feats = torch.cat([x[src], x[dst], edge_features], dim=1)
        return self.edge_mlp(feats).squeeze(-1)  # logits

def safe_metrics(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(np.float32)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    # guard AUC/AP if single-class
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = np.nan
    try:
        auprc = average_precision_score(y_true, y_score)
    except Exception:
        auprc = np.nan
    return acc, f1, auc, auprc

# ---------------------------
# Train across seeds (same split; different inits)
# ---------------------------
metrics_list = []
for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GAT(x_tensor.shape[1], 128, edge_features_full.shape[1]).to(device)
    # class imbalance from TRAIN only
    pos = float(edge_labels_full[train_idx].sum().item())
    neg = float(len(train_idx_np) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1e-8)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tensor, edge_index, edge_features_full)
        loss = criterion(logits[train_idx], edge_labels_full[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_tensor, edge_index, edge_features_full)
        prob = torch.sigmoid(logits)  # apply sigmoid only for scoring
        y_true = edge_labels_full[test_idx].detach().cpu().numpy()
        y_score = prob[test_idx].detach().cpu().numpy()
        y_pred = (y_score >= 0.5).astype(np.float32)

    acc, f1, auc, auprc = safe_metrics(y_true, y_score, thr=0.5)
    metrics_list.append((acc, f1, auc, auprc))

# ---------------------------
# Aggregate & save
# ---------------------------
metrics_np = np.array(metrics_list)
means = metrics_np.mean(axis=0)
stds  = metrics_np.std(axis=0)

with open(output_file, 'w') as f:
    f.write("GAT Edge Prediction\n")
    f.write(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}\n")
    f.write(f"F1 Score:  {means[1]:.4f} ± {stds[1]:.4f}\n")
    f.write(f"AUROC:     {means[2]:.4f} ± {stds[2]:.4f}\n")
    f.write(f"AUPRC:     {means[3]:.4f} ± {stds[3]:.4f}\n")

print("\nResults saved to:", output_file)
print("\nFinal Bootstrapped Metrics (mean ± std over seeds):")
print(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}")
print(f"F1 Score:  {means[1]:.4f} ± {stds[1]:.4f}")
print(f"AUROC:     {means[2]:.4f} ± {stds[2]:.4f}")
print(f"AUPRC:     {means[3]:.4f} ± {stds[3]:.4f}")
