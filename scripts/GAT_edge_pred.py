import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
import sys
import os

# Directory
dataset_dir = sys.argv[1]
output_file = os.path.join(dataset_dir, "gat_edge_results.txt")
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]

# Load data
node_data = np.load(f"{dataset_dir}/node_features_pca.npz")
valid_indices = np.load(f"{dataset_dir}/valid_indices.npy")
embedding = node_data['embedding'][valid_indices]
mutation_pca = node_data['mutation_vector_pca'][valid_indices]
depth = node_data['depth'][valid_indices].reshape(-1, 1)
x = np.concatenate([embedding, mutation_pca, depth], axis=1)

x_tensor = torch.tensor(x, dtype=torch.float)

edge_data = np.load(f"{dataset_dir}/edge_pred_data.npz", allow_pickle=True)
orig_edge_index = edge_data['edge_index']
edge_labels_np = edge_data['label']

edge_attr_data = np.load(f"{dataset_dir}/edge_features.npz")
edge_features_np = np.stack([
    edge_attr_data['mutation_count'],
    edge_attr_data['edit_distance'],
    edge_attr_data['mutation_similarity'],
    edge_attr_data['reverse']
], axis=1)

mask_src = np.isin(orig_edge_index[0], valid_indices)
mask_dst = np.isin(orig_edge_index[1], valid_indices)
mask_edge = mask_src & mask_dst
filtered_edges = orig_edge_index[:, mask_edge]
filtered_labels = edge_labels_np[mask_edge]
edge_mask_indices = np.where(mask_edge)[0]
edge_mask_indices = edge_mask_indices[edge_mask_indices < edge_features_np.shape[0]]
filtered_edge_feats = edge_features_np[edge_mask_indices]

min_len = min(len(filtered_labels), len(filtered_edge_feats))
filtered_labels = filtered_labels[:min_len]
filtered_edges = filtered_edges[:, :min_len]
filtered_edge_feats = filtered_edge_feats[:min_len]

old_to_new = {old: new for new, old in enumerate(valid_indices)}
src_remapped = np.array([old_to_new[int(u)] for u in filtered_edges[0]])
dst_remapped = np.array([old_to_new[int(v)] for v in filtered_edges[1]])
edge_index_np = np.vstack([src_remapped, dst_remapped])

edge_index = torch.tensor(edge_index_np, dtype=torch.long)
edge_labels = torch.tensor(filtered_labels.astype(int), dtype=torch.float)
edge_features = torch.tensor(filtered_edge_feats, dtype=torch.float)

x_tensor = x_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
edge_index = edge_index.to(x_tensor.device)
edge_labels = edge_labels.to(x_tensor.device)
edge_features = edge_features.to(x_tensor.device)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_feature_dim):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.6)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 8 + edge_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, edge_features):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        src, dst = edge_index
        feats = torch.cat([x[src], x[dst], edge_features], dim=1)
        return torch.sigmoid(self.edge_mlp(feats)).squeeze(-1)

metrics_list = []
for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)

    indices = np.arange(len(edge_labels))
    np.random.shuffle(indices)
    train_end = int(0.6 * len(indices))
    val_end = int(0.8 * len(indices))

    train_idx = torch.tensor(indices[:train_end], dtype=torch.long)
    val_idx = torch.tensor(indices[train_end:val_end], dtype=torch.long)
    test_idx = torch.tensor(indices[val_end:], dtype=torch.long)

    model = GAT(x_tensor.shape[1], 128, edge_features.shape[1]).to(x_tensor.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.BCELoss()

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        out = model(x_tensor, edge_index, edge_features)
        loss = criterion(out[train_idx], edge_labels[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(x_tensor, edge_index, edge_features)
        pred = (out >= 0.5).float()
        y_true = edge_labels[test_idx].cpu().numpy()
        y_pred = pred[test_idx].cpu().numpy()
        y_score = out[test_idx].cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    metrics_list.append((acc, f1, auc, auprc))

metrics_np = np.array(metrics_list)
means = metrics_np.mean(axis=0)
stds = metrics_np.std(axis=0)

with open(output_file, 'w') as f:
    f.write("GAT Edge Prediction with Bootstrapping (10 runs)\n")
    f.write(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}\n")
    f.write(f"F1 Score:  {means[1]:.4f} ± {stds[1]:.4f}\n")
    f.write(f"AUROC:     {means[2]:.4f} ± {stds[2]:.4f}\n")
    f.write(f"AUPRC:     {means[3]:.4f} ± {stds[3]:.4f}\n")

print("\n✅ Results saved to:", output_file)
print("\nFinal Bootstrapped Metrics:")
print(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}")
print(f"F1 Score:  {means[1]:.4f} ± {stds[1]:.4f}")
print(f"AUROC:     {means[2]:.4f} ± {stds[2]:.4f}")
print(f"AUPRC:     {means[3]:.4f} ± {stds[3]:.4f}")

'''
import numpy as np
import torch
from torch_geometric.data import Data
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import os

# Directory
# Usage: python GAT_edge_pred.py <dataset_dir>

dataset_dir = sys.argv[1]

def load_node_features_and_labels(dataset_dir):
    node_data = np.load(f"{dataset_dir}/node_features_pca.npz")
    valid_indices = np.load(f"{dataset_dir}/valid_indices.npy")

    embedding = node_data['embedding'][valid_indices]
    mutation_pca = node_data['mutation_vector_pca'][valid_indices]
    depth = node_data['depth'][valid_indices].reshape(-1, 1)
    x = np.concatenate([embedding, mutation_pca, depth], axis=1)

    y = node_data['lineage_label'][valid_indices]
    unique_labels = np.unique(y)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y], dtype=np.int64)

    x_tensor = torch.tensor(x, dtype=torch.float)
    return x_tensor, y, valid_indices

def load_edge_data_and_remap(dataset_dir, valid_indices):
    edge_data = np.load(f"{dataset_dir}/edge_pred_data.npz", allow_pickle=True)
    orig_edge_index = edge_data['edge_index']
    edge_labels_np = edge_data['label']

    edge_attr_data = np.load(f"{dataset_dir}/edge_features.npz")
    mutation_count = edge_attr_data['mutation_count']
    edit_distance = edge_attr_data['edit_distance']
    mutation_similarity = edge_attr_data['mutation_similarity']
    is_reverse = edge_attr_data['reverse']

    edge_features_np = np.stack([
        mutation_count,
        edit_distance,
        mutation_similarity,
        is_reverse
    ], axis=1)

    mask_src = np.isin(orig_edge_index[0], valid_indices)
    mask_dst = np.isin(orig_edge_index[1], valid_indices)
    mask_edge = mask_src & mask_dst

    filtered_edges = orig_edge_index[:, mask_edge]
    filtered_labels = edge_labels_np[mask_edge]

    edge_mask_indices = np.where(mask_edge)[0]
    edge_mask_indices = edge_mask_indices[edge_mask_indices < edge_features_np.shape[0]]
    filtered_edge_feats = edge_features_np[edge_mask_indices]

    if len(filtered_labels) != len(filtered_edge_feats):
        min_len = min(len(filtered_labels), len(filtered_edge_feats))
        filtered_labels = filtered_labels[:min_len]
        filtered_edges = filtered_edges[:, :min_len]
        filtered_edge_feats = filtered_edge_feats[:min_len]

    old_to_new = {old: new for new, old in enumerate(valid_indices)}
    src_remapped = np.array([old_to_new[int(u)] for u in filtered_edges[0]])
    dst_remapped = np.array([old_to_new[int(v)] for v in filtered_edges[1]])
    edge_index_np = np.vstack([src_remapped, dst_remapped])

    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_labels = torch.tensor(filtered_labels.astype(int), dtype=torch.float)
    edge_features = torch.tensor(filtered_edge_feats, dtype=torch.float)

    return edge_index, edge_labels, edge_labels.numpy().astype(int), edge_features

# Load data
x_tensor, y, valid_indices = load_node_features_and_labels(dataset_dir)
edge_index, edge_labels, edge_labels_np, edge_features = load_edge_data_and_remap(dataset_dir, valid_indices)

if edge_features.shape[0] != edge_index.shape[1]:
    raise ValueError("Mismatch: edge_features and edge_index must have same number of edges")

data = Data(x=x_tensor, edge_index=edge_index, y=edge_labels)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
splits = list(skf.split(np.zeros(len(edge_labels_np)), edge_labels_np))
train_idx, test_idx = splits[0]
val_idx = test_idx[:len(test_idx)//2]
test_idx = test_idx[len(test_idx)//2:]

train_idx = torch.tensor(train_idx, dtype=torch.long)
val_idx = torch.tensor(val_idx, dtype=torch.long)
test_idx = torch.tensor(test_idx, dtype=torch.long)

train_y = np.array(edge_labels_np[train_idx.numpy()], dtype=np.int64)
class_weights_arr = compute_class_weight(class_weight='balanced', classes=np.unique(train_y), y=train_y)

weight_tensor = torch.ones_like(edge_labels)
if len(class_weights_arr) == 2:
    cw_dict = {cls: w for cls, w in zip(np.unique(train_y), class_weights_arr)}
    for cls in [0,1]:
        weight_tensor[edge_labels == cls] = cw_dict[cls]

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_feature_dim):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.6)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 8 + edge_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, edge_features):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        src, dst = edge_index
        if edge_features.shape[0] != src.shape[0]:
            raise RuntimeError(f"Mismatch in edge count: edge_features {edge_features.shape[0]} vs edge_index {src.shape[0]}")
        feats = torch.cat([x[src], x[dst], edge_features.to(x.device)], dim=1)
        return torch.sigmoid(self.edge_mlp(feats))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(x_tensor.shape[1], 128, edge_features.shape[1]).to(device)
data = data.to(device)
edge_labels = edge_labels.to(device)
edge_features = edge_features.to(device)
weight_tensor = weight_tensor.to(device)

criterion = nn.BCELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, edge_features).squeeze()
    loss_raw = criterion(out[train_idx], edge_labels[train_idx])
    loss = (loss_raw * weight_tensor[train_idx]).mean()
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index, edge_features).squeeze().cpu()
    pred = (out >= 0.5).float()
    y_true = edge_labels.cpu()

    results = {}
    for name, idx in zip(['Train','Val','Test'], [train_idx, val_idx, test_idx]):
        idx_cpu = idx.cpu()
        true = y_true[idx_cpu].numpy()
        pred_ = pred[idx_cpu].numpy()
        prob_ = out[idx_cpu].numpy()
        precision = precision_score(true, pred_, zero_division=0)
        recall = recall_score(true, pred_, zero_division=0)
        f1 = f1_score(true, pred_, zero_division=0)
        try:
            auc = roc_auc_score(true, prob_)
        except ValueError:
            auc = float('nan')
        acc = float((pred_ == true).sum() / len(true))
        results[name] = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}
    return results

best_val = 0.0
best_metrics = None
for epoch in range(1, 201):
    loss = train()
    metrics = test()
    val_acc = metrics['Val']['acc']
    if val_acc > best_val:
        best_val = val_acc
        best_metrics = metrics['Test']
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")

print(f"\n✅ Best Val Acc: {best_val:.4f}")
print(f"Test Acc: {best_metrics['acc']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall: {best_metrics['recall']:.4f}")
print(f"F1: {best_metrics['f1']:.4f}")
print(f"AUC: {best_metrics['auc']:.4f}")
'''