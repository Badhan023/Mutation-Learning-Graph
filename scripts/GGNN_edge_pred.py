import numpy as np
import torch
import sys
import os
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GatedGraphConv

# ------------------ Config ------------------
dataset_dir = sys.argv[1]
result_file = os.path.join(dataset_dir, "ggnn_edge_results.txt")
use_date = True
use_is_hypothetical = True

# ------------------ Load Node Features ------------------
node_data = np.load(f"{dataset_dir}/node_features_pca.npz")
valid_indices = np.load(f"{dataset_dir}/valid_indices.npy")

embedding = node_data['embedding'][valid_indices]
mutation_pca = node_data['mutation_vector_pca'][valid_indices]
depth = node_data['depth'][valid_indices].reshape(-1, 1)
features = [embedding, mutation_pca, depth]

if use_date:
    date = node_data['date'][valid_indices].reshape(-1, 1)
    date_mean = date[~np.isnan(date)].mean()
    date[np.isnan(date)] = date_mean
    date = (date - date.mean()) / date.std()
    features.append(date)

if use_is_hypothetical:
    is_hypothetical = node_data['is_hypothetical'][valid_indices].reshape(-1, 1).astype(np.float32)
    features.append(is_hypothetical)

x = np.concatenate(features, axis=1)
y = node_data['lineage_label'][valid_indices]

unique_labels = np.unique(y)
label_map = {old: new for new, old in enumerate(unique_labels)}
y = np.array([label_map[label] for label in y], dtype=np.int64)

x_tensor = torch.tensor(x, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

# ------------------ Load and Remap Edge Index ------------------
edge_data = np.load(f"{dataset_dir}/edge_features.npz")
old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
filtered_edge_index_list = []

for src, tgt in zip(edge_data['source'], edge_data['target']):
    if src in old_to_new and tgt in old_to_new:
        filtered_edge_index_list.append([old_to_new[src], old_to_new[tgt]])

filtered_edge_index = torch.tensor(filtered_edge_index_list, dtype=torch.long).T  # shape (2, E)
edge_index = filtered_edge_index

# ------------------ PyG Data Object ------------------
data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

# ------------------ Define GGNN ------------------
class GGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ggnn = GatedGraphConv(out_channels, num_layers=3)
        self.lin_in = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin_in(x)
        return self.ggnn(x, edge_index)

# ------------------ Edge Classifier ------------------
class EdgeMLP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)

# ------------------ Prepare Training Data ------------------
edge_labels = edge_data['reverse']
edge_mask = [(src in old_to_new and tgt in old_to_new) for src, tgt in zip(edge_data['source'], edge_data['target'])]
edge_labels = edge_labels[edge_mask]
y_bin = torch.tensor(edge_labels, dtype=torch.long)

# Bootstrap settings
seeds = list(range(10))
acc_list, f1_list, auroc_list, auprc_list = [], [], [], []

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(edge_labels)
    idx = np.random.permutation(n)
    split1 = int(n * 0.6)
    split2 = int(n * 0.8)
    train_idx = torch.tensor(idx[:split1], dtype=torch.long)
    val_idx = torch.tensor(idx[split1:split2], dtype=torch.long)
    test_idx = torch.tensor(idx[split2:], dtype=torch.long)

    model = GGNN(x.shape[1], 64).to('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = EdgeMLP(64).to('cuda' if torch.cuda.is_available() else 'cpu')
    device = next(model.parameters()).device
    data = data.to(device)
    x_tensor = x_tensor.to(device)
    y_bin = y_bin.to(device)
    filtered_edge_index = filtered_edge_index.to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    def train():
        model.train()
        classifier.train()
        optimizer.zero_grad()
        node_embed = model(data.x, data.edge_index)
        src_embed = node_embed[filtered_edge_index[0]]
        tgt_embed = node_embed[filtered_edge_index[1]]
        pred = classifier(src_embed[train_idx], tgt_embed[train_idx])
        loss = criterion(pred, y_bin[train_idx])
        loss.backward()
        optimizer.step()

    def evaluate():
        model.eval()
        classifier.eval()
        node_embed = model(data.x, data.edge_index)
        src_embed = node_embed[filtered_edge_index[0]]
        tgt_embed = node_embed[filtered_edge_index[1]]
        pred = classifier(src_embed[test_idx], tgt_embed[test_idx])
        pred_label = pred.argmax(dim=1)
        prob = F.softmax(pred, dim=1)[:, 1].detach().cpu().numpy()
        true = y_bin[test_idx].cpu().numpy()
        acc = accuracy_score(true, pred_label.cpu().numpy())
        f1 = f1_score(true, pred_label.cpu().numpy())
        try:
            auroc = roc_auc_score(true, prob)
        except:
            auroc = float('nan')
        try:
            auprc = average_precision_score(true, prob)
        except:
            auprc = float('nan')
        return acc, f1, auroc, auprc

    for epoch in range(1, 101):
        train()

    acc, f1, auroc, auprc = evaluate()
    acc_list.append(acc)
    f1_list.append(f1)
    auroc_list.append(auroc)
    auprc_list.append(auprc)

acc_avg, acc_std = np.mean(acc_list), np.std(acc_list)
f1_avg, f1_std = np.mean(f1_list), np.std(f1_list)
auroc_avg, auroc_std = np.nanmean(auroc_list), np.nanstd(auroc_list)
auprc_avg, auprc_std = np.nanmean(auprc_list), np.nanstd(auprc_list)

result_str = f"""
✅ GGNN Bootstrap Results (10 seeds)
Accuracy:  {acc_avg:.4f} ± {acc_std:.4f}
F1 Score:  {f1_avg:.4f} ± {f1_std:.4f}
AUROC:     {auroc_avg:.4f} ± {auroc_std:.4f}
AUPRC:     {auprc_avg:.4f} ± {auprc_std:.4f}
"""
print(result_str)
with open(result_file, 'w') as f:
    f.write(result_str)
