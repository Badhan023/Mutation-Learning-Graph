import sys
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

#===================Inputs=====================================
directory = sys.argv[1]
node_feature_file = f"{directory}/node_features_pca.npz"
edge_feature_file = f"{directory}/edge_features.npz"
valid_indices_file = f"{directory}/valid_indices.npy"

#====================Outputs===================================
edge_index_file = f"{directory}/edge_index_filtered.npz"

#===================Load node features=========================
node_data = np.load(node_feature_file)
embedding = node_data['embedding']
mutation_pca = node_data['mutation_vector_pca']
depth = node_data['depth']
date = node_data['date']
is_hypothetical = node_data['is_hypothetical']
y_all = node_data['lineage_label']

valid_indices = np.load(valid_indices_file)

# Concatenate all features
x_all = np.concatenate([embedding, mutation_pca, depth, date, is_hypothetical], axis=1)

# Filter only observed (non-hypothetical) nodes
observed_mask = (y_all != -1)
x = x_all[observed_mask]
y = y_all[observed_mask]

# Remap class labels
y_unique = np.unique(y)
y_map = {old: new for new, old in enumerate(y_unique)}
y = np.array([y_map[label] for label in y], dtype=np.int64)

# Keep corresponding valid indices for edge filtering
observed_indices = np.nonzero(observed_mask)[0]
valid_observed_indices = np.intersect1d(valid_indices, observed_indices)
valid_set = set(valid_observed_indices)

# Load and reindex edge list
edge_data = np.load(edge_feature_file)
source = edge_data['source']
target = edge_data['target']

edge_mask = np.isin(source, list(valid_set)) & np.isin(target, list(valid_set))
source_filtered = source[edge_mask]
target_filtered = target[edge_mask]

old_to_new = {old: new for new, old in enumerate(valid_observed_indices)}
source_reindexed = np.array([old_to_new[s] for s in source_filtered])
target_reindexed = np.array([old_to_new[t] for t in target_filtered])

edge_index = torch.tensor([source_reindexed, target_reindexed], dtype=torch.long)
np.savez_compressed(edge_index_file, edge_index=edge_index.cpu().numpy())

# Slice final x and y for the observed valid nodes
x = x_all[valid_observed_indices]
y = y_all[valid_observed_indices]

# Final label mapping after slicing
y_unique = np.unique(y)
y_map = {old: new for new, old in enumerate(y_unique)}
y = np.array([y_map[label] for label in y], dtype=np.int64)

# Convert node data
y_tensor = torch.tensor(y, dtype=torch.long)
x_tensor = torch.tensor(x, dtype=torch.float)

#===================GCN model definition========================
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(x, edge_index)
        return x

#===================Bootstrap Evaluation========================
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
all_acc, all_f1, all_auroc, all_auprc = [], [], [], []

for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    train_idx, val_idx, test_idx = splits[0][0], splits[1][1], splits[2][1]
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = torch.zeros(len(y), dtype=torch.bool)
    data.val_mask = torch.zeros(len(y), dtype=torch.bool)
    data.test_mask = torch.zeros(len(y), dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(x.shape[1], 64, len(np.unique(y))).to(device)
    data = data.to(device)

    y_train_np = y_tensor[train_idx].cpu().numpy().astype(np.int64)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_np), y=y_train_np)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
        prob = out.softmax(dim=1).cpu().numpy()
        true = data.y.cpu().numpy()
        mask = data.test_mask.cpu().numpy()
        y_true, y_pred, y_score = true[mask], pred[mask], prob[mask]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        try:
            auroc = roc_auc_score(y_true, y_score, multi_class='ovr')
        except:
            auroc = float('nan')
        try:
            auprc = average_precision_score(y_true, y_score, average='weighted')
        except:
            auprc = float('nan')
        return acc, f1, auroc, auprc

    for epoch in range(1, 201):
        loss = train()

    acc, f1, auroc, auprc = test()
    all_acc.append(acc)
    all_f1.append(f1)
    all_auroc.append(auroc)
    all_auprc.append(auprc)

#===================Print and Save Summary=====================
def summarize(metric_list):
    return np.mean(metric_list), np.std(metric_list)

summary_text = []
summary_text.append("\n✅ Bootstrap Results (10 seeds)")
summary_text.append("Accuracy:  {:.4f} ± {:.4f}".format(*summarize(all_acc)))
summary_text.append("F1 Score:  {:.4f} ± {:.4f}".format(*summarize(all_f1)))
summary_text.append("AUROC:     {:.4f} ± {:.4f}".format(*summarize(all_auroc)))
summary_text.append("AUPRC:     {:.4f} ± {:.4f}".format(*summarize(all_auprc)))

for line in summary_text:
    print(line)

# Save to file
output_file = os.path.join(directory, "GCN_results.txt")
with open(output_file, "w") as f:
    for line in summary_text:
        f.write(line + "\n")


'''
import sys
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

#===================Inputs=====================================
directory = sys.argv[1]
node_feature_file = f"{directory}/node_features_pca.npz"
edge_feature_file = f"{directory}/edge_features.npz"
valid_indices_file = f"{directory}/valid_indices.npy"

#====================Outputs===================================
edge_index_file = f"{directory}/edge_index_filtered.npz"

#===================Load node features=========================
node_data = np.load(node_feature_file)
embedding = node_data['embedding']
mutation_pca = node_data['mutation_vector_pca']
depth = node_data['depth']
date = node_data['date']
is_hypothetical = node_data['is_hypothetical']
y = node_data['lineage_label']

valid_indices = np.load(valid_indices_file)

# Concatenate all features
x = np.concatenate([embedding, mutation_pca, depth, date, is_hypothetical], axis=1)
x = x[valid_indices]
y = y[valid_indices]

# Remap class labels
y_unique = np.unique(y)
y_map = {old: new for new, old in enumerate(y_unique)}
y = np.array([y_map[label] for label in y], dtype=np.int64)

# Load and reindex edge list
edge_data = np.load(edge_feature_file)
source = edge_data['source']
target = edge_data['target']

# Filter and reindex edges
valid_set = set(valid_indices)
edge_mask = np.isin(source, valid_indices) & np.isin(target, valid_indices)
source_filtered = source[edge_mask]
target_filtered = target[edge_mask]
old_to_new = {old: new for new, old in enumerate(valid_indices)}
source_reindexed = np.array([old_to_new[s] for s in source_filtered])
target_reindexed = np.array([old_to_new[t] for t in target_filtered])
edge_index = torch.tensor([source_reindexed, target_reindexed], dtype=torch.long)
np.savez_compressed(edge_index_file, edge_index=edge_index.cpu().numpy())

# Convert node data
y_tensor = torch.tensor(y, dtype=torch.long)
x_tensor = torch.tensor(x, dtype=torch.float)

#===================GCN model definition========================
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(x, edge_index)
        return x

#===================Bootstrap Evaluation========================
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
all_acc, all_f1, all_auroc, all_auprc = [], [], [], []

for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    train_idx, val_idx, test_idx = splits[0][0], splits[1][1], splits[2][1]
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = torch.zeros(len(y), dtype=torch.bool)
    data.val_mask = torch.zeros(len(y), dtype=torch.bool)
    data.test_mask = torch.zeros(len(y), dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(x.shape[1], 64, len(np.unique(y))).to(device)
    data = data.to(device)

    y_train_np = y_tensor[train_idx].cpu().numpy()
    y_train_np = np.array(y_train_np, dtype=np.int64)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_np), y=y_train_np)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
        prob = out.softmax(dim=1).cpu().numpy()
        true = data.y.cpu().numpy()
        mask = data.test_mask.cpu().numpy()
        y_true, y_pred, y_score = true[mask], pred[mask], prob[mask]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        try:
            auroc = roc_auc_score(y_true, y_score, multi_class='ovr')
        except:
            auroc = float('nan')
        try:
            auprc = average_precision_score(y_true, y_score, average='weighted')
        except:
            auprc = float('nan')
        return acc, f1, auroc, auprc

    for epoch in range(1, 201):
        loss = train()

    acc, f1, auroc, auprc = test()
    all_acc.append(acc)
    all_f1.append(f1)
    all_auroc.append(auroc)
    all_auprc.append(auprc)

#===================Print and Save Summary=====================
def summarize(metric_list):
    return np.mean(metric_list), np.std(metric_list)

summary_text = []
summary_text.append("\n✅ Bootstrap Results (10 seeds)")
summary_text.append("Accuracy:  {:.4f} ± {:.4f}".format(*summarize(all_acc)))
summary_text.append("F1 Score:  {:.4f} ± {:.4f}".format(*summarize(all_f1)))
summary_text.append("AUROC:     {:.4f} ± {:.4f}".format(*summarize(all_auroc)))
summary_text.append("AUPRC:     {:.4f} ± {:.4f}".format(*summarize(all_auprc)))

for line in summary_text:
    print(line)

# Save to file
output_file = os.path.join(directory, "GCN_results.txt")
with open(output_file, "w") as f:
    for line in summary_text:
        f.write(line + "\n")

'''