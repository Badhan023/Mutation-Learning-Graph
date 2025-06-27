import numpy as np
import torch
from torch_geometric.data import Data
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch.nn as nn
import os

# ------------------ Config ------------------
dataset_dir = sys.argv[1]

# ------------------ Load Node Features ------------------
node_data = np.load(f"{dataset_dir}/node_features_pca.npz")
valid_indices = np.load(f"{dataset_dir}/valid_indices.npy")

embedding = node_data['embedding']
mutation_pca = node_data['mutation_vector_pca']
depth = node_data['depth']
date = node_data['date']
is_hypothetical = node_data['is_hypothetical']
y_all = node_data['lineage_label']

# Filter observed (non-hypothetical) and valid indices
observed_mask = (y_all != -1)
observed_indices = np.nonzero(observed_mask)[0]
valid_observed_indices = np.intersect1d(valid_indices, observed_indices)

embedding = embedding[valid_observed_indices]
mutation_pca = mutation_pca[valid_observed_indices]
depth = depth[valid_observed_indices].reshape(-1, 1)
date = date[valid_observed_indices].reshape(-1, 1)
is_hypothetical = is_hypothetical[valid_observed_indices].reshape(-1, 1).astype(np.float32)
y = y_all[valid_observed_indices]

# Normalize and fill date
date_mean = date[~np.isnan(date)].mean()
date[np.isnan(date)] = date_mean
date = (date - date.mean()) / date.std()

# Concatenate features
x = np.concatenate([embedding, mutation_pca, depth, date, is_hypothetical], axis=1)

# Relabel classes
y_unique = np.unique(y)
y_map = {old: new for new, old in enumerate(y_unique)}
y = np.array([y_map[label] for label in y], dtype=np.int64)

x_tensor = torch.tensor(x, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

# ------------------ Load Edge Index ------------------
edge_data = np.load(f"{dataset_dir}/edge_index_filtered.npz")
edge_index = torch.tensor(edge_data['edge_index'], dtype=torch.long)

# ------------------ Define GraphSAGE Model ------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ------------------ Bootstrap Seeds ------------------
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
all_acc, all_f1, all_auroc, all_auprc = [], [], [], []

for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Split indices
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    train_idx, test_idx = splits[0]
    val_idx = test_idx[:len(test_idx) // 2]
    test_idx = test_idx[len(test_idx) // 2:]

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx   = torch.tensor(val_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx, dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = torch.zeros(len(y), dtype=torch.bool)
    data.val_mask = torch.zeros(len(y), dtype=torch.bool)
    data.test_mask = torch.zeros(len(y), dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    # Class weights
    y_train_np = y_tensor[train_idx].cpu().numpy().astype(np.int64)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_np), y=y_train_np)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(x.shape[1], 128, len(y_unique)).to(device)
    data = data.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
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
        pred = out.argmax(dim=1)
        y_true = data.y.cpu().numpy()
        y_score = out.softmax(dim=1).cpu().numpy()
        y_pred = pred.cpu().numpy()

        mask = data.test_mask.cpu().numpy()
        true = y_true[mask]
        pred_ = y_pred[mask]
        score_ = y_score[mask]

        acc = accuracy_score(true, pred_)
        f1 = f1_score(true, pred_, average='weighted', zero_division=0)

        true_bin = label_binarize(true, classes=np.arange(len(y_unique)))
        try:
            auroc = roc_auc_score(true_bin, score_, average='macro', multi_class='ovr')
        except:
            auroc = float('nan')
        try:
            auprc = average_precision_score(true_bin, score_, average='macro')
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

# ------------------ Report ------------------
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
with open(os.path.join(dataset_dir, "graphSAGE_node_results.txt"), "w") as f:
    for line in summary_text:
        f.write(line + "\n")


'''
import numpy as np
import torch
from torch_geometric.data import Data
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch.nn as nn
import os

# ------------------ Config ------------------
dataset_dir = sys.argv[1]
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

# ------------------ Load Edge Index ------------------
edge_data = np.load(f"{dataset_dir}/edge_index_filtered.npz")
edge_index = torch.tensor(edge_data['edge_index'], dtype=torch.long)

# ------------------ Define GraphSAGE Model ------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ------------------ Bootstrap Seeds ------------------
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
all_acc, all_f1, all_auroc, all_auprc = [], [], [], []

for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Split indices
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    train_idx, test_idx = splits[0]
    val_idx = test_idx[:len(test_idx) // 2]
    test_idx = test_idx[len(test_idx) // 2:]

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx   = torch.tensor(val_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx, dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = torch.zeros(len(y), dtype=torch.bool)
    data.val_mask = torch.zeros(len(y), dtype=torch.bool)
    data.test_mask = torch.zeros(len(y), dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    # Class weights
    y_train_np = y_tensor[train_idx].detach().cpu().numpy().astype(int).tolist()
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_np), y=y_train_np)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(x.shape[1], 128, len(unique_labels)).to(device)
    data = data.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
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
        pred = out.argmax(dim=1)
        y_true = data.y.cpu().numpy()
        y_score = out.softmax(dim=1).cpu().numpy()
        y_pred = pred.cpu().numpy()

        mask = data.test_mask.cpu().numpy()
        true = y_true[mask]
        pred_ = y_pred[mask]
        score_ = y_score[mask]

        acc = accuracy_score(true, pred_)
        f1 = f1_score(true, pred_, average='weighted', zero_division=0)

        true_bin = label_binarize(true, classes=np.arange(len(unique_labels)))
        try:
            auroc = roc_auc_score(true_bin, score_, average='macro', multi_class='ovr')
        except:
            auroc = float('nan')
        try:
            auprc = average_precision_score(true_bin, score_, average='macro')
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

# ------------------ Report ------------------
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
with open(os.path.join(dataset_dir, "graphSAGE_node_results.txt"), "w") as f:
    for line in summary_text:
        f.write(line + "\n")

'''