import sys
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

#===================Inputs=====================================
directory = sys.argv[1]
node_feature_file = f"{directory}/node_features_pca.npz"
edge_feature_file = f"{directory}/edge_features.npz"
valid_indices_file = f"{directory}/valid_indices.npy"

#====================Outputs===================================
edge_index_file = f"{directory}/edge_index_filtered.npz"
summary_file = os.path.join(directory, "GCN_results.txt")

#===================Load node features=========================
node_data = np.load(node_feature_file)
embedding = node_data["embedding"]                         # (N, d1)
mutation_pca = node_data["mutation_vector_pca"]            # (N, d2)
depth = node_data["depth"]                                 # (N,) or (N,1)
date = node_data["date"]                                   # (N,) or (N,1)
is_hypothetical = node_data["is_hypothetical"]             # (N,) boole/int
y_all = node_data["lineage_label"]                         # (N,) -1 for hypo

# Ensure 2D for concat (robust to 1D inputs)
def col2d(a):
    a = np.asarray(a)
    return a if a.ndim == 2 else a.reshape(-1, 1)

x_all = np.concatenate([
    embedding,
    mutation_pca,
    col2d(depth),
    col2d(date),
    col2d(is_hypothetical.astype(np.float32)),
], axis=1)

valid_indices = np.load(valid_indices_file)

# Filter only observed (non-hypothetical) nodes
observed_mask = (y_all != -1)
observed_indices = np.nonzero(observed_mask)[0]

# Build the set of valid observed nodes
valid_observed_indices = np.intersect1d(valid_indices, observed_indices)  # sorted
valid_set = set(valid_observed_indices.tolist())

#===================Load and reindex edge list=================
edge_data = np.load(edge_feature_file)
source = edge_data["source"]
target = edge_data["target"]

edge_mask = np.isin(source, valid_observed_indices) & np.isin(target, valid_observed_indices)
source_filtered = source[edge_mask]
target_filtered = target[edge_mask]

old_to_new = {old: new for new, old in enumerate(valid_observed_indices)}
source_reindexed = np.fromiter((old_to_new[s] for s in source_filtered), dtype=np.int64)
target_reindexed = np.fromiter((old_to_new[t] for t in target_filtered), dtype=np.int64)

edge_index = torch.tensor([source_reindexed, target_reindexed], dtype=torch.long)
# Make undirected for GCN stability (optional but recommended)
edge_index = to_undirected(edge_index)

# Save filtered edge_index for record-keeping
np.savez_compressed(edge_index_file, edge_index=edge_index.cpu().numpy())

#===================Slice final node arrays====================
x = x_all[valid_observed_indices]
y = y_all[valid_observed_indices]

# Final label mapping after slicing
y_unique = np.unique(y)
y_map = {old: new for new, old in enumerate(y_unique)}
y = np.array([y_map[label] for label in y], dtype=np.int64)

# Tensors
x_tensor = torch.tensor(x, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

#===================GCN model definition========================
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(x, edge_index)
        return x

#===================Bootstrap Evaluation========================
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
all_acc, all_f1, all_auroc, all_auprc = [], [], [], []

num_classes = len(np.unique(y))
classes_arr = np.arange(num_classes)

for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build 3 disjoint folds: fold0=train, fold1=val, fold2=test (no leakage)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    folds = []
    for _, test_idx_ in skf.split(np.zeros(len(y)), y):
        folds.append(test_idx_)
    # folds[0], folds[1], folds[2] are the three disjoint index sets
    train_idx = np.concatenate([folds[0], folds[1]])   # two folds
    val_idx   = folds[2]                               # one fold
    test_idx  = folds[1]                               # use a held-out fold (different from val)
    # If you prefer train=fold0+fold2, val=fold1, test=fold2, adjust accordingly.

    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t   = torch.tensor(val_idx, dtype=torch.long)
    test_idx_t  = torch.tensor(test_idx, dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = torch.zeros(len(y), dtype=torch.bool)
    data.val_mask   = torch.zeros(len(y), dtype=torch.bool)
    data.test_mask  = torch.zeros(len(y), dtype=torch.bool)
    data.train_mask[train_idx_t] = True
    data.val_mask[val_idx_t]     = True
    data.test_mask[test_idx_t]   = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(x.shape[1], 64, num_classes).to(device)
    data = data.to(device)

    # Class weights from the actual train labels present in this split
    y_train_np = y_tensor[train_idx].cpu().numpy().astype(np.int64)
    present_classes = np.unique(y_train_np)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train_np
    )
    # Map weights to full class set
    weight_full = np.ones(num_classes, dtype=np.float32)
    for c, w in zip(present_classes, class_weights):
        weight_full[c] = w
    class_weights_t = torch.tensor(weight_full, dtype=torch.float, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train_one():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def eval_split(mask):
        model.eval()
        out = model(data.x, data.edge_index)
        prob = out.softmax(dim=1).cpu().numpy()
        pred = out.argmax(dim=1).cpu().numpy()
        true = data.y.cpu().numpy()
        m = mask.cpu().numpy()

        y_true = true[m]
        y_pred = pred[m]
        y_score = prob[m]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Safe multiclass AUROC (may be undefined if a class absent in y_true)
        try:
            auroc = roc_auc_score(
                y_true,
                y_score,
                multi_class="ovr",
                labels=classes_arr if len(np.unique(y_true)) > 1 else None
            )
        except Exception:
            auroc = float("nan")

        # Multiclass AUPRC: binarize labels One-vs-Rest and do macro average
        try:
            y_true_bin = label_binarize(y_true, classes=classes_arr)
            auprc_per_class = []
            for c in range(num_classes):
                if y_true_bin[:, c].sum() == 0:
                    continue
                auprc_c = average_precision_score(y_true_bin[:, c], y_score[:, c])
                auprc_per_class.append(auprc_c)
            auprc = float(np.mean(auprc_per_class)) if auprc_per_class else float("nan")
        except Exception:
            auprc = float("nan")

        return acc, f1, auroc, auprc

    # Train
    for epoch in range(1, 201):
        loss = train_one()
        if epoch % 10 == 0:
            print(f"Seed {seed} | Epoch {epoch:03d} | Loss: {loss:.4f}")

    # Test on held-out test fold
    acc, f1, auroc, auprc = eval_split(data.test_mask)
    all_acc.append(acc); all_f1.append(f1); all_auroc.append(auroc); all_auprc.append(auprc)

#===================Print and Save Summary=====================
def summarize(metric_list):
    arr = np.array(metric_list, dtype=np.float64)
    return np.nanmean(arr), np.nanstd(arr)

summary_lines = []
summary_lines.append("\nBootstrap Results (10 seeds)")
summary_lines.append("Accuracy:  {:.4f} ± {:.4f}".format(*summarize(all_acc)))
summary_lines.append("F1 Score:  {:.4f} ± {:.4f}".format(*summarize(all_f1)))
summary_lines.append("AUROC:     {:.4f} ± {:.4f}".format(*summarize(all_auroc)))
summary_lines.append("AUPRC:     {:.4f} ± {:.4f}".format(*summarize(all_auprc)))

for line in summary_lines:
    print(line)

with open(summary_file, "w") as f:
    for line in summary_lines:
        f.write(line + "\n")
