import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected
import sys, os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import torch.nn as nn

# ------------------ Config ------------------
dataset_dir = sys.argv[1]

# ------------------ Tiny-class safe 60/20/20 split ------------------
def stratified_split_60_20_20_safe(y, seed):
    """
    Returns (train_idx, val_idx, test_idx) as np.int64 arrays.
    - n==1 in a class: assign to train
    - n==2: 1 train, 1 test
    - n>=3: ~60/20/20 with guards to ensure non-empty test
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    classes = np.unique(y)
    train, val, test = [], [], []
    for c in classes:
        c_idx = idx[y == c].copy()
        rng.shuffle(c_idx)
        n = len(c_idx)
        if n == 1:
            train.extend(c_idx)
        elif n == 2:
            train.append(c_idx[0]); test.append(c_idx[1])
        else:
            n_train = int(round(0.6 * n))
            n_val   = int(round(0.2 * n))
            n_train = max(1, min(n - 2, n_train))
            n_val   = max(1, min(n - n_train - 1, n_val))
            n_test  = n - n_train - n_val
            if n_test == 0:
                if n_val > 1:
                    n_val -= 1; n_test = 1
                else:
                    n_train = max(1, n_train - 1); n_test = 1
            tr = c_idx[:n_train]
            va = c_idx[n_train:n_train + n_val]
            te = c_idx[n_train + n_val:]
            train.extend(tr); val.extend(va); test.extend(te)
    return (np.array(train, dtype=np.int64),
            np.array(val,   dtype=np.int64),
            np.array(test,  dtype=np.int64))

# ------------------ Load Node Features ------------------
node_data = np.load(f"{dataset_dir}/node_features_pca.npz")
valid_indices = np.load(f"{dataset_dir}/valid_indices.npy")

embedding       = node_data['embedding']
mutation_pca    = node_data['mutation_vector_pca']
depth           = node_data['depth']
date            = node_data['date']
is_hypothetical = node_data['is_hypothetical']
y_all           = node_data['lineage_label']

# Filter observed (non-hypothetical) and valid indices
observed_mask = (y_all != -1)
observed_indices = np.nonzero(observed_mask)[0]
valid_observed_indices = np.intersect1d(valid_indices, observed_indices)

embedding       = embedding[valid_observed_indices]
mutation_pca    = mutation_pca[valid_observed_indices]
depth           = depth[valid_observed_indices].reshape(-1, 1)
date            = date[valid_observed_indices].reshape(-1, 1)
is_hypothetical = is_hypothetical[valid_observed_indices].reshape(-1, 1).astype(np.float32)
y               = y_all[valid_observed_indices]

# Normalize and fill date (safe)
date_mean = np.nanmean(date)
date[np.isnan(date)] = date_mean
date_std = float(np.std(date))
if date_std < 1e-8:
    date_std = 1.0
date = (date - float(np.mean(date))) / date_std

# Concatenate features
x = np.concatenate([embedding, mutation_pca, depth, date, is_hypothetical], axis=1)

# Relabel classes -> 0..C-1
y_unique = np.unique(y)
y_map = {old: new for new, old in enumerate(y_unique)}
y = np.array([y_map[label] for label in y], dtype=np.int64)
num_classes = len(np.unique(y))
classes_arr = np.arange(num_classes)

x_tensor = torch.tensor(x, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

# ------------------ Load Edge Index ------------------
edge_data = np.load(f"{dataset_dir}/edge_index_filtered.npz")
edge_index = torch.tensor(edge_data['edge_index'], dtype=torch.long)
edge_index = to_undirected(edge_index)  # recommended for SAGE

# ------------------ GraphSAGE ------------------
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

def eval_on(model, data, mask, num_classes, classes_arr):
    model.eval()
    out = model(data.x, data.edge_index)
    prob = out.softmax(dim=1).detach().cpu().numpy()     # detach fix
    pred = out.argmax(dim=1).detach().cpu().numpy()      # detach fix
    true = data.y.cpu().numpy()
    m = mask.cpu().numpy()
    y_true, y_pred, y_score = true[m], pred[m], prob[m]

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # AUROC: integer labels + probs with OVR; guard missing classes
    try:
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro', labels=classes_arr)
        else:
            auroc = float('nan')
    except Exception:
        auroc = float('nan')

    # AUPRC: binarize and macro-average over present classes
    try:
        y_true_bin = label_binarize(y_true, classes=classes_arr)
        auprc_list = []
        for c in range(num_classes):
            if y_true_bin[:, c].sum() == 0:
                continue
            auprc_list.append(average_precision_score(y_true_bin[:, c], y_score[:, c]))
        auprc = float(np.mean(auprc_list)) if auprc_list else float('nan')
    except Exception:
        auprc = float('nan')

    return acc, f1, auroc, auprc

for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 60/20/20 stratified split with safe fallback for tiny classes
    try:
        idx = np.arange(len(y))
        train_idx, temp_idx, y_train, y_temp = train_test_split(
            idx, y, test_size=0.4, stratify=y, random_state=seed
        )
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
        )
    except Exception:
        train_idx, val_idx, test_idx = stratified_split_60_20_20_safe(y, seed)

    # Build masks
    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t   = torch.tensor(val_idx, dtype=torch.long)
    test_idx_t  = torch.tensor(test_idx, dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = torch.zeros(len(y), dtype=torch.bool); data.train_mask[train_idx_t] = True
    data.val_mask   = torch.zeros(len(y), dtype=torch.bool); data.val_mask[val_idx_t]     = True
    data.test_mask  = torch.zeros(len(y), dtype=torch.bool); data.test_mask[test_idx_t]   = True

    # Class weights → full length C
    y_train_np = y[train_idx]
    present = np.unique(y_train_np)
    w_present = compute_class_weight(class_weight='balanced', classes=present, y=y_train_np)
    w_full = np.ones(num_classes, dtype=np.float32)
    for c, w in zip(present, w_present):
        w_full[c] = w
    class_weights = torch.tensor(w_full, dtype=torch.float)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(x.shape[1], 128, num_classes).to(device)
    data = data.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train_one():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    for epoch in range(1, 201):
        _ = train_one()

    acc, f1, auroc, auprc = eval_on(model, data, data.test_mask, num_classes, classes_arr)
    all_acc.append(acc); all_f1.append(f1); all_auroc.append(auroc); all_auprc.append(auprc)

# ------------------ Report ------------------
def summarize(v):
    arr = np.array(v, dtype=np.float64)
    return np.nanmean(arr), np.nanstd(arr)

summary = [
    "\nGraphSAGE Bootstrap Results",
    "Accuracy:  {:.4f} ± {:.4f}".format(*summarize(all_acc)),
    "F1 Score:  {:.4f} ± {:.4f}".format(*summarize(all_f1)),
    "AUROC:     {:.4f} ± {:.4f}".format(*summarize(all_auroc)),
    "AUPRC:     {:.4f} ± {:.4f}".format(*summarize(all_auprc)),
]
print("\n".join(summary))
with open(os.path.join(dataset_dir, "graphSAGE_node_results.txt"), "w") as f:
    f.write("\n".join(summary) + "\n")
