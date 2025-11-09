import numpy as np
import torch
import sys, os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Config ------------------
dataset_dir = sys.argv[1]
use_date = True
use_is_hypothetical = True

# ------------------ Tiny-class safe 60/20/20 split ------------------
def stratified_split_60_20_20_safe(y, seed):
    """
    Returns (train_idx, val_idx, test_idx) as np.int64 arrays.
    - n==1: assign to train
    - n==2: 1 train, 1 test
    - n>=3: ~60/20/20 with guards
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

embedding = node_data['embedding'][valid_indices]
mutation_pca = node_data['mutation_vector_pca'][valid_indices]
depth = node_data['depth'][valid_indices].reshape(-1, 1)
labels_all = node_data['lineage_label'][valid_indices]

# Filter observed (non-hypothetical) nodes
observed_mask = (labels_all != -1)
embedding = embedding[observed_mask]
mutation_pca = mutation_pca[observed_mask]
depth = depth[observed_mask]
labels_all = labels_all[observed_mask]

features = [embedding, mutation_pca, depth]

if use_date:
    date = node_data['date'][valid_indices].reshape(-1, 1)[observed_mask]
    date_mean = np.nanmean(date)
    date[np.isnan(date)] = date_mean
    date_std = float(np.std(date))
    if date_std < 1e-8:
        date_std = 1.0
    date = (date - float(np.mean(date))) / date_std
    features.append(date)

if use_is_hypothetical:
    # Here it should be all zeros after filtering, but kept for consistency
    is_hypothetical = node_data['is_hypothetical'][valid_indices].reshape(-1, 1).astype(np.float32)[observed_mask]
    features.append(is_hypothetical)

x = np.concatenate(features, axis=1)
y = labels_all

# Remap class labels to 0..C-1
unique_labels = np.unique(y)
label_map = {old: new for new, old in enumerate(unique_labels)}
y = np.array([label_map[label] for label in y], dtype=np.int64)
num_classes = len(unique_labels)
classes_arr = np.arange(num_classes)

# Tensors (we'll move to device later)
x_tensor = torch.tensor(x, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

# ------------------ Define MLP Model ------------------
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)

# ------------------ Bootstrap Seeds ------------------
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
all_acc, all_f1, all_auroc, all_auprc = [], [], [], []

def eval_on(model, x_tensor, y_tensor, mask, classes_arr):
    model.eval()
    with torch.no_grad():
        out = model(x_tensor)
        prob = out.softmax(dim=1).detach().cpu().numpy()
        pred = out.argmax(dim=1).detach().cpu().numpy()
        true = y_tensor.detach().cpu().numpy()

    m = mask.detach().cpu().numpy().astype(bool)
    y_true, y_pred, y_score = true[m], pred[m], prob[m]

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # AUROC: integer labels + prob with OVR; guard missing classes
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
        for c in classes_arr:
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

    # ---- 60/20/20 stratified split with safe fallback ----
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

    # Convert to boolean masks (device-safe)
    n = len(y)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[torch.tensor(train_idx, dtype=torch.long)] = True
    val_mask[torch.tensor(val_idx, dtype=torch.long)]     = True
    test_mask[torch.tensor(test_idx, dtype=torch.long)]   = True

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
    model = MLP(x.shape[1], hidden_channels=64, out_channels=num_classes).to(device)
    x_dev = x_tensor.to(device)
    y_dev = y_tensor.to(device)
    train_mask_dev = train_mask.to(device)
    val_mask_dev   = val_mask.to(device)
    test_mask_dev  = test_mask.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train_one():
        model.train()
        optimizer.zero_grad()
        out = model(x_dev)
        loss = criterion(out[train_mask_dev], y_dev[train_mask_dev])
        loss.backward()
        optimizer.step()
        return loss.item()

    for epoch in range(1, 201):
        _ = train_one()

    acc, f1, auroc, auprc = eval_on(model, x_dev, y_dev, test_mask_dev, classes_arr)
    all_acc.append(acc); all_f1.append(f1); all_auroc.append(auroc); all_auprc.append(auprc)

# ------------------ Report ------------------
def summarize(v):
    arr = np.array(v, dtype=np.float64)
    return np.nanmean(arr), np.nanstd(arr)

summary_text = [
    "\nMLP Bootstrap Results",
    "Accuracy:  {:.4f} ± {:.4f}".format(*summarize(all_acc)),
    "F1 Score:  {:.4f} ± {:.4f}".format(*summarize(all_f1)),
    "AUROC:     {:.4f} ± {:.4f}".format(*summarize(all_auroc)),
    "AUPRC:     {:.4f} ± {:.4f}".format(*summarize(all_auprc)),
]
print("\n".join(summary_text))
with open(os.path.join(dataset_dir, "mlp_node_results.txt"), "w") as f:
    f.write("\n".join(summary_text) + "\n")

