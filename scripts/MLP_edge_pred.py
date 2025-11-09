# MLP_edge_pred.py  (fair, symmetric pairwise features; unique negatives; train-only standardization)
import numpy as np
import torch
import sys, os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Config ------------------
dataset_dir = sys.argv[1]
output_file = os.path.join(dataset_dir, "mlp_edge_results.txt")
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
batch_size = 4096
epochs = 200
lr = 0.01
weight_decay = 5e-4

# ------------------ Load Node Features ------------------
node = np.load(f"{dataset_dir}/node_features_pca.npz")
valid_idx = np.load(f"{dataset_dir}/valid_indices.npy")

embedding = node['embedding'][valid_idx]
mutation_pca = node['mutation_vector_pca'][valid_idx]
depth = node['depth'][valid_idx].reshape(-1, 1)

X_nodes = np.concatenate([embedding, mutation_pca, depth], axis=1).astype(np.float32)
N, d_node = X_nodes.shape

# Map original -> compact idx
old_to_new = {int(old): int(new) for new, old in enumerate(valid_idx)}

# ------------------ Load positive edges (aligned to features) ------------------
ei = np.load(f"{dataset_dir}/edge_index_filtered.npz")['edge_index']
pos_src, pos_dst = [], []
valid_edge_attr_indices = []
for i, (u, v) in enumerate(zip(ei[0], ei[1])):
    u = int(u); v = int(v)
    if u in old_to_new and v in old_to_new and old_to_new[u] != old_to_new[v]:
        pos_src.append(old_to_new[u])
        pos_dst.append(old_to_new[v])
        valid_edge_attr_indices.append(i)

pos_pairs = np.stack([np.array(pos_src, dtype=np.int64),
                      np.array(pos_dst, dtype=np.int64)], axis=1)
num_pos = pos_pairs.shape[0]

# (We do NOT use edge_features.npz here to avoid unfairness for negatives.)

# ------------------ Utilities ------------------
pos_set = set((int(a), int(b)) for a, b in pos_pairs)

def sample_random_negatives(m, rng):
    """Unique random negatives not in pos_set and no self-loops."""
    neg = set()
    while len(neg) < m:
        a = int(rng.integers(0, N))
        b = int(rng.integers(0, N))
        if a != b and (a, b) not in pos_set and (a, b) not in neg:
            neg.add((a, b))
    neg = np.array(list(neg), dtype=np.int64)
    return neg

def pairwise_features(X_nodes, pairs):
    """Symmetric pairwise features for both pos/neg:
       [abs(Xu - Xv), Xu * Xv, cosine_sim, l2_dist]
    """
    u = pairs[:, 0]
    v = pairs[:, 1]
    Xu = X_nodes[u]   # [M, d]
    Xv = X_nodes[v]   # [M, d]

    # elementwise symmetric components
    abs_diff = np.abs(Xu - Xv)
    prod = Xu * Xv

    # cosine similarity (scalar)
    Xu_n = Xu / (np.linalg.norm(Xu, axis=1, keepdims=True) + 1e-8)
    Xv_n = Xv / (np.linalg.norm(Xv, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.sum(Xu_n * Xv_n, axis=1, keepdims=True)

    # l2 distance (scalar)
    l2 = np.linalg.norm(Xu - Xv, axis=1, keepdims=True)

    feats = np.concatenate([abs_diff, prod, cos_sim, l2], axis=1).astype(np.float32)
    return feats

def iterate_minibatches(X, y, batch_size, shuffle=True, rng_=None):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        (rng_ if rng_ is not None else np.random).shuffle(idx)
    for start in range(0, n, batch_size):
        sel = idx[start:start+batch_size]
        yield X[sel], y[sel]

@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    logits_list = []
    for xb, _ in iterate_minibatches(X, y, batch_size, shuffle=False):
        xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
        logits_list.append(model(xb_t).detach().cpu().numpy())
    logits = np.vstack(logits_list).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    acc = accuracy_score(y, preds)
    f1b = f1_score(y, preds, average='binary', zero_division=0)
    try:
        auroc = roc_auc_score(y, probs)
    except Exception:
        auroc = float('nan')
    try:
        auprc = average_precision_score(y, probs)
    except Exception:
        auprc = float('nan')
    return acc, f1b, auroc, auprc

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=1):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)

# ------------------ Bootstrapped Runs ------------------
all_acc, all_f1, all_auroc, all_auprc = [], [], [], []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    np.random.seed(seed); torch.manual_seed(seed)

    # 60/20/20 split over positives
    idx = np.arange(num_pos)
    rng.shuffle(idx)
    train_end = int(0.6 * num_pos)
    val_end   = int(0.8 * num_pos)

    pos_train = pos_pairs[idx[:train_end]]
    pos_val   = pos_pairs[idx[train_end:val_end]]
    pos_test  = pos_pairs[idx[val_end:]]

    # 1:1 unique random negatives per split
    neg_train = sample_random_negatives(len(pos_train), rng)
    neg_val   = sample_random_negatives(len(pos_val),   rng)
    neg_test  = sample_random_negatives(len(pos_test),  rng)

    # Build symmetric pairwise features
    X_pos_tr = pairwise_features(X_nodes, pos_train)
    X_neg_tr = pairwise_features(X_nodes, neg_train)
    X_pos_va = pairwise_features(X_nodes, pos_val)
    X_neg_va = pairwise_features(X_nodes, neg_val)
    X_pos_te = pairwise_features(X_nodes, pos_test)
    X_neg_te = pairwise_features(X_nodes, neg_test)

    X_train = np.concatenate([X_pos_tr, X_neg_tr], axis=0)
    y_train = np.concatenate([np.ones(len(X_pos_tr), dtype=np.int64),
                              np.zeros(len(X_neg_tr), dtype=np.int64)], axis=0)
    X_val = np.concatenate([X_pos_va, X_neg_va], axis=0)
    y_val = np.concatenate([np.ones(len(X_pos_va), dtype=np.int64),
                            np.zeros(len(X_neg_va), dtype=np.int64)], axis=0)
    X_test = np.concatenate([X_pos_te, X_neg_te], axis=0)
    y_test = np.concatenate([np.ones(len(X_pos_te), dtype=np.int64),
                             np.zeros(len(X_neg_te), dtype=np.int64)], axis=0)

    # Standardize (fit on train only)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Model
    in_dim = X_train.shape[1]  # 2*d_node + 2 scalars
    model = MLP(in_channels=in_dim, hidden_channels=128, out_channels=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()

    local_rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, shuffle=True, rng_=local_rng):
            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            yb_t = torch.tensor(yb.reshape(-1, 1).astype(np.float32), dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = crit(model(xb_t), yb_t)
            loss.backward()
            opt.step()

    acc, f1b, auroc, auprc = evaluate(model, X_test, y_test, device)
    all_acc.append(acc); all_f1.append(f1b); all_auroc.append(auroc); all_auprc.append(auprc)

def summarize(a):
    arr = np.array(a, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))

acc_m, acc_s = summarize(all_acc)
f1_m, f1_s = summarize(all_f1)
auc_m, auc_s = summarize(all_auroc)
auprc_m, auprc_s = summarize(all_auprc)

lines = [
    "MLP Edge Prediction",
    "Accuracy:  {:.4f} ± {:.4f}".format(acc_m, acc_s),
    "F1 Score:  {:.4f} ± {:.4f}".format(f1_m, f1_s),
    "AUROC:     {:.4f} ± {:.4f}".format(auc_m, auc_s),
    "AUPRC:     {:.4f} ± {:.4f}".format(auprc_m, auprc_s),
]

with open(output_file, "w") as f:
    for ln in lines: f.write(ln + "\n")
for ln in lines: print(ln)

