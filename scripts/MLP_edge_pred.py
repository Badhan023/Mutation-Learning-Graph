# MLP_edge_pred.py  (fixed alignment with edge_features)
import numpy as np
import torch
import sys, os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
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

# Map original -> compact idx (old graph indices -> [0..len(valid_idx)-1])
old_to_new = {int(old): int(new) for new, old in enumerate(valid_idx)}

# ------------------ Load positive edges (and align features!) ------------------
# We iterate over edge_index_filtered *in its original order* and remember which rows are valid.
ei = np.load(f"{dataset_dir}/edge_index_filtered.npz")['edge_index']
pos_src, pos_dst = [], []
valid_edge_attr_indices = []  # indices into edge_features arrays that correspond to the kept edges

for i, (u, v) in enumerate(zip(ei[0], ei[1])):
    u = int(u); v = int(v)
    if u in old_to_new and v in old_to_new and old_to_new[u] != old_to_new[v]:
        pos_src.append(old_to_new[u])
        pos_dst.append(old_to_new[v])
        valid_edge_attr_indices.append(i)

pos_pairs = np.stack([np.array(pos_src, dtype=np.int64),
                      np.array(pos_dst, dtype=np.int64)], axis=1)
num_pos = pos_pairs.shape[0]

# Edge features: take the SAME rows (valid_edge_attr_indices) so they align with pos_pairs
ef = np.load(f"{dataset_dir}/edge_features.npz")
mutation_count      = ef['mutation_count'][valid_edge_attr_indices]
edit_distance       = ef['edit_distance'][valid_edge_attr_indices]
mutation_similarity = ef['mutation_similarity'][valid_edge_attr_indices]
reverse_flag        = ef['reverse'][valid_edge_attr_indices]
edge_attr_pos = np.stack([mutation_count, edit_distance, mutation_similarity, reverse_flag],
                         axis=1).astype(np.float32)

# ------------------ Utilities ------------------
pos_set = set((int(a), int(b)) for a, b in pos_pairs)

def sample_random_negatives(m, rng):
    neg = []
    while len(neg) < m:
        a = int(rng.integers(0, N))
        b = int(rng.integers(0, N))
        if a != b and (a, b) not in pos_set:
            neg.append((a, b))
    return np.array(neg, dtype=np.int64)

def build_edge_rows(pos_pairs, neg_pairs, pos_edge_attrs):
    """
    Rows: [X_u || X_v || edge_attr]. For negatives, we cycle positive attrs to keep shapes consistent
    (mirrors practice in your other scripts).
    """
    def node_pair_rows(pairs):
        u, v = pairs[:, 0], pairs[:, 1]
        return np.concatenate([X_nodes[u], X_nodes[v]], axis=1).astype(np.float32)

    X_pos = node_pair_rows(pos_pairs)
    A_pos = pos_edge_attrs.astype(np.float32)

    if len(neg_pairs) > 0:
        X_neg = node_pair_rows(neg_pairs)
        rep = int(np.ceil(len(neg_pairs) / max(1, len(A_pos))))
        A_neg = np.tile(A_pos, (rep, 1))[:len(neg_pairs)]
    else:
        X_neg = np.zeros((0, X_nodes.shape[1]*2), dtype=np.float32)
        A_neg = np.zeros((0, A_pos.shape[1] if A_pos.ndim == 2 else 4), dtype=np.float32)

    X_pos_full = np.concatenate([X_pos, A_pos], axis=1)
    X_neg_full = np.concatenate([X_neg, A_neg], axis=1)
    return X_pos_full, X_neg_full

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)

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
    f1w = f1_score(y, preds, average='weighted', zero_division=0)
    try:
        auroc = roc_auc_score(y, probs)
    except Exception:
        auroc = float('nan')
    try:
        auprc = average_precision_score(y, probs)
    except Exception:
        auprc = float('nan')
    return acc, f1w, auroc, auprc

# ------------------ Bootstrapped Runs ------------------
all_acc, all_f1, all_auroc, all_auprc = [], [], [], []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    np.random.seed(seed); torch.manual_seed(seed)

    # Random 60/20/20 split over *positives*, and slice edge_attr_pos the same way
    idx = np.arange(num_pos)
    rng.shuffle(idx)
    train_end = int(0.6 * num_pos)
    val_end   = int(0.8 * num_pos)

    pos_train = pos_pairs[idx[:train_end]]
    pos_val   = pos_pairs[idx[train_end:val_end]]
    pos_test  = pos_pairs[idx[val_end:]]

    attr_train = edge_attr_pos[idx[:train_end]]
    attr_val   = edge_attr_pos[idx[train_end:val_end]]
    attr_test  = edge_attr_pos[idx[val_end:]]

    # 1:1 random negatives (per split)
    neg_train = sample_random_negatives(len(pos_train), rng)
    neg_val   = sample_random_negatives(len(pos_val),   rng)
    neg_test  = sample_random_negatives(len(pos_test),  rng)

    # Build edge rows
    X_pos_tr, X_neg_tr = build_edge_rows(pos_train, neg_train, attr_train)
    X_pos_va, X_neg_va = build_edge_rows(pos_val,   neg_val,   attr_val)
    X_pos_te, X_neg_te = build_edge_rows(pos_test,  neg_test,  attr_test)

    X_train = np.concatenate([X_pos_tr, X_neg_tr], axis=0)
    y_train = np.concatenate([np.ones(len(X_pos_tr), dtype=np.int64),
                              np.zeros(len(X_neg_tr), dtype=np.int64)], axis=0)

    X_val = np.concatenate([X_pos_va, X_neg_va], axis=0)
    y_val = np.concatenate([np.ones(len(X_pos_va), dtype=np.int64),
                            np.zeros(len(X_neg_va), dtype=np.int64)], axis=0)

    X_test = np.concatenate([X_pos_te, X_neg_te], axis=0)
    y_test = np.concatenate([np.ones(len(X_pos_te), dtype=np.int64),
                             np.zeros(len(X_neg_te), dtype=np.int64)], axis=0)

    # Model
    in_dim = X_train.shape[1]  # 2*d_node + 4 edge features
    model = MLP(in_channels=in_dim, hidden_channels=64, out_channels=1).to(device)
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

    acc, f1w, auroc, auprc = evaluate(model, X_test, y_test, device)
    all_acc.append(acc); all_f1.append(f1w); all_auroc.append(auroc); all_auprc.append(auprc)

def summarize(a):
    arr = np.array(a, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))

acc_m, acc_s = summarize(all_acc)
f1_m, f1_s = summarize(all_f1)
auc_m, auc_s = summarize(all_auroc)
auprc_m, auprc_s = summarize(all_auprc)

lines = [
    "MLP Edge Prediction (Random split + Random negatives + Edge feats) — 10 seeds",
    "Accuracy:  {:.4f} ± {:.4f}".format(acc_m, acc_s),
    "F1 Score:  {:.4f} ± {:.4f}".format(f1_m, f1_s),
    "AUROC:     {:.4f} ± {:.4f}".format(auc_m, auc_s),
    "AUPRC:     {:.4f} ± {:.4f}".format(auprc_m, auprc_s),
]

with open(output_file, "w") as f:
    for ln in lines: f.write(ln + "\n")
for ln in lines: print(ln)
