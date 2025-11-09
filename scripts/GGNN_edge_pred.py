import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GatedGraphConv
from torch_geometric.utils import add_self_loops

# ------------------ Config ------------------
dataset_dir = sys.argv[1]
result_file = os.path.join(dataset_dir, "ggnn_edge_results.txt")
use_date = True
use_is_hypothetical = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42,52,62,72,82,92,102,112,122,132]

# Hard negative sampling
USE_HARD_NEG = True     # True => pick negatives among similar nodes
K_NEG = 1               # negatives per positive
SIM_TOPK = 50           # consider top-k most similar nodes per source
MAX_N_FOR_HARD = 12000  # fall back to random if graph is huge

# ------------------ Helpers ------------------
def sample_negatives_hard(edge_pos_np, x_np, num_nodes, pos_set, k_neg=1, topk=50, seed=123):
    """
    For each positive (u, v), pick up to k_neg nodes t among u's TOP-K most similar
    (cosine over node features) such that (u, t) is NOT an existing edge and t!=u.
    Returns array [2, k_neg*E_pos].
    """
    E_pos = edge_pos_np.shape[1]
    need = E_pos * k_neg
    neg_edges = []
    # cosine normalize
    x_norm = x_np / (np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(seed)

    for u in edge_pos_np[0]:
        u = int(u)
        sim = x_norm @ x_norm[u]
        sim[u] = -1.0
        k = min(topk, num_nodes - 1)
        if k <= 0:
            continue
        cand_idx = np.argpartition(-sim, kth=k-1)[:k]
        rng.shuffle(cand_idx)

        picked = 0
        for t in cand_idx:
            t = int(t)
            if (u, t) in pos_set or u == t:
                continue
            neg_edges.append((u, t))
            picked += 1
            if picked >= k_neg:
                break
        if len(neg_edges) >= need:
            break

    # pad with random if needed
    if len(neg_edges) < need:
        seen = set(neg_edges)
        while len(neg_edges) < need:
            s = int(rng.integers(0, num_nodes))
            t = int(rng.integers(0, num_nodes))
            if s == t or (s, t) in pos_set or (s, t) in seen:
                continue
            neg_edges.append((s, t))
            seen.add((s, t))

    neg_edges = np.array(neg_edges[:need], dtype=np.int64).T
    return neg_edges

def pair_features_from_nodes(x_np, edges_2xE):
    """
    Build fair edge features for ANY edges (pos or neg) from node features:
      - cosine similarity
      - L2 distance
    Returns [E, F_edge].
    """
    s = edges_2xE[0].astype(int)
    t = edges_2xE[1].astype(int)
    xs = x_np[s]; xt = x_np[t]
    xs_n = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
    xt_n = xt / (np.linalg.norm(xt, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.sum(xs_n * xt_n, axis=1, keepdims=True)
    l2 = np.linalg.norm(xs - xt, axis=1, keepdims=True)
    feats = np.hstack([cos_sim, l2]).astype(np.float32)
    return feats

def safe_metrics(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(np.float32)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    try: auc = roc_auc_score(y_true, y_score)
    except Exception: auc = np.nan
    try: auprc = average_precision_score(y_true, y_score)
    except Exception: auprc = np.nan
    return acc, f1, auc, auprc

def stratified_split(labels, seed=42, test_size=0.2, val_size=0.2):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(len(labels)), labels))
    # ensure both classes in test
    if len(np.unique(labels[test_idx])) < 2:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed+1)
        trainval_idx, test_idx = next(sss1.split(np.zeros(len(labels)), labels))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=seed+2)
    tr_sub, va_sub = next(sss2.split(np.zeros(len(trainval_idx)), labels[trainval_idx]))
    return trainval_idx[tr_sub], trainval_idx[va_sub], test_idx

# ------------------ Load Node Features ------------------
node_data = np.load(f"{dataset_dir}/node_features_pca.npz")
valid_indices = np.load(f"{dataset_dir}/valid_indices.npy")

embedding = node_data['embedding'][valid_indices]
mutation_pca = node_data['mutation_vector_pca'][valid_indices]
depth = node_data['depth'][valid_indices].reshape(-1, 1)

features = [embedding, mutation_pca, depth]

if use_date:
    date = node_data['date'][valid_indices].reshape(-1, 1)
    date_mean = np.nanmean(date)
    date = np.where(np.isnan(date), date_mean, date)
    date = (date - date.mean()) / (date.std() + 1e-8)
    features.append(date)

if use_is_hypothetical:
    is_hyp = node_data['is_hypothetical'][valid_indices].reshape(-1, 1).astype(np.float32)
    features.append(is_hyp)

x_np = np.concatenate(features, axis=1)
x = torch.tensor(x_np, dtype=torch.float32, device=device)
num_nodes = x.shape[0]

# ------------------ Load Edges (positives) ------------------
ef = np.load(f"{dataset_dir}/edge_features.npz")
src_orig = ef['source'].astype(np.int64)
dst_orig = ef['target'].astype(np.int64)

# remap to compact ids [0..N_valid-1]
old2new = {int(o): i for i, o in enumerate(valid_indices)}
keep = np.array([(int(u) in old2new) and (int(v) in old2new) for u,v in zip(src_orig, dst_orig)], dtype=bool)
src = np.array([old2new[int(u)] for u in src_orig[keep]], dtype=np.int64)
dst = np.array([old2new[int(v)] for v in dst_orig[keep]], dtype=np.int64)

edge_pos = np.stack([src, dst], axis=0)   # [2, E_pos]
E_pos = edge_pos.shape[1]

# ------------------ Build negatives (hard or random) ------------------
pos_set = set(zip(edge_pos[0].tolist(), edge_pos[1].tolist()))
if USE_HARD_NEG and num_nodes <= MAX_N_FOR_HARD:
    edge_neg = sample_negatives_hard(edge_pos, x_np, num_nodes, pos_set, k_neg=K_NEG, topk=SIM_TOPK, seed=123)
else:
    need = edge_pos.shape[1] * K_NEG
    rng = np.random.default_rng(123)
    neg = set()
    while len(neg) < need:
        s = int(rng.integers(0, num_nodes))
        t = int(rng.integers(0, num_nodes))
        if s==t or (s,t) in pos_set or (s,t) in neg:
            continue
        neg.add((s,t))
    neg_src, neg_dst = zip(*neg)
    edge_neg = np.stack([np.array(neg_src), np.array(neg_dst)], axis=0)

# ------------------ Candidates & labels ------------------
edge_label_index_np = np.concatenate([edge_pos, edge_neg], axis=1)  # [2, E_all]
y_np = np.concatenate([np.ones(edge_pos.shape[1], dtype=np.int64),
                       np.zeros(edge_neg.shape[1], dtype=np.int64)])

# ------------------ FAIR edge features for BOTH classes ------------------
edge_feat_pos_np = pair_features_from_nodes(x_np, edge_pos)  # [E_pos, F_edge]
edge_feat_neg_np = pair_features_from_nodes(x_np, edge_neg)  # [E_neg, F_edge]
edge_feat_all_np = np.vstack([edge_feat_pos_np, edge_feat_neg_np])  # [E_all, F_edge]
F_edge = edge_feat_all_np.shape[1]

# ------------------ Split (stratified) ------------------
train_idx_np, val_idx_np, test_idx_np = stratified_split(y_np, seed=42)
# Standardize edge features using TRAIN only (no leakage)
scaler = StandardScaler().fit(edge_feat_all_np[train_idx_np])
edge_feat_all_np = scaler.transform(edge_feat_all_np)

# Torchify
edge_label_index = torch.tensor(edge_label_index_np, dtype=torch.long, device=device)
y = torch.tensor(y_np.astype(np.float32), dtype=torch.float32, device=device)
edge_feat_all = torch.tensor(edge_feat_all_np, dtype=torch.float32, device=device)

train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=device)
val_idx   = torch.tensor(val_idx_np,   dtype=torch.long, device=device)
test_idx  = torch.tensor(test_idx_np,  dtype=torch.long, device=device)

# ------------------ Message-passing graph (no leakage) ------------------
is_pos_train = (y[train_idx] == 1.0).detach().cpu().numpy().astype(bool)
train_pos_idx = train_idx[torch.tensor(is_pos_train, dtype=torch.bool, device=device)]
edge_index_msg = edge_label_index[:, train_pos_idx]
edge_index_msg = torch.cat([edge_index_msg, edge_index_msg.flip(0)], dim=1)  # undirected
edge_index_msg, _ = add_self_loops(edge_index_msg, num_nodes=num_nodes)

data = Data(x=x, edge_index=edge_index_msg).to(device)

# ------------------ Models ------------------
class GGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, hidden)
        self.ggnn = GatedGraphConv(hidden, num_layers=3)
    def forward(self, x, edge_index):
        h = self.lin_in(x)
        h = self.ggnn(h, edge_index)
        return h

class EdgeMLP(nn.Module):
    def __init__(self, hidden, edge_feat_dim):
        super().__init__()
        in_dim = 2*hidden + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)  # logits
        )
    def forward(self, h, edge_index, edge_feats):
        s, t = edge_index
        z = torch.cat([h[s], h[t], edge_feats], dim=1)
        return self.mlp(z).squeeze(-1)

# ------------------ Train & Eval across seeds ------------------
accs, f1s, aucs, aprs = [], [], [], []
HIDDEN = 64
LR = 0.01
WD = 5e-4
EPOCHS = 100

for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    ggnn = GGNN(x.shape[1], HIDDEN).to(device)
    head = EdgeMLP(HIDDEN, F_edge).to(device)
    params = list(ggnn.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WD)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS+1):
        ggnn.train(); head.train(); optimizer.zero_grad()
        h = ggnn(data.x, data.edge_index)
        logits = head(h, edge_label_index[:, train_idx], edge_feat_all[train_idx])
        loss = criterion(logits, y[train_idx])
        loss.backward(); optimizer.step()

    # evaluate
    ggnn.eval(); head.eval()
    with torch.no_grad():
        h = ggnn(data.x, data.edge_index)
        logits_te = head(h, edge_label_index[:, test_idx], edge_feat_all[test_idx])
        prob_te = torch.sigmoid(logits_te).cpu().numpy()
    yt = y[test_idx].cpu().numpy()
    acc, f1, auc, auprc = safe_metrics(yt, prob_te, thr=0.5)
    accs.append(acc); f1s.append(f1); aucs.append(auc); aprs.append(auprc)

# ------------------ Report ------------------
acc_avg, acc_std = np.mean(accs), np.std(accs)
f1_avg, f1_std   = np.mean(f1s), np.std(f1s)
auc_avg, auc_std = np.nanmean(aucs), np.nanstd(aucs)
apr_avg, apr_std = np.nanmean(aprs), np.nanstd(aprs)

result_str = (
    "GGNN Edge Prediction\n"
    "Msg-pass graph: train positives (undirected + self-loops); negatives sampled hard/random\n"
    f"Accuracy:  {acc_avg:.4f} ± {acc_std:.4f}\n"
    f"F1 Score:  {f1_avg:.4f} ± {f1_std:.4f}\n"
    f"AUROC:     {auc_avg:.4f} ± {auc_std:.4f}\n"
    f"AUPRC:     {apr_avg:.4f} ± {apr_std:.4f}\n"
)
print("\n" + result_str)
with open(result_file, "w") as f:
    f.write(result_str)
