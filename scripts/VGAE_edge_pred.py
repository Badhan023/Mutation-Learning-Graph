import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops

# ---------------------------
# Args / paths / device
# ---------------------------
dataset_dir = sys.argv[1]
output_file = os.path.join(dataset_dir, "vgae_edge_results.txt")
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# Node features
# ---------------------------
node_data = np.load(os.path.join(dataset_dir, "node_features_pca.npz"))
valid_indices = np.load(os.path.join(dataset_dir, "valid_indices.npy"))

embedding = node_data['embedding'][valid_indices]
mutation_pca = node_data['mutation_vector_pca'][valid_indices]
depth = node_data['depth'][valid_indices].reshape(-1, 1)

x_np = np.concatenate([embedding, mutation_pca, depth], axis=1)
x = torch.tensor(x_np, dtype=torch.float32, device=device)
num_nodes = x.shape[0]

# depth column index in x_np
DEPTH_COL = embedding.shape[1] + mutation_pca.shape[1]  # single column

# ---------------------------
# Positive edges (may already be remapped)
# ---------------------------
edge_pos_np = np.load(os.path.join(dataset_dir, "edge_index_filtered.npz"))['edge_index']
# Detect if already compact [0..N_valid-1]; if not, remap from original IDs
if edge_pos_np.max() < len(valid_indices):
    edge_index_pos = torch.tensor(edge_pos_np, dtype=torch.long, device=device)
else:
    old_to_new = {int(old): int(new) for new, old in enumerate(valid_indices)}
    filtered_src, filtered_dst = [], []
    for u, v in zip(edge_pos_np[0], edge_pos_np[1]):
        if int(u) in old_to_new and int(v) in old_to_new:
            filtered_src.append(old_to_new[int(u)])
            filtered_dst.append(old_to_new[int(v)])
    edge_index_pos = torch.tensor(np.stack([filtered_src, filtered_dst], axis=0),
                                  dtype=torch.long, device=device)

E_pos = edge_index_pos.shape[1]

# ---------------------------
# Negatives: balanced, no collisions (random; hard-neg can be added later)
# ---------------------------
pos_set = set(zip(edge_index_pos[0].tolist(), edge_index_pos[1].tolist()))
rng = np.random.default_rng(123)
neg_edges = set()
while len(neg_edges) < E_pos:
    s = int(rng.integers(0, num_nodes))
    t = int(rng.integers(0, num_nodes))
    if s == t or (s, t) in pos_set or (s, t) in neg_edges:
        continue
    neg_edges.add((s, t))
neg_src, neg_dst = zip(*neg_edges)
edge_index_neg = torch.tensor([neg_src, neg_dst], dtype=torch.long, device=device)

# ---------------------------
# Candidates & labels
# ---------------------------
edge_label_index = torch.cat([edge_index_pos, edge_index_neg], dim=1)  # [2, 2*E_pos]
edge_label = torch.cat([torch.ones(E_pos), torch.zeros(E_pos)], dim=0).to(device)
E_all = edge_label.shape[0]
labels_np = edge_label.detach().cpu().numpy().astype(int)

# ---------------------------
# FAIR edge features (for BOTH classes)
#   - cosine similarity
#   - L2 distance
#   - |depth Δ|
# ---------------------------
def pair_feats_from_nodes(x_np, edges_2xE, depth_col=None):
    s = edges_2xE[0].astype(int)
    t = edges_2xE[1].astype(int)
    xs, xt = x_np[s], x_np[t]
    # cosine
    xs_n = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
    xt_n = xt / (np.linalg.norm(xt, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.sum(xs_n * xt_n, axis=1, keepdims=True)
    # l2
    l2 = np.linalg.norm(xs - xt, axis=1, keepdims=True)
    feats = [cos_sim, l2]
    # |depth delta|
    if depth_col is not None:
        d_abs = np.abs(x_np[s, depth_col] - x_np[t, depth_col]).reshape(-1, 1)
        feats.append(d_abs)
    return np.hstack(feats).astype(np.float32)

edge_pos_np_compact = edge_index_pos.detach().cpu().numpy()
edge_neg_np_compact = edge_index_neg.detach().cpu().numpy()
edge_feat_pos_np = pair_feats_from_nodes(x_np, edge_pos_np_compact, depth_col=DEPTH_COL)  # [E_pos, F]
edge_feat_neg_np = pair_feats_from_nodes(x_np, edge_neg_np_compact, depth_col=DEPTH_COL)  # [E_pos, F]
edge_feat_all_np = np.vstack([edge_feat_pos_np, edge_feat_neg_np])   # [E_all, F]
F_edge = edge_feat_all_np.shape[1]

# ---------------------------
# Stratified 60/20/20 split (fixed once)
# ---------------------------
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
trainval_idx, test_idx_np = next(sss1.split(np.zeros(E_all), labels_np))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=43)  # 0.25 of 0.8 -> 0.2
tr_sub, va_sub = next(sss2.split(np.zeros(len(trainval_idx)), labels_np[trainval_idx]))
train_idx_np = trainval_idx[tr_sub]
val_idx_np   = trainval_idx[va_sub]

# Train-only standardization (no leakage)
scaler = StandardScaler().fit(edge_feat_all_np[train_idx_np])
edge_feat_all_np = scaler.transform(edge_feat_all_np)

# Torch tensors
train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=device)
val_idx   = torch.tensor(val_idx_np,   dtype=torch.long, device=device)
test_idx  = torch.tensor(test_idx_np,  dtype=torch.long, device=device)
edge_feat_all = torch.tensor(edge_feat_all_np, dtype=torch.float32, device=device)

# ---------------------------
# No-leakage message-passing graph: positive TRAIN edges only
# ---------------------------
is_pos_train = (edge_label[train_idx] == 1.0).detach().cpu().numpy().astype(bool)
train_pos_idx = train_idx[torch.tensor(is_pos_train, dtype=torch.bool, device=device)]
edge_index_msg = edge_label_index[:, train_pos_idx]
edge_index_msg = torch.cat([edge_index_msg, edge_index_msg.flip(0)], dim=1)  # undirected
edge_index_msg, _ = add_self_loops(edge_index_msg, num_nodes=num_nodes)

# ---------------------------
# VGAE with GraphSAGE encoder
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels, z_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 2*z_dim)
        self.conv_mu = SAGEConv(2*z_dim, z_dim)
        self.conv_logstd = SAGEConv(2*z_dim, z_dim)
    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)

class VGAEEdgeFeat(nn.Module):
    def __init__(self, encoder, edge_feat_dim, z_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Linear(2*z_dim + edge_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # logits
        )
        self.mu = None
        self.logstd = None
    def encode(self, x, edge_index):
        mu, logstd = self.encoder(x, edge_index)
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        self.mu, self.logstd = mu, logstd
        return z
    def decode_logits(self, z, edge_index, edge_attr):
        s, t = edge_index
        z_s, z_t = z[s], z[t]
        inp = torch.cat([z_s, z_t, edge_attr], dim=1)
        return self.decoder(inp).squeeze(-1)  # logits
    def kl_loss(self):
        # mean over nodes
        return -0.5 * torch.mean(torch.sum(1 + 2*self.logstd - self.mu**2 - torch.exp(2*self.logstd), dim=1))

def safe_metrics(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(np.float32)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    try: auc = roc_auc_score(y_true, y_score)
    except Exception: auc = np.nan
    try: auprc = average_precision_score(y_true, y_score)
    except Exception: auprc = np.nan
    return acc, f1, auc, auprc

# ---------------------------
# Train across seeds (KL warm-up + scheduler + val-threshold)
# ---------------------------
metrics = []
Z_DIM = 64
LR = 0.003        # <— tuned
EPOCHS = 200
BETA_FINAL = 1.0
WARMUP_EPOCHS = 40

for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    enc = Encoder(x.shape[1], Z_DIM)
    model = VGAEEdgeFeat(enc, edge_feat_dim=edge_feat_all.shape[1], z_dim=Z_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-4
    )


    best_val = float('inf'); best_state = None; patience_ctr = 0; PATIENCE = 25

    for epoch in range(1, EPOCHS+1):
        model.train(); optimizer.zero_grad()
        z = model.encode(x, edge_index_msg)
        logits_tr = model.decode_logits(z, edge_label_index[:, train_idx], edge_feat_all[train_idx])
        loss_rec = F.binary_cross_entropy_with_logits(logits_tr, edge_label[train_idx])

        beta = BETA_FINAL * min(1.0, epoch / WARMUP_EPOCHS)
        loss = loss_rec + (beta / max(1, num_nodes)) * model.kl_loss()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        # validation loss for scheduler/early-stop
        with torch.no_grad():
            model.eval()
            z_val = model.encode(x, edge_index_msg)
            logits_val = model.decode_logits(z_val, edge_label_index[:, val_idx], edge_feat_all[val_idx])
            val_loss = F.binary_cross_entropy_with_logits(logits_val, edge_label[val_idx]).item()
        scheduler.step(val_loss)

        if val_loss + 1e-4 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # threshold from validation to maximize F1
    with torch.no_grad():
        model.eval()
        z = model.encode(x, edge_index_msg)
        pv = torch.sigmoid(model.decode_logits(z, edge_label_index[:, val_idx], edge_feat_all[val_idx])).cpu().numpy()
        yv = edge_label[val_idx].cpu().numpy()
    thr_grid = np.linspace(0.1, 0.9, 33)
    f1s = [f1_score(yv, (pv >= t).astype(np.float32)) for t in thr_grid]
    best_thr = float(thr_grid[int(np.argmax(f1s))])

    # Test
    with torch.no_grad():
        z = model.encode(x, edge_index_msg)
        pt = torch.sigmoid(model.decode_logits(z, edge_label_index[:, test_idx], edge_feat_all[test_idx])).cpu().numpy()
    yt = edge_label[test_idx].cpu().numpy()
    acc, f1, auc, auprc = safe_metrics(yt, pt, thr=best_thr)
    metrics.append((acc, f1, auc, auprc))

# ---------------------------
# Aggregate & save
# ---------------------------
metrics_np = np.array(metrics, dtype=np.float64)
means, stds = metrics_np.mean(axis=0), metrics_np.std(axis=0)

with open(output_file, 'w') as f:
    f.write("VGAE Edge Prediction\n")
    f.write(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}\n")
    f.write(f"F1 Score:  {means[1]:.4f} ± {stds[1]:.4f}\n")
    f.write(f"AUROC:     {means[2]:.4f} ± {stds[2]:.4f}\n")
    f.write(f"AUPRC:     {means[3]:.4f} ± {stds[3]:.4f}\n")

print("\nResults saved to:", output_file)
print("Final Bootstrapped Metrics (mean ± std):")
print(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}")
print(f"F1 Score:  {means[1]:.4f} ± {stds[1]:.4f}")
print(f"AUROC:     {means[2]:.4f} ± {stds[2]:.4f}")
print(f"AUPRC:     {means[3]:.4f} ± {stds[3]:.4f}")

