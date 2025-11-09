import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops

# ---------------------------
# Args & config
# ---------------------------
dataset_dir = sys.argv[1]
output_file = os.path.join(dataset_dir, "graphSAGE_edge_results.txt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]

# ---------------------------
# Load node features
# ---------------------------
node_data = np.load(os.path.join(dataset_dir, "node_features_pca.npz"))
valid_indices = np.load(os.path.join(dataset_dir, "valid_indices.npy"))

embedding = node_data['embedding'][valid_indices]
mutation_pca = node_data['mutation_vector_pca'][valid_indices]
depth = node_data['depth'][valid_indices].reshape(-1, 1)
x_np = np.concatenate([embedding, mutation_pca, depth], axis=1)

# ---------------------------
# Load positive base edges (REMAPPED ids) & features (aligned)
# ---------------------------
edge_pos = np.load(os.path.join(dataset_dir, "edge_index_filtered.npz"))['edge_index']  # [2, E_pos]
E_pos = edge_pos.shape[1]

ef = np.load(os.path.join(dataset_dir, "edge_features.npz"))
edge_feat_pos_np = np.stack([ef['mutation_count'], ef['edit_distance'], ef['mutation_similarity'], ef['reverse']], axis=1)  # [E_pos, 4]

num_nodes = x_np.shape[0]
pos_set = set(zip(edge_pos[0].tolist(), edge_pos[1].tolist()))

# ---------------------------
# Build candidate set with k negatives per positive (k=3)
# ---------------------------
K_NEG = 3
rng = np.random.default_rng(123)

def sample_negatives(num_needed):
    neg = set()
    while len(neg) < num_needed:
        src = int(rng.integers(0, num_nodes))
        dst = int(rng.integers(0, num_nodes))
        if src == dst: continue
        if (src, dst) in pos_set or (src, dst) in neg: continue
        neg.add((src, dst))
    return np.array(list(neg), dtype=np.int64).T  # [2, num_needed]

# Construct once (we’ll also resample each epoch inside training for hardness)
neg_edges_np = sample_negatives(E_pos * K_NEG)  # [2, E_pos*K]
edge_label_index_np = np.concatenate([edge_pos, neg_edges_np], axis=1)  # [2, E_all]
edge_label_np = np.concatenate([np.ones(E_pos, dtype=np.int64), np.zeros(E_pos*K_NEG, dtype=np.int64)], axis=0)  # [E_all]

# Edge feature matrix for candidates: positives = real feats; negatives = zeros
edge_feat_neg_np = np.zeros((E_pos*K_NEG, edge_feat_pos_np.shape[1]), dtype=np.float32)
edge_feat_all_np = np.vstack([edge_feat_pos_np, edge_feat_neg_np])  # [E_all, F]

E_all = edge_label_np.shape[0]

# ---------------------------
# Stratified 60/20/20 split (fixed once)
# ---------------------------
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
trainval_idx, test_idx = next(sss1.split(np.zeros(E_all), edge_label_np))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=43)  # 0.25 of 0.8 -> 0.2
tr_sub, va_sub = next(sss2.split(np.zeros(len(trainval_idx)), edge_label_np[trainval_idx]))
train_idx = trainval_idx[tr_sub]
val_idx   = trainval_idx[va_sub]

# ---------------------------
# Standardize features
# - Node features: fit on all nodes (no label info)
# - Edge features: fit on TRAIN edges only (strict)
# ---------------------------
node_scaler = StandardScaler().fit(x_np)
x_np = node_scaler.transform(x_np)

edge_scaler = StandardScaler().fit(edge_feat_all_np[train_idx])
edge_feat_all_np = edge_scaler.transform(edge_feat_all_np)

# ---------------------------
# Torchify static tensors
# ---------------------------
x = torch.tensor(x_np, dtype=torch.float32, device=device)
edge_label_index = torch.tensor(edge_label_index_np, dtype=torch.long, device=device)
edge_label = torch.tensor(edge_label_np.astype(np.float32), dtype=torch.float32, device=device)
edge_feat_all = torch.tensor(edge_feat_all_np, dtype=torch.float32, device=device)

train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
val_idx_t   = torch.tensor(val_idx,   dtype=torch.long, device=device)
test_idx_t  = torch.tensor(test_idx,  dtype=torch.long, device=device)

# ---------------------------
# Build message-passing graph from positive TRAIN edges only,
# then densify: make undirected + add self-loops
# ---------------------------
is_pos_train = (edge_label[train_idx_t] == 1.0).detach().cpu().numpy().astype(bool)
train_pos_idx = train_idx_t[torch.tensor(is_pos_train, dtype=torch.bool, device=device)]
edge_index_msg = edge_label_index[:, train_pos_idx]  # [2, E_train_pos]
# undirected
edge_index_msg = torch.cat([edge_index_msg, edge_index_msg.flip(0)], dim=1)
# self-loops
edge_index_msg, _ = add_self_loops(edge_index_msg, num_nodes=num_nodes)

# ---------------------------
# Model
# ---------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_feat_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1   = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2   = nn.BatchNorm1d(hidden_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + edge_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # logits
        )

    def encode(self, x, edge_index):
        z = self.conv1(x, edge_index); z = self.bn1(z); z = F.relu(z)
        z = self.conv2(z, edge_index); z = self.bn2(z); z = F.relu(z)
        return z

    def decode(self, z, edge_index, edge_attr):
        src, dst = edge_index
        edge_feat = torch.cat([z[src], z[dst], edge_attr], dim=1)
        return self.edge_mlp(edge_feat).squeeze(-1)  # logits

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
# Train across seeds (same split; different inits)
# ---------------------------
metrics_list = []
EPOCHS = 200
LR = 0.003
WEIGHT_DECAY = 5e-4
PATIENCE = 20

for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    model = GraphSAGE(x.shape[1], 128, edge_feat_dim=edge_feat_all.shape[1]).to(device)

    # Weighted BCE from TRAIN
    ytr = edge_label[train_idx_t]
    n_pos = float((ytr == 1).sum().item()); n_neg = float((ytr == 0).sum().item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1e-8)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float('inf'); best_state = None; no_improve = 0

    for epoch in range(1, EPOCHS+1):
        model.train(); optimizer.zero_grad()

        # OPTIONAL: resample negatives each epoch to make task harder
        # (keeps splits the same; only training negatives are reshaped)
        # Comment out next 10 lines if you prefer static negatives.
        neg_needed = E_pos * K_NEG
        neg_edges_epoch = sample_negatives(neg_needed)
        edge_label_index_epoch = np.concatenate([edge_pos, neg_edges_epoch], axis=1)
        edge_feat_neg_epoch = np.zeros((neg_needed, edge_feat_pos_np.shape[1]), dtype=np.float32)
        edge_feat_all_epoch = np.vstack([edge_feat_pos_np, edge_feat_neg_epoch])
        edge_feat_all_epoch = edge_scaler.transform(edge_feat_all_epoch)  # use same scaler
        edge_label_index_t = torch.tensor(edge_label_index_epoch, dtype=torch.long, device=device)
        edge_feat_all_t    = torch.tensor(edge_feat_all_epoch, dtype=torch.float32, device=device)

        z = model.encode(x, edge_index_msg)
        logits = model.decode(z, edge_label_index_t[:, train_idx_t], edge_feat_all_t[train_idx_t])
        loss = criterion(logits, edge_label[train_idx_t])
        loss.backward(); optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            z = model.encode(x, edge_index_msg)
            logits_val = model.decode(z, edge_label_index[:, val_idx_t], edge_feat_all[val_idx_t])
            val_loss = criterion(logits_val, edge_label[val_idx_t]).item()

        if val_loss < best_val - 1e-4:
            best_val = val_loss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                # print(f"Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Test
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index_msg)
        logits_test = model.decode(z, edge_label_index[:, test_idx_t], edge_feat_all[test_idx_t])
        prob_test = torch.sigmoid(logits_test).cpu().numpy()

    y_true = edge_label[test_idx_t].cpu().numpy()
    acc, f1, auc, auprc = safe_metrics(y_true, prob_test, thr=0.5)
    metrics_list.append((acc, f1, auc, auprc))

# ---------------------------
# Aggregate & save
# ---------------------------
metrics_np = np.array(metrics_list, dtype=np.float64)
means = metrics_np.mean(axis=0); stds = metrics_np.std(axis=0)

with open(output_file, 'w') as f:
    f.write("GraphSAGE Edge Prediction\n")
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
