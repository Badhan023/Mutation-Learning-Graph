import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
import sys
import os

# Directory
dataset_dir = sys.argv[1]
output_file = os.path.join(dataset_dir, "graphSAGE_edge_results.txt")

# Set deterministic seeds for bootstrapping
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]

# Load node features and labels
node_data = np.load(f"{dataset_dir}/node_features_pca.npz")
valid_indices = np.load(f"{dataset_dir}/valid_indices.npy")

embedding = node_data['embedding'][valid_indices]
mutation_pca = node_data['mutation_vector_pca'][valid_indices]
depth = node_data['depth'][valid_indices].reshape(-1, 1)
x = np.concatenate([embedding, mutation_pca, depth], axis=1)

y = node_data['lineage_label'][valid_indices]
is_hypothetical = node_data['is_hypothetical'][valid_indices]

unique_labels = np.unique(y)
label_map = {old: new for new, old in enumerate(unique_labels)}
y = np.array([label_map[label] for label in y], dtype=np.int64)

x_tensor = torch.tensor(x, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

edge_data = np.load(f"{dataset_dir}/edge_index_filtered.npz")
edge_index_np = edge_data['edge_index']
edge_index = torch.tensor(edge_index_np, dtype=torch.long)

edge_features = np.load(f"{dataset_dir}/edge_features.npz")
mutation_count = edge_features['mutation_count']
edit_distance = edge_features['edit_distance']
mutation_similarity = edge_features['mutation_similarity']
reverse = edge_features['reverse']

edge_attr_all = np.stack([
    mutation_count,
    edit_distance,
    mutation_similarity,
    reverse
], axis=1)
edge_attr_all = torch.tensor(edge_attr_all, dtype=torch.float)

data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

num_nodes = data.num_nodes
num_pos_edges = data.edge_index.size(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
x = data.x
edge_index = data.edge_index
edge_attr_all = edge_attr_all.to(device)

metrics_list = []
for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)

    neg_edges = set()
    while len(neg_edges) < num_pos_edges:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst and (src, dst) not in neg_edges:
            neg_edges.add((src, dst))
    neg_src, neg_dst = zip(*neg_edges)
    neg_edge_index = torch.tensor([neg_src, neg_dst], dtype=torch.long, device=device)

    edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([torch.ones(num_pos_edges), torch.zeros(num_pos_edges)], dim=0).to(device)

    indices = np.arange(len(edge_label))
    np.random.shuffle(indices)
    train_end = int(0.6 * len(indices))
    val_end = int(0.8 * len(indices))

    train_idx = torch.tensor(indices[:train_end], dtype=torch.long, device='cpu')
    val_idx = torch.tensor(indices[train_end:val_end], dtype=torch.long, device='cpu')
    test_idx = torch.tensor(indices[val_end:], dtype=torch.long, device='cpu')

    edge_label_index_train = edge_label_index[:, train_idx.to(device)]
    edge_label_train = edge_label[train_idx.to(device)]
    edge_attr_train = edge_attr_all[train_idx % num_pos_edges]

    edge_label_index_test = edge_label_index[:, test_idx.to(device)]
    edge_label_test = edge_label[test_idx.to(device)]
    edge_attr_test = edge_attr_all[test_idx % num_pos_edges]

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, edge_feat_dim):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * hidden_channels + edge_feat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        def encode(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x

        def decode(self, z, edge_index, edge_attr):
            src, dst = edge_index
            edge_feat = torch.cat([z[src], z[dst], edge_attr], dim=1)
            return self.edge_mlp(edge_feat).squeeze(-1)

    model = GraphSAGE(x.shape[1], 128, edge_feat_dim=edge_attr_all.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        out = model.decode(z, edge_label_index_train, edge_attr_train)
        loss = F.binary_cross_entropy_with_logits(out, edge_label_train)
        loss.backward()
        optimizer.step()

    model.eval()
    z = model.encode(x, edge_index)
    out = model.decode(z, edge_label_index_test, edge_attr_test)
    pred = torch.sigmoid(out)
    pred_binary = (pred > 0.5).float()

    y_true = edge_label_test.cpu().numpy()
    y_pred = pred_binary.cpu().numpy()
    y_score = pred.detach().cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    metrics_list.append((acc, f1, auc, auprc))

metrics_np = np.array(metrics_list)
means = metrics_np.mean(axis=0)
stds = metrics_np.std(axis=0)

with open(output_file, 'w') as f:
    f.write("GraphSAGE Edge Prediction with Bootstrapping (10 runs)\n")
    f.write(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}\n")
    f.write(f"F1 Score:  {means[1]:.4f} ± {stds[1]:.4f}\n")
    f.write(f"AUROC:     {means[2]:.4f} ± {stds[2]:.4f}\n")
    f.write(f"AUPRC:     {means[3]:.4f} ± {stds[3]:.4f}\n")

print("\n✅ Results saved to:", output_file)
print("Final Bootstrapped Metrics:")
print(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}")
print(f"F1 Score:  {means[1]:.4f} ± {stds[1]:.4f}")
print(f"AUROC:     {means[2]:.4f} ± {stds[2]:.4f}")
print(f"AUPRC:     {means[3]:.4f} ± {stds[3]:.4f}")

'''
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import sys

# Directory
dataset_dir = sys.argv[1]

# Load node features and labels
node_data = np.load(f"{dataset_dir}/node_features_pca.npz")
valid_indices = np.load(f"{dataset_dir}/valid_indices.npy")

embedding = node_data['embedding'][valid_indices]
mutation_pca = node_data['mutation_vector_pca'][valid_indices]
depth = node_data['depth'][valid_indices].reshape(-1, 1)
x = np.concatenate([embedding, mutation_pca, depth], axis=1)

y = node_data['lineage_label'][valid_indices]
is_hypothetical = node_data['is_hypothetical'][valid_indices]

# Remap class labels
unique_labels = np.unique(y)
label_map = {old: new for new, old in enumerate(unique_labels)}
y = np.array([label_map[label] for label in y], dtype=np.int64)

# Torch tensors
x_tensor = torch.tensor(x, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

# Edge index
edge_data = np.load(f"{dataset_dir}/edge_index_filtered.npz")
edge_index_np = edge_data['edge_index']
edge_index = torch.tensor(edge_index_np, dtype=torch.long)

# Data object
data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

# Generate improved negative samples (avoid self-loops and ensure uniqueness)
num_nodes = data.num_nodes
num_pos_edges = data.edge_index.size(1)
neg_edges = set()
while len(neg_edges) < num_pos_edges:
    src = np.random.randint(0, num_nodes)
    dst = np.random.randint(0, num_nodes)
    if src != dst and (src, dst) not in neg_edges:
        neg_edges.add((src, dst))
neg_src, neg_dst = zip(*neg_edges)
neg_edge_index = torch.tensor([neg_src, neg_dst], dtype=torch.long)

# Combine positive and negative edges for edge prediction
edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=1)
edge_label = torch.cat([torch.ones(num_pos_edges), torch.zeros(num_pos_edges)], dim=0)

# Split into train/val/test using positive edges only
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
pos_labels = np.ones(num_pos_edges)
train_idx_np, test_idx_np = list(skf.split(np.zeros(len(pos_labels)), pos_labels))[0]
val_idx_np = test_idx_np[:len(test_idx_np)//2]
test_idx_np = test_idx_np[len(test_idx_np)//2:]

train_idx = torch.tensor(train_idx_np, dtype=torch.long)
val_idx = torch.tensor(val_idx_np, dtype=torch.long)
test_idx = torch.tensor(test_idx_np, dtype=torch.long)

edge_label_index_train = torch.cat([data.edge_index[:, train_idx], neg_edge_index[:, train_idx]], dim=1)
edge_label_train = torch.cat([torch.ones(len(train_idx)), torch.zeros(len(train_idx))], dim=0)

edge_label_index_val = torch.cat([data.edge_index[:, val_idx], neg_edge_index[:, val_idx]], dim=1)
edge_label_val = torch.cat([torch.ones(len(val_idx)), torch.zeros(len(val_idx))], dim=0)

edge_label_index_test = torch.cat([data.edge_index[:, test_idx], neg_edge_index[:, test_idx]], dim=1)
edge_label_test = torch.cat([torch.ones(len(test_idx)), torch.zeros(len(test_idx))], dim=0)

# Define GraphSAGE with MLP decoder
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        edge_feat = torch.cat([z[src], z[dst]], dim=1)
        return self.edge_mlp(edge_feat).squeeze(-1)

# Training and evaluation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(x.shape[1], 128).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x = data.x
edge_index = data.edge_index

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, edge_index)
    out = model.decode(z, edge_label_index_train.to(device))
    loss = F.binary_cross_entropy_with_logits(out, edge_label_train.to(device))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
z = model.encode(x, edge_index)
out = model.decode(z, edge_label_index_test.to(device))
pred = (out > 0).float()

# Compute metrics
y_true = edge_label_test.cpu().numpy()
y_pred = pred.cpu().numpy()
y_score = out.detach().cpu().numpy()

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_score)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nMetrics for Edge Prediction:")
print(f"Accuracy:  {acc:.4f}")
print(f"AUC:       {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
'''