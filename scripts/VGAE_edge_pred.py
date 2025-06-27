import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
import sys
import os

# Directory
dataset_dir = sys.argv[1]
output_file = os.path.join(dataset_dir, "vgae_edge_results.txt")
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]

# Load node features and labels
node_data = np.load(f"{dataset_dir}/node_features_pca.npz")
valid_indices = np.load(f"{dataset_dir}/valid_indices.npy")
embedding = node_data['embedding'][valid_indices]
mutation_pca = node_data['mutation_vector_pca'][valid_indices]
depth = node_data['depth'][valid_indices].reshape(-1, 1)
x = np.concatenate([embedding, mutation_pca, depth], axis=1)
x_tensor = torch.tensor(x, dtype=torch.float)

# Load and safely remap edge index
edge_data = np.load(f"{dataset_dir}/edge_index_filtered.npz")
edge_index_np = edge_data['edge_index']
old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}

filtered_src = []
filtered_dst = []
valid_edge_attr_indices = []

for i, (src, dst) in enumerate(zip(edge_index_np[0], edge_index_np[1])):
    if src in old_to_new and dst in old_to_new:
        filtered_src.append(old_to_new[src])
        filtered_dst.append(old_to_new[dst])
        valid_edge_attr_indices.append(i)

edge_index_remapped = np.stack([filtered_src, filtered_dst], axis=0)
edge_index = torch.tensor(edge_index_remapped, dtype=torch.long)

# Load edge features and align with valid edges
edge_feat_data = np.load(f"{dataset_dir}/edge_features.npz")
mutation_count = edge_feat_data['mutation_count'][valid_edge_attr_indices]
edit_distance = edge_feat_data['edit_distance'][valid_edge_attr_indices]
mutation_similarity = edge_feat_data['mutation_similarity'][valid_edge_attr_indices]
reverse = edge_feat_data['reverse'][valid_edge_attr_indices]

edge_attr_all = np.stack([
    mutation_count,
    edit_distance,
    mutation_similarity,
    reverse
], axis=1)
edge_attr_all = torch.tensor(edge_attr_all, dtype=torch.float)

num_nodes = x_tensor.shape[0]
num_pos_edges = edge_index.shape[1]

pos_edge_set = set((int(u), int(v)) for u, v in zip(edge_index[0], edge_index[1]))

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)
        self.conv_mu = SAGEConv(2 * out_channels, out_channels)
        self.conv_logstd = SAGEConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class VGAEEdgeFeat(torch.nn.Module):
    def __init__(self, encoder, edge_feat_dim, z_dim):
        super().__init__()
        self.encoder = encoder
        self.z_dim = z_dim
        self.decoder_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * z_dim + edge_feat_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def encode(self, x, edge_index):
        mu, logstd = self.encoder(x, edge_index)
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        self.mu = mu
        self.logstd = logstd
        return z

    def decode(self, z, edge_index, edge_attr):
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        edge_input = torch.cat([z_src, z_dst, edge_attr], dim=1)
        return torch.sigmoid(self.decoder_mlp(edge_input)).squeeze()

    def kl_loss(self):
        return -0.5 * torch.mean(torch.sum(1 + 2 * self.logstd - self.mu**2 - torch.exp(2 * self.logstd), dim=1))

metrics_list = []
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_tensor = x_tensor.to(device)
edge_index = edge_index.to(device)
edge_attr_all = edge_attr_all.to(device)

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate safe negative edges
    neg_edges = set()
    while len(neg_edges) < num_pos_edges:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst and (src, dst) not in pos_edge_set and (src, dst) not in neg_edges:
            neg_edges.add((src, dst))

    neg_src, neg_dst = zip(*neg_edges)
    neg_edge_index = torch.tensor([neg_src, neg_dst], dtype=torch.long, device=device)

    edge_label_index = torch.cat([edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([torch.ones(num_pos_edges), torch.zeros(num_pos_edges)], dim=0).to(device)
    edge_attr_combined = torch.cat([edge_attr_all, edge_attr_all[:num_pos_edges]], dim=0)

    indices = np.arange(edge_label.shape[0])
    np.random.shuffle(indices)
    train_end = int(0.6 * len(indices))
    test_start = int(0.8 * len(indices))
    train_idx = torch.tensor(indices[:train_end], dtype=torch.long)
    test_idx = torch.tensor(indices[test_start:], dtype=torch.long)

    encoder = Encoder(x_tensor.shape[1], 64)
    model = VGAEEdgeFeat(encoder, edge_feat_dim=edge_attr_all.shape[1], z_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        try:
            z = model.encode(x_tensor, edge_index)
            edge_index_train = edge_label_index[:, train_idx]
            edge_attr_train = edge_attr_combined[train_idx]
            prob = model.decode(z, edge_index_train, edge_attr_train)
            prob = prob.clamp(min=1e-7, max=1 - 1e-7)
            target = edge_label[train_idx]
            if torch.any(torch.isnan(prob)) or torch.any(torch.isnan(target)):
                print("⚠️ NaNs detected — skipping this epoch")
                continue
            loss = F.binary_cross_entropy(prob, target) + (1 / num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"🚨 Skipping epoch due to error: {e}")
            continue

    model.eval()
    with torch.no_grad():
        z = model.encode(x_tensor, edge_index)
        edge_index_test = edge_label_index[:, test_idx]
        edge_attr_test = edge_attr_combined[test_idx]
        prob = model.decode(z, edge_index_test, edge_attr_test)
        pred = (prob > 0.5).float()
        y_true = edge_label[test_idx].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_score = prob.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    metrics_list.append((acc, f1, auc, auprc))

metrics_np = np.array(metrics_list)
means = metrics_np.mean(axis=0)
stds = metrics_np.std(axis=0)

with open(output_file, 'w') as f:
    f.write("VGAE Edge Prediction with GraphSAGE Encoder + Edge Features (10 runs)\n")
    f.write(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}\n")
    f.write(f"F1 Score:  {means[1]:.4f} ± {stds[1]:.4f}\n")
    f.write(f"AUROC:     {means[2]:.4f} ± {stds[2]:.4f}\n")
    f.write(f"AUPRC:     {means[3]:.4f} ± {stds[3]:.4f}\n")

print("\n✅ Results saved to:", output_file)
print("\nFinal Bootstrapped Metrics:")
print(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}")
print(f"F1 Score:  {means[1]:.4f} ± {stds[1]:.4f}")
print(f"AUROC:     {means[2]:.4f} ± {stds[2]:.4f}")
print(f"AUPRC:     {means[3]:.4f} ± {stds[3]:.4f}")

'''
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
import sys
import os

# Directory
dataset_dir = sys.argv[1]
output_file = os.path.join(dataset_dir, "vgae_edge_results.txt")
SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]

# Load node features and labels
node_data = np.load(f"{dataset_dir}/node_features_pca.npz")
valid_indices = np.load(f"{dataset_dir}/valid_indices.npy")
embedding = node_data['embedding'][valid_indices]
mutation_pca = node_data['mutation_vector_pca'][valid_indices]
depth = node_data['depth'][valid_indices].reshape(-1, 1)
x = np.concatenate([embedding, mutation_pca, depth], axis=1)
x_tensor = torch.tensor(x, dtype=torch.float)

# Load and safely remap edge index
edge_data = np.load(f"{dataset_dir}/edge_index_filtered.npz")
edge_index_np = edge_data['edge_index']
old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}

filtered_src = []
filtered_dst = []
valid_edge_attr_indices = []

for i, (src, dst) in enumerate(zip(edge_index_np[0], edge_index_np[1])):
    if src in old_to_new and dst in old_to_new:
        filtered_src.append(old_to_new[src])
        filtered_dst.append(old_to_new[dst])
        valid_edge_attr_indices.append(i)

edge_index_remapped = np.stack([filtered_src, filtered_dst], axis=0)
edge_index = torch.tensor(edge_index_remapped, dtype=torch.long)

# Load edge features and align with valid edges
edge_feat_data = np.load(f"{dataset_dir}/edge_features.npz")
mutation_count = edge_feat_data['mutation_count'][valid_edge_attr_indices]
edit_distance = edge_feat_data['edit_distance'][valid_edge_attr_indices]
mutation_similarity = edge_feat_data['mutation_similarity'][valid_edge_attr_indices]
reverse = edge_feat_data['reverse'][valid_edge_attr_indices]

edge_attr_all = np.stack([
    mutation_count,
    edit_distance,
    mutation_similarity,
    reverse
], axis=1)
edge_attr_all = torch.tensor(edge_attr_all, dtype=torch.float)

num_nodes = x_tensor.shape[0]
num_pos_edges = edge_index.shape[1]

pos_edge_set = set((int(u), int(v)) for u, v in zip(edge_index[0], edge_index[1]))

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class VGAEEdgeFeat(torch.nn.Module):
    def __init__(self, encoder, edge_feat_dim, z_dim):
        super().__init__()
        self.encoder = encoder
        self.z_dim = z_dim
        self.decoder_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * z_dim + edge_feat_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def encode(self, x, edge_index):
        mu, logstd = self.encoder(x, edge_index)
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        self.mu = mu
        self.logstd = logstd
        return z

    def decode(self, z, edge_index, edge_attr):
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        edge_input = torch.cat([z_src, z_dst, edge_attr], dim=1)
        return torch.sigmoid(self.decoder_mlp(edge_input)).squeeze()

    def kl_loss(self):
        return -0.5 * torch.mean(torch.sum(1 + 2 * self.logstd - self.mu**2 - torch.exp(2 * self.logstd), dim=1))

metrics_list = []
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_tensor = x_tensor.to(device)
edge_index = edge_index.to(device)
edge_attr_all = edge_attr_all.to(device)

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate safe negative edges
    neg_edges = set()
    while len(neg_edges) < num_pos_edges:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst and (src, dst) not in pos_edge_set and (src, dst) not in neg_edges:
            neg_edges.add((src, dst))

    neg_src, neg_dst = zip(*neg_edges)
    neg_edge_index = torch.tensor([neg_src, neg_dst], dtype=torch.long, device=device)

    edge_label_index = torch.cat([edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([torch.ones(num_pos_edges), torch.zeros(num_pos_edges)], dim=0).to(device)
    edge_attr_combined = torch.cat([edge_attr_all, edge_attr_all[:num_pos_edges]], dim=0)

    indices = np.arange(edge_label.shape[0])
    np.random.shuffle(indices)
    train_end = int(0.6 * len(indices))
    test_start = int(0.8 * len(indices))
    train_idx = torch.tensor(indices[:train_end], dtype=torch.long)
    test_idx = torch.tensor(indices[test_start:], dtype=torch.long)

    encoder = Encoder(x_tensor.shape[1], 64)
    model = VGAEEdgeFeat(encoder, edge_feat_dim=edge_attr_all.shape[1], z_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        try:
            z = model.encode(x_tensor, edge_index)
            edge_index_train = edge_label_index[:, train_idx]
            edge_attr_train = edge_attr_combined[train_idx]
            prob = model.decode(z, edge_index_train, edge_attr_train)
            prob = prob.clamp(min=1e-7, max=1 - 1e-7)
            target = edge_label[train_idx]
            if torch.any(torch.isnan(prob)) or torch.any(torch.isnan(target)):
                print("⚠️ NaNs detected — skipping this epoch")
                continue
            loss = F.binary_cross_entropy(prob, target) + (1 / num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"🚨 Skipping epoch due to error: {e}")
            continue

    model.eval()
    with torch.no_grad():
        z = model.encode(x_tensor, edge_index)
        edge_index_test = edge_label_index[:, test_idx]
        edge_attr_test = edge_attr_combined[test_idx]
        prob = model.decode(z, edge_index_test, edge_attr_test)
        pred = (prob > 0.5).float()
        y_true = edge_label[test_idx].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_score = prob.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    metrics_list.append((acc, f1, auc, auprc))

metrics_np = np.array(metrics_list)
means = metrics_np.mean(axis=0)
stds = metrics_np.std(axis=0)

with open(output_file, 'w') as f:
    f.write("VGAE Edge Prediction with GCN Encoder + Edge Features (10 runs)\n")
    f.write(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}\n")
    f.write(f"F1 Score:  {means[1]:.4f} ± {stds[1]:.4f}\n")
    f.write(f"AUROC:     {means[2]:.4f} ± {stds[2]:.4f}\n")
    f.write(f"AUPRC:     {means[3]:.4f} ± {stds[3]:.4f}\n")

print("\n✅ Results saved to:", output_file)
print("\nFinal Bootstrapped Metrics:")
print(f"Accuracy:  {means[0]:.4f} ± {stds[0]:.4f}")
print(f"F1 Score:  {means[1]:.4f} ± {stds[1]:.4f}")
print(f"AUROC:     {means[2]:.4f} ± {stds[2]:.4f}")
print(f"AUPRC:     {means[3]:.4f} ± {stds[3]:.4f}")
'''