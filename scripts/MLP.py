import numpy as np
import torch
import sys
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Config ------------------
dataset_dir = sys.argv[1]
use_date = True
use_is_hypothetical = True

# ------------------ Load Node Features ------------------
node_data = np.load(f"{dataset_dir}/node_features_pca.npz")
valid_indices = np.load(f"{dataset_dir}/valid_indices.npy")

embedding = node_data['embedding'][valid_indices]
mutation_pca = node_data['mutation_vector_pca'][valid_indices]
depth = node_data['depth'][valid_indices].reshape(-1, 1)
features = [embedding, mutation_pca, depth]

if use_date:
    date = node_data['date'][valid_indices].reshape(-1, 1)
    date_mean = date[~np.isnan(date)].mean()
    date[np.isnan(date)] = date_mean
    date = (date - date.mean()) / date.std()
    features.append(date)

if use_is_hypothetical:
    is_hypothetical = node_data['is_hypothetical'][valid_indices].reshape(-1, 1).astype(np.float32)
    features.append(is_hypothetical)

x = np.concatenate(features, axis=1)
y = node_data['lineage_label'][valid_indices]

# Remap class labels to 0-based
unique_labels = np.unique(y)
label_map = {old: new for new, old in enumerate(unique_labels)}
y = np.array([label_map[label] for label in y], dtype=np.int64)

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

for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Stratified Split
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    train_idx, test_idx = splits[0]
    val_idx = test_idx[:len(test_idx) // 2]
    test_idx = test_idx[len(test_idx) // 2:]

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx   = torch.tensor(val_idx, dtype=torch.long)
    test_idx  = torch.tensor(test_idx, dtype=torch.long)

    y_train_np = y_tensor[train_idx].detach().cpu().numpy().astype(int).tolist()
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_np), y=y_train_np)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(x.shape[1], hidden_channels=64, out_channels=len(unique_labels)).to(device)
    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(x_tensor[train_idx])
        loss = criterion(out, y_tensor[train_idx])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = model(x_tensor)
        pred = out.argmax(dim=1)
        score = out.softmax(dim=1)
        y_true = y_tensor.cpu().numpy()
        y_pred = pred.cpu().numpy()
        score_ = score.detach().cpu().numpy()

        mask = test_idx.cpu().numpy()
        true = y_true[mask]
        pred_ = y_pred[mask]
        score_ = score_[mask]

        acc = accuracy_score(true, pred_)
        f1 = f1_score(true, pred_, average='weighted', zero_division=0)

        true_bin = label_binarize(true, classes=np.arange(len(unique_labels)))
        try:
            auroc = roc_auc_score(true_bin, score_, average='macro', multi_class='ovr')
        except:
            auroc = float('nan')
        try:
            auprc = average_precision_score(true_bin, score_, average='macro')
        except:
            auprc = float('nan')

        return acc, f1, auroc, auprc

    for epoch in range(1, 201):
        train()

    acc, f1, auroc, auprc = evaluate()
    all_acc.append(acc)
    all_f1.append(f1)
    all_auroc.append(auroc)
    all_auprc.append(auprc)

# ------------------ Report ------------------
def summarize(metric_list):
    return np.mean(metric_list), np.std(metric_list)

summary_text = []
summary_text.append("\n✅ MLP Bootstrap Results (10 seeds)")
summary_text.append("Accuracy:  {:.4f} ± {:.4f}".format(*summarize(all_acc)))
summary_text.append("F1 Score:  {:.4f} ± {:.4f}".format(*summarize(all_f1)))
sum_auroc = summarize(all_auroc)
summary_text.append("AUROC:     {:.4f} ± {:.4f}".format(*sum_auroc))
sum_auprc = summarize(all_auprc)
summary_text.append("AUPRC:     {:.4f} ± {:.4f}".format(*sum_auprc))

for line in summary_text:
    print(line)

with open(os.path.join(dataset_dir, "mlp_node_results.txt"), "w") as f:
    for line in summary_text:
        f.write(line + "\n")
