# train_fixed_pipeline.py
"""
Fixed end-to-end pipeline: BERT (text) + GAT (graph) + MLP fusion classifier
Run: python train_fixed_pipeline.py
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, auc

# Transformers (BERT)
from transformers import AutoTokenizer, AutoModel

# PyG (Graph Attention Network)
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data as GeomData
except Exception as e:
    raise ImportError("torch_geometric is required. Install it following https://pytorch-geometric.readthedocs.io/") from e

# -----------------------
# Config
# -----------------------
DATA_DIR = Path("/Users/dhruvthakkar/Downloads/446Project")

NODES_CSV = str(DATA_DIR / "Nodes_Data.csv")
TWEETS_CSV = str(DATA_DIR / "Tweet_Contents.csv")
EDGES_CSV = str(DATA_DIR / "Edges_Data.csv")



SEED = 42
BERT_MODEL_NAME = "distilbert-base-uncased"  # faster; switch to 'bert-base-uncased' if you have GPU/time
BERT_MAX_LEN = 64
BERT_BATCH = 32
EMBED_CACHE = DATA_DIR / "user_bert_embs.npy"

GAT_NODE_IN_DIM = 3   # followers_norm, followed_norm, verified
GAT_HIDDEN = 64
GAT_HEADS = 4
GAT_OUT_DIM = 128

FUSION_HIDDEN = 256
CLASSIFIER_EPOCHS = 20
GAT_PRETRAIN_EPOCHS = 30  # per fold (train on train nodes only)
GAT_LR = 1e-3
FUSION_LR = 1e-4

N_SPLITS = 5   # set to 10 if you want 10-fold but for speed default 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1

# Reproducibility
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# -----------------------
# Utilities: load data
# -----------------------
def load_csvs(nodes_csv, tweets_csv, edges_csv):
    nodes = pd.read_csv(nodes_csv)
    tweets = pd.read_csv(tweets_csv)
    edges = pd.read_csv(edges_csv)
    # Basic sanity checks
    assert 'User ID' in nodes.columns, "Nodes CSV must have 'User ID'"
    assert 'label' in nodes.columns, "Nodes CSV must have 'label' column"
    # Ensure tweets have 'User ID' and text column (we'll autodetect text column)
    if 'User ID' not in tweets.columns:
        raise ValueError("Tweets CSV must contain 'User ID' column")
    # Detect text column (commonly 'Text' or last column)
    text_col = 'Text' if 'Text' in tweets.columns else tweets.columns[-1]
    tweets = tweets.rename(columns={text_col: 'Text'})
    # Edges expected columns
    if 'User ID (Source)' not in edges.columns or 'User ID (Destination)' not in edges.columns:
        raise ValueError("Edges CSV must contain 'User ID (Source)' and 'User ID (Destination)' columns")
    return nodes, tweets, edges

# -----------------------
# Prepare node features and mapping
# -----------------------
def prepare_node_features(nodes_df):
    # Normalize followers/followed by log1p
    nodes_df['Followers'] = pd.to_numeric(nodes_df['Followers'].fillna(0))
    nodes_df['Followed'] = pd.to_numeric(nodes_df['Followed'].fillna(0))
    nodes_df['Verified'] = pd.to_numeric(nodes_df.get('Verified', 0).fillna(0))
    nodes_df['followers_norm'] = np.log1p(nodes_df['Followers'].astype(float))
    nodes_df['followed_norm'] = np.log1p(nodes_df['Followed'].astype(float))
    # Build mapping user_id -> index (0..N-1)
    user_ids = nodes_df['User ID'].astype(str).tolist()
    uid2idx = {uid: i for i, uid in enumerate(user_ids)}
    # Node feature matrix
    feat = np.stack([
        nodes_df['followers_norm'].values,
        nodes_df['followed_norm'].values,
        nodes_df['Verified'].astype(int).values
    ], axis=1).astype(np.float32)
    labels = nodes_df['label'].astype(int).values
    return uid2idx, torch.tensor(feat, dtype=torch.float), labels, nodes_df

# -----------------------
# Build edge index
# -----------------------
def build_edge_index(edges_df, uid2idx):
    # Map source/dest to indices, filter unknowns
    src_raw = edges_df['User ID (Source)'].astype(str)
    dst_raw = edges_df['User ID (Destination)'].astype(str)
    src_idx = src_raw.map(uid2idx)
    dst_idx = dst_raw.map(uid2idx)
    valid_mask = src_idx.notna() & dst_idx.notna()
    src_idx = src_idx[valid_mask].astype(int).tolist()
    dst_idx = dst_idx[valid_mask].astype(int).tolist()
    if len(src_idx) == 0:
        # no edges -> create empty tensors
        edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    return edge_index

# -----------------------
# BERT embeddings per user (cache)
# -----------------------
class BertEmbedder:
    def __init__(self, model_name=BERT_MODEL_NAME, max_len=BERT_MAX_LEN, device=DEVICE):
        print(f"Loading tokenizer/model {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.max_len = max_len
        self.device = device

    def user_embeddings(self, tweets_df, uid2idx, num_users, cache_path=EMBED_CACHE, batch_size=BERT_BATCH):
        # If cache exists, load
        if cache_path.exists():
            try:
                arr = np.load(cache_path)
                if arr.shape[0] == num_users:
                    print("Loaded cached BERT embeddings:", cache_path)
                    return torch.tensor(arr, dtype=torch.float)
            except:
                pass

        # Group tweets by user index
        grouped = defaultdict(list)
        for row in tweets_df.itertuples(index=False):
            uid = str(row._asdict().get('User ID') if hasattr(row, '_asdict') else row[0])
            text = row.Text if hasattr(row, 'Text') else row[1]
            idx = uid2idx.get(str(uid))
            if idx is not None:
                grouped[idx].append(str(text))

        # For each user, keep up to K tweets and compute CLS embedding per tweet, then mean across that user's tweets
        max_per_user = 5
        embeddings = np.zeros((num_users, self.model.config.hidden_size), dtype=np.float32)
        users_with_text = 0
        all_user_indices = list(range(num_users))
        self.model.eval()
        with torch.no_grad():
            # Prepare batches of (user_idx, list_of_texts->list strings)
            items = []
            for u in all_user_indices:
                texts = grouped.get(u, [])
                if len(texts) == 0:
                    items.append((u, []))
                else:
                    items.append((u, texts[:max_per_user]))
            # Process in batches by flattening tweets
            for i in tqdm(range(0, len(items), batch_size), desc="BERT batches"):
                batch = items[i:i+batch_size]
                # flatten texts into list, keep mapping to user
                flat_texts = []
                mapping = []
                for u, texts in batch:
                    if len(texts) == 0:
                        mapping.append((u, []))
                    else:
                        mapping.append((u, list(range(len(flat_texts), len(flat_texts)+len(texts)))))
                        flat_texts.extend(texts)
                if len(flat_texts) == 0:
                    # nothing to encode in this batch
                    for u, idxs in mapping:
                        if len(idxs) == 0:
                            continue
                    continue
                enc = self.tokenizer(flat_texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
                enc = {k: v.to(self.device) for k,v in enc.items()}
                out = self.model(**enc, return_dict=True)
                # take CLS token: last_hidden_state[:,0,:]
                cls = out.last_hidden_state[:,0,:].cpu().numpy()  # shape [num_flat_texts, hidden]
                # assign back to users in this batch by averaging their tweet CLSs
                for (u, idxs) in mapping:
                    if len(idxs) == 0:
                        continue
                    user_vec = cls[idxs].mean(axis=0)
                    embeddings[u] = user_vec
                    users_with_text += 1
        print(f"Users with text embeddings: {users_with_text}/{num_users}")
        np.save(cache_path, embeddings)
        return torch.tensor(embeddings, dtype=torch.float)

# -----------------------
# GAT model and small head
# -----------------------
class GATModel(nn.Module):
    def __init__(self, in_dim=GAT_NODE_IN_DIM, hidden=GAT_HIDDEN, heads=GAT_HEADS, out_dim=GAT_OUT_DIM, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden*heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.act = nn.ELU()
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.act(x)
        x = self.gat2(x, edge_index)
        return x

# Fusion classifier
class FusionClassifier(nn.Module):
    def __init__(self, text_dim, graph_dim=GAT_OUT_DIM, hidden=FUSION_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + graph_dim, hidden),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, text_emb, graph_emb):
        x = torch.cat([text_emb, graph_emb], dim=1)
        return self.net(x).view(-1)

# -----------------------
# Training utilities
# -----------------------
def train_gat_on_train_nodes(gat_model, head, node_feats, edge_index, train_idx, labels, device, epochs=GAT_PRETRAIN_EPOCHS):
    """
    Train GAT + head on the graph using labels only for train_idx nodes.
    head: nn.Linear(GAT_OUT_DIM, 1)
    """
    gat_model.to(device); head.to(device)
    node_feats = node_feats.to(device)
    edge_index = edge_index.to(device)
    y = torch.tensor(labels, dtype=torch.float, device=device)
    optimizer = torch.optim.Adam(list(gat_model.parameters()) + list(head.parameters()), lr=GAT_LR, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = 1e9
    gat_model.train(); head.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = gat_model(node_feats, edge_index)  # [N, out_dim]
        logits = head(out).view(-1)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                pred = torch.sigmoid(logits[train_idx]).cpu().numpy()
                cur_ap = average_precision_score(y[train_idx].cpu().numpy(), pred) if len(train_idx) > 1 else 0.0
            print(f"  [GAT pretrain] epoch {epoch+1}/{epochs} loss={loss.item():.4f} train_AP={cur_ap:.4f}")
    # return trained model and head
    return gat_model, head

def train_fusion_classifier(train_text, train_graph, train_labels, val_text, val_graph, val_labels, text_dim, device, epochs=CLASSIFIER_EPOCHS):
    classifier = FusionClassifier(text_dim=text_dim, graph_dim=train_graph.shape[1]).to(device)
    opt = torch.optim.Adam(classifier.parameters(), lr=FUSION_LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    train_text = train_text.to(device); train_graph = train_graph.to(device)
    train_y = torch.tensor(train_labels, dtype=torch.float, device=device)
    val_text = val_text.to(device); val_graph = val_graph.to(device)
    best_val_ap = -1
    best_state = None
    for epoch in range(epochs):
        classifier.train()
        opt.zero_grad()
        logits = classifier(train_text, train_graph)
        loss = criterion(logits, train_y)
        loss.backward()
        opt.step()
        if (epoch+1) % 5 == 0 or epoch == 0:
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(val_text, val_graph)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_ap = average_precision_score(val_labels, val_probs) if len(val_labels) > 0 else 0.0
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_state = classifier.state_dict()
            print(f"    [Fusion] epoch {epoch+1}/{epochs} loss={loss.item():.4f} val_AP={val_ap:.4f}")
    # load best
    if best_state is not None:
        classifier.load_state_dict(best_state)
    return classifier

# -----------------------
# Evaluation helpers
# -----------------------
def compute_metrics(y_true, y_probs):
    # y_probs: floats 0-1
    y_pred = (y_probs >= 0.5).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    ap = average_precision_score(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    return {"macro_f1": macro_f1, "avg_precision": ap, "pr_auc": pr_auc}

# -----------------------
# Main CV experiment
# -----------------------
def main():
    set_seed(SEED)
    print("DEVICE:", DEVICE)
    print("Loading CSVs...")
    nodes_df, tweets_df, edges_df = load_csvs(NODES_CSV, TWEETS_CSV, EDGES_CSV)
    uid2idx, node_feats, labels, nodes_df = prepare_node_features(nodes_df)
    num_nodes = node_feats.shape[0]
    print(f"Num nodes: {num_nodes}")

    edge_index = build_edge_index(edges_df, uid2idx)
    print(f"Edge index shape: {edge_index.shape}")

    # BERT embedder and compute embeddings (cached)
    embedder = BertEmbedder(model_name=BERT_MODEL_NAME, device=DEVICE)
    print("Computing/Loading user BERT embeddings (this may take a while)...")
    user_text_emb = embedder.user_embeddings(tweets_df, uid2idx, num_nodes, cache_path=EMBED_CACHE, batch_size=BERT_BATCH)
    text_dim = user_text_emb.shape[1]
    print("Text embedding shape:", user_text_emb.shape)

    # Stratified CV
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    all_metrics = []
    fold = 0
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        fold += 1
        print("\n" + "="*40)
        print(f"FOLD {fold}/{N_SPLITS} (train {len(train_idx)} / val {len(val_idx)})")
        # Train GAT on training nodes only (supervised pretrain)
        gat = GATModel(in_dim=node_feats.shape[1], hidden=GAT_HIDDEN, heads=GAT_HEADS, out_dim=GAT_OUT_DIM).to(DEVICE)
        head = nn.Linear(GAT_OUT_DIM, 1).to(DEVICE)

        # Train GAT+head on training nodes
        if edge_index.numel() == 0:
            # no edges: fallback to identity (use node feats through MLP-like mapping)
            print("No edges found: skipping GAT training and using linear projection of node features")
            with torch.no_grad():
                graph_emb = torch.tanh(nn.Linear(node_feats.shape[1], GAT_OUT_DIM)(node_feats)).detach()
        else:
            gat, head = train_gat_on_train_nodes(
                gat, head, node_feats, edge_index, torch.tensor(train_idx, dtype=torch.long, device=DEVICE),
                labels, DEVICE, epochs=GAT_PRETRAIN_EPOCHS
            )
            # compute final node embeddings (use eval mode)
            gat.eval()
            with torch.no_grad():
                graph_emb = gat(node_feats.to(DEVICE), edge_index.to(DEVICE)).cpu()

        # Prepare per-fold train/val tensors
        train_text = user_text_emb[train_idx]
        val_text = user_text_emb[val_idx]
        train_graph = graph_emb[train_idx]
        val_graph = graph_emb[val_idx]
        train_y = labels[train_idx]
        val_y = labels[val_idx]

        # Train fusion classifier
        classifier = train_fusion_classifier(train_text, train_graph, train_y, val_text, val_graph, val_y, text_dim, DEVICE, epochs=CLASSIFIER_EPOCHS)

        # Evaluate on validation set
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_text.to(DEVICE), val_graph.to(DEVICE))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        metrics = compute_metrics(val_y, val_probs)
        print(f"Fold {fold} metrics: Macro-F1={metrics['macro_f1']:.4f}, AP={metrics['avg_precision']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")
        all_metrics.append(metrics)

    # aggregate
    mean_macro_f1 = np.mean([m['macro_f1'] for m in all_metrics])
    std_macro_f1 = np.std([m['macro_f1'] for m in all_metrics])
    mean_ap = np.mean([m['avg_precision'] for m in all_metrics])
    std_ap = np.std([m['avg_precision'] for m in all_metrics])
    summary = {
        "n_splits": N_SPLITS,
        "mean_macro_f1": float(mean_macro_f1),
        "std_macro_f1": float(std_macro_f1),
        "mean_avg_precision": float(mean_ap),
        "std_avg_precision": float(std_ap),
        "n_nodes": int(num_nodes)
    }
    os.makedirs("results", exist_ok=True)
    with open("results/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "="*40)
    print("CV Summary:")
    print(json.dumps(summary, indent=2))
    print("Saved to results/metrics_summary.json")

if __name__ == "__main__":
    main()
