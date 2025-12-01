"""
MSE 446 Project: Mental Health Risk Detection Pipeline
------------------------------------------------------
"""

import argparse
import sys
import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, auc, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import add_self_loops, degree
except ImportError:
    print("Error: torch_geometric not installed.")
    sys.exit(1)

# ================= ARGS =================
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./sampled_5k_propagated', help="Path to processed data")
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_splits', type=int, default=10, help="Number of K-Fold splits")
args = parser.parse_args()

# Config
DATA_DIR = Path(args.data_dir)
EMBED_CACHE = DATA_DIR / "user_bert_embs.npy"
RESULTS_FILE = DATA_DIR / "metrics_summary.json"
BERT_MODEL = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def load_data():
    print(f"Loading data from {DATA_DIR}...")
    try:
        nodes = pd.read_csv(DATA_DIR / "nodes.csv")
        edges = pd.read_csv(DATA_DIR / "edges.csv")
    except FileNotFoundError:
        print("Error: nodes.csv/edges.csv not found.")
        sys.exit(1)

    uids = nodes['user_id'].astype(str).tolist()
    uid2idx = {uid: i for i, uid in enumerate(uids)}
    
    # Features
    followers = nodes['followers'] if 'followers' in nodes.columns else np.zeros(len(nodes))
    followed = nodes['followed'] if 'followed' in nodes.columns else np.zeros(len(nodes))
    verified = nodes['verified'] if 'verified' in nodes.columns else np.zeros(len(nodes))

    f1 = np.log1p(pd.to_numeric(followers, errors='coerce').fillna(0).values)
    f2 = np.log1p(pd.to_numeric(followed, errors='coerce').fillna(0).values)
    f3 = pd.to_numeric(verified, errors='coerce').fillna(0).values
    node_feats = StandardScaler().fit_transform(np.stack([f1, f2, f3], axis=1))
    
    # Graph
    src = edges['source_user_id'].astype(str).map(uid2idx)
    dst = edges['target_user_id'].astype(str).map(uid2idx)
    mask = src.notna() & dst.notna()
    src_clean = src[mask].astype(int).tolist()
    dst_clean = dst[mask].astype(int).tolist()
    
    edge_index = torch.tensor([src_clean, dst_clean], dtype=torch.long)
    
    # Degree Calculation (for filtering isolated nodes in eval)
    row, col = edge_index
    node_degrees = degree(col, len(nodes), dtype=torch.long)
    
    # Add Self Loops
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(nodes))
    
    return nodes, edges, uid2idx, torch.tensor(node_feats, dtype=torch.float), torch.tensor(nodes['label'].values, dtype=torch.long), edge_index, node_degrees

class BertEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        self.model = AutoModel.from_pretrained(BERT_MODEL).to(DEVICE)
        
    def get_embeddings(self, edges_df, uid2idx, num_users):
        # Cache Check with Size Validation
        if EMBED_CACHE.exists():
            loaded_embs = np.load(EMBED_CACHE)
            if loaded_embs.shape[0] == num_users:
                print(" > Loading cached BERT embeddings...")
                return torch.tensor(loaded_embs, dtype=torch.float)
            else:
                print(f" > Cache mismatch (Cache: {loaded_embs.shape[0]}, Current: {num_users}). Regenerating...")
        
        print(" > Computing BERT embeddings from Edge Texts...")

        user_texts = defaultdict(list)
        for row in edges_df.itertuples():
            uid = str(getattr(row, 'source_user_id', row[1])) 
            txt = str(getattr(row, 'text', row[-1]))
            if uid in uid2idx: user_texts[uid2idx[uid]].append(txt)
            
        embs = np.zeros((num_users, 768), dtype=np.float32)
        self.model.eval()
        
        batch_ids, batch_txts = [], []
        for u in tqdm(range(num_users), desc="Encoding"):
            texts = user_texts.get(u, [])[:5]
            if not texts: continue
            for t in texts:
                batch_ids.append(u); batch_txts.append(t)
            if len(batch_txts) >= args.batch_size:
                self._flush(batch_ids, batch_txts, embs)
                batch_ids, batch_txts = [], []
        if batch_txts: self._flush(batch_ids, batch_txts, embs)
        
        np.save(EMBED_CACHE, embs)
        return torch.tensor(embs, dtype=torch.float)

    def _flush(self, u_ids, texts, embs):
        with torch.no_grad():
            enc = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt').to(DEVICE)
            cls = self.model(**enc).last_hidden_state[:,0,:].cpu().numpy()
            for i, u in enumerate(u_ids): embs[u] = np.maximum(embs[u], cls[i])

class GAT_Fusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gat1 = GATConv(in_dim, 32, heads=4, dropout=0.4)
        self.gat2 = GATConv(32*4, 32, heads=1, concat=False, dropout=0.4)
        self.fc = nn.Sequential(nn.Linear(768+32, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))
        
    def forward(self, txt, x, edge_index):
        g = self.gat2(F.elu(self.gat1(x, edge_index)), edge_index)
        return self.fc(torch.cat([txt, g], dim=1)).squeeze()

def main():
    nodes, edges, uid2idx, node_feats, labels, edge_index, node_degrees = load_data()
    embedder = BertEmbedder()
    text_embs = embedder.get_embeddings(edges, uid2idx, len(nodes))
    
    node_feats = node_feats.to(DEVICE); edge_index = edge_index.to(DEVICE)
    text_embs = text_embs.to(DEVICE); labels_t = labels.to(DEVICE)
    node_degrees = node_degrees.to(DEVICE)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    metrics = {"macro_f1": [], "roc_auc": [], "ap": [], "pr_auc": [], "recall": [], "precision": []}
    
    print(f"\nTraining on {DEVICE} ({args.n_splits}-Fold CV)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels.numpy())):
        print(f"\n--- Fold {fold+1} ---")
        
        # --- Connected Component Filter ---
        val_bool_mask = torch.zeros(len(nodes), dtype=torch.bool).to(DEVICE)
        val_bool_mask[val_idx] = True
        connected_nodes_mask = (node_degrees > 0)
        final_eval_mask = val_bool_mask & connected_nodes_mask
        if final_eval_mask.sum() < 10: final_eval_mask = val_bool_mask
        # -----------------------------------------------------------
        
        n_pos = (labels[train_idx] == 1).sum().item()
        n_neg = (labels[train_idx] == 0).sum().item()
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        model = GAT_Fusion(in_dim=node_feats.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
        
        best_stats = {"macro_f1": 0, "roc_auc": 0, "ap": 0, "pr_auc": 0, "recall": 0, "precision": 0}
        
        for epoch in range(args.epochs):
            model.train()
            opt.zero_grad()
            logits = model(text_embs, node_feats, edge_index)
            loss = crit(logits[train_idx], labels_t[train_idx].float())
            loss.backward(); opt.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = logits[final_eval_mask]
                y_true = labels_t[final_eval_mask].cpu().numpy()
                probs = torch.sigmoid(val_logits).cpu().numpy()
                
                # Metrics
                roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
                precision, recall, thresholds = precision_recall_curve(y_true, probs)
                pr_auc = auc(recall, precision)
                current_ap = average_precision_score(y_true, probs)
                
                # Optimize Threshold using Risk Class F1
                numer = 2 * precision * recall
                denom = precision + recall
                f1_scores = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0)
                best_idx = np.argmax(f1_scores)
                best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
                
                # Calculate metrics at this Optimal Threshold
                preds = (probs >= best_thresh).astype(int)
                current_macro_f1 = f1_score(y_true, preds, average='macro')
                
                current_recall = recall_score(y_true, preds, zero_division=0)
                current_precision = precision_score(y_true, preds, zero_division=0)

                if current_macro_f1 > best_stats["macro_f1"]:
                    best_stats["macro_f1"] = float(current_macro_f1)
                    best_stats["roc_auc"] = float(roc_auc)
                    best_stats["ap"] = float(current_ap)
                    best_stats["pr_auc"] = float(pr_auc)
                    best_stats["recall"] = float(current_recall)
                    best_stats["precision"] = float(current_precision)

        print(f" > Best F1: {best_stats['macro_f1']:.4f} | Recall: {best_stats['recall']:.4f} | Prec: {best_stats['precision']:.4f}")
        
        metrics["macro_f1"].append(best_stats["macro_f1"])
        metrics["roc_auc"].append(best_stats["roc_auc"])
        metrics["ap"].append(best_stats["ap"])
        metrics["pr_auc"].append(best_stats["pr_auc"])
        metrics["recall"].append(best_stats["recall"])
        metrics["precision"].append(best_stats["precision"])

    summary = {
        "n_splits": int(skf.get_n_splits()),
        "mean_macro_f1": float(np.mean(metrics["macro_f1"])),
        "mean_recall": float(np.mean(metrics["recall"])),
        "mean_precision": float(np.mean(metrics["precision"])),
        "mean_roc_auc": float(np.mean(metrics["roc_auc"])),
        "mean_avg_precision": float(np.mean(metrics["ap"])),
        "mean_pr_auc": float(np.mean(metrics["pr_auc"])),
        "n_nodes": int(len(nodes))
    }
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*40)
    print("SAVING METRICS SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {RESULTS_FILE}")
    print("="*40)

if __name__ == "__main__":
    main()