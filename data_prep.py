# data_prep.py
import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

# ================= ARGS =================
parser = argparse.ArgumentParser(description="Step 1: Data Prep & Snowball Sampling")
parser.add_argument('--data_dir', type=str, default='.', help="Path to folder containing raw .xlsx files")
parser.add_argument('--target_nodes', type=int, default=5000, help="Number of nodes to sample")
args = parser.parse_args()

RAW_DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = RAW_DATA_DIR / "sampled_5k_propagated"

RISK_KEYWORDS = [
    "hopeless", "depressed", "anxiety", "help me", "pain", 
    "suicide", "die", "sad", "lonely", "overwhelmed"
]

def clean_id(col):
    """Helper to ensure IDs are strings without decimals"""
    return pd.to_numeric(col, errors='coerce').fillna(0).astype(int).astype(str)

def run_pipeline():
    print(f"Reading raw Excel data from: {RAW_DATA_DIR.resolve()}")
    
    try:
        print(" > Loading Nodes_Data.xlsx...")
        nodes = pd.read_excel(RAW_DATA_DIR / "Nodes_Data.xlsx", engine='openpyxl')
        
        print(" > Loading Tweet_Contents.xlsx...")
        tweets = pd.read_excel(RAW_DATA_DIR / "Tweet_Contents.xlsx", engine='openpyxl')
        
        print(" > Loading Edges_Data.xlsx...")
        edges = pd.read_excel(RAW_DATA_DIR / "Edges_Data.xlsx", engine='openpyxl')
        
    except FileNotFoundError:
        print(f"Error: Could not find .xlsx files in '{RAW_DATA_DIR.resolve()}'.")
        sys.exit(1)

    # --- 0. JOIN EDGES AND TWEETS ---
    print("Step 0: Joining Edges and Tweets...")
    
    # Check if counts match (Sanity check)
    if len(edges) != len(tweets):
        print(f"Warning: Edges count ({len(edges)}) != Tweets count ({len(tweets)}). Joining by index anyway.")
    
    edges = edges.reset_index(drop=True)
    tweets = tweets.reset_index(drop=True)
    
    # Combine into one 'interactions' dataframe
    interactions = pd.concat([edges, tweets], axis=1)
    
    # --- CLEANING & STANDARDIZING INPUTS ---
    print(" > Cleaning IDs...")
    # Clean Nodes
    nodes['User ID'] = clean_id(nodes['User ID'])
    
    # Clean Interactions
    # Handle the raw column names (User ID (Source), etc.)
    interactions['User ID (Source)'] = clean_id(interactions['User ID (Source)'])
    interactions['User ID (Destination)'] = clean_id(interactions['User ID (Destination)'])
    
    # Identify Text Column
    text_col = 'Tweet' if 'Tweet' in interactions.columns else 'Text'
    if text_col not in interactions.columns:
        text_col = tweets.columns[0] # Fallback
    print(f" > Using '{text_col}' as text source.")

    # --- 1. WEAK SUPERVISION ---
    print("Step 1: Identifying Seed Nodes...")
    
    # Find risk interactions
    risk_interactions = interactions[interactions[text_col].astype(str).str.contains('|'.join(RISK_KEYWORDS), case=False, na=False)]
    
    seed_ids = risk_interactions['User ID (Source)'].unique().tolist()
    print(f" > Found {len(seed_ids)} seed users based on keyword matches.")
    
    if len(seed_ids) < 10:
        seed_ids.extend(nodes['User ID'].sample(50).tolist())

    node_labels = {uid: 0 for uid in nodes['User ID']}
    for uid in seed_ids:
        node_labels[uid] = 1

    # --- 2. SNOWBALL SAMPLING ---
    print(f"Step 2: Snowball Sampling to {args.target_nodes} nodes...")
    
    adj = {}
    for row in interactions.itertuples():
        # Robust column access
        try:
            src = str(getattr(row, "User ID (Source)"))
            dst = str(getattr(row, "User ID (Destination)"))
        except AttributeError:
            src = str(getattr(row, "_1"))
            dst = str(getattr(row, "_2"))
            
        if src not in adj: adj[src] = []
        if dst not in adj: adj[dst] = []
        adj[src].append(dst)
        adj[dst].append(src)

    selected_users = set(seed_ids)
    queue = list(seed_ids)
    
    while len(selected_users) < args.target_nodes and len(queue) > 0:
        curr = queue.pop(0)
        neighbors = adj.get(curr, [])
        for n in neighbors:
            if n not in selected_users:
                selected_users.add(n)
                queue.append(n)
                if len(selected_users) >= args.target_nodes:
                    break

    # --- 3. LABEL PROPAGATION ---
    print("Step 3: Label Propagation...")
    final_labels = {}
    for uid in selected_users:
        if node_labels.get(uid, 0) == 1:
            final_labels[uid] = 1
            continue
        neighbors = adj.get(uid, [])
        risk_neighbors = sum(1 for n in neighbors if n in selected_users and node_labels.get(n, 0) == 1)
        final_labels[uid] = 1 if risk_neighbors >= 2 else 0

    print(f" > Total At Risk Users: {sum(final_labels.values())}")

    # --- 4. STANDARDIZED SAVING ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Step 4: Saving standardized CSVs...")

    # 1. NODES
    final_nodes = nodes[nodes['User ID'].isin(selected_users)].copy()
    final_nodes['label'] = final_nodes['User ID'].map(final_labels)
    # RENAME COLUMNS
    final_nodes = final_nodes.rename(columns={
        'User ID': 'user_id',
        'Followers': 'followers',
        'Followed': 'followed',
        'Verified': 'verified'
    })
    # Keep only relevant columns
    cols_to_keep = ['user_id', 'label', 'followers', 'followed', 'verified']
    # Filter only if they exist (Verified might be missing in some datasets)
    cols_to_keep = [c for c in cols_to_keep if c in final_nodes.columns]
    final_nodes[cols_to_keep].to_csv(OUTPUT_DIR / "nodes.csv", index=False)

    # 2. EDGES
    final_interactions = interactions[
        (interactions['User ID (Source)'].isin(selected_users)) & 
        (interactions['User ID (Destination)'].isin(selected_users))
    ].copy()
    
    final_edges = final_interactions[['User ID (Source)', 'User ID (Destination)']].copy()
    # RENAME COLUMNS
    final_edges = final_edges.rename(columns={
        'User ID (Source)': 'source_user_id',
        'User ID (Destination)': 'target_user_id'
    })
    final_edges.to_csv(OUTPUT_DIR / "edges.csv", index=False)

    # 3. TWEETS
    final_tweets = final_interactions[['User ID (Source)', text_col]].copy()
    # RENAME COLUMNS
    final_tweets = final_tweets.rename(columns={
        'User ID (Source)': 'user_id', 
        text_col: 'text'
    })
    final_tweets.to_csv(OUTPUT_DIR / "tweets.csv", index=False)
    
    print(f"Done. Saved standardized files to {OUTPUT_DIR}")
    print("Files created: nodes.csv, edges.csv, tweets.csv")

if __name__ == "__main__":
    run_pipeline()