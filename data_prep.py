import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# ================= ARGS =================
parser = argparse.ArgumentParser(description="Step 1: Full Data Prep (No Sampling)")
parser.add_argument('--data_dir', type=str, default='.', help="Path to folder containing raw .xlsx files")
args = parser.parse_args()

RAW_DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = RAW_DATA_DIR / "data"

RISK_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "killing myself", "kms", "end my life",
    "end it all", "want to die", "wanna die", "wish i was dead", "better off dead",
    "no reason to live", "tired of living", "pointless existence", "goodbye forever",
    "suicide note", "planning my death", "taking my own life", "ready to go",
    "sleep forever", "never wake up", "my last day", "final goodbye", "unalive",
    "sewer slide", "commit suicide", "catch the bus", "jump off", "hang myself",
    "self harm", "self-harm", "sh", "cutting myself", "cut myself", "cutter",
    "razor blade", "wrists", "bleeding out", "burn myself", "burning skin",
    "hitting myself", "punish myself", "overdose", "oding", "pills", "swallow pills",
    "starving myself", "not eating", "skinny enough", "anorexia", "bulimia",
    "purging", "binge eating", "body dysmorphia", "hate my body", "ugly",
    "fat", "disgusting", "scars", "fresh cuts", "styro", "beans", 
    "hopeless", "hopelessness", "despair", "empty", "hollow", "numb",
    "worthless", "useless", "failure", "fail", "loser", "burden",
    "waste of space", "waste of air", "nobody cares", "no one cares",
    "invisible", "ignored", "alone", "lonely", "loneliness", "isolated",
    "abandoned", "rejected", "unloved", "hated", "everyone hates me",
    "hate myself", "loathe myself", "disappointed in myself", "can't do this anymore",
    "giving up", "gave up", "done trying", "lost cause", "broken",
    "damaged", "shattered", "falling apart", "falling down", "drowning",
    "suffocating", "can't breathe", "trapped", "stuck", "no way out",
    "darkness", "black hole", "void", "abyss", "misery", "miserable",
    "anxiety", "anxious", "panic attack", "panicking", "freaking out",
    "scared", "terrified", "fear", "dread", "doom", "overwhelmed",
    "stressed", "stress", "pressure", "can't cope", "can't handle it",
    "breaking down", "breakdown", "mental breakdown", "nervous breakdown",
    "shaking", "trembling", "heart racing", "chest pain", "nauseous",
    "spiraling", "out of control", "losing my mind", "going crazy",
    "going mad", "insane", "manic", "mania", "paranoid", "paranoia",
    "intrusive thoughts", "voices in my head", "demons", "haunted",
    "depressed", "depression", "sad", "sadness", "crying", "tears",
    "sobbing", "cry myself to sleep", "can't stop crying", "exhausted",
    "tired", "fatigue", "lethargic", "no energy", "can't get out of bed",
    "sleeping all day", "insomnia", "can't sleep", "up all night",
    "don't want to wake up", "nightmare", "emotional pain", "hurts so much",
    "pain inside", "aching", "agony", "torture", "suffering", "suffer",
    "why me", "what's the point", "nothing matters", "apathy", "don't care",
    "help me", "need help", "please help", "save me", "someone talk to me",
    "anyone there", "listening", "vent", "venting", "rant", "ranting",
    "advice", "support", "therapy", "therapist", "psychiatrist", "meds",
    "medication", "antidepressants", "withdrawals", "relapse", "relapsing",
    "triggered", "triggering", "tw", "trigger warning", "cw", 
    "pain", "hurt", "dying", "death", "dead", "kill", "blood", "knife",
    "gun", "rope", "pills", "drugs", "alcohol", "drunk", "high",
    "fucked up", "ruined", "destroy", "destroyed", "gone", "leave me alone",
    "go away", "farewell", "sorry", "i'm sorry", "regret", "guilt", "shame"
]

def clean_id(col):
    return pd.to_numeric(col, errors='coerce').fillna(0).astype(int).astype(str)

def run_pipeline():
    print(f"Reading FULL RAW DATA from: {RAW_DATA_DIR.resolve()}")
    try:
        print(" > Loading Nodes_Data.xlsx...")
        nodes = pd.read_excel(RAW_DATA_DIR / "Nodes_Data.xlsx", engine='openpyxl')
        print(" > Loading Tweet_Contents.xlsx...")
        tweets = pd.read_excel(RAW_DATA_DIR / "Tweet_Contents.xlsx", engine='openpyxl')
        print(" > Loading Edges_Data.xlsx...")
        edges = pd.read_excel(RAW_DATA_DIR / "Edges_Data.xlsx", engine='openpyxl')
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    # --- 0. JOIN EDGES AND TWEETS ---
    print("Step 0: Joining Edges and Tweets...")
    if len(edges) != len(tweets):
        print(f"Warning: Counts differ. Edges: {len(edges)}, Tweets: {len(tweets)}. Joining by index.")
    
    edges = edges.reset_index(drop=True)
    tweets = tweets.reset_index(drop=True)
    interactions = pd.concat([edges, tweets], axis=1)
    
    print(" > Cleaning IDs...")
    nodes['User ID'] = clean_id(nodes['User ID'])
    interactions['User ID (Source)'] = clean_id(interactions['User ID (Source)'])
    interactions['User ID (Destination)'] = clean_id(interactions['User ID (Destination)'])
    
    text_col = 'Tweet' if 'Tweet' in interactions.columns else 'Text'
    if text_col not in interactions.columns: text_col = tweets.columns[0]

    # --- 1. WEAK SUPERVISION (ALL NODES) ---
    print("Step 1: Identifying Seeds (Full Dataset)...")
    # Vectorized string search on the full column
    risk_mask = interactions[text_col].astype(str).str.contains('|'.join(RISK_KEYWORDS), case=False, na=False)
    risk_interactions = interactions[risk_mask]
    seed_ids = set(risk_interactions['User ID (Source)'].unique())
    
    print(f" > Found {len(seed_ids)} initial risk seeds out of {len(nodes)} total users.")

    node_labels = {uid: 0 for uid in nodes['User ID']}
    for uid in seed_ids: 
        if uid in node_labels:
            node_labels[uid] = 1

    # --- 2. BUILD GRAPH (NO SNOWBALL - JUST BUILD IT) ---
    print("Step 2: Building Full Adjacency Map...")
    adj = {}
    
    # Pre-filter: We only care about edges where Source AND Dest exist in our Nodes file
    # This prevents "Ghost Users" from 500k dataset crashing the logic
    valid_users = set(nodes['User ID'])
    
    # We use a loop with tqdm for visibility on large data
    for row in tqdm(interactions.itertuples(), total=len(interactions), desc="Mapping Edges"):
        try:
            src = str(getattr(row, "User ID (Source)"))
            dst = str(getattr(row, "User ID (Destination)"))
        except AttributeError:
            src = str(getattr(row, "_1"))
            dst = str(getattr(row, "_2"))
        
        # Only map valid users to keep graph clean
        if src in valid_users and dst in valid_users:
            if src not in adj: adj[src] = []
            if dst not in adj: adj[dst] = []
            adj[src].append(dst)
            adj[dst].append(src)

    # --- 3. LABEL PROPAGATION ---
    print("Step 3: Label Propagation on Full Graph...")
    final_labels = {}
    
    # Process all valid users
    for uid in tqdm(valid_users, desc="Propagating Labels"):
        # Keep original labels
        if node_labels.get(uid, 0) == 1:
            final_labels[uid] = 1
            continue
        
        # Check neighbors
        neighbors = adj.get(uid, [])
        risk_neighbors = 0
        for n in neighbors:
            if node_labels.get(n, 0) == 1:
                risk_neighbors += 1
        
        # Threshold rule
        final_labels[uid] = 1 if risk_neighbors >= 2 else 0

    # --- 4. SAVING ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Step 4: Saving Full Dataset...")

    # Nodes
    final_nodes = nodes.copy()
    final_nodes['label'] = final_nodes['User ID'].map(final_labels)
    final_nodes = final_nodes.rename(columns={'User ID':'user_id', 'Followers':'followers', 'Followed':'followed', 'Verified':'verified'})
    cols_to_keep = [c for c in ['user_id', 'label', 'followers', 'followed', 'verified'] if c in final_nodes.columns]
    final_nodes[cols_to_keep].to_csv(OUTPUT_DIR / "nodes.csv", index=False)

    # Edges (Filtered to valid users only)
    final_interactions = interactions[
        (interactions['User ID (Source)'].isin(valid_users)) & 
        (interactions['User ID (Destination)'].isin(valid_users))
    ].copy()
    
    final_edges = final_interactions[['User ID (Source)', 'User ID (Destination)', text_col]].copy()
    final_edges = final_edges.rename(columns={
        'User ID (Source)': 'source_user_id', 
        'User ID (Destination)': 'target_user_id',
        text_col: 'text'
    })
    final_edges.to_csv(OUTPUT_DIR / "edges.csv", index=False)
    
    # Stats
    n_high_risk = final_nodes['label'].sum()
    n_total = len(final_nodes)
    ratio = (n_high_risk / n_total) * 100 if n_total > 0 else 0
    
    print("\n" + "="*40)
    print("FULL DATA PREPARATION COMPLETE")
    print(f"Total Nodes:      {n_total}")
    print(f"Total Edges:      {len(final_edges)}")
    print(f"High Risk Users:  {n_high_risk} ({ratio:.1f}%)")
    print(f"Saved to:         {OUTPUT_DIR.resolve()}")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_pipeline()