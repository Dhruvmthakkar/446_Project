# data_prep.py
import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

# ================= ARGS =================
parser = argparse.ArgumentParser(description="Step 1: Data Prep & Snowball Sampling")
parser.add_argument('--data_dir', type=str, default='.', help="Path to folder containing raw .xlsx files")
parser.add_argument('--target_nodes', type=int, default=5000, help="Total number of nodes to sample")
parser.add_argument('--risk_ratio', type=float, default=0.2, help="Target ratio of High Risk users (e.g. 0.2 = 20%)")
args = parser.parse_args()

RAW_DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = RAW_DATA_DIR / "data"

# --- EXHAUSTIVE KEYWORD LIST ---
RISK_KEYWORDS = [
    # SUICIDAL IDEATION
    "suicide", "suicidal", "kill myself", "killing myself", "kms", "end my life",
    "end it all", "want to die", "wanna die", "wish i was dead", "better off dead",
    "no reason to live", "tired of living", "pointless existence", "goodbye forever",
    "suicide note", "planning my death", "taking my own life", "ready to go",
    "sleep forever", "never wake up", "my last day", "final goodbye", "unalive",
    "sewer slide", "commit suicide", "catch the bus", "jump off", "hang myself",
    # SELF HARM
    "self harm", "self-harm", "sh", "cutting myself", "cut myself", "cutter",
    "razor blade", "wrists", "bleeding out", "burn myself", "burning skin",
    "hitting myself", "punish myself", "overdose", "oding", "pills", "swallow pills",
    "starving myself", "not eating", "skinny enough", "anorexia", "bulimia",
    "purging", "binge eating", "body dysmorphia", "hate my body", "ugly",
    "fat", "disgusting", "scars", "fresh cuts", "styro", "beans", 
    # DESPAIR
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
    # ANXIETY
    "anxiety", "anxious", "panic attack", "panicking", "freaking out",
    "scared", "terrified", "fear", "dread", "doom", "overwhelmed",
    "stressed", "stress", "pressure", "can't cope", "can't handle it",
    "breaking down", "breakdown", "mental breakdown", "nervous breakdown",
    "shaking", "trembling", "heart racing", "chest pain", "nauseous",
    "spiraling", "out of control", "losing my mind", "going crazy",
    "going mad", "insane", "manic", "mania", "paranoid", "paranoia",
    "intrusive thoughts", "voices in my head", "demons", "haunted",
    # DEPRESSION
    "depressed", "depression", "sad", "sadness", "crying", "tears",
    "sobbing", "cry myself to sleep", "can't stop crying", "exhausted",
    "tired", "fatigue", "lethargic", "no energy", "can't get out of bed",
    "sleeping all day", "insomnia", "can't sleep", "up all night",
    "don't want to wake up", "nightmare", "emotional pain", "hurts so much",
    "pain inside", "aching", "agony", "torture", "suffering", "suffer",
    "why me", "what's the point", "nothing matters", "apathy", "don't care",
    # HELP / CONTEXT
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
    if len(edges) != len(tweets):
        print(f"Warning: Edges count ({len(edges)}) != Tweets count ({len(tweets)}). Joining by index.")
    
    edges = edges.reset_index(drop=True)
    tweets = tweets.reset_index(drop=True)
    interactions = pd.concat([edges, tweets], axis=1)
    
    # --- CLEANING ---
    print(" > Cleaning IDs...")
    nodes['User ID'] = clean_id(nodes['User ID'])
    interactions['User ID (Source)'] = clean_id(interactions['User ID (Source)'])
    interactions['User ID (Destination)'] = clean_id(interactions['User ID (Destination)'])
    
    text_col = 'Tweet' if 'Tweet' in interactions.columns else 'Text'
    if text_col not in interactions.columns: text_col = tweets.columns[0]
    print(f" > Using '{text_col}' as text source.")

    # --- 1. WEAK SUPERVISION (WITH CAP) ---
    print("Step 1: Identifying Seed Nodes...")
    risk_interactions = interactions[interactions[text_col].astype(str).str.contains('|'.join(RISK_KEYWORDS), case=False, na=False)]
    seed_ids = risk_interactions['User ID (Source)'].unique().tolist()
    print(f" > Found {len(seed_ids)} potential high-risk seeds.")
    
    # --- SEED CAPPING ---
    target_seeds = int(args.target_nodes * args.risk_ratio)
    
    if len(seed_ids) > target_seeds:
        # print(f" > Capping seeds to {target_seeds} to maintain target ratio ({args.risk_ratio*100}%).")
        seed_ids = list(np.random.choice(seed_ids, target_seeds, replace=False))
    elif len(seed_ids) < 10:
        print(" > Warning: Very few seeds found. Adding random seeds.")
        seed_ids.extend(nodes['User ID'].sample(50).tolist())

    node_labels = {uid: 0 for uid in nodes['User ID']}
    for uid in seed_ids: node_labels[uid] = 1

    # --- 2. SNOWBALL SAMPLING ---
    print(f"Step 2: Snowball Sampling (Target: {args.target_nodes})...")
    adj = {}
    for row in interactions.itertuples():
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
    
    # Standard Snowball
    while len(selected_users) < args.target_nodes and len(queue) > 0:
        curr = queue.pop(0)
        neighbors = adj.get(curr, [])
        for n in neighbors:
            if n not in selected_users:
                selected_users.add(n)
                queue.append(n)
                if len(selected_users) >= args.target_nodes: break
    
    print(f" > Snowball reached {len(selected_users)} nodes.")

    # --- 3. RANDOM BACKFILL ---
    # If we haven't hit the target size, fill with random Safe users from the Metadata file
    if len(selected_users) < args.target_nodes:
        remaining_needed = args.target_nodes - len(selected_users)
        print(f" > Backfilling with {remaining_needed} random safe users to hit target size...")
        
        all_users = set(nodes['User ID'])
        available = list(all_users - selected_users)
        
        if len(available) >= remaining_needed:
            fill = np.random.choice(available, remaining_needed, replace=False)
            selected_users.update(fill)
        else:
            print(" > Warning: Not enough users in metadata to fill target. Using all available.")
            selected_users.update(available)

    # --- 4. LABEL PROPAGATION ---
    print("Step 4: Label Propagation...")
    final_labels = {}
    for uid in selected_users:
        if node_labels.get(uid, 0) == 1:
            final_labels[uid] = 1
            continue
        neighbors = adj.get(uid, [])
        risk_neighbors = sum(1 for n in neighbors if n in selected_users and node_labels.get(n, 0) == 1)
        final_labels[uid] = 1 if risk_neighbors >= 2 else 0

    # --- 5. STANDARDIZED SAVING ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Step 5: Saving standardized CSVs...")

    # SAVE NODES (Filter Ghosts: only keep selected users that actually exist in Nodes Data)
    final_nodes = nodes[nodes['User ID'].isin(selected_users)].copy()
    final_nodes['label'] = final_nodes['User ID'].map(final_labels)
    final_nodes = final_nodes.rename(columns={'User ID':'user_id', 'Followers':'followers', 'Followed':'followed', 'Verified':'verified'})
    cols_to_keep = [c for c in ['user_id', 'label', 'followers', 'followed', 'verified'] if c in final_nodes.columns]
    final_nodes[cols_to_keep].to_csv(OUTPUT_DIR / "nodes.csv", index=False)

    # SAVE EDGES
    final_interactions = interactions[
        (interactions['User ID (Source)'].isin(selected_users)) & 
        (interactions['User ID (Destination)'].isin(selected_users))
    ].copy()
    
    final_edges = final_interactions[['User ID (Source)', 'User ID (Destination)', text_col]].copy()
    final_edges = final_edges.rename(columns={
        'User ID (Source)': 'source_user_id', 
        'User ID (Destination)': 'target_user_id',
        text_col: 'text'
    })
    final_edges.to_csv(OUTPUT_DIR / "edges.csv", index=False)
    
    # --- FINAL SUMMARY STATS ---
    print("\n" + "="*40)
    print("DATA PREPARATION COMPLETE")
    
    n_high_risk = final_nodes['label'].sum()
    n_total = len(final_nodes)
    ratio = (n_high_risk / n_total) * 100 if n_total > 0 else 0
    
    print(f"Total Nodes:      {n_total}")
    print(f"Total Interactions: {len(final_edges)}")
    print(f"High Risk Users:  {n_high_risk} ({ratio:.1f}%)")
    print(f"Safe Users:       {n_total - n_high_risk}")
    print(f"Saved to:         {OUTPUT_DIR.resolve()}")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_pipeline()