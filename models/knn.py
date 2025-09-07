
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

"""
Simple item-kNN recommender using cosine similarity over implicit feedback.
We treat ratings >= 4 as implicit positives.
"""

def build_ui_matrix(train: pd.DataFrame, min_items=5, min_users=5):
    # filter sparse users/items a bit for speed
    user_counts = train.groupby('user_id').size()
    item_counts = train.groupby('item_id').size()
    keep_users = user_counts[user_counts >= min_items].index
    keep_items = item_counts[item_counts >= min_users].index
    f = train[train['user_id'].isin(keep_users) & train['item_id'].isin(keep_items)]

    # implicit
    f = f.assign(implicit=(f['rating'] >= 4).astype(int))
    # pivot to user-item binary matrix
    ui = f.pivot_table(index='user_id', columns='item_id', values='implicit', fill_value=0)
    return ui

def recommend(ui: pd.DataFrame, topk=10, k=200):
    item_vecs = ui.values.T  # items x users
    sim = cosine_similarity(item_vecs)
    items = ui.columns.to_numpy()

    user_pos = {u: set(ui.columns[ui.loc[u].to_numpy().nonzero()[0]]) for u in ui.index}

    rows = []
    for u in tqdm(ui.index, desc="recommending"):
        # user profile as mean of positive items
        pos = list(user_pos[u])
        if not pos:
            continue
        pos_idx = np.isin(items, pos)
        # score each item by mean similarity to positives
        scores = sim[:, pos_idx].mean(axis=1)
        # filter seen
        seen_mask = np.isin(items, pos)
        scores[seen_mask] = -1e9
        top_idx = np.argpartition(scores, -topk)[-topk:]
        top_sorted = top_idx[np.argsort(scores[top_idx])[::-1]]
        for rank, idx in enumerate(top_sorted, 1):
            rows.append((u, items[idx], rank, float(scores[idx])))
    recs = pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "score"]) 
    return recs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    train = pd.read_csv(Path(args.data_dir) / "train.csv")
    ui = build_ui_matrix(train)
    recs = recommend(ui, topk=args.topk, k=args.k)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    outp = Path(args.out_dir) / f"preds_top{args.topk}.csv"
    recs.to_csv(outp, index=False)
    print("Saved", outp)
