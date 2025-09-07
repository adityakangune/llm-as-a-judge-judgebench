
import argparse
import json
import pandas as pd

"""
Compute Precision@K, Recall@K, and NDCG@K for leave-one-out test.
Test file has one held-out item per user.
"""

def dcg(rel):
    import math
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel))

def ndcg_at_k(preds, truth, k=10):
    # truth: one item per user
    truth_item = dict(zip(truth.user_id, truth.item_id))
    grouped = preds.groupby('user_id')
    ndcgs = []
    for u, dfu in grouped:
        items = dfu.sort_values('rank').item_id.tolist()[:k]
        rel = [1 if it == truth_item.get(u) else 0 for it in items]
        idcg = 1.0  # ideal DCG is 1 at rank 1 in leave-one-out
        ndcgs.append(dcg(rel) / idcg)
    return sum(ndcgs) / len(ndcgs)

def precision_recall_at_k(preds, truth, k=10):
    truth_item = dict(zip(truth.user_id, truth.item_id))
    grouped = preds.groupby('user_id')
    hits = 0
    for u, dfu in grouped:
        items = dfu.sort_values('rank').item_id.tolist()[:k]
        if truth_item.get(u) in items:
            hits += 1
    n_users = grouped.ngroups
    prec = hits / (n_users * k)
    rec = hits / n_users
    return prec, rec

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    preds = pd.read_csv(args.preds)
    truth = pd.read_csv(args.truth)

    ndcg = ndcg_at_k(preds, truth, k=args.k)
    prec, rec = precision_recall_at_k(preds, truth, k=args.k)

    out = {"k": args.k, "ndcg": ndcg, "precision": prec, "recall": rec}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved classic metrics to", args.out)
