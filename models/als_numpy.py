import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import trange

"""
Simple implicit ALS (Hu, Koren, Volinsky 2008) in NumPy.
Treats ratings >=4 as implicit positives with confidence c = 1 + alpha * r.

Outputs top-K recommendations per user in the same CSV format as kNN:
user_id,item_id,rank
"""

def build_matrices(train_csv, min_items=5, min_users=5):
    df = pd.read_csv(train_csv)
    # keep a bit denser subset for speed
    uc = df.groupby("user_id").size()
    ic = df.groupby("item_id").size()
    keep_u = uc[uc >= min_items].index
    keep_i = ic[ic >= min_users].index
    df = df[df.user_id.isin(keep_u) & df.item_id.isin(keep_i)]

    # reindex to [0..U-1], [0..I-1]
    uids = {u:i for i,u in enumerate(sorted(df.user_id.unique()))}
    iids = {i:j for j,i in enumerate(sorted(df.item_id.unique()))}

    rows = df["user_id"].map(uids).to_numpy()
    cols = df["item_id"].map(iids).to_numpy()
    vals = (df["rating"] >= 4).astype(float).to_numpy()  # implicit 0/1

    U, I = len(uids), len(iids)
    X = csr_matrix((vals, (rows, cols)), shape=(U, I))
    return X, uids, iids

def als_implicit(X, factors=64, reg=0.1, alpha=40.0, iters=10, seed=42):
    """
    X: csr_matrix of shape (U, I) with 0/1 implicit data
    Returns: user_factors (U x F), item_factors (I x F)
    """
    rng = np.random.default_rng(seed)
    U, I = X.shape
    # latent factors
    P = 0.1 * rng.standard_normal((U, factors))
    Q = 0.1 * rng.standard_normal((I, factors))

    # precompute identity
    I_F = np.eye(factors)

    # convert to CSR/CSC for fast row/col access
    X_csr = X.tocsr()
    X_csc = X.tocsc()

    # confidence = 1 + alpha * x
    for _ in trange(iters, desc="ALS"):
        # fix Q, solve for P
        QtQ = Q.T @ Q + reg * I_F
        for u in range(U):
            start, end = X_csr.indptr[u], X_csr.indptr[u+1]
            idx = X_csr.indices[start:end]           # items user u interacted with
            if idx.size == 0:
                continue
            Cu = 1.0 + alpha * np.ones_like(idx, dtype=float)
            Pu = np.ones_like(idx, dtype=float)      # preference 1 for observed
            Qu = Q[idx]                               # |idx| x F
            A = QtQ + (Qu.T * (Cu - 1)) @ Qu
            b = (Qu.T * Cu) @ Pu
            P[u] = np.linalg.solve(A, b)

        # fix P, solve for Q
        PtP = P.T @ P + reg * I_F
        for i in range(I):
            start, end = X_csc.indptr[i], X_csc.indptr[i+1]
            idx = X_csc.indices[start:end]           # users who interacted with item i
            if idx.size == 0:
                continue
            Ci = 1.0 + alpha * np.ones_like(idx, dtype=float)
            Pi = np.ones_like(idx, dtype=float)
            Pu = P[idx]                               # |idx| x F
            A = PtP + (Pu.T * (Ci - 1)) @ Pu
            b = (Pu.T * Ci) @ Pi
            Q[i] = np.linalg.solve(A, b)

    return P, Q

def topk_recs(P, Q, X, K=10):
    """Return top-K unseen item ids per user."""
    U, I = X.shape
    scores = P @ Q.T                                  # U x I
    # mask seen items with -inf
    X_coo = X.tocoo()
    scores[X_coo.row, X_coo.col] = -1e9

    recs = []
    for u in range(U):
        row = scores[u]
        # get top K indices
        idx = np.argpartition(row, -K)[-K:]
        idx = idx[np.argsort(row[idx])][::-1]
        for rank, i in enumerate(idx, 1):
            recs.append((u, i, rank))
    return recs

def main(train_csv, out_csv, topk, factors, reg, alpha, iters):
    X, uids, iids = build_matrices(train_csv)
    P, Q = als_implicit(X, factors=factors, reg=reg, alpha=alpha, iters=iters)

    # map back to original ids
    inv_u = {v:k for k,v in uids.items()}
    inv_i = {v:k for k,v in iids.items()}

    recs = topk_recs(P, Q, X, K=topk)
    rows = [(int(inv_u[u]), int(inv_i[i]), int(r)) for (u,i,r) in recs]
    df = pd.DataFrame(rows, columns=["user_id","item_id","rank"]).sort_values(["user_id","rank"])
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} recommendations to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--out", default="results/preds_als_top10.csv")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=40.0)
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()
    main(args.train, args.out, args.topk, args.factors, args.reg, args.alpha, args.iters)
