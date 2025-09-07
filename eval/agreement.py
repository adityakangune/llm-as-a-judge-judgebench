import argparse
import json
import pandas as pd
from scipy.stats import spearmanr, pointbiserialr

def per_user_signals(preds: pd.DataFrame, truth: pd.DataFrame, k: int = 10):
    """
    Compute per-user classic signals:
      - hit@k (0/1)
      - ndcg@k (0 if miss, 1/log2(rank+1) if hit)
    Assumes preds has columns: user_id,item_id,rank (1=best)
    Assumes truth has columns: user_id,item_id (one test item per user)
    """
    # keep only top-k
    p = preds[preds["rank"] <= k].copy()

    # truth map
    tmap = dict(zip(truth.user_id, truth.item_id))

    # find hit rank per user (if any)
    hit_ranks = (
        p.assign(is_hit=lambda df: df.apply(lambda r: int(r.item_id == tmap.get(r.user_id)), axis=1))
         .query("is_hit == 1")
         .groupby("user_id")["rank"].min()
    )

    # all users in preds (groupby preserves users with at least one row)
    users = p["user_id"].unique()

    import math
    hit_at_k = {}
    ndcg_at_k = {}
    for u in users:
        if u in hit_ranks:
            r = int(hit_ranks[u])
            hit_at_k[u] = 1
            ndcg_at_k[u] = 1.0 / math.log2(r + 1)
        else:
            hit_at_k[u] = 0
            ndcg_at_k[u] = 0.0

    df = pd.DataFrame({
        "user_id": users,
        "hit_at_k": [hit_at_k[u] for u in users],
        "ndcg_user": [ndcg_at_k[u] for u in users],
    })
    return df

def main(classic_json: str, llm_csv: str, preds_csv: str, truth_csv: str, out_json: str, k: int):
    # summaries (still useful to show)
    with open(classic_json, "r") as f:
        classic = json.load(f)

    llm = pd.read_csv(llm_csv)
    preds = pd.read_csv(preds_csv)
    truth = pd.read_csv(truth_csv)

    # per-user LLM score (mean over that user's list)
    user_llm = llm.groupby("user_id")["score_llm"].mean().rename("llm_score")

    # per-user classic signals that vary
    per_user = per_user_signals(preds, truth, k=k).set_index("user_id")

    df = per_user.join(user_llm, how="inner").dropna()

    # Spearman between LLM score and per-user NDCG (continuous)
    rho_ndcg, p_ndcg = spearmanr(df["llm_score"], df["ndcg_user"])

    # Point-biserial correlation between LLM score and hit@k (binary)
    # (equivalent to Pearson with a binary variable)
    try:
        r_hit, p_hit = pointbiserialr(df["hit_at_k"], df["llm_score"])
    except Exception:
        r_hit, p_hit = float("nan"), float("nan")

    out = {
        "classic_summary": classic,
        "counts": {
            "users_evaluated": int(df.shape[0]),
            "hits_at_k": int(df["hit_at_k"].sum()),
        },
        "correlations": {
            "spearman_llm_vs_user_ndcg": {"rho": float(rho_ndcg), "p_value": float(p_ndcg)},
            "pointbiserial_llm_vs_hit_at_k": {"r": float(r_hit), "p_value": float(p_hit)},
        },
        "notes": "Per-user signals avoid constant-vector issue; k used for top-K={}".format(k),
    }

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved agreement stats to", out_json)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classic", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    main(args.classic, args.llm, args.preds, args.truth, args.out, args.k)
