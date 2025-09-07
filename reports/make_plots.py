import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def per_user_signals(preds: pd.DataFrame, truth: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Per-user hit@k (0/1) and NDCG@k (0 if miss, 1/log2(rank+1) if hit)."""
    p = preds.loc[preds["rank"] <= k, ["user_id", "item_id", "rank"]].copy()
    tmap = dict(zip(truth.user_id, truth.item_id))

    # hit rank for each user (if any)
    hits = (
        p.assign(is_hit=lambda df: (df["item_id"] == df["user_id"].map(tmap)).astype(int))
         .query("is_hit == 1")
         .groupby("user_id")["rank"].min()
    )

    users = p["user_id"].unique()
    rows = []
    for u in users:
        if u in hits:
            r = int(hits[u])
            rows.append((u, 1, 1.0 / math.log2(r + 1)))
        else:
            rows.append((u, 0, 0.0))
    return pd.DataFrame(rows, columns=["user_id", "hit_at_k", "ndcg_user"])


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_llm_hist(llm: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    ax.hist(llm["score_llm"], bins=bins)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("LLM score (1–5)")
    ax.set_ylabel("Count")
    ax.set_title("LLM-as-a-Judge Score Distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_box_llm_vs_hit(per_user: pd.DataFrame, llm: pd.DataFrame, out_path: Path):
    df = per_user.merge(llm.groupby("user_id")["score_llm"].mean().rename("llm_score"),
                        on="user_id", how="inner")
    g0 = df.loc[df["hit_at_k"] == 0, "llm_score"].dropna()
    g1 = df.loc[df["hit_at_k"] == 1, "llm_score"].dropna()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([g0, g1], labels=["miss", "hit"], showmeans=True)
    ax.set_ylabel("Per-user mean LLM score")
    ax.set_title("LLM Score vs Hit@K")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_scatter_ndcg_vs_llm(per_user: pd.DataFrame, llm: pd.DataFrame, out_path: Path):
    df = per_user.merge(llm.groupby("user_id")["score_llm"].mean().rename("llm_score"),
                        on="user_id", how="inner")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["ndcg_user"], df["llm_score"], s=8)
    ax.set_xlabel("Per-user NDCG@K")
    ax.set_ylabel("Per-user mean LLM score")
    ax.set_title("Per-user NDCG vs LLM Score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_classic_bar(classic_json_path: Path, label: str, out_path: Path):
    with open(classic_json_path) as f:
        cm = json.load(f)

    metrics = ["precision", "recall", "ndcg"]
    values = [cm.get(m, 0.0) for m in metrics]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metrics, values)
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"Classic Metrics ({label})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(
    preds_csv="results/preds_als_top10.csv",
    truth_csv="data/processed/test.csv",
    llm_csv="results/llm_scores.csv",
    classic_json="results/classic_metrics.json",
    k=10,
    label="ALS@10",
    out_dir="results",
):
    ensure_dir(out_dir)
    preds = pd.read_csv(preds_csv)
    truth = pd.read_csv(truth_csv)
    llm = pd.read_csv(llm_csv)

    per_user = per_user_signals(preds, truth, k=int(k))

    # 1) Histogram
    plot_llm_hist(llm, Path(out_dir) / f"plot_llm_hist_{label}.png")

    # 2) Boxplot vs hit
    plot_box_llm_vs_hit(per_user, llm, Path(out_dir) / f"plot_llm_vs_hit_{label}.png")

    # 3) Scatter NDCG vs LLM
    plot_scatter_ndcg_vs_llm(per_user, llm, Path(out_dir) / f"plot_ndcg_vs_llm_{label}.png")

    # 4) Classic metrics bar
    plot_classic_bar(Path(classic_json), label, Path(out_dir) / f"plot_classic_metrics_{label}.png")

    print("Saved figures in", out_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/preds_als_top10.csv")
    ap.add_argument("--truth", default="data/processed/test.csv")
    ap.add_argument("--llm", default="results/llm_scores.csv")
    ap.add_argument("--classic", default="results/classic_metrics.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--label", default="ALS@10")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    main(
        preds_csv=args.preds,
        truth_csv=args.truth,
        llm_csv=args.llm,
        classic_json=args.classic,
        k=args.k,
        label=args.label,
        out_dir=args.out_dir,
    )
