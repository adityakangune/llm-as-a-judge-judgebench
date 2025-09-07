import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="JudgeBench", layout="wide")
st.title("JudgeBench — Classic vs LLM-as-a-Judge")

# -----------------------------
# Helpers
# -----------------------------
def load_json(path):
    try:
        return json.load(open(path, "r"))
    except Exception:
        return None

def per_user_signals(preds: pd.DataFrame, truth: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Compute per-user classic signals:
      - hit@k (0/1)
      - ndcg_user@k (0 if miss, 1/log2(rank+1) if hit)
    Requires preds with columns: user_id,item_id,rank (1=best)
             truth with columns: user_id,item_id (one per user)
    """
    if preds is None or truth is None or preds.empty or truth.empty:
        return pd.DataFrame(columns=["user_id", "hit_at_k", "ndcg_user"])

    p = preds[preds["rank"] <= k].copy()
    tmap = dict(zip(truth.user_id, truth.item_id))

    def is_hit_row(r):
        return int(r.item_id == tmap.get(r.user_id))

    hits = (
        p.assign(is_hit=lambda df: df.apply(is_hit_row, axis=1))
         .query("is_hit == 1")
         .groupby("user_id")["rank"].min()
    )

    import math
    users = p["user_id"].unique()
    hit_at_k = {}
    ndcg_at_k = {}
    for u in users:
        if u in hits:
            r = int(hits[u])
            hit_at_k[u] = 1
            ndcg_at_k[u] = 1.0 / math.log2(r + 1)
        else:
            hit_at_k[u] = 0
            ndcg_at_k[u] = 0.0

    return pd.DataFrame({
        "user_id": users,
        "hit_at_k": [hit_at_k[u] for u in users],
        "ndcg_user": [ndcg_at_k[u] for u in users],
    })


# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Inputs")
    preds_path = st.text_input("Predictions CSV", value="results/preds_top10.csv")
    truth_path = st.text_input("Test CSV", value="data/processed/test.csv")
    classic_json = st.text_input("Classic metrics JSON", value="results/classic_metrics.json")
    llm_scores_path = st.text_input("LLM scores CSV", value="results/llm_scores.csv")
    k_val = st.number_input("K", min_value=1, max_value=100, value=10, step=1)

# -----------------------------
# Load data
# -----------------------------
classic = load_json(classic_json) if Path(classic_json).exists() else None
preds = pd.read_csv(preds_path) if Path(preds_path).exists() else None
truth = pd.read_csv(truth_path) if Path(truth_path).exists() else None
llm = pd.read_csv(llm_scores_path) if Path(llm_scores_path).exists() else None

# -----------------------------
# Overview
# -----------------------------
st.subheader("Overview")
cols = st.columns(2)
with cols[0]:
    if classic is not None:
        st.markdown("**Classic metrics**")
        st.json(classic)
    else:
        st.info("Load classic metrics JSON to view summary.")
with cols[1]:
    if llm is not None and not llm.empty:
        st.markdown("**LLM scores (sample)**")
        st.dataframe(llm.head(20))
    else:
        st.info("Load LLM scores CSV to view sample.")

st.markdown("---")

# -----------------------------
# Disagreement explorer (existing)
# -----------------------------
st.subheader("Disagreement explorer")
if llm is not None and not llm.empty:
    # per-user aggregates
    user_llm = llm.groupby('user_id')['score_llm'].mean().rename('llm_score')
    inv_rank = llm.groupby('user_id')['rank'].apply(lambda s: (1 / s).mean()).rename('inv_mean_rank')
    agg = pd.concat([user_llm, inv_rank], axis=1).reset_index()

    st.caption("Scatter: per-user inverse-mean-rank (proxy for classic strength) vs LLM score")
    st.scatter_chart(agg, x='inv_mean_rank', y='llm_score')

    if not agg.empty:
        uid = st.number_input(
            "Inspect user_id", 
            min_value=int(agg.user_id.min()), 
            max_value=int(agg.user_id.max()), 
            value=int(agg.user_id.iloc[0])
        )
        if uid in llm.user_id.values:
            urows = llm[llm.user_id == uid].sort_values('rank')
            st.write("Recommendations for user", uid)
            st.dataframe(urows)
            rats = urows['rationale'].dropna().unique()
            if len(rats):
                st.info(rats[0][:500])
        else:
            st.info("User not in current LLM scores.")
else:
    st.info("Load LLM scores to view disagreement explorer.")

st.markdown("---")

# -----------------------------
# New: LLM score histogram + boxplot vs hit@K
# -----------------------------
st.subheader("LLM score distribution & relation to classic signals")

if llm is not None and preds is not None and truth is not None and not llm.empty and not preds.empty and not truth.empty:
    # 1) Histogram of score_llm
    st.markdown("**Histogram of LLM scores**")
    fig1, ax1 = plt.subplots()
    llm["score_llm"].plot(kind="hist", bins=[0.5,1.5,2.5,3.5,4.5,5.5], ax=ax1)
    ax1.set_xlabel("score_llm (1..5)")
    ax1.set_ylabel("count")
    ax1.set_title("LLM score distribution")
    st.pyplot(fig1)

    # 2) Boxplot of LLM score vs hit@K (per user)
    per_user = per_user_signals(preds, truth, k=int(k_val))
    df = per_user.merge(llm.groupby("user_id")["score_llm"].mean().reset_index(), on="user_id", how="inner")
    if not df.empty:
        st.markdown(f"**Boxplot: LLM score vs hit@{int(k_val)} (per-user)**")
        # Prepare data for 0/1 groups
        g0 = df.loc[df["hit_at_k"] == 0, "score_llm"]
        g1 = df.loc[df["hit_at_k"] == 1, "score_llm"]

        fig2, ax2 = plt.subplots()
        ax2.boxplot([g0.dropna(), g1.dropna()], labels=["miss", "hit"], showmeans=True)
        ax2.set_ylabel("per-user mean LLM score")
        ax2.set_title(f"LLM score by hit@{int(k_val)}")
        st.pyplot(fig2)

        # 3) Scatter: per-user ndcg_user vs LLM score
        st.markdown(f"**Scatter: per-user NDCG@{int(k_val)} vs LLM score**")
        fig3, ax3 = plt.subplots()
        ax3.scatter(df["ndcg_user"], df["score_llm"], s=8)
        ax3.set_xlabel(f"NDCG@{int(k_val)} per user")
        ax3.set_ylabel("per-user mean LLM score")
        ax3.set_title("Per-user NDCG vs LLM score")
        st.pyplot(fig3)
    else:
        st.info("Could not compute per-user signals (no overlap between users in preds/llm/truth).")
else:
    st.info("Load predictions, truth, and LLM scores to view these plots.")

st.markdown("— Made with a weekend spirit.")
