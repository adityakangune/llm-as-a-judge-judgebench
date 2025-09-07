
# JudgeBench

JudgeBench is a weekend-sized framework to study agreement and disagreement between classic recommender metrics (Precision@K, Recall@K, NDCG) and LLM-as-a-Judge scores. It includes a Streamlit app for interactive exploration and simple calibration using a small human-labeled set.

## What this repo shows
- Train a simple item-kNN recommender on MovieLens 1M
- Compute classic top-K metrics
- Score recommendations with an LLM-judge **or** a built-in mock-judge for offline runs
- Compare correlation and agreement, and visualize disagreements in an app
- Calibrate LLM scores to human labels using isotonic or logistic mapping

## Quickstart
```bash
# 1) setup
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) data
python data/download_movielens.py --dest data/raw
python data/prepare_splits.py --raw_dir data/raw --out_dir data/processed --seed 42

# 3) train + evaluate (classic)
python models/knn.py --data_dir data/processed --out_dir results --k 200 --topk 10
python eval/classic.py --preds results/preds_top10.csv --truth data/processed/test.csv --out results/classic_metrics.json

# 4) get LLM or mock scores
# mock (no API needed)
python eval/llm_judge.py --mode mock --preds results/preds_top10.csv --out results/llm_scores.csv
# or real LLM (set your provider/key via env)
# export OPENAI_API_KEY=... or ANTHROPIC_API_KEY=...
python eval/llm_judge.py --mode openai --preds results/preds_top10.csv --out results/llm_scores.csv

# 5) agreement + calibration
python eval/agreement.py --classic results/classic_metrics.json --llm results/llm_scores.csv --out results/agreement.json
# optional: if you have a small human label CSV
python eval/calibrate.py --llm results/llm_scores.csv --human data/human_labels.csv --out results/calibrated_scores.csv

# 6) run app
streamlit run app/streamlit_app.py
```
