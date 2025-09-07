
.PHONY: setup data classic llm mock agreement calibrate app

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:
	python data/download_movielens.py --dest data/raw
	python data/prepare_splits.py --raw_dir data/raw --out_dir data/processed --seed 42

classic:
	python models/knn.py --data_dir data/processed --out_dir results --k 200 --topk 10
	python eval/classic.py --preds results/preds_top10.csv --truth data/processed/test.csv --out results/classic_metrics.json

mock:
	python eval/llm_judge.py --mode mock --preds results/preds_top10.csv --out results/llm_scores.csv

llm:
	python eval/llm_judge.py --mode openai --preds results/preds_top10.csv --out results/llm_scores.csv

agreement:
	python eval/agreement.py --classic results/classic_metrics.json --llm results/llm_scores.csv --out results/agreement.json

calibrate:
	python eval/calibrate.py --llm results/llm_scores.csv --human data/human_labels.csv --out results/calibrated_scores.csv

app:
	streamlit run app/streamlit_app.py
