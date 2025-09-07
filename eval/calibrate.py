
import argparse
import pandas as pd
from sklearn.isotonic import IsotonicRegression

"""
Fit a simple calibration mapping from LLM scores (1..5) to human labels (1..5).
Outputs calibrated_scores.csv with an extra column 'score_calibrated'.
"""

def main(llm_csv: str, human_csv: str, out_csv: str):
    llm = pd.read_csv(llm_csv)
    human = pd.read_csv(human_csv)  # columns: user_id,item_id,rank,label
    # join on user_id,item_id,rank
    df = llm.merge(human, on=["user_id", "item_id", "rank"], how="inner")

    x = df['score_llm'].astype(float)
    y = df['label'].astype(float)

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(x, y)

    llm['score_calibrated'] = iso.predict(llm['score_llm'].astype(float))
    llm.to_csv(out_csv, index=False)
    print("Saved calibrated scores to", out_csv)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.llm, args.human, args.out)
