import argparse
import pandas as pd
from surprise import Dataset, Reader, SVD
from collections import defaultdict

def get_top_n(predictions, n=10):
    """Return top-N recommendation list for each user from a set of predictions."""
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    # sort and take top-n
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def main(train_csv, out_csv, topk=10, n_factors=100, n_epochs=20):
    df = pd.read_csv(train_csv)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

    trainset = data.build_full_trainset()

    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, verbose=True)
    algo.fit(trainset)

    # Predict all user–item pairs not in training set
    anti_testset = trainset.build_anti_testset()
    predictions = algo.test(anti_testset)

    top_n = get_top_n(predictions, n=topk)

    # Flatten to DataFrame
    rows = []
    for uid, recs in top_n.items():
        for rank, (iid, _) in enumerate(recs, start=1):
            rows.append((int(uid), int(iid), rank))
    out = pd.DataFrame(rows, columns=["user_id", "item_id", "rank"])
    out.to_csv(out_csv, index=False)
    print(f"Saved top-{topk} predictions to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--out", default="results/preds_svd_top10.csv")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--factors", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    main(args.train, args.out, args.topk, args.factors, args.epochs)
