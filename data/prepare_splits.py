
import argparse
from pathlib import Path
import pandas as pd

# MovieLens 1M format: ratings.dat (UserID::MovieID::Rating::Timestamp)

def load_ml1m(raw_dir: Path) -> pd.DataFrame:
    ratings = pd.read_csv(raw_dir / "ml-1m" / "ratings.dat", sep="::", engine="python",
                          names=["user_id", "item_id", "rating", "ts"])
    return ratings

def leave_one_out_split(df: pd.DataFrame):
    # last interaction per user into test
    df = df.sort_values(["user_id", "ts"])
    last = df.groupby("user_id").tail(1)
    train = pd.concat([df, last]).drop_duplicates(keep=False)
    test = last[["user_id", "item_id"]].copy()
    return train, test

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_ml1m(raw)
    train, test = leave_one_out_split(df)

    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print("Saved train/test to", out)
