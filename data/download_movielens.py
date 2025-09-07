
import argparse
import os
import zipfile
import requests
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

def download(url: str, dest_dir: str):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    zip_path = Path(dest_dir) / "ml-1m.zip"
    if zip_path.exists():
        print("Already downloaded")
        return zip_path
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    return zip_path

def extract(zip_path: Path, dest_dir: str):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()
    zp = download(URL, args.dest)
    extract(zp, args.dest)
    print("Done. Raw data in", args.dest)
