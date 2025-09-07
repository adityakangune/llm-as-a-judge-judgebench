import argparse, os, json, time, re, csv
import pandas as pd
from tqdm import tqdm

# ----- flexible import for prompts -----
try:
    from . import prompts as P  # type: ignore
except Exception:
    import importlib.util, pathlib
    here = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location("prompts", here / "prompts.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    P = mod  # type: ignore

# -----------------------
# Mock judge
# -----------------------
import random
def mock_score(group):
    pos = set(group.get('positives', []))
    recs = group['items']
    overlap = len(pos.intersection(recs))
    diversity_penalty = max(0, len(recs) - len(set(recs)))
    base = 3 + (1 if overlap > 0 else 0) - (1 if diversity_penalty > 0 else 0)
    jitter = random.choice([-1, 0, 1])
    score = max(1, min(5, base + jitter))
    rationale = f"mock: overlap={overlap}, dup_penalty={diversity_penalty}, jitter={jitter}"
    return score, rationale

# -----------------------
# OpenAI judge
# -----------------------
def openai_call(system_prompt: str, user_prompt: str):
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role":"system","content":P.SYSTEM_PROMPT},
                  {"role":"user","content":user_prompt}],
    )
    msg = resp.choices[0].message.content
    try:
        data = json.loads(msg)
        return int(data.get("score", 3)), str(data.get("rationale",""))
    except Exception:
        return 3, f"fallback: could not parse JSON (got: {msg[:100]})"

# -----------------------
# Gemini judge (with retries)
# -----------------------
def gemini_call(system_prompt: str, user_prompt: str):
    import google.generativeai as genai
    from google.api_core import exceptions as gex
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"{system_prompt}\n\n{user_prompt}"
    backoff = 8
    for attempt in range(6):
        try:
            resp = model.generate_content(prompt, generation_config={"temperature": 0})
            msg = (resp.text or "").strip()
            data = json.loads(msg)
            return int(data.get("score", 3)), str(data.get("rationale", ""))
        except gex.ResourceExhausted as e:
            # Parse suggested retry delay if present; else sleep a safe default.
            m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", str(e))
            wait = int(m.group(1)) if m else max(20, backoff)
            time.sleep(wait)
            backoff = min(backoff * 2, 120)
        except Exception as e:
            # Could not parse JSON or transient error; return neutral score
            msg = str(e)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            return 3, f"fallback: error/parsing issue ({msg[:100]})"

    return 3, "fallback: retries exhausted"

# -----------------------
# Utilities
# -----------------------
def build_user_histories(train_path: str) -> dict:
    train = pd.read_csv(train_path)
    pos = train[train["rating"] >= 4]
    return pos.groupby("user_id")["item_id"].apply(list).to_dict()

def ensure_rate(last_ts: float, rpm: float) -> float:
    """sleep so we don't exceed requests-per-minute"""
    if rpm <= 0: return time.time()
    min_interval = 60.0 / rpm
    now = time.time()
    elapsed = now - last_ts
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
        now = time.time()
    return now

# -----------------------
# Main
# -----------------------
def main(mode: str, preds_path: str, train_path: str, out_csv: str,
         rpm: float, max_users: int, resume: bool):
    preds = pd.read_csv(preds_path)
    user_hist = build_user_histories(train_path)

    # figure out which users still need scoring (resume support)
    done_users = set()
    if resume and os.path.exists(out_csv):
        try:
            existing = pd.read_csv(out_csv, usecols=["user_id"])
            done_users = set(existing["user_id"].unique())
        except Exception:
            done_users = set()

    grouped = list(preds.groupby("user_id"))
    if max_users > 0:
        grouped = grouped[:max_users]

    # Prepare output writer (append-safe)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(out_csv) or not resume
    out_f = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.writer(out_f)
    if write_header:
        writer.writerow(["user_id","item_id","rank","score_llm","rationale"])

    last_ts = 0.0
    count_since_flush = 0
    try:
        for uid, g in tqdm(grouped, desc=f"{mode} judging"):
            if uid in done_users:
                continue
            items = g.sort_values("rank")["item_id"].tolist()
            positives = user_hist.get(uid, [])[:10]
            user_prompt = P.USER_PROMPT_TEMPLATE.format(
                positives=positives, recommendations=items)

            # rate-limit before API call
            if mode in ("openai","gemini"):
                last_ts = ensure_rate(last_ts, rpm)

            if mode == "mock":
                score, rat = mock_score({"positives": set(positives), "items": items})
            elif mode == "openai":
                score, rat = openai_call(P.SYSTEM_PROMPT, user_prompt)
            elif mode == "gemini":
                score, rat = gemini_call(P.SYSTEM_PROMPT, user_prompt)
            else:
                raise ValueError("mode must be 'mock', 'openai', or 'gemini'")

            # write rows for this user
            for _, row in g.iterrows():
                writer.writerow([uid, int(row["item_id"]), int(row["rank"]), score, rat])
            count_since_flush += 1
            if count_since_flush >= 25:
                out_f.flush()
                count_since_flush = 0
    finally:
        out_f.flush()
        out_f.close()

    print("Saved LLM scores to", out_csv)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock","openai","gemini"], required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--rpm", type=float, default=12.0, help="requests per minute cap (Gemini free-tier ~15 rpm)")
    ap.add_argument("--max_users", type=int, default=0, help="limit number of users (0 = all)")
    ap.add_argument("--resume", action="store_true", help="append to existing CSV and skip completed users")
    args = ap.parse_args()
    main(args.mode, args.preds, args.train, args.out, args.rpm, args.max_users, args.resume)

