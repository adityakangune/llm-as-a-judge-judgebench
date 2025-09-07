import pandas as pd
df = pd.read_csv("results/llm_scores.csv")
print(df["score_llm"].value_counts())
