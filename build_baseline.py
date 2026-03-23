import pandas as pd

df = pd.read_json("data/python_splits/test_verified.jsonl", lines=True)
df["output"] = df["input"]
df.to_json("results/baseline/baseline.jsonl", lines=True, orient="records")