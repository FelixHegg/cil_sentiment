import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import glob
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# 1. Sample 100 sentences (without labels) and split into 4 files of 25 each
# ---------------------------------------------------------------------------
df = pd.read_csv('../data/training.csv')
df100 = df.sample(n=100, random_state=42).reset_index(drop=True)
df100_nolabel = df100[["id","sentence"]]
df100_nolabel["label"] = ""        # ← empty column for your teammates
chunks = np.array_split(df100_nolabel, 4)
for i, chunk in enumerate(chunks, start=1):
    chunk.to_csv(f"group_member_{i}_to_label.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Read back the annotated files and assemble predictions
# ---------------------------------------------------------------------------
files = glob.glob("group_member_*_to_label.csv")
labeled = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
labeled["label"] = labeled["label"].str.lower().str.strip()
labeled = labeled.rename(columns={"label": "pred_label"})
labeled.to_csv("100_labeled_by_team.csv", index=False)

# --------------------------------------------------------------------------
# 3. Compute human‑annotation accuracy
# --------------------------------------------------------------------------
pred = pd.read_csv("100_labeled_by_team.csv")  
true = pd.read_csv("../data/training.csv")[["id","label"]]
df = pred.merge(true, on="id", how="left")
acc = accuracy_score(df["label"], df["pred_label"])
print(f"Accuracy: {acc:.2%}")

# --------------------------------------------------------------------------
# 4. Compute competition scoring function L
# --------------------------------------------------------------------------
mapping = {"negative": -1, "neutral": 0, "positive": 1}
y_true = df["label"].map(mapping).values
y_pred = df["pred_label"].map(mapping).values
mae = np.mean(np.abs(y_true - y_pred))
L = 0.5 * (2 - mae)
print(f"Competition score L: {L:.3f}")