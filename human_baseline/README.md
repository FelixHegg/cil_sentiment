# Human Baseline Annotation Script

This script helps you establish a human‐annotation baseline on a sentiment‐analysis dataset by:

1. Sampling 100 sentences from your training data and splitting them into 4 files (25 each) for manual labeling.
2. Reading back those labeled files and assembling them into one predictions file.
3. Computing human‐annotation accuracy against the gold labels.
4. Computing the competition scoring function $L$.

> **Note:** The first step needs to be run separately from the others. Before running Steps 2–4, comment out the sampling block under **Step 1** so you don’t overwrite your labeled files.

---

## Prerequisites

* **Python** 3.6 or higher
* **Packages**:

  ```bash
  pip install pandas numpy scikit-learn
  ```

---

## Usage

### Step 1: Generate the labeling files

Run the script to sample and split the data:

```bash
python human_baseline.py
```

This creates four CSV files in your working directory:

* `group_member_1_to_label.csv`
* `group_member_2_to_label.csv`
* `group_member_3_to_label.csv`
* `group_member_4_to_label.csv`

Each file contains:

```
id, sentence, label
```

(where `label` is empty for manual annotation).

---

### Step 2: Label the data

Share each CSV file with a group member. They should fill in their `label` column with one of:

```
negative, neutral, positive
```

---

### Step 3: Assemble labels & compute metrics

1. **Comment out** (or remove) the sampling block at the top of `human_baseline.py` so your labeled files aren’t overwritten:

   ```python
   # --- Step 1: Sampling (comment out before Steps 2–4) ---
   # df = pd.read_csv('data/training.csv')
   # df100 = df.sample(n=100, random_state=42).reset_index(drop=True)
   # chunks = np.array_split(df100, 4)
   # for i, chunk in enumerate(chunks, 1):
   #     chunk.to_csv(f'group_member_{i}_to_label.csv', index=False)
   ```

2. Re-run the script:

   ```bash
   python human_baseline.py
   ```

This will:

* Concatenate all `group_member_*_to_label.csv` into `100_labeled_by_team.csv`
* Print **Accuracy** (percentage correct vs. `data/training.csv`)
* Print the **Competition Kaggle score L**

---

## Output

* **100\_labeled\_by\_team.csv**
  Contains all 100 human‐labeled sentences with columns:

  ```
  id, sentence, pred_label
  ```

* **Console Output**

  ```
  Accuracy: XX.XX%
  Competition score L: Y.YYY
  ```

---
