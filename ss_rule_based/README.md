# Rule-Based Sentiment Analysis Script

This script applies simple, rule‐based sentiment analysis to your dataset using **TextBlob** and **VADER**. It will:

1. Load training and test data.  
2. Apply a TextBlob‐based sentiment classifier.  
3. Apply an NLTK VADER‐based sentiment classifier.  
4. Compute and print accuracy for both methods on the training set.  
5. Apply both methods to the test set and save submission files.

> **Note:** On first run, NLTK will download the VADER lexicon.

---

## Prerequisites

- **Python** 3.6 or higher  
- **Packages**:

  ```bash
  pip install pandas textblob nltk
  ```

---

## Usage

Simply run:

```bash
python RuleBased.py
```

What happens:

1. **VADER lexicon** is downloaded (if not already present).
2. **Training data** (`data/training.csv`) is read and both classifiers are applied:

   * Prints accuracy on the training data

3. **Test data** (`data/test.csv`) is read and both classifiers predict labels.

4. **Submission files** are created in your working directory:

   * `submission_textblob.csv`
   * `submission_vader.csv`
---
