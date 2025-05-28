# Human Baseline Annotation Script

This script helps you establish a human‐annotation baseline on a sentiment‐analysis dataset by:

1. Sampling 100 sentences from your training data and splitting them into 4 files (25 each) for manual labeling.  
2. Reading back those labeled files and assembling them into one predictions file.  
3. Computing human‐annotation accuracy against the gold labels.  
4. Computing the competition scoring function \(L\).


