import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import warnings
import math

warnings.filterwarnings("ignore")

# Define Constants and Device
DATA_DIR = '../data'
# --- Define Hugging Face Model Name ---
HF_MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Using Hugging Face model: {HF_MODEL_NAME}")

# Load Test Data
print("Loading test data...")
test_data_path = os.path.join(DATA_DIR, 'test.csv')
try:
    test_data = pd.read_csv(test_data_path, index_col=0)
    print("Test data loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{test_data_path}' not found. Please ensure the file exists.")
    exit()

# Ensure sentences are strings
test_sentences = test_data['sentence'].astype(str).tolist() # Get sentences as a list
print(f"Loaded {len(test_sentences)} test examples.")

# Load Pre-trained Tokenizer and Model from Hugging Face
print(f"Loading tokenizer and model for '{HF_MODEL_NAME}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    model.to(device) # Move model to the GPU or CPU
    model.eval() # Set model to evaluation mode
    print("Tokenizer and model loaded successfully.")
except Exception as e:
    print(f"Error loading model/tokenizer from Hugging Face: {e}")
    print("Please ensure you have an internet connection and the 'transformers' library is installed.")
    exit()

# Define Label Mapping
# Mapping from the 5 classes predicted by the HF model
hf_sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
# Mapping from the 5 HF classes to the 3 project classes
five_to_three_map = {
    "Very Negative": "negative",
    "Negative": "negative",
    "Neutral": "neutral",
    "Positive": "positive",
    "Very Positive": "positive"
}
print("Label mappings defined.")

# Prediction Function
def predict_sentiment_batch(texts, model, tokenizer, batch_size=32):
    """ Predicts sentiment for a list of texts in batches """
    predictions_3_class = []
    num_batches = math.ceil(len(texts) / batch_size)
    print(f"Starting prediction for {len(texts)} texts in {num_batches} batches...")

    with torch.no_grad(): # Disable gradient calculations for inference
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512 # Use max length suitable for the model
            ).to(device) # Move batch inputs to the same device as the model

            outputs = model(**inputs)
            predicted_indices = torch.argmax(outputs.logits, dim=-1).tolist()

            # Map indices to 5-class labels, then to 3-class labels
            for idx in predicted_indices:
                hf_label = hf_sentiment_map.get(idx, "Neutral") # Default to Neutral if index unexpected
                project_label = five_to_three_map.get(hf_label, "neutral") # Default to neutral
                predictions_3_class.append(project_label)

            if (i // batch_size + 1) % 10 == 0: # Print progress every 10 batches
                 print(f"  Processed batch {i // batch_size + 1}/{num_batches}")

    print("Prediction finished.")
    return predictions_3_class

# Generate Predictions for Test Set
test_predictions_labels = predict_sentiment_batch(test_sentences, model, tokenizer)

# Format Predictions for Submission
print("Formatting predictions for submission...")
submission_df = pd.DataFrame({'id': test_data.index, 'label': test_predictions_labels})
print("Submission DataFrame created:")
print(submission_df.head())

# Save Submission File
submission_filename = f'test_predictions_HF_{HF_MODEL_NAME.split("/")[-1]}.csv'
try:
    submission_df.to_csv(submission_filename, index=False)
    print(f"\nTest predictions saved successfully to '{submission_filename}'")
except Exception as e:
    print(f"\nError saving submission file: {e}")

print("\nScript finished.")