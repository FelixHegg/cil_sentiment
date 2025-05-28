import pandas as pd
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

nltk.download('vader_lexicon')

# Define the TextBlob sentiment function
def textblob_sentiment(text, pos_threshold=0.1, neg_threshold=-0.1):
    """
    Analyze text sentiment using TextBlob.
    Returns "positive", "negative", or "neutral" based on polarity.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > pos_threshold:
        return "positive"
    elif polarity < neg_threshold:
        return "negative"
    else:
        return "neutral"

# Initialize VADER's SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Define the VADER sentiment function
def vader_sentiment(text):
    """
    Analyze text sentiment using VADER.
    Returns "positive", "negative", or "neutral" using the compound score.
    """
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

# --------------------------------------------------------------
# 1. Load Training and Test Data
# --------------------------------------------------------------
training_data = pd.read_csv('../data/training.csv')
test_data = pd.read_csv('../data/test.csv')

# --------------------------------------------------------------
# 2. Apply Sentiment Analysis on the Training Data
# --------------------------------------------------------------
training_data['textblob_sentiment'] = training_data['sentence'].apply(textblob_sentiment)
training_data['vader_sentiment'] = training_data['sentence'].apply(vader_sentiment)

print("\nTraining Data with Sentiment Analysis:")
print(training_data[['sentence', 'textblob_sentiment', 'vader_sentiment']].head())

# ---------------------------------------------------------------------
# 3. Calculate the Error (Accuracy) on the Training Data
# ---------------------------------------------------------------------
textblob_correct = (training_data['textblob_sentiment'] == training_data['label']).sum()
vader_correct = (training_data['vader_sentiment'] == training_data['label']).sum()
total_training = len(training_data)

textblob_accuracy = textblob_correct / total_training
vader_accuracy = vader_correct / total_training

print(f"\nTextBlob: {textblob_correct} out of {total_training} predictions are correct, accuracy: {textblob_accuracy:.2f}")
print(f"VADER: {vader_correct} out of {total_training} predictions are correct, accuracy: {vader_accuracy:.2f}")

# ------------------------------------------------------------------------------
# 4. Apply Sentiment Analysis on the Test Data and Create Submissions
# ------------------------------------------------------------------------------
test_data['textblob_pred'] = test_data['sentence'].apply(textblob_sentiment)
test_data['vader_pred'] = test_data['sentence'].apply(vader_sentiment)

print("\nTest Data with Both Predicted Labels:")
print(test_data[['id', 'textblob_pred', 'vader_pred']].head())

# ------------------------------------------------------------------------------
# 5. Create and Save the Submission Files
# ------------------------------------------------------------------------------
submission_textblob = test_data[['id']].copy()
submission_textblob['label'] = test_data['textblob_pred']
submission_textblob.to_csv("submission_textblob.csv", index=False)
print("\nSubmission file 'submission_textblob.csv' has been created:")
print(submission_textblob.head())


submission_vader = test_data[['id']].copy()
submission_vader['label'] = test_data['vader_pred']
submission_vader.to_csv("submission_vader.csv", index=False)
print("\nSubmission file 'submission_vader.csv' has been created:")
print(submission_vader.head())