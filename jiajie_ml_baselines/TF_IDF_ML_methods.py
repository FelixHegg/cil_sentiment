import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Model Selection & Evaluation
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, make_scorer

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier

import os
import time
import warnings

warnings.filterwarnings("ignore")

nltk_resources = ['punkt', 'wordnet', 'stopwords', 'punkt_tab']
print("Checking/downloading NLTK resources...")
for resource in nltk_resources:
    try:
        _ = nltk.data.find(f'corpora/{resource}' if resource in ['wordnet', 'stopwords'] else f'tokenizers/{resource}')
        # print(f"Resource '{resource}' already downloaded.")
    except LookupError:
        print(f"Downloading NLTK resource: {resource}...")
        try:
            nltk.download(resource, quiet=False)
            print(f"'{resource}' downloaded successfully.")
        except Exception as e:
            print(f"Error downloading '{resource}': {e}")
            print("Please try downloading manually: nltk.download('{resource}')")
            exit()

# Define Data Directory
DATA_DIR = '../data'

# Load Training Data
print("Loading training data...")
train_data_path = os.path.join(DATA_DIR, 'training.csv')
try:
    training_data = pd.read_csv(train_data_path, index_col=0)
    print("Training data loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{train_data_path}' not found. Please ensure the file exists.")
    exit()

# Encode Labels
label_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
training_data['label_encoded'] = training_data['label'].map(label_mapping)

sentences = training_data['sentence']
labels = training_data['label_encoded']
print(f"Loaded {len(sentences)} training examples.")

# Define Custom Tokenizer with Lemmatization
print("Setting up custom tokenizer with lemmatization...")
lemmatizer = WordNetLemmatizer()
stop_words_list = set(stopwords.words('english'))

def lemmatizing_tokenizer(text):
    text = re.sub(r'[\d\W]+', ' ', text.lower()).strip()
    tokens = word_tokenize(text)
    lemmatized_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in stop_words_list and len(word) > 1
    ]
    return lemmatized_tokens

# Define Evaluation Metric (L-Score based on MAE)
def l_score_func(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return 0.5 * (2 - mae)

custom_scorer = make_scorer(l_score_func, greater_is_better=True)
print("Custom evaluation scorer defined (L-Score).")

# Define Models and Parameter Grids to Test
print("Defining models and hyperparameter grids...")

models_to_test = [
    {
        'name': 'Logistic Regression',
        'estimator': LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        'param_grid': {
            'classifier__C': [0.1, 1.0, 10.0]
        }
    },
    {
        'name': 'Multinomial Naive Bayes',
        'estimator': MultinomialNB(),
        'param_grid': {
            'classifier__alpha': [0.1, 0.5, 1.0]
        }
    },
    {
        'name': 'Linear SVM',
        'estimator': LinearSVC(random_state=42, max_iter=2000, dual=True),
        'param_grid': {
            'classifier__C': [0.1, 1.0, 10.0]
        }
    },
    {
        'name': 'Gradient Boosting',
        'estimator': GradientBoostingClassifier(random_state=42),
        'param_grid': {
            'classifier__n_estimators': [100, 200], # Number of boosting stages
            'classifier__learning_rate': [0.1, 0.05], # Step size shrinkage
            'classifier__max_depth': [3, 5] # Max depth of individual trees
        }
    },
]

# Setup K-Fold Cross-Validation (used for all models)
print("Setting up 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Loop Through Models, Run GridSearchCV, and Store Results
results = []
best_overall_score = -1
best_overall_estimator = None
best_overall_model_name = ""

print("\n--- Starting Model Evaluation (Using TF-IDF) ---")

for model_config in models_to_test:
    model_name = model_config['name']
    estimator = model_config['estimator']
    param_grid = model_config['param_grid']
    print(f"\nEvaluating Model: {model_name}")

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            tokenizer=lemmatizing_tokenizer,
            ngram_range=(1, 2),
            max_features=10000,
            lowercase=False
        )),
        ('classifier', estimator)
    ])

    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=custom_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    try:
        grid_search.fit(sentences, labels)
        end_time = time.time()
        fit_time = end_time - start_time

        print(f"Finished Grid Search for {model_name} in {fit_time:.2f} seconds.")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation L-Score: {grid_search.best_score_:.5f}")

        results.append({
            'model_name': model_name,
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'best_estimator': grid_search.best_estimator_,
            'fit_time_seconds': fit_time
        })

        if grid_search.best_score_ > best_overall_score:
            best_overall_score = grid_search.best_score_
            best_overall_estimator = grid_search.best_estimator_
            best_overall_model_name = model_name

    except Exception as e:
        print(f"!!! Error during GridSearchCV fit for {model_name} !!!")
        print(e)
        results.append({
            'model_name': model_name,
            'best_score': -1,
            'best_params': {},
            'best_estimator': None,
            'fit_time_seconds': time.time() - start_time
        })

print("\n--- Overall Best Model (Using TF-IDF) ---")
if best_overall_estimator is not None:
    print(f"Best performing model type: {best_overall_model_name}")
    print(f"Best Cross-Validation L-Score: {best_overall_score:.5f}")
    best_result = next((r for r in results if r['model_name'] == best_overall_model_name), None)
    if best_result:
         print(f"Best Parameters for {best_overall_model_name}: {best_result['best_params']}")
else:
    print("No model completed successfully.")
    exit()

# Load Test Data
print("\nLoading test data...")
test_data_path = os.path.join(DATA_DIR, 'test.csv')
try:
    test_data = pd.read_csv(test_data_path, index_col=0)
    print("Test data loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{test_data_path}' not found. Please ensure the file exists.")
    exit()

test_sentences = test_data['sentence']
print(f"Loaded {len(test_sentences)} test examples.")

# Make Predictions on Test Data using the Overall Best Model
print(f"\nMaking predictions on the test set using the best model ({best_overall_model_name} with TF-IDF)...")
test_predictions_numerical = best_overall_estimator.predict(test_sentences)
print("Predictions generated.")

# Format Predictions for Submission
print("Formatting predictions for submission...")
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
test_predictions_labels = pd.Series(test_predictions_numerical).map(inverse_label_mapping)

submission_df = pd.DataFrame({'id': test_data.index, 'label': test_predictions_labels})
print("Submission DataFrame created:")
print(submission_df.head())

# Save Submission File
submission_filename = f'test_predictions_best_TFIDF_{best_overall_model_name.replace(" ", "_")}.csv'
try:
    submission_df.to_csv(submission_filename, index=False)
    print(f"\nTest predictions saved successfully to '{submission_filename}'")
except Exception as e:
    print(f"\nError saving submission file: {e}")

print("\nScript finished.")