import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
import gensim # For Word2Vec
import gensim.downloader as api

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight

import os
import time
import warnings
import wandb

warnings.filterwarnings("ignore")

nltk_resources = ['punkt', 'punkt_tab']
print("Checking/downloading NLTK resources...")
for resource in nltk_resources:
    try:
        _ = nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        print(f"Downloading NLTK resource: {resource}...")
        try:
            nltk.download(resource, quiet=False)
            print(f"'{resource}' downloaded successfully.")
        except Exception as e:
            print(f"Error downloading '{resource}': {e}")
            exit()

# Define Data Directory, Device, and Model Save Path
DATA_DIR = '../data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = 'best_mlp_pretrained_w2v_simple_prep_weighted_loss_model.pth'
WANDB_RUN_NAME = "MLP_PretrainedW2V_WeightedLoss_Run1" # <-- SET YOUR RUN NAME HERE
print(f"Using device: {device}")
print(f"Best model will be saved to: {BEST_MODEL_PATH}")
print(f"Wandb run name: {WANDB_RUN_NAME}")


# Load Training Data
print("Loading training data...")
train_data_path = os.path.join(DATA_DIR, 'training.csv')
try:
    training_data = pd.read_csv(train_data_path, index_col=0)
    print("Training data loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{train_data_path}' not found. Please ensure the file exists.")
    exit()
sentences = training_data['sentence'].astype(str)
labels = training_data['label']
print(f"Loaded {len(sentences)} training examples.")
# --- Display Class Distribution ---
print("Original Label Distribution:")
print(labels.value_counts(normalize=True))

# Define Simplified Tokenizer
print("Setting up simplified tokenizer...")
def simple_tokenizer(text):
    text = re.sub(r'[\d\W]+', ' ', text.lower()).strip()
    tokens = word_tokenize(text)
    return tokens

print("Tokenizing training sentences with simplified tokenizer...")
tokenized_sentences = [simple_tokenizer(sent) for sent in sentences]

# Load Pre-trained Word2Vec Model
print("Loading pre-trained Word2Vec model (word2vec-google-news-300)...")
try:
    w2v_model = api.load('word2vec-google-news-300')
    W2V_DIM = w2v_model.vector_size
    print(f"Pre-trained Word2Vec model loaded. Vector size: {W2V_DIM}")
    if not hasattr(w2v_model, 'wv'): w2v_model.wv = w2v_model
except Exception as e:
    print(f"Error loading pre-trained Word2Vec model: {e}")
    exit()

# Vectorize Sentences by Averaging Word Vectors
print("Vectorizing sentences using pre-trained Word2Vec model...")
def vectorize_sentence(tokens, model):
    vectors = []
    for word in tokens:
        try: vectors.append(model.wv[word])
        except KeyError: continue
    if not vectors: return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)

X_vectors = np.array([vectorize_sentence(tokens, w2v_model) for tokens in tokenized_sentences])
print(f"Sentence vectors created with shape: {X_vectors.shape}")

# Prepare Labels
print("Preparing labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels) # 0, 1, 2
num_classes = len(label_encoder.classes_)
print(f"Label mapping: {list(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
index_to_mae_label_map = {
    label_encoder.transform(['negative'])[0]: -1,
    label_encoder.transform(['neutral'])[0]: 0,
    label_encoder.transform(['positive'])[0]: 1
}
print(f"Index to MAE Label Map: {index_to_mae_label_map}")

# Create Train/Validation Split
print("Creating train/validation split...")
X_train, X_val, y_train, y_val = train_test_split(
    X_vectors, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42
)
print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

print("Calculating class weights for loss function...")
# Calculate weights based on the training set distribution
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train), # Class labels (0, 1, 2)
    y=y_train                 # Training labels used to compute frequencies
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Calculated class weights: {class_weights_tensor}")

# Create Datasets and DataLoaders
print("Creating PyTorch Datasets and DataLoaders...")
class SentimentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

BATCH_SIZE = 64
train_dataset = SentimentDataset(X_train, y_train)
val_dataset = SentimentDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("Defining PyTorch MLP model architecture...")
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 128)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.5)
        self.layer_2 = nn.Linear(128, 256)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.3)
        self.layer_3 = nn.Linear(256, 128)
        self.relu_3 = nn.ReLU()
        self.dropout_3 = nn.Dropout(0.2)
        self.layer_4 = nn.Linear(128, 64)
        self.relu_4 = nn.ReLU()
        self.dropout_4 = nn.Dropout(0.1)
        self.output_layer = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.relu_1(self.layer_1(x))
        x = self.dropout_1(x)
        x = self.relu_2(self.layer_2(x))
        x = self.dropout_2(x)
        x = self.relu_3(self.layer_3(x))
        x = self.dropout_3(x)
        x = self.relu_4(self.layer_4(x))
        x = self.dropout_4(x)
        x = self.output_layer(x)
        return x

mlp_model = MLP(input_dim=W2V_DIM, num_classes=num_classes).to(device)
print(mlp_model)

# Define Loss, Optimizer, and LR Scheduler
print("Defining loss function, optimizer, and LR scheduler...")
LEARNING_RATE = 1e-3
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True) # Adjusted patience back

print("Initializing Weights & Biases...")
wandb.init(
    project="cil-sentiment-analysis", # CHANGE YOUR PROJECT NAME IF NEEDED
    name=WANDB_RUN_NAME,
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": 150, # Reset epochs slightly
        "batch_size": BATCH_SIZE,
        "architecture": "MLP",
        "embedding": "Pretrained Word2Vec (Google News 300d)",
        "vector_strategy": "Average",
        "tokenizer": "Simple (lower, no-punct/num, word_tokenize)",
        "optimizer": "Adam",
        "w2v_dim": W2V_DIM,
        "early_stopping_patience": 10,
        "lr_scheduler_patience": 3,
        "validation_split": 0.1,
        "loss_function": "Weighted CrossEntropyLoss", # Log loss type
        "class_weights": class_weights_tensor.cpu().tolist() # Log weights used
    }
)

# Training Loop
print("\n--- Starting PyTorch MLP Model Training (with Weighted Loss) ---")
EPOCHS = wandb.config.epochs
EARLY_STOPPING_PATIENCE = wandb.config.early_stopping_patience

best_val_loss = float('inf')
epochs_no_improve = 0

start_time = time.time()

for epoch in range(EPOCHS):
    # Training Phase
    mlp_model.train()
    running_loss = 0.0
    all_train_preds_indices = []
    all_train_labels_indices = []
    for i, data in enumerate(train_loader):
        inputs, labels_batch = data; inputs, labels_batch = inputs.to(device), labels_batch.to(device)
        optimizer.zero_grad(); outputs = mlp_model(inputs)
        loss = criterion(outputs, labels_batch) # Loss calculation now uses weights implicitly
        loss.backward(); optimizer.step()
        running_loss += loss.item(); _, predicted_train = torch.max(outputs.data, 1)
        all_train_preds_indices.append(predicted_train.cpu().numpy())
        all_train_labels_indices.append(labels_batch.cpu().numpy())
    epoch_train_loss = running_loss / len(train_loader)
    all_train_preds_indices = np.concatenate(all_train_preds_indices)
    all_train_labels_indices = np.concatenate(all_train_labels_indices)
    epoch_train_acc = 100 * (all_train_preds_indices == all_train_labels_indices).sum() / len(all_train_labels_indices)
    mapped_train_preds = np.vectorize(index_to_mae_label_map.get)(all_train_preds_indices)
    mapped_train_labels = np.vectorize(index_to_mae_label_map.get)(all_train_labels_indices)
    train_mae = mean_absolute_error(mapped_train_labels, mapped_train_preds)
    epoch_train_l_score = 0.5 * (2 - train_mae)

    # Validation Phase
    mlp_model.eval()
    val_loss = 0.0
    all_val_preds_indices = []
    all_val_labels_indices = []
    with torch.no_grad():
        for data in val_loader:
            inputs, labels_batch = data; inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = mlp_model(inputs)
            loss = criterion(outputs, labels_batch) # Loss calculation now uses weights implicitly
            val_loss += loss.item(); _, predicted_val = torch.max(outputs.data, 1)
            all_val_preds_indices.append(predicted_val.cpu().numpy())
            all_val_labels_indices.append(labels_batch.cpu().numpy())
    epoch_val_loss = val_loss / len(val_loader)
    all_val_preds_indices = np.concatenate(all_val_preds_indices)
    all_val_labels_indices = np.concatenate(all_val_labels_indices)
    epoch_val_acc = 100 * (all_val_preds_indices == all_val_labels_indices).sum() / len(all_val_labels_indices)
    mapped_val_preds = np.vectorize(index_to_mae_label_map.get)(all_val_preds_indices)
    mapped_val_labels = np.vectorize(index_to_mae_label_map.get)(all_val_labels_indices)
    val_mae = mean_absolute_error(mapped_val_labels, mapped_val_preds)
    epoch_val_l_score = 0.5 * (2 - val_mae)

    print(f'Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, Train L-Score: {epoch_train_l_score:.4f} - Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%, Val L-Score: {epoch_val_l_score:.4f}')

    # Log Metrics to Wandb
    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"epoch": epoch + 1, "train_loss": epoch_train_loss, "train_accuracy": epoch_train_acc, "train_l_score": epoch_train_l_score, "val_loss": epoch_val_loss, "val_accuracy": epoch_val_acc, "val_l_score": epoch_val_l_score, "learning_rate": current_lr})

    # LR Scheduling & Early Stopping
    scheduler.step(epoch_val_loss)
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        torch.save(mlp_model.state_dict(), BEST_MODEL_PATH)
        print(f'Validation loss decreased ({best_val_loss:.4f}). Saving model to {BEST_MODEL_PATH}')
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_val_l_score"] = epoch_val_l_score
        wandb.run.summary["best_epoch"] = epoch + 1
    else:
        epochs_no_improve += 1
        print(f'Validation loss did not improve. Counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}')
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f'Early stopping triggered after {epoch+1} epochs.')
        break

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")
wandb.run.summary["training_time_seconds"] = end_time - start_time

# Load Best Model and Evaluate
print(f"\nLoading best model weights from {BEST_MODEL_PATH}...")
best_model = MLP(input_dim=W2V_DIM, num_classes=num_classes).to(device)
try:
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print("Best model loaded successfully.")
    best_model.eval()
    final_val_loss = 0.0; all_final_val_preds_indices = []; all_final_val_labels_indices = []
    with torch.no_grad():
        for data in val_loader:
            inputs, labels_batch = data; inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = best_model(inputs); loss = criterion(outputs, labels_batch) # Weighted loss applied here too
            final_val_loss += loss.item(); _, predicted = torch.max(outputs.data, 1)
            all_final_val_preds_indices.append(predicted.cpu().numpy())
            all_final_val_labels_indices.append(labels_batch.cpu().numpy())
    final_val_loss /= len(val_loader)
    all_final_val_preds_indices = np.concatenate(all_final_val_preds_indices)
    all_final_val_labels_indices = np.concatenate(all_final_val_labels_indices)
    final_val_accuracy = 100 * (all_final_val_preds_indices == all_final_val_labels_indices).sum() / len(all_final_val_labels_indices)
    mapped_final_val_preds = np.vectorize(index_to_mae_label_map.get)(all_final_val_preds_indices)
    mapped_final_val_labels = np.vectorize(index_to_mae_label_map.get)(all_final_val_labels_indices)
    final_val_mae = mean_absolute_error(mapped_final_val_labels, mapped_final_val_preds)
    final_val_l_score = 0.5 * (2 - final_val_mae)
    print(f'\nFinal Validation Loss (best model): {final_val_loss:.4f}')
    print(f'Final Validation Accuracy (best model): {final_val_accuracy:.2f}%')
    print(f'Final Validation L-Score (best model): {final_val_l_score:.4f}')
    wandb.run.summary["final_val_loss"] = final_val_loss
    wandb.run.summary["final_val_accuracy"] = final_val_accuracy
    wandb.run.summary["final_val_l_score"] = final_val_l_score
except FileNotFoundError: print(f"Error: Model file not found at {BEST_MODEL_PATH}."); exit()
except Exception as e: print(f"Error loading model state: {e}."); exit()

# Load Test Data
print("\nLoading test data...")
test_data_path = os.path.join(DATA_DIR, 'test.csv')
try: test_data = pd.read_csv(test_data_path, index_col=0); print("Test data loaded successfully.")
except FileNotFoundError: print(f"Error: '{test_data_path}' not found."); exit()
test_sentences = test_data['sentence'].astype(str)
print(f"Loaded {len(test_sentences)} test examples.")

# Preprocess and Vectorize Test Data
print("Preprocessing and vectorizing test data using simple tokenizer...")
tokenized_test_sentences = [simple_tokenizer(sent) for sent in test_sentences]
X_test_vectors = np.array([vectorize_sentence(tokens, w2v_model) for tokens in tokenized_test_sentences], dtype=np.float32)
X_test_tensor = torch.tensor(X_test_vectors, dtype=torch.float32).to(device)
print(f"Test vectors created with shape: {X_test_tensor.shape}")

# Make Predictions on Test Data using the Best Model
print(f"\nMaking predictions on the test set using the best model...")
test_predictions_indices = []
best_model.eval()
test_predictions_list = []
test_batch_size = 256
with torch.no_grad():
    for i in range(0, len(X_test_tensor), test_batch_size):
        batch = X_test_tensor[i:i+test_batch_size]
        outputs = best_model(batch)
        _, predicted = torch.max(outputs.data, 1)
        test_predictions_list.append(predicted.cpu().numpy())
test_predictions_indices = np.concatenate(test_predictions_list)
print("Predictions generated.")

# Format Predictions for Submission
print("Formatting predictions for submission...")
test_predictions_labels = label_encoder.inverse_transform(test_predictions_indices)
submission_df = pd.DataFrame({'id': test_data.index, 'label': test_predictions_labels})
print("Submission DataFrame created:")
print(submission_df.head())

# Save Submission File
submission_filename = f'test_predictions_{WANDB_RUN_NAME}.csv' # Use wandb run name
try:
    submission_df.to_csv(submission_filename, index=False)
    print(f"\nTest predictions saved successfully to '{submission_filename}'")
except Exception as e:
    print(f"\nError saving submission file: {e}")

# Finish Wandb Run
print("\nFinishing Wandb run...")
wandb.finish()

print("\nScript finished.")