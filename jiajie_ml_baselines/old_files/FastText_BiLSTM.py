import pandas as pd
import numpy as np
import re
import nltk

from nltk.tokenize import word_tokenize
import gensim
import gensim.downloader as api

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    try: _ = nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        print(f"Downloading NLTK resource: {resource}...")
        try: nltk.download(resource, quiet=False); print(f"'{resource}' downloaded successfully.")
        except Exception as e: print(f"Error downloading '{resource}': {e}"); exit()

# Define Data Directory, Device, Model Save Path, and Sequence Length
DATA_DIR = '../data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = 'best_bilstm_fasttext_simple_prep_weighted_loss_model.pth'
WANDB_RUN_NAME = "BiLSTM_FastText_WeightedLoss_Run1" # Use BiLSTM in name
print(f"Using device: {device}")
print(f"Best model will be saved to: {BEST_MODEL_PATH}")
print(f"Wandb run name: {WANDB_RUN_NAME}")

# Load Training Data
print("Loading training data...")
train_data_path = os.path.join(DATA_DIR, 'training.csv')
try: training_data = pd.read_csv(train_data_path, index_col=0); print("Training data loaded successfully.")
except FileNotFoundError: print(f"Error: '{train_data_path}' not found."); exit()
sentences = training_data['sentence'].astype(str)
labels = training_data['label']
print(f"Loaded {len(sentences)} training examples.")
print("Original Label Distribution:"); print(labels.value_counts(normalize=True))

# Define Simplified Tokenizer
print("Setting up simplified tokenizer...")
def simple_tokenizer(text):
    text = re.sub(r'[\d\W]+', ' ', text.lower()).strip()
    tokens = word_tokenize(text)
    return tokens
print("Tokenizing training sentences with simplified tokenizer...")
tokenized_sentences = [simple_tokenizer(sent) for sent in sentences]

# Calculate MAX_SEQ_LEN based on 95th percentile
print("Calculating sequence length based on 95th percentile...")
sentence_lengths = [len(tokens) for tokens in tokenized_sentences]
MAX_SEQ_LEN = int(np.percentile(sentence_lengths, 95))
if MAX_SEQ_LEN == 0: MAX_SEQ_LEN = 10 # Add a minimum length
print(f"Using MAX_SEQ_LEN based on 95th percentile: {MAX_SEQ_LEN}")

# Load Pre-trained FastText Model
print("Loading pre-trained FastText model (fasttext-wiki-news-subwords-300)...")
try:
    ft_model = api.load('fasttext-wiki-news-subwords-300')
    FT_DIM = ft_model.vector_size
    print(f"Pre-trained FastText model loaded. Vector size: {FT_DIM}")
except Exception as e: print(f"Error loading pre-trained FastText model: {e}"); exit()

# Vectorize Sentences as Sequences with Padding (Check for word existence)
print(f"Vectorizing sentences as sequences using FastText (padding/truncating to {MAX_SEQ_LEN})...")
def vectorize_sentence_sequence_ft(tokens, model, max_len):
    vectors = []
    for word in tokens:
        if word in model: vectors.append(model[word]) # Direct access, skip OOV that model itself can't handle
    vectors = vectors[:max_len]
    num_vectors = len(vectors)
    if num_vectors < max_len:
        padding = np.zeros((max_len - num_vectors, model.vector_size), dtype=np.float32)
        vectors_np = np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, model.vector_size), dtype=np.float32)
        vectors = np.concatenate((vectors_np, padding), axis=0)
    elif num_vectors == 0: return np.zeros((max_len, model.vector_size), dtype=np.float32)
    else: vectors = np.array(vectors, dtype=np.float32)
    if vectors.shape[0] != max_len or vectors.shape[1] != model.vector_size: return np.zeros((max_len, model.vector_size), dtype=np.float32) # Safeguard
    return vectors

X_sequences = np.array([vectorize_sentence_sequence_ft(tokens, ft_model, MAX_SEQ_LEN) for tokens in tokenized_sentences])
print(f"Sentence sequences created with shape: {X_sequences.shape}") # Should be (num_sentences, MAX_SEQ_LEN, FT_DIM)

# Prepare Labels
print("Preparing labels...")
label_encoder = LabelEncoder(); y_encoded = label_encoder.fit_transform(labels); num_classes = len(label_encoder.classes_)
print(f"Label mapping: {list(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
index_to_mae_label_map = { label_encoder.transform(['negative'])[0]: -1, label_encoder.transform(['neutral'])[0]: 0, label_encoder.transform(['positive'])[0]: 1}
print(f"Index to MAE Label Map: {index_to_mae_label_map}")

# Create Train/Validation Split
print("Creating train/validation split...")
X_train, X_val, y_train, y_val = train_test_split(
    X_sequences, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42
)
print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

# Calculate Class Weights
print("Calculating class weights for loss function...")
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Calculated class weights: {class_weights_tensor}")

# Create PyTorch Datasets and DataLoaders for Sequences
print("Creating PyTorch Datasets and DataLoaders for sequences...")
class SentimentSequenceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

BATCH_SIZE = 64
train_dataset = SentimentSequenceDataset(X_train, y_train)
val_dataset = SentimentSequenceDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# --- Define BiLSTM Model Architecture ---
print("Defining PyTorch BiLSTM model architecture...")
class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # Input tensors will have shape (batch_size, seq_len, embedding_dim)
            bidirectional=True, # Make it bidirectional
            dropout=dropout if num_layers > 1 else 0 # Add dropout between LSTM layers if stacked
        )
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Define the output layer
        # Input features = hidden_dim * 2 because it's bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        # Initialize hidden and cell states (optional, defaults to zeros)
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device) # *2 for bidirectional
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)

        # Pass sequence through LSTM
        # lstm_out, (hidden, cell) = self.lstm(x, (h0, c0))
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)

        # We'll use the final hidden state of the last layer
        # Concatenate the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # hidden_cat shape: (batch_size, hidden_dim * 2)

        # Apply dropout and pass through the final fully connected layer
        out = self.dropout(hidden_cat)
        out = self.fc(out)
        # out shape: (batch_size, num_classes)
        return out

# Hyperparameters for BiLSTM
LSTM_HIDDEN_DIM = 128 # Size of LSTM hidden state
LSTM_NUM_LAYERS = 2   # Number of LSTM layers (stacked)
LSTM_DROPOUT = 0.5

# Instantiate the BiLSTM model
lstm_model = BiLSTMClassifier(
    embedding_dim=FT_DIM,
    hidden_dim=LSTM_HIDDEN_DIM,
    num_classes=num_classes,
    num_layers=LSTM_NUM_LAYERS,
    dropout=LSTM_DROPOUT
).to(device)

print(lstm_model)
print(f"Model Parameters: {sum(p.numel() for p in lstm_model.parameters() if p.requires_grad):,}")

# Define Loss, Optimizer, and LR Scheduler
print("Defining loss function, optimizer (with weight decay), and LR scheduler...")
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
# --- Pass lstm_model parameters to optimizer ---
optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

# Initialize Weights & Biases
print("Initializing Weights & Biases...")
wandb.init(
    project="cil-sentiment-analysis",
    name=WANDB_RUN_NAME,
    config={ # --- CHANGE: Updated config for BiLSTM ---
        "learning_rate": LEARNING_RATE, "epochs": 50, "batch_size": BATCH_SIZE,
        "architecture": "BiLSTM",
        "max_seq_len_strategy": "95th Percentile", "max_seq_len": MAX_SEQ_LEN,
        "embedding": "Pretrained FastText (wiki-news-subwords-300)",
        "vector_strategy": "Sequence",
        "tokenizer": "Simple (lower, no-punct/num, word_tokenize)",
        "optimizer": "Adam", "weight_decay": WEIGHT_DECAY, "w2v_dim": FT_DIM,
        "early_stopping_patience": 5, "lr_scheduler_patience": 3,
        "validation_split": 0.15, "loss_function": "Weighted CrossEntropyLoss",
        "class_weights": class_weights_tensor.cpu().tolist(),
        "lstm_hidden_dim": LSTM_HIDDEN_DIM,
        "lstm_num_layers": LSTM_NUM_LAYERS,
        "lstm_dropout": LSTM_DROPOUT
    }
)


# Training Loop
print("\n--- Starting PyTorch BiLSTM Model Training ---") # Updated print
EPOCHS = wandb.config.epochs
EARLY_STOPPING_PATIENCE = wandb.config.early_stopping_patience

best_val_loss = float('inf'); epochs_no_improve = 0; start_time = time.time()

for epoch in range(EPOCHS):
    # --- Training Phase ---
    lstm_model.train() # Use lstm_model
    running_loss = 0.0; all_train_preds_indices = []; all_train_labels_indices = []
    for i, data in enumerate(train_loader):
        inputs, labels_batch = data; inputs, labels_batch = inputs.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        outputs = lstm_model(inputs) # Use lstm_model
        loss = criterion(outputs, labels_batch); loss.backward(); optimizer.step()
        running_loss += loss.item(); _, predicted_train = torch.max(outputs.data, 1)
        all_train_preds_indices.append(predicted_train.cpu().numpy()); all_train_labels_indices.append(labels_batch.cpu().numpy())
    
    epoch_train_loss = running_loss / len(train_loader); all_train_preds_indices = np.concatenate(all_train_preds_indices); all_train_labels_indices = np.concatenate(all_train_labels_indices)
    epoch_train_acc = 100*(all_train_preds_indices == all_train_labels_indices).sum()/len(all_train_labels_indices)
    mapped_train_preds = np.vectorize(index_to_mae_label_map.get)(all_train_preds_indices); mapped_train_labels = np.vectorize(index_to_mae_label_map.get)(all_train_labels_indices)
    train_mae = mean_absolute_error(mapped_train_labels, mapped_train_preds); epoch_train_l_score = 0.5 * (2 - train_mae)

    # --- Validation Phase ---
    lstm_model.eval() 
    val_loss = 0.0; all_val_preds_indices = []; all_val_labels_indices = []
    with torch.no_grad():
        for data in val_loader:
            inputs, labels_batch = data; inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = lstm_model(inputs) 
            loss = criterion(outputs, labels_batch); val_loss += loss.item()
            _, predicted_val = torch.max(outputs.data, 1); all_val_preds_indices.append(predicted_val.cpu().numpy()); all_val_labels_indices.append(labels_batch.cpu().numpy())
   
    epoch_val_loss = val_loss / len(val_loader); all_val_preds_indices = np.concatenate(all_val_preds_indices); all_val_labels_indices = np.concatenate(all_val_labels_indices)
    epoch_val_acc = 100*(all_val_preds_indices == all_val_labels_indices).sum()/len(all_val_labels_indices)
    mapped_val_preds = np.vectorize(index_to_mae_label_map.get)(all_val_preds_indices); mapped_val_labels = np.vectorize(index_to_mae_label_map.get)(all_val_labels_indices)
    val_mae = mean_absolute_error(mapped_val_labels, mapped_val_preds); epoch_val_l_score = 0.5 * (2 - val_mae)

    print(f'Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}%, L: {epoch_train_l_score:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%, L: {epoch_val_l_score:.4f}')
    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"epoch": epoch + 1, "train_loss": epoch_train_loss, "train_accuracy": epoch_train_acc, "train_l_score": epoch_train_l_score, "val_loss": epoch_val_loss, "val_accuracy": epoch_val_acc, "val_l_score": epoch_val_l_score, "learning_rate": current_lr})

    # --- LR Scheduling & Early Stopping ---
    scheduler.step(epoch_val_loss)
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss; epochs_no_improve = 0
        torch.save(lstm_model.state_dict(), BEST_MODEL_PATH) 
        print(f'Val loss decreased ({best_val_loss:.4f}). Saving model to {BEST_MODEL_PATH}')
        wandb.run.summary["best_val_loss"] = best_val_loss; wandb.run.summary["best_val_l_score"] = epoch_val_l_score; wandb.run.summary["best_epoch"] = epoch + 1
    else:
        epochs_no_improve += 1; print(f'Val loss did not improve. Counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}')
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE: print(f'Early stopping triggered after {epoch+1} epochs.'); break

end_time = time.time(); print(f"\nTraining finished in {end_time - start_time:.2f} seconds."); wandb.run.summary["training_time_seconds"] = end_time - start_time

# Load Best Model and Evaluate
print(f"\nLoading best model weights from {BEST_MODEL_PATH}...")
# --- Load BiLSTMClassifier model ---
best_model = BiLSTMClassifier(
                embedding_dim=FT_DIM, hidden_dim=wandb.config.lstm_hidden_dim,
                num_classes=num_classes, num_layers=wandb.config.lstm_num_layers,
                dropout=wandb.config.lstm_dropout # Use logged config values
            ).to(device)
try:
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device)); print("Best model loaded successfully.")
    best_model.eval()
   
    final_val_loss = 0.0; all_final_val_preds_indices = []; all_final_val_labels_indices = []
    with torch.no_grad():
        for data in val_loader:
            inputs, labels_batch = data; inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = best_model(inputs); loss = criterion(outputs, labels_batch); final_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1); all_final_val_preds_indices.append(predicted.cpu().numpy()); all_final_val_labels_indices.append(labels_batch.cpu().numpy())
    final_val_loss /= len(val_loader); all_final_val_preds_indices = np.concatenate(all_final_val_preds_indices); all_final_val_labels_indices = np.concatenate(all_final_val_labels_indices)
    final_val_accuracy = 100*(all_final_val_preds_indices == all_final_val_labels_indices).sum()/len(all_final_val_labels_indices)
    mapped_final_val_preds = np.vectorize(index_to_mae_label_map.get)(all_final_val_preds_indices); mapped_final_val_labels = np.vectorize(index_to_mae_label_map.get)(all_final_val_labels_indices)
    final_val_mae = mean_absolute_error(mapped_final_val_labels, mapped_final_val_preds); final_val_l_score = 0.5 * (2 - final_val_mae)
    print(f'\nFinal Validation Loss: {final_val_loss:.4f}, Accuracy: {final_val_accuracy:.2f}%, L-Score: {final_val_l_score:.4f}')
    wandb.run.summary["final_val_loss"] = final_val_loss; wandb.run.summary["final_val_accuracy"] = final_val_accuracy; wandb.run.summary["final_val_l_score"] = final_val_l_score
except FileNotFoundError: print(f"Error: Model file not found at {BEST_MODEL_PATH}."); exit()
except Exception as e: print(f"Error loading model state: {e}."); exit()

# Load Test Data
print("\nLoading test data...")
test_data_path = os.path.join(DATA_DIR, 'test.csv')
try: test_data = pd.read_csv(test_data_path, index_col=0); print("Test data loaded successfully.")
except FileNotFoundError: print(f"Error: '{test_data_path}' not found."); exit()
test_sentences = test_data['sentence'].astype(str)
print(f"Loaded {len(test_sentences)} test examples.")

# Preprocess and Vectorize Test Data as Sequences
print(f"Preprocessing and vectorizing test data using FastText...")
tokenized_test_sentences = [simple_tokenizer(sent) for sent in test_sentences]
X_test_sequences = np.array([vectorize_sentence_sequence_ft(tokens, ft_model, MAX_SEQ_LEN) for tokens in tokenized_test_sentences], dtype=np.float32)
X_test_tensor = torch.tensor(X_test_sequences, dtype=torch.float32).to(device)
print(f"Test sequences created with shape: {X_test_tensor.shape}")

# Make Predictions on Test Data using the Best Model
print(f"\nMaking predictions on the test set using the best BiLSTM model...")

best_model.eval(); test_predictions_list = []; test_batch_size = 256
with torch.no_grad():
    for i in range(0, len(X_test_tensor), test_batch_size):
        batch = X_test_tensor[i:i+test_batch_size]
        outputs = best_model(batch); _, predicted = torch.max(outputs.data, 1)
        test_predictions_list.append(predicted.cpu().numpy())
test_predictions_indices = np.concatenate(test_predictions_list)
print("Predictions generated.")

# Format Predictions for Submission
print("Formatting predictions for submission...")
test_predictions_labels = label_encoder.inverse_transform(test_predictions_indices)
submission_df = pd.DataFrame({'id': test_data.index, 'label': test_predictions_labels})
print("Submission DataFrame created:"); print(submission_df.head())

# Save Submission File
submission_filename = f'test_predictions_{WANDB_RUN_NAME}.csv'
try:
    submission_df.to_csv(submission_filename, index=False)
    print(f"\nTest predictions saved successfully to '{submission_filename}'")
except Exception as e: print(f"\nError saving submission file: {e}")

# Finish Wandb Run
print("\nFinishing Wandb run...")
wandb.finish()

print("\nScript finished.")