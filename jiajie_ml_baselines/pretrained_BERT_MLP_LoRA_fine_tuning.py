import os
import pandas as pd
import numpy as np
import re
import nltk

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import time
import warnings
from torch.utils.tensorboard import SummaryWriter
import math
import json 

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

nltk_resources = ['punkt', 'punkt_tab']
print("Checking/downloading NLTK resources...")
for resource in nltk_resources:
    try: _ = nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        print(f"Downloading NLTK resource: {resource}...")
        try: nltk.download(resource, quiet=False); print(f"'{resource}' downloaded successfully.")
        except Exception as e: print(f"Error downloading '{resource}': {e}"); exit()

# Define Data Directory, Device, Model Info, Sequence Length, Paths
DATA_DIR = 'data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_MODEL_NAME = "distilbert-base-multilingual-cased"
CLASSIFIER_TYPE = "MLP"
PEFT_METHOD = "LoRA"
LORA_RANK = 64; LORA_ALPHA = 128; TARGET_MODULES = ["q_lin", "v_lin", "k_lin", "out_lin"]
RUN_SUFFIX = f"R{LORA_RANK}_Alpha{LORA_ALPHA}_TargetsAll" 
BEST_LORA_ADAPTER_PATH = f'best_{PEFT_METHOD}_adapters_{HF_MODEL_NAME}_{RUN_SUFFIX}'
BEST_MLP_HEAD_PATH = f'best_{PEFT_METHOD}_{HF_MODEL_NAME}_{CLASSIFIER_TYPE}_head_{RUN_SUFFIX}.pth'
# --- Use TENSORBOARD_RUN_NAME consistent with TensorBoard logging ---
TENSORBOARD_RUN_NAME = f"{PEFT_METHOD}_{HF_MODEL_NAME}_{CLASSIFIER_TYPE}_{RUN_SUFFIX}"
MAX_SEQ_LEN = 128
print(f"Using device: {device}")
print(f"Best LoRA adapters will be saved to: {BEST_LORA_ADAPTER_PATH}")
print(f"Best MLP head will be saved to: {BEST_MLP_HEAD_PATH}")
print(f"TensorBoard Log Directory: /root/tf-logs/{TENSORBOARD_RUN_NAME}")
print(f"Using Hugging Face model for PEFT: {HF_MODEL_NAME}")
print(f"Classifier head: {CLASSIFIER_TYPE}")
print(f"PEFT Method: {PEFT_METHOD} (r={LORA_RANK}, alpha={LORA_ALPHA}, targets={TARGET_MODULES})")
print(f"Max sequence length: {MAX_SEQ_LEN}")

# Load Training Data
print("Loading training data...")
train_data_path = os.path.join(DATA_DIR, 'training.csv')
try: training_data = pd.read_csv(train_data_path, index_col=0); print("Training data loaded successfully.")
except FileNotFoundError: print(f"Error: '{train_data_path}' not found."); exit()
sentences = training_data['sentence'].astype(str).tolist(); labels = training_data['label']
print(f"Loaded {len(sentences)} training examples."); print("Original Label Distribution:"); print(labels.value_counts(normalize=True))

# Load Pre-trained Tokenizer
print(f"Loading tokenizer for '{HF_MODEL_NAME}'...")
try: tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME); VOCAB_SIZE = tokenizer.vocab_size; print(f"Tokenizer '{HF_MODEL_NAME}' loaded.")
except Exception as e: print(f"Error loading tokenizer '{HF_MODEL_NAME}': {e}"); exit()

# Tokenize ALL Sentences
print(f"Tokenizing all sentences using HF tokenizer (padding/truncating to {MAX_SEQ_LEN})...")
encodings = tokenizer(sentences, truncation=True, padding='max_length', max_length=MAX_SEQ_LEN, return_tensors='pt')
input_ids = encodings['input_ids']; attention_masks = encodings['attention_mask']
print(f"Input IDs shape: {input_ids.shape}")

# Prepare Labels
print("Preparing labels...")
label_encoder = LabelEncoder(); y_encoded = label_encoder.fit_transform(labels); num_classes = len(label_encoder.classes_)
print(f"Label mapping: {list(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
index_to_mae_label_map = { label_encoder.transform(['negative'])[0]: -1, label_encoder.transform(['neutral'])[0]: 0, label_encoder.transform(['positive'])[0]: 1}
print(f"Index to MAE Label Map: {index_to_mae_label_map}")

# Create Train/Validation Split
print("Creating train/validation split...")
train_idx, val_idx = train_test_split(np.arange(len(y_encoded)), test_size=0.15, stratify=y_encoded, random_state=42)

# Calculate Class Weights
print("Calculating class weights for loss function...")
y_train_for_weights = y_encoded[train_idx]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_for_weights), y=y_train_for_weights)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Calculated class weights: {class_weights_tensor}")

# Create Dataset for Transformer Inputs
print("Creating PyTorch Datasets and DataLoaders for Transformer inputs...")
class TransformerSentimentDataset(Dataset):
    def __init__(self, input_ids, masks, labels): self.input_ids = input_ids; self.masks = masks; self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return {'input_ids': self.input_ids[idx], 'attention_mask': self.masks[idx], 'labels': self.labels[idx]}

train_dataset = TransformerSentimentDataset(input_ids[train_idx], attention_masks[train_idx], y_encoded[train_idx])
val_dataset = TransformerSentimentDataset(input_ids[val_idx], attention_masks[val_idx], y_encoded[val_idx])

BATCH_SIZE = 256
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

# Load Base Model and Apply LoRA
print(f"Loading base model '{HF_MODEL_NAME}' for LoRA...")
try: bert_model = AutoModel.from_pretrained(HF_MODEL_NAME); BERT_HIDDEN_SIZE = bert_model.config.hidden_size; print("Base model loaded.")
except Exception as e: print(f"Error loading base model: {e}"); exit()

print("Applying LoRA configuration...")
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES, lora_dropout=0.1, bias="none"
)
peft_model = get_peft_model(bert_model, lora_config)
print("LoRA applied. Trainable parameters:")
peft_model.print_trainable_parameters()

# Define Combined Model using PEFT model
print("Defining PyTorch Model (PEFT Base + MLP Head)...")
class PeftMLPClassifier(nn.Module):
    def __init__(self, peft_base_model, num_classes, mlp_hidden_1=128, mlp_hidden_2=64, mlp_dropout=0.5):
        super(PeftMLPClassifier, self).__init__()
        self.bert = peft_base_model; self.bert_hidden_size = self.bert.config.hidden_size
        self.mlp_head = nn.Sequential(
            nn.Linear(self.bert_hidden_size, mlp_hidden_1), nn.ReLU(), nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_1, mlp_hidden_2), nn.ReLU(), nn.Dropout(mlp_dropout * 0.6),
            nn.Linear(mlp_hidden_2, num_classes) )
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.mlp_head(cls_output)
        return logits

MLP_HIDDEN_1 = 128; MLP_HIDDEN_2 = 64; MLP_DROPOUT = 0.5
combined_model = PeftMLPClassifier(
                    peft_base_model=peft_model, num_classes=num_classes,
                    mlp_hidden_1=MLP_HIDDEN_1, mlp_hidden_2=MLP_HIDDEN_2,
                    mlp_dropout=MLP_DROPOUT
                 ).to(device)
print(combined_model); print(f"Combined Model Trainable Parameters: {sum(p.numel() for p in combined_model.parameters() if p.requires_grad):,}")

# Define Loss, Optimizer, and LR Scheduler
print("Defining loss function, optimizer (with weight decay), and LR scheduler...")
LEARNING_RATE = 1e-4; WEIGHT_DECAY = 1e-4
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
# --- Use optim.AdamW ---
optimizer = optim.AdamW(combined_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

# Initialize TensorBoard Writer
print("Initializing TensorBoard writer...")
log_dir = os.path.join("/root/tf-logs/", TENSORBOARD_RUN_NAME)
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard logs will be saved to: {log_dir}")

# Log Hyperparameters to TensorBoard
hparams = {
    "learning_rate": LEARNING_RATE, "epochs": 50, "batch_size": BATCH_SIZE,
    "architecture": f"PEFT_{PEFT_METHOD}_{HF_MODEL_NAME}+MLP", "peft_method": PEFT_METHOD,
    "lora_r": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_dropout": lora_config.lora_dropout,
    "lora_target_modules": str(TARGET_MODULES), "max_seq_len": MAX_SEQ_LEN,
    "embedding": f"PEFT {HF_MODEL_NAME}", "tokenizer": HF_MODEL_NAME, "classifier": CLASSIFIER_TYPE,
    "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
    "bert_hidden_size": combined_model.bert_hidden_size, "mlp_hidden_1": MLP_HIDDEN_1,
    "mlp_hidden_2": MLP_HIDDEN_2, "mlp_dropout": MLP_DROPOUT,
    "early_stopping_patience": 5, "lr_scheduler_patience": 3, "validation_split": 0.15,
    "loss_function": "Weighted CrossEntropyLoss", "class_weights": str(class_weights_tensor.cpu().tolist()),
}
hparams_text = json.dumps(hparams, indent=2)
writer.add_text("Hyperparameters", f"<pre>{hparams_text}</pre>", 0)


# Training Loop
print(f"\n--- Starting PyTorch {PEFT_METHOD} Fine-tuning + MLP Training ---")
EPOCHS = hparams['epochs']; EARLY_STOPPING_PATIENCE = hparams['early_stopping_patience']
best_val_loss = float('inf'); epochs_no_improve = 0; start_time = time.time()

for epoch in range(EPOCHS):
    # Training Phase
    combined_model.train(); running_loss = 0.0; all_train_preds_indices = []; all_train_labels_indices = []
    for i, batch_data in enumerate(train_loader):
        input_ids_batch = batch_data['input_ids'].to(device); attention_mask_batch = batch_data['attention_mask'].to(device); labels_batch = batch_data['labels'].to(device)
        optimizer.zero_grad(); outputs = combined_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
        loss = criterion(outputs, labels_batch); loss.backward(); optimizer.step()
        running_loss += loss.item(); _, predicted_train = torch.max(outputs.data, 1)
        all_train_preds_indices.append(predicted_train.cpu().numpy()); all_train_labels_indices.append(labels_batch.cpu().numpy())
    epoch_train_loss = running_loss / len(train_loader); all_train_preds_indices = np.concatenate(all_train_preds_indices); all_train_labels_indices = np.concatenate(all_train_labels_indices)
    epoch_train_acc = 100*(all_train_preds_indices == all_train_labels_indices).sum()/len(all_train_labels_indices)
    mapped_train_preds = np.vectorize(index_to_mae_label_map.get)(all_train_preds_indices); mapped_train_labels = np.vectorize(index_to_mae_label_map.get)(all_train_labels_indices)
    train_mae = mean_absolute_error(mapped_train_labels, mapped_train_preds); epoch_train_l_score = 0.5 * (2 - train_mae)

    # Validation Phase
    combined_model.eval(); val_loss = 0.0; all_val_preds_indices = []; all_val_labels_indices = []
    with torch.no_grad():
        for batch_data in val_loader:
            input_ids_batch = batch_data['input_ids'].to(device); attention_mask_batch = batch_data['attention_mask'].to(device); labels_batch = batch_data['labels'].to(device)
            outputs = combined_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
            loss = criterion(outputs, labels_batch); val_loss += loss.item()
            _, predicted_val = torch.max(outputs.data, 1); all_val_preds_indices.append(predicted_val.cpu().numpy()); all_val_labels_indices.append(labels_batch.cpu().numpy())
    epoch_val_loss = val_loss / len(val_loader); all_val_preds_indices = np.concatenate(all_val_preds_indices); all_val_labels_indices = np.concatenate(all_val_labels_indices)
    epoch_val_acc = 100*(all_val_preds_indices == all_val_labels_indices).sum()/len(all_val_labels_indices)
    mapped_val_preds = np.vectorize(index_to_mae_label_map.get)(all_val_preds_indices); mapped_val_labels = np.vectorize(index_to_mae_label_map.get)(all_val_labels_indices)
    val_mae = mean_absolute_error(mapped_val_labels, mapped_val_preds); epoch_val_l_score = 0.5 * (2 - val_mae)

    print(f'Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}%, L: {epoch_train_l_score:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%, L: {epoch_val_l_score:.4f}')
    current_lr = optimizer.param_groups[0]['lr']
    # Log Metrics to TensorBoard
    writer.add_scalar('Loss/Train', epoch_train_loss, epoch + 1); writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch + 1); writer.add_scalar('L-Score/Train', epoch_train_l_score, epoch + 1)
    writer.add_scalar('Loss/Validation', epoch_val_loss, epoch + 1); writer.add_scalar('Accuracy/Validation', epoch_val_acc, epoch + 1); writer.add_scalar('L-Score/Validation', epoch_val_l_score, epoch + 1)
    writer.add_scalar('LearningRate', current_lr, epoch + 1)

    # LR Scheduling & Early Stopping
    scheduler.step(epoch_val_loss)
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss; epochs_no_improve = 0
        try:
            combined_model.bert.save_pretrained(BEST_LORA_ADAPTER_PATH)
            torch.save(combined_model.mlp_head.state_dict(), BEST_MLP_HEAD_PATH)
            print(f'Val loss decreased ({best_val_loss:.4f}). Saving adapters to {BEST_LORA_ADAPTER_PATH} and MLP head to {BEST_MLP_HEAD_PATH}')
        except Exception as e: print(f"Error saving model/adapters: {e}")
    else:
        epochs_no_improve += 1; print(f'Val loss did not improve. Counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}')
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE: print(f'Early stopping triggered after {epoch+1} epochs.'); break

end_time = time.time(); print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")
writer.add_scalar('Training/TotalTimeSeconds', end_time - start_time, EPOCHS)


# Load Best Model using PEFT
print(f"\nLoading base model '{HF_MODEL_NAME}' and best LoRA adapters from '{BEST_LORA_ADAPTER_PATH}'...")
try:
    base_model = AutoModel.from_pretrained(HF_MODEL_NAME)
    best_peft_model = PeftModel.from_pretrained(base_model, BEST_LORA_ADAPTER_PATH).to(device)
    print("LoRA adapters loaded successfully.")
    best_model = PeftMLPClassifier(
                    peft_base_model=best_peft_model, num_classes=num_classes,
                    mlp_hidden_1=hparams['mlp_hidden_1'], mlp_hidden_2=hparams['mlp_hidden_2'],
                    mlp_dropout=hparams['mlp_dropout'] ).to(device)
    best_model.mlp_head.load_state_dict(torch.load(BEST_MLP_HEAD_PATH, map_location=device))
    print("Best MLP head loaded successfully.")
    best_model.eval()

    # Final Validation Evaluation
    final_val_loss = 0.0; all_final_val_preds_indices = []; all_final_val_labels_indices = []
    with torch.no_grad():
        for batch_data in val_loader:
            input_ids_batch = batch_data['input_ids'].to(device); attention_mask_batch = batch_data['attention_mask'].to(device); labels_batch = batch_data['labels'].to(device)
            outputs = best_model(input_ids_batch, attention_mask_batch); loss = criterion(outputs, labels_batch); final_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1); all_final_val_preds_indices.append(predicted.cpu().numpy()); all_final_val_labels_indices.append(labels_batch.cpu().numpy())
    final_val_loss /= len(val_loader); all_final_val_preds_indices = np.concatenate(all_final_val_preds_indices); all_final_val_labels_indices = np.concatenate(all_final_val_labels_indices)
    final_val_accuracy = 100*(all_final_val_preds_indices == all_final_val_labels_indices).sum()/len(all_final_val_labels_indices)
    mapped_final_val_preds = np.vectorize(index_to_mae_label_map.get)(all_final_val_preds_indices); mapped_final_val_labels = np.vectorize(index_to_mae_label_map.get)(all_final_val_labels_indices)
    final_val_mae = mean_absolute_error(mapped_final_val_labels, mapped_final_val_preds); final_val_l_score = 0.5 * (2 - final_val_mae)
    print(f'\nFinal Validation Loss: {final_val_loss:.4f}, Accuracy: {final_val_accuracy:.2f}%, L-Score: {final_val_l_score:.4f}')
    writer.add_scalar('FinalVal/Loss', final_val_loss, EPOCHS); writer.add_scalar('FinalVal/Accuracy', final_val_accuracy, EPOCHS); writer.add_scalar('FinalVal/L-Score', final_val_l_score, EPOCHS)

except FileNotFoundError: print(f"Error: Model/Adapter files not found. Cannot proceed."); exit()
except Exception as e: print(f"Error loading model state: {e}. Cannot proceed."); exit()


# Load Test Data
print("\nLoading test data...")
test_data_path = os.path.join(DATA_DIR, 'test.csv')
try: test_data = pd.read_csv(test_data_path, index_col=0); print("Test data loaded successfully.")
except FileNotFoundError: print(f"Error: '{test_data_path}' not found."); exit()
test_sentences = test_data['sentence'].astype(str).tolist()
print(f"Loaded {len(test_sentences)} test examples.")

# Tokenize Test Data
print(f"Tokenizing test data using HF tokenizer...")
test_encodings = tokenizer(test_sentences, truncation=True, padding='max_length', max_length=MAX_SEQ_LEN, return_tensors='pt')
test_input_ids = test_encodings['input_ids'].to(device)
test_attention_masks = test_encodings['attention_mask'].to(device)
print(f"Test input_ids created with shape: {test_input_ids.shape}")

# Make Predictions on Test Data using the Best PEFT Model
print(f"\nMaking predictions on the test set using the best {PEFT_METHOD} model...")
best_model.eval(); test_predictions_list = []; test_batch_size = BATCH_SIZE
with torch.no_grad():
    for i in range(0, len(test_input_ids), test_batch_size):
        input_ids_batch = test_input_ids[i:i+test_batch_size]
        attention_mask_batch = test_attention_masks[i:i+test_batch_size]
        outputs = best_model(input_ids_batch, attention_mask_batch)
        _, predicted = torch.max(outputs.data, 1)
        test_predictions_list.append(predicted.cpu().numpy())
test_predictions_indices = np.concatenate(test_predictions_list)
print("Predictions generated.")

# Format Predictions for Submission
print("Formatting predictions for submission...")
test_predictions_labels = label_encoder.inverse_transform(test_predictions_indices)
submission_df = pd.DataFrame({'id': test_data.index, 'label': test_predictions_labels})
print("Submission DataFrame created:"); print(submission_df.head())

# Save Submission File
submission_filename = f'test_predictions_{TENSORBOARD_RUN_NAME}.csv'
try:
    submission_df.to_csv(submission_filename, index=False)
    print(f"\nTest predictions saved successfully to '{submission_filename}'")
except Exception as e: print(f"\nError saving submission file: {e}")

# Finish TensorBoard Writer
print("\nClosing TensorBoard writer...")
writer.close()

print("\nScript finished.")