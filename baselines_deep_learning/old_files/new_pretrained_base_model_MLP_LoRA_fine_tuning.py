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
from sklearn.metrics import mean_absolute_error # accuracy_score was imported but not used directly
from sklearn.utils.class_weight import compute_class_weight

import time
import warnings
import math
import json
import wandb # Added for Weights & Biases

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

nltk_resources = ['punkt', 'punkt_tab']
print("Checking/downloading NLTK resources...")
for resource in nltk_resources:
    try:
        if resource in ['punkt', 'punkt_tab']:
            _ = nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        print(f"Downloading NLTK resource: {resource}...")
        try:
            nltk.download(resource, quiet=False)
            print(f"'{resource}' downloaded successfully.")
        except Exception as e:
            print(f"Error downloading '{resource}': {e}")
            exit()

# --- Configuration Section ---
# Define Data Directory, Device, Model Info, Sequence Length, Paths
DATA_DIR = '../../data'  # Adjust if your data directory is elsewhere
WANDB_PROJECT_NAME = "cil-sentiment-analysis" # Customizable WandB project name
WANDB_SAVE_DIR = "./wandb_standalone_runs" # Local directory for wandb files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
TARGET_MODULES = ["query", "key", "value", "attention.output.dense", "pooler.dense"]

CLASSIFIER_TYPE = "MLP"
PEFT_METHOD = "LoRA"
LORA_RANK = 64
LORA_ALPHA = 128
HF_MODEL_FILENAME_SUFFIX = HF_MODEL_NAME.split('/')[-1]

# Construct a unique run name for WandB and file saving
# You can customize this further, e.g., by adding a timestamp or a specific experiment tag
WANDB_RUN_NAME = f"PEFT_{HF_MODEL_FILENAME_SUFFIX}_R{LORA_RANK}_A{LORA_ALPHA}_{int(time.time())}"

RUN_SUFFIX = f"R{LORA_RANK}_Alpha{LORA_ALPHA}_Targets_{'-'.join(TARGET_MODULES).replace('.', '_')}"
BASE_SAVE_PATH = f"./trained_models_standalone/{HF_MODEL_FILENAME_SUFFIX}_{WANDB_RUN_NAME}" # Save models in a run-specific folder
BEST_LORA_ADAPTER_PATH = os.path.join(BASE_SAVE_PATH, f'best_{PEFT_METHOD}_adapters')
BEST_MLP_HEAD_PATH = os.path.join(BASE_SAVE_PATH, f'best_{PEFT_METHOD}_{CLASSIFIER_TYPE}_head.pth')
HPARAMS_FILE_PATH = os.path.join(BASE_SAVE_PATH, "hparams.json")
SUBMISSION_FILE_PATH = os.path.join(BASE_SAVE_PATH, f"submission_{HF_MODEL_FILENAME_SUFFIX}.csv")


MAX_SEQ_LEN = 128

print(f"Using device: {device}")
print(f"WandB Run Name: {WANDB_RUN_NAME}")
print(f"Models and artifacts will be saved in: {BASE_SAVE_PATH}")
print(f"Using Hugging Face model for PEFT: {HF_MODEL_NAME}")
print(f"Classifier head: {CLASSIFIER_TYPE}")
print(f"PEFT Method: {PEFT_METHOD} (r={LORA_RANK}, alpha={LORA_ALPHA}, targets={TARGET_MODULES})")
print(f"Max sequence length: {MAX_SEQ_LEN}")

# Ensure save directories exist
os.makedirs(BASE_SAVE_PATH, exist_ok=True)
os.makedirs(WANDB_SAVE_DIR, exist_ok=True)


# --- Data Loading and Preprocessing ---
print("Loading training data...")
train_data_path = os.path.join(DATA_DIR, 'training.csv')
try:
    training_data = pd.read_csv(train_data_path, index_col=0)
    print("Training data loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{train_data_path}' not found. Ensure path is correct.")
    exit()
sentences = training_data['sentence'].astype(str).tolist()
labels = training_data['label']
print(f"Loaded {len(sentences)} training examples.")
print("Original Label Distribution:")
print(labels.value_counts(normalize=True))

print(f"Loading tokenizer for '{HF_MODEL_NAME}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    print(f"Tokenizer '{HF_MODEL_NAME}' loaded.")
except Exception as e:
    print(f"Error loading tokenizer '{HF_MODEL_NAME}': {e}")
    exit()

print(f"Tokenizing all sentences (padding/truncating to {MAX_SEQ_LEN})...")
encodings = tokenizer(sentences, truncation=True, padding='max_length', max_length=MAX_SEQ_LEN, return_tensors='pt')
input_ids = encodings['input_ids']
attention_masks = encodings['attention_mask']
print(f"Input IDs shape: {input_ids.shape}")

print("Preparing labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
print(f"Label mapping: {list(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
index_to_mae_label_map = {
    label_encoder.transform(['negative'])[0]: -1,
    label_encoder.transform(['neutral'])[0]: 0,
    label_encoder.transform(['positive'])[0]: 1
}
print(f"Index to MAE Label Map: {index_to_mae_label_map}")

print("Creating train/validation split...")
train_idx, val_idx = train_test_split(np.arange(len(y_encoded)), test_size=0.15, stratify=y_encoded, random_state=42)

print("Calculating class weights...")
y_train_for_weights = y_encoded[train_idx]
class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_for_weights), y=y_train_for_weights)
class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
print(f"Calculated class weights: {class_weights_tensor}")

class TransformerSentimentDataset(Dataset):
    def __init__(self, input_ids, masks, labels):
        self.input_ids = input_ids
        self.masks = masks
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return {'input_ids': self.input_ids[idx], 'attention_mask': self.masks[idx], 'labels': self.labels[idx]}

train_dataset = TransformerSentimentDataset(input_ids[train_idx], attention_masks[train_idx], y_encoded[train_idx])
val_dataset = TransformerSentimentDataset(input_ids[val_idx], attention_masks[val_idx], y_encoded[val_idx])

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 200 # Default epochs
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.2
NUM_WORKERS = 6 # Adjust based on your system

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# --- Model Definition and PEFT Setup ---
print(f"Loading base model '{HF_MODEL_NAME}' for LoRA...")
try:
    base_model = AutoModel.from_pretrained(HF_MODEL_NAME)
    BASE_MODEL_HIDDEN_SIZE = base_model.config.hidden_size
    print(f"Base model loaded. Hidden size: {BASE_MODEL_HIDDEN_SIZE}")
except Exception as e:
    print(f"Error loading base model: {e}")
    exit()

print("Applying LoRA configuration...")
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES, lora_dropout=0.1, bias="none"
)
peft_model = get_peft_model(base_model, lora_config)
print("LoRA applied. Trainable parameters:")
peft_model.print_trainable_parameters()

class PeftMLPClassifier(nn.Module):
    def __init__(self, peft_base_model, base_model_hidden_size, num_classes, mlp_hidden_1=128, mlp_hidden_2=64, mlp_dropout=0.5):
        super(PeftMLPClassifier, self).__init__()
        self.peft_base = peft_base_model
        self.base_model_hidden_size = base_model_hidden_size
        self.mlp_head = nn.Sequential(
            nn.Linear(self.base_model_hidden_size, mlp_hidden_1), nn.ReLU(), nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_1, mlp_hidden_2), nn.ReLU(), nn.Dropout(mlp_dropout * 0.6),
            nn.Linear(mlp_hidden_2, num_classes)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.peft_base(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.mlp_head(cls_output)
        return logits

MLP_HIDDEN_1 = 128
MLP_HIDDEN_2 = 64
MLP_DROPOUT = 0.5

combined_model = PeftMLPClassifier(
    peft_base_model=peft_model, base_model_hidden_size=BASE_MODEL_HIDDEN_SIZE,
    num_classes=num_classes, mlp_hidden_1=MLP_HIDDEN_1, mlp_hidden_2=MLP_HIDDEN_2,
    mlp_dropout=MLP_DROPOUT
).to(device)
print(combined_model)
print(f"Combined Model Trainable Parameters: {sum(p.numel() for p in combined_model.parameters() if p.requires_grad):,}")

# --- Optimizer, Scheduler, Loss ---
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(combined_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, verbose=True)

# --- WandB and Hyperparameter Logging ---
hparams_to_log = {
    "learning_rate": LEARNING_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
    "architecture": f"PEFT_{PEFT_METHOD}_{HF_MODEL_NAME}+MLP", "peft_method": PEFT_METHOD,
    "lora_r": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_dropout": lora_config.lora_dropout,
    "lora_target_modules": str(TARGET_MODULES), "max_seq_len": MAX_SEQ_LEN,
    "base_hf_model": HF_MODEL_NAME, "tokenizer": HF_MODEL_NAME, "classifier_head": CLASSIFIER_TYPE,
    "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
    "base_model_hidden_size": BASE_MODEL_HIDDEN_SIZE, "mlp_hidden_1": MLP_HIDDEN_1,
    "mlp_hidden_2": MLP_HIDDEN_2, "mlp_dropout": MLP_DROPOUT,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE, "lr_scheduler_patience": LR_SCHEDULER_PATIENCE,
    "lr_scheduler_factor": LR_SCHEDULER_FACTOR,
    "validation_split": 0.15, "random_state_split": 42,
    "loss_function": "Weighted CrossEntropyLoss",
    "class_weights_values": str(class_weights_tensor.cpu().tolist()),
    "num_workers_dataloader": NUM_WORKERS
}
try:
    with open(HPARAMS_FILE_PATH, 'w') as f:
        json.dump(hparams_to_log, f, indent=4)
    print(f"Hyperparameters saved to {HPARAMS_FILE_PATH}")
except Exception as e:
    print(f"Error saving hyperparameters: {e}")

# Initialize WandB
try:
    wandb.init(
        project=WANDB_PROJECT_NAME,
        name=WANDB_RUN_NAME,
        config=hparams_to_log,
        dir=WANDB_SAVE_DIR # Saves wandb files locally
    )
    print(f"WandB initialized successfully for run: {WANDB_RUN_NAME}")
except Exception as e:
    print(f"WandB initialization failed: {e}. Training will proceed without WandB logging.")
    wandb = None # Ensure wandb calls are skipped if init fails


# --- Training Loop ---
print(f"\n--- Starting PyTorch {PEFT_METHOD} Fine-tuning ({HF_MODEL_NAME}) + MLP Training ---")
best_val_loss = float('inf')
epochs_no_improve = 0
start_time = time.time()

try: # Wrap training in a try-finally to ensure wandb.finish() is called
    for epoch in range(EPOCHS):
        # Training Phase
        combined_model.train()
        running_loss = 0.0
        all_train_preds_indices = []
        all_train_labels_indices = []
        for i, batch_data in enumerate(train_loader):
            input_ids_batch = batch_data['input_ids'].to(device)
            attention_mask_batch = batch_data['attention_mask'].to(device)
            labels_batch = batch_data['labels'].to(device)

            optimizer.zero_grad()
            outputs = combined_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted_train = torch.max(outputs.data, 1)
            all_train_preds_indices.append(predicted_train.cpu().numpy())
            all_train_labels_indices.append(labels_batch.cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader)
        all_train_preds_indices = np.concatenate(all_train_preds_indices)
        all_train_labels_indices = np.concatenate(all_train_labels_indices)
        epoch_train_acc = 100 * (all_train_preds_indices == all_train_labels_indices).sum() / len(all_train_labels_indices)
        
        mapped_train_preds = np.vectorize(index_to_mae_label_map.get)(all_train_preds_indices)
        mapped_train_labels = np.vectorize(index_to_mae_label_map.get)(all_train_labels_indices)
        valid_train_indices = (mapped_train_preds != None) & (mapped_train_labels != None)
        train_mae = mean_absolute_error(mapped_train_preds[valid_train_indices], mapped_train_labels[valid_train_indices]) if np.sum(valid_train_indices) > 0 else 0
        epoch_train_l_score = 0.5 * (2 - train_mae)

        # Validation Phase
        combined_model.eval()
        val_loss = 0.0
        all_val_preds_indices = []
        all_val_labels_indices = []
        with torch.no_grad():
            for batch_data in val_loader:
                input_ids_batch = batch_data['input_ids'].to(device)
                attention_mask_batch = batch_data['attention_mask'].to(device)
                labels_batch = batch_data['labels'].to(device)
                outputs = combined_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                _, predicted_val = torch.max(outputs.data, 1)
                all_val_preds_indices.append(predicted_val.cpu().numpy())
                all_val_labels_indices.append(labels_batch.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        all_val_preds_indices = np.concatenate(all_val_preds_indices)
        all_val_labels_indices = np.concatenate(all_val_labels_indices)
        epoch_val_acc = 100 * (all_val_preds_indices == all_val_labels_indices).sum() / len(all_val_labels_indices)
        
        mapped_val_preds = np.vectorize(index_to_mae_label_map.get)(all_val_preds_indices)
        mapped_val_labels = np.vectorize(index_to_mae_label_map.get)(all_val_labels_indices)
        valid_val_indices = (mapped_val_preds != None) & (mapped_val_labels != None)
        val_mae = mean_absolute_error(mapped_val_preds[valid_val_indices], mapped_val_labels[valid_val_indices]) if np.sum(valid_val_indices) > 0 else 0
        epoch_val_l_score = 0.5 * (2 - val_mae)

        print(f'Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}%, L: {epoch_train_l_score:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%, L: {epoch_val_l_score:.4f}')
        current_lr = optimizer.param_groups[0]['lr']
        
        if wandb and wandb.run: # Check if wandb was initialized
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss, "train_accuracy": epoch_train_acc, "train_l_score": epoch_train_l_score,
                "val_loss": epoch_val_loss, "val_accuracy": epoch_val_acc, "val_l_score": epoch_val_l_score,
                "learning_rate": current_lr
            })

        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            try:
                os.makedirs(BEST_LORA_ADAPTER_PATH, exist_ok=True) # Ensure directory exists
                combined_model.peft_base.save_pretrained(BEST_LORA_ADAPTER_PATH)
                torch.save(combined_model.mlp_head.state_dict(), BEST_MLP_HEAD_PATH)
                print(f'Val loss decreased ({best_val_loss:.4f}). Saving adapters to {BEST_LORA_ADAPTER_PATH} and MLP head to {BEST_MLP_HEAD_PATH}')
                if wandb and wandb.run:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_val_l_score"] = epoch_val_l_score
                    wandb.run.summary["best_epoch"] = epoch + 1
            except Exception as e:
                print(f"Error saving model/adapters: {e}")
        else:
            epochs_no_improve += 1
            print(f'Val loss did not improve. Counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}')

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
finally:
    if wandb and wandb.run: # Ensure wandb.finish is called if init was successful
        training_duration = time.time() - start_time
        print(f"\nTraining finished in {training_duration:.2f} seconds.")
        wandb.run.summary["training_time_seconds"] = training_duration
        wandb.finish()
        print("WandB run finished.")
    elif 'start_time' in locals(): # If training started but wandb failed
        training_duration = time.time() - start_time
        print(f"\nTraining finished in {training_duration:.2f} seconds (WandB not used or failed).")


# --- Final Evaluation and Test Prediction ---
best_model_for_eval = None # Initialize to handle cases where training might not complete
if os.path.exists(BEST_LORA_ADAPTER_PATH) and os.path.exists(BEST_MLP_HEAD_PATH):
    print(f"\nLoading best model '{HF_MODEL_NAME}' with LoRA adapters from '{BEST_LORA_ADAPTER_PATH}' for final validation...")
    try:
        final_base_model = AutoModel.from_pretrained(HF_MODEL_NAME)
        best_peft_model_for_eval = PeftModel.from_pretrained(final_base_model, BEST_LORA_ADAPTER_PATH).to(device)
        print("LoRA adapters loaded successfully.")

        best_model_for_eval = PeftMLPClassifier(
            peft_base_model=best_peft_model_for_eval, base_model_hidden_size=BASE_MODEL_HIDDEN_SIZE,
            num_classes=num_classes, mlp_hidden_1=MLP_HIDDEN_1, mlp_hidden_2=MLP_HIDDEN_2,
            mlp_dropout=MLP_DROPOUT
        ).to(device)
        best_model_for_eval.mlp_head.load_state_dict(torch.load(BEST_MLP_HEAD_PATH, map_location=device))
        print("Best MLP head loaded successfully.")
        best_model_for_eval.eval()

        final_val_loss, final_val_acc, final_val_l_score, _, _ = 0.0, 0.0, 0.0 # Initialize
        temp_val_loss = 0.0
        all_final_val_preds_indices, all_final_val_labels_indices = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                input_ids_batch, attention_mask_batch, labels_batch = batch_data['input_ids'].to(device), batch_data['attention_mask'].to(device), batch_data['labels'].to(device)
                outputs = best_model_for_eval(input_ids_batch, attention_mask_batch)
                loss = criterion(outputs, labels_batch)
                temp_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_final_val_preds_indices.append(predicted.cpu().numpy()); all_final_val_labels_indices.append(labels_batch.cpu().numpy())
        
        final_val_loss = temp_val_loss / len(val_loader)
        all_final_val_preds_indices = np.concatenate(all_final_val_preds_indices); all_final_val_labels_indices = np.concatenate(all_final_val_labels_indices)
        final_val_accuracy = 100 * (all_final_val_preds_indices == all_final_val_labels_indices).sum() / len(all_final_val_labels_indices)
        
        mapped_final_val_preds = np.vectorize(index_to_mae_label_map.get)(all_final_val_preds_indices)
        mapped_final_val_labels = np.vectorize(index_to_mae_label_map.get)(all_final_val_labels_indices)
        valid_final_val_indices = (mapped_final_val_preds != None) & (mapped_final_val_labels != None)
        final_val_mae = mean_absolute_error(mapped_final_val_preds[valid_final_val_indices], mapped_final_val_labels[valid_final_val_indices]) if np.sum(valid_final_val_indices) > 0 else 0
        final_val_l_score = 0.5 * (2 - final_val_mae)
        
        print(f'\nFinal Validation Loss (Best Model): {final_val_loss:.4f}, Accuracy: {final_val_accuracy:.2f}%, L-Score: {final_val_l_score:.4f}')
        if wandb and wandb.run:
            wandb.run.summary["final_eval_val_loss"] = final_val_loss
            wandb.run.summary["final_eval_val_accuracy"] = final_val_accuracy
            wandb.run.summary["final_eval_val_l_score"] = final_val_l_score

    except FileNotFoundError: print(f"Error: Model/Adapter files not found. Skipping final validation.")
    except Exception as e: print(f"Error during final model validation: {e}")


    if best_model_for_eval:
        print("\nLoading test data for submission...")
        try:
            test_data = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), index_col=0)
            print("Test data loaded successfully.")
            test_sentences = test_data['sentence'].astype(str).tolist()
            print(f"Loaded {len(test_sentences)} test examples.")

            print(f"Tokenizing test data using HF tokenizer '{HF_MODEL_NAME}'...")
            test_encodings = tokenizer(test_sentences, truncation=True, padding='max_length', max_length=MAX_SEQ_LEN, return_tensors='pt')
            test_input_ids = test_encodings['input_ids'].to(device)
            test_attention_masks = test_encodings['attention_mask'].to(device)
            print(f"Test input_ids created with shape: {test_input_ids.shape}")

            print(f"\nMaking predictions on the test set...")
            best_model_for_eval.eval()
            test_predictions_list = []
            inference_batch_size = BATCH_SIZE # Use training BATCH_SIZE or a specific inference batch_size
            with torch.no_grad():
                for i in range(0, len(test_input_ids), inference_batch_size):
                    input_ids_batch = test_input_ids[i:i + inference_batch_size]
                    attention_mask_batch = test_attention_masks[i:i + inference_batch_size]
                    outputs = best_model_for_eval(input_ids_batch, attention_mask_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    test_predictions_list.append(predicted.cpu().numpy())
            test_predictions_indices = np.concatenate(test_predictions_list)
            print("Predictions generated.")

            test_predictions_labels = label_encoder.inverse_transform(test_predictions_indices)
            submission_df = pd.DataFrame({'id': test_data.index, 'label': test_predictions_labels})
            print("Submission DataFrame created:")
            print(submission_df.head())

            os.makedirs(os.path.dirname(SUBMISSION_FILE_PATH), exist_ok=True) # Ensure submission dir in model folder exists
            submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
            print(f"\nTest predictions saved successfully to '{SUBMISSION_FILE_PATH}'")
            if wandb and wandb.run:
                 wandb.save(SUBMISSION_FILE_PATH) # Save submission file to wandb
                 print(f"Submission file also uploaded to WandB run: {WANDB_RUN_NAME}")


        except FileNotFoundError: print(f"Error: Test data file not found.")
        except Exception as e: print(f"\nError during test prediction or submission saving: {e}")
    else:
        print("Error: Best model for evaluation was not loaded/available. Skipping test set prediction.")

print("\nScript finished.")