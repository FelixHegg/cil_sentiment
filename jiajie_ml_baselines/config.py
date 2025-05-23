import torch
import os

# --- Base Directories ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SRC_DIR)

# --- Data Path ---
DATA_DIR = os.path.join(PARENT_DIR, 'data')
TRAIN_CSV = os.path.join(DATA_DIR, 'training.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
MODEL_SAVE_DIR = os.path.join(SRC_DIR, 'testing_checkpoints')
SUBMISSION_DIR = os.path.join(SRC_DIR, 'submissions')
WANDB_LOCAL_SAVE_DIR = os.path.join(SRC_DIR, 'wandb_local_runs')

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Preprocessing ---
SEQUENCE_LENGTH_PERCENTILE = 95
MIN_SEQUENCE_LENGTH = 10
# Max sequence length specifically for the Hugging Face Base model
HF_MAX_SEQ_LEN = 128

# --- Embeddings & Pre-trained Models ---
W2V_MODEL_NAME = 'word2vec-google-news-300'
FASTTEXT_MODEL_NAME = 'fasttext-wiki-news-subwords-300'
HF_PRETRAINED_SENTIMENT_MODEL_NAME = "tabularisai/multilingual-sentiment-analysis" # For direct inference
# Base model for PEFT fine-tuning
HF_BASE_MODEL_FOR_PEFT = "sentence-transformers/all-mpnet-base-v2"


# --- Training General ---
VALIDATION_SPLIT = 0.15
RANDOM_STATE = 42
EPOCHS = 50 
BATCH_SIZE = 64 
LEARNING_RATE = 1e-3 
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.2
LR_SCHEDULER_PATIENCE = 3
NUM_WORKERS = 8

# --- WandB ---
WANDB_PROJECT = "cil-sentiment-analysis"

# --- Model Specific Configs ---
MLP_CONFIG = {
    "model_type": "mlp",
    "embedding_model": W2V_MODEL_NAME,
    "vectorization_strategy": "average",
    "hidden_dims": [128, 256, 128, 64],
    "dropouts": [0.5, 0.3, 0.2, 0.1],
    "learning_rate": 1e-3,
    "weight_decay": 0,
    "early_stopping_patience": 10,
    "epochs": 100,
    "lr_scheduler_factor": 0.5
}

CNN_CONFIG = {
    "model_type": "cnn",
    "embedding_model": FASTTEXT_MODEL_NAME,
    "vectorization_strategy": "sequence",
    "num_filters": 80,
    "filter_sizes": [3, 4, 5],
    "dropout": 0.6,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 5,
    "epochs": 100,
}

BILSTM_CONFIG = {
    "model_type": "bilstm",
    "embedding_model": FASTTEXT_MODEL_NAME,
    "vectorization_strategy": "sequence",
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 5,
    "epochs": 100,
}

HF_PRETRAINED_SENTIMENT_ANALYSIS_MODEL_CONFIG = {
    "model_type": "hf_pretrained_sentiment_analysis_model",
    "hf_model_name": HF_PRETRAINED_SENTIMENT_MODEL_NAME
}

# --- Config for PEFT + MPNet + MLP Head ---
PEFT_MPNET_MLP_CONFIG = {
    "model_type": "peft_mpnet_mlp",
    "hf_base_model_name": HF_BASE_MODEL_FOR_PEFT,
    "max_seq_len": HF_MAX_SEQ_LEN,
    "peft_method": "LoRA",
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_target_modules": ["query", "key", "value", "attention.output.dense", "pooler.dense"],
    "lora_dropout": 0.1,
    "lora_bias": "none",
    "classifier_mlp_hidden_dims": [128, 64],
    "classifier_mlp_dropout": 0.5,
    "learning_rate": 1e-4, 
    "weight_decay": 1e-4,
    "optimizer_type": "AdamW",
    "batch_size": 256,
    "epochs": 100,
    "early_stopping_patience": 5,
    "lr_scheduler_patience": 3,
    "lr_scheduler_factor": 0.2,
}

def get_model_config(model_type):
    model_type_lower = model_type.lower()
    if model_type_lower == 'mlp':
        return MLP_CONFIG
    elif model_type_lower == 'cnn':
        return CNN_CONFIG
    elif model_type_lower == 'bilstm':
        return BILSTM_CONFIG
    elif model_type_lower == 'hf_pretrained_sentiment_analysis_model':
        return HF_PRETRAINED_SENTIMENT_ANALYSIS_MODEL_CONFIG
    elif model_type_lower == 'peft_mpnet_mlp':
        return PEFT_MPNET_MLP_CONFIG
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# --- Create directories if they don't exist ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WANDB_LOCAL_SAVE_DIR, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Model save directory: {MODEL_SAVE_DIR}")
print(f"Submission directory: {SUBMISSION_DIR}")
print(f"WandB local runs directory: {WANDB_LOCAL_SAVE_DIR}")
