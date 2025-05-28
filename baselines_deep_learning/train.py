import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import wandb
import os

import config
import data_loader as dl
import preprocessing as pp
import dataset as ds
import models as m
import trainer as t

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    print(f"Starting training: {args.model_type}, Run: {args.run_name}, Device: {config.DEVICE}")

    training_df = dl.load_training_data()
    sentences = training_df['sentence'].astype(str).tolist()
    labels_text = training_df['label'].tolist()
    print(f"Loaded {len(sentences)} examples. Label dist:\n{pd.Series(labels_text).value_counts(normalize=True)}")

    label_encoder, index_to_mae_label_map, num_classes = pp.get_label_utils(labels_text)
    y_encoded = label_encoder.transform(labels_text)

    model_specific_config = config.get_model_config(args.model_type)
    run_details = {
        "model_type": args.model_type, "run_name": args.run_name,
        "num_classes": num_classes, "validation_split": config.VALIDATION_SPLIT,
        "random_state": config.RANDOM_STATE,
        "batch_size": model_specific_config.get('batch_size', config.BATCH_SIZE),
        **model_specific_config
    }

    if args.model_type == 'peft_mpnet_mlp':
        hf_tokenizer_name = model_specific_config['hf_base_model_name']
        max_seq_len = model_specific_config['max_seq_len']
        input_ids, attention_masks, _ = pp.preprocess_data_hf(sentences, hf_tokenizer_name, max_seq_len)
        train_idx, val_idx = train_test_split(np.arange(len(y_encoded)), test_size=config.VALIDATION_SPLIT, stratify=y_encoded, random_state=config.RANDOM_STATE)
        X_train_ids, X_val_ids = input_ids[train_idx], input_ids[val_idx]
        X_train_masks, X_val_masks = attention_masks[train_idx], attention_masks[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        train_dataset = ds.TransformerSentimentDataset(X_train_ids, X_train_masks, y_train)
        val_dataset = ds.TransformerSentimentDataset(X_val_ids, X_val_masks, y_val)
        run_details["hf_tokenizer_name"] = hf_tokenizer_name
        run_details["max_seq_len_used"] = max_seq_len
        model_init_embedding_dim = None
    else:
        embedding_model_name = model_specific_config['embedding_model']
        vectorization_strategy = model_specific_config['vectorization_strategy']
        X_processed, max_seq_len_used = pp.preprocess_data_non_hf(sentences, vectorization_strategy, embedding_model_name)
        X_train, X_val, y_train, y_val = train_test_split(X_processed, y_encoded, test_size=config.VALIDATION_SPLIT, stratify=y_encoded, random_state=config.RANDOM_STATE)
        train_dataset = ds.SentimentDataset(X_train, y_train)
        val_dataset = ds.SentimentDataset(X_val, y_val)
        run_details["embedding_model"] = embedding_model_name
        run_details["vectorization_strategy"] = vectorization_strategy
        run_details["max_seq_len_used"] = max_seq_len_used
        temp_emb_model = pp.load_embedding_model(embedding_model_name)
        model_init_embedding_dim = temp_emb_model.vector_size
        run_details["embedding_dim"] = model_init_embedding_dim

    print(f"Train samples: {len(y_train)}, Val samples: {len(y_val)}")
    current_batch_size = run_details["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    model = m.get_model(args.model_type, num_classes, model_specific_config, embedding_dim=model_init_embedding_dim)
    model.to(config.DEVICE)

    learning_rate = model_specific_config.get('learning_rate', config.LEARNING_RATE)
    weight_decay = model_specific_config.get('weight_decay', config.WEIGHT_DECAY)
    optimizer_type = model_specific_config.get('optimizer_type', 'Adam')

    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print(f"Using optimizer: {optimizer_type}")

    lr_scheduler_factor = model_specific_config.get('lr_scheduler_factor', config.LR_SCHEDULER_FACTOR)
    lr_scheduler_patience = model_specific_config.get('lr_scheduler_patience', config.LR_SCHEDULER_PATIENCE)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        verbose=True
    )
    print(f"Using ReduceLROnPlateau scheduler for {args.model_type} with factor {lr_scheduler_factor} and patience {lr_scheduler_patience}.")

    wandb_config_cleaned = {k: v for k, v in run_details.items() if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))}
    print("Initializing WandB...")
    try:
        wandb.init(project=config.WANDB_PROJECT, 
                   name=args.run_name, 
                   config=wandb_config_cleaned, 
                   dir=config.WANDB_LOCAL_SAVE_DIR)
        if wandb.run:
             print(f"WandB initialized successfully. Run ID: {wandb.run.id}. Local files in: {config.WANDB_LOCAL_SAVE_DIR}")
        else:
            print("WandB init called but no run object created. Logging will be skipped.")
    except Exception as e:
        print(f"WandB init failed: {e}. Training will continue without logging.")

    try:
        best_model_path_info = t.train_model(
            model, args.model_type, train_loader, val_loader, optimizer, scheduler, config.DEVICE,
            index_to_mae_label_map, y_train, args.run_name, label_encoder, run_details
        )
        print(f"\nTraining complete. Best model/adapters saved. Info: {best_model_path_info}")
    except Exception as e:
        print(f"\nError during training: {e}"); 
        import traceback
        traceback.print_exc()
    finally:
        if wandb.run is not None:
            print("Finishing Wandb run...")
            wandb.finish()
    print("\nScript finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model.")
    parser.add_argument("model_type", type=str, choices=['mlp', 'cnn', 'bilstm', 'peft_mpnet_mlp'],
                        help="Type of model to train.")
    parser.add_argument("--run_name", type=str, required=True, help="Unique name for this run.")
    args = parser.parse_args()
    main(args)
