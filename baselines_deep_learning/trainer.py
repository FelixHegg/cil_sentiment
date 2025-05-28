import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
import time
import wandb
import os
import json
import config

def calculate_l_score(y_true_indices, y_pred_indices, index_to_mae_label_map):
    """Calculates the L-Score based on MAE."""
    # Convert map keys to integers if they are strings from JSON loading
    # Also handle the case where index_to_mae_label_map might be empty or None
    if not index_to_mae_label_map:
        print("Warning: index_to_mae_label_map is empty or None. Cannot calculate L-score accurately.")
        return 0.0

    # Check if keys are already integers, if not, try to convert
    # Ensure keys exist before trying to access list(index_to_mae_label_map.keys())[0]
    map_int_keys = index_to_mae_label_map
    if index_to_mae_label_map and isinstance(list(index_to_mae_label_map.keys())[0], str):
        try:
            map_int_keys = {int(k): v for k, v in index_to_mae_label_map.items()}
        except ValueError:
            print("Warning: Could not convert all keys in index_to_mae_label_map to int. L-score might be inaccurate.")
            return 0.0
    try:
        # Map integer indices (0, 1, 2) to MAE labels (-1, 0, 1)
        mapped_true = np.vectorize(map_int_keys.get)(y_true_indices)
        mapped_pred = np.vectorize(map_int_keys.get)(y_pred_indices)

        if None in mapped_true or None in mapped_pred:
             # This can happen if an index from y_true_indices or y_pred_indices is not in map_int_keys
             print(f"Warning: Mapping resulted in None values. Check if all predicted/true indices are in map_int_keys.")
             print(f"Map keys: {list(map_int_keys.keys())}")
             print(f"Unique true indices: {np.unique(y_true_indices)}")
             print(f"Unique pred indices: {np.unique(y_pred_indices)}")
             # Count how many Nones to understand the extent
             none_in_true = np.sum(mapped_true == None) # Using == None because np.vectorize might return Python None
             none_in_pred = np.sum(mapped_pred == None)
             if none_in_true > 0 or none_in_pred > 0:
                 print(f"Number of Nones in mapped_true: {none_in_true}, in mapped_pred: {none_in_pred}")
                 return 0.0
            
    except Exception as e:
        print(f"Error during MAE label mapping: {e}")
        print(f"Map: {map_int_keys}, True unique: {np.unique(y_true_indices)}, Pred unique: {np.unique(y_pred_indices)}")
        return 0.0

    # Ensure no None values before MAE calculation if not caught above
    if any(x is None for x in mapped_true) or any(x is None for x in mapped_pred):
        print("Error: None values still present in mapped true/pred before MAE calculation. L-score calculation aborted.")
        return 0.0

    mae = mean_absolute_error(mapped_true, mapped_pred)
    return 0.5 * (2 - mae)

def evaluate(model, loader, criterion, device, index_to_mae_label_map, model_type):
    """Evaluates the model on the given data loader."""
    model.eval()
    total_loss = 0.0
    all_preds_indices, all_labels_indices = [], []
    with torch.no_grad():
        for batch_data in loader:
            if model_type == 'peft_mpnet_mlp':
                input_ids = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                labels_batch = batch_data['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else: # For older models (MLP, CNN, BiLSTM)
                inputs, labels_batch = batch_data
                inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                outputs = model(inputs)

            if isinstance(criterion, torch.nn.Module): criterion = criterion.to(device)
            loss = criterion(outputs, labels_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds_indices.append(predicted.cpu().numpy())
            all_labels_indices.append(labels_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    all_preds_indices = np.concatenate(all_preds_indices)
    all_labels_indices = np.concatenate(all_labels_indices)
    accuracy = 100 * (all_preds_indices == all_labels_indices).sum() / len(all_labels_indices)
    l_score = calculate_l_score(all_labels_indices, all_preds_indices, index_to_mae_label_map)
    return avg_loss, accuracy, l_score, all_labels_indices, all_preds_indices

def train_one_epoch(model, loader, optimizer, criterion, device, index_to_mae_label_map, model_type):
    """Performs one epoch of training."""
    model.train()
    running_loss = 0.0
    all_preds_indices, all_labels_indices = [], []
    for i, batch_data in enumerate(loader):
        optimizer.zero_grad()
        if model_type == 'peft_mpnet_mlp':
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels_batch = batch_data['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else: # For older models
            inputs, labels_batch = batch_data
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = model(inputs)

        if isinstance(criterion, torch.nn.Module): criterion = criterion.to(device)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds_indices.append(predicted.cpu().numpy())
        all_labels_indices.append(labels_batch.cpu().numpy())

    avg_loss = running_loss / len(loader)
    all_preds_indices = np.concatenate(all_preds_indices)
    all_labels_indices = np.concatenate(all_labels_indices)
    accuracy = 100 * (all_preds_indices == all_labels_indices).sum() / len(all_labels_indices)
    l_score = calculate_l_score(all_labels_indices, all_preds_indices, index_to_mae_label_map)
    return avg_loss, accuracy, l_score

def get_class_weights(y_train_indices, device):
    """Computes class weights for imbalanced datasets."""
    print("Calculating class weights...")
    unique_classes = np.unique(y_train_indices)
    if len(unique_classes) <= 1:
        print("Warning: Only one class found in training data for weight calculation. Using uniform weights.")
        return torch.ones(len(unique_classes) if len(unique_classes) > 0 else 1, dtype=torch.float32).to(device)
    try:
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_indices)
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"Class weights for classes {unique_classes}: {cw_tensor}")
        return cw_tensor
    except Exception as e:
        print(f"Error in get_class_weights: {e}. Using uniform weights for {len(unique_classes)} classes.")
        return torch.ones(len(unique_classes), dtype=torch.float32).to(device)

def train_model(model, model_type, train_loader, val_loader, optimizer, scheduler, device,
                index_to_mae_label_map, y_train_indices, run_name,
                label_encoder, inference_details):
    """Main training loop with validation, early stopping, WandB logging, and config saving."""
    model_cfg = config.get_model_config(model_type)
    epochs = inference_details.get('epochs', model_cfg.get('epochs', config.EPOCHS))
    early_stopping_patience = inference_details.get('early_stopping_patience', model_cfg.get('early_stopping_patience', config.EARLY_STOPPING_PATIENCE))
    
    run_suffix = "" # Initialize run_suffix
    if model_type == 'peft_mpnet_mlp':
        base_model_name_suffix = model_cfg.get('hf_base_model_name', 'unknown_base').split("/")[-1]
        lora_rank = model_cfg.get('lora_rank', 'Rna')
        lora_alpha = model_cfg.get('lora_alpha', 'Ana')
        run_suffix = f"{base_model_name_suffix}_R{lora_rank}_A{lora_alpha}" # Define run_suffix here
        best_lora_adapter_dir = os.path.join(config.MODEL_SAVE_DIR, f"best_lora_adapters_{run_name}_{run_suffix}")
        best_mlp_head_path = os.path.join(config.MODEL_SAVE_DIR, f"best_mlp_head_{run_name}_{run_suffix}.pth")
        best_model_path = {"adapters": best_lora_adapter_dir, "head": best_mlp_head_path}
        print(f"PEFT Adapters will be saved to: {best_lora_adapter_dir}")
        print(f"PEFT MLP Head will be saved to: {best_mlp_head_path}")
    else:
        best_model_filename = f"best_{model_type}_{run_name}.pth"
        best_model_path = os.path.join(config.MODEL_SAVE_DIR, best_model_filename)
        print(f"Best model will be saved to: {best_model_path}")

    class_weights_tensor = get_class_weights(y_train_indices, device)
    
    # Determine num_classes from model's final layer more robustly
    final_layer = None
    if hasattr(model, 'output_layer') and isinstance(model.output_layer, nn.Linear):
        final_layer = model.output_layer
    elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
         final_layer = model.fc
    elif hasattr(model, 'mlp_head') and isinstance(model.mlp_head, nn.Sequential) and \
         len(model.mlp_head) > 0 and isinstance(model.mlp_head[-1], nn.Linear):
         final_layer = model.mlp_head[-1]
    
    if final_layer is None:
        # Fallback if num_classes is in inference_details (e.g. from label_encoder)
        if 'num_classes' in inference_details:
            num_classes_model = inference_details['num_classes']
            print(f"Warning: Could not directly determine num_classes from model's final layer. Using num_classes from inference_details: {num_classes_model}")
        else:
            raise ValueError("Could not determine number of classes from model's final Linear layer or inference_details.")
    else:
        num_classes_model = final_layer.out_features


    if len(class_weights_tensor) != num_classes_model:
         print(f"Warning: Number of class weights ({len(class_weights_tensor)}) does not match model output classes ({num_classes_model}). Check label encoding and model definition. Using uniform weights.")
         criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
         criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)

    print(f"\n--- Starting Training: {model_type.upper()} Model ({run_name}) ---")
    print(f"Epochs: {epochs}, Batch Size: {inference_details.get('batch_size', config.BATCH_SIZE)}, Device: {device}")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc, train_l_score = train_one_epoch(
            model, train_loader, optimizer, criterion, device, index_to_mae_label_map, model_type
        )
        val_loss, val_acc, val_l_score, _, _ = evaluate(
            model, val_loader, criterion, device, index_to_mae_label_map, model_type
        )
        print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, L: {train_l_score:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, L: {val_l_score:.4f}')

        current_lr = optimizer.param_groups[0]['lr']
        if wandb and wandb.run: # Check if wandb.run is not None
            wandb.log({
                "epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc,
                "train_l_score": train_l_score, "val_loss": val_loss, "val_accuracy": val_acc,
                "val_l_score": val_l_score, "learning_rate": current_lr
            })
    
        scheduler.step(val_loss) # Assuming val_loss is the metric
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            try:
                if model_type == 'peft_mpnet_mlp':
                    os.makedirs(best_lora_adapter_dir, exist_ok=True)
                    model.peft.save_pretrained(best_lora_adapter_dir)
                    torch.save(model.mlp_head.state_dict(), best_mlp_head_path)
                    print(f'---> Val Loss decreased to {best_val_loss:.4f}. Saving PEFT adapters to {best_lora_adapter_dir} and MLP head to {best_mlp_head_path}')
                else:
                    torch.save(model.state_dict(), best_model_path)
                    print(f'---> Val Loss decreased to {best_val_loss:.4f}. Saving model to {best_model_path}')

                # Define inf_config_filename_base before using it
                inf_config_filename_base = f"best_{model_type}_{run_name}"
                if model_type == 'peft_mpnet_mlp':
                     inf_config_filename_base = f"best_peft_config_{run_name}_{run_suffix}"
                
                inf_config_path = os.path.join(config.MODEL_SAVE_DIR, f"{inf_config_filename_base}_config.json")
                config_to_save = inference_details.copy()
                config_to_save["label_encoder_classes"] = label_encoder.classes_.tolist()
                config_to_save["index_to_mae_label_map"] = {str(k): v for k, v in index_to_mae_label_map.items()}
                with open(inf_config_path, 'w') as f: json.dump(config_to_save, f, indent=4)
                print(f"Inference config saved to {inf_config_path}")

            except Exception as e: print(f"Error during model/config saving: {e}")

            if wandb and wandb.run: # Check if wandb.run is not None
                wandb.run.summary["best_val_l_score_at_best_loss"] = val_l_score
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch + 1
        else:
            epochs_no_improve += 1
            print(f'Validation Loss did not improve. Counter: {epochs_no_improve}/{early_stopping_patience}')
        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs.'); break

    training_duration = time.time() - start_time
    print(f"\nTraining finished in {training_duration:.2f} seconds.")
    if wandb and wandb.run: wandb.run.summary["training_time_seconds"] = training_duration

    print(f"\nLoading best model for final evaluation...")
    try:
        final_model_instance = None
        if model_type == 'peft_mpnet_mlp':
            from models import PeftWithMLPHeadClassifier
            from transformers import AutoModel
            from peft import PeftModel
            
            hf_base_model_name_for_load = inference_details.get('hf_base_model_name', config.HF_BASE_MODEL_FOR_PEFT)
            base_model_for_eval = AutoModel.from_pretrained(hf_base_model_name_for_load)
            peft_model_for_eval = PeftModel.from_pretrained(base_model_for_eval, best_lora_adapter_dir)
            
            final_model_instance = PeftWithMLPHeadClassifier(
                peft_base_model=peft_model_for_eval,
                num_classes=inference_details['num_classes'],
                mlp_hidden_dims=inference_details.get('classifier_mlp_hidden_dims', [128, 64]),
                mlp_dropout=inference_details.get('classifier_mlp_dropout', 0.5)
            ).to(device)
            final_model_instance.mlp_head.load_state_dict(torch.load(best_mlp_head_path, map_location=device))
            print("Best PEFT model (adapters + head) loaded for final evaluation.")
        else:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            final_model_instance = model
            print(f"Best model {model_type} loaded for final evaluation.")

        final_model_instance.eval()
        final_val_loss, final_val_acc, final_val_l_score, _, _ = evaluate(
            final_model_instance, val_loader, criterion, device, index_to_mae_label_map, model_type
        )
        print(f'\nFinal Validation Metrics (best model): Loss: {final_val_loss:.4f}, Acc: {final_val_acc:.2f}%, L-Score: {final_val_l_score:.4f}')
        if wandb and wandb.run:
            wandb.run.summary["final_val_loss"] = final_val_loss
            wandb.run.summary["final_val_accuracy"] = final_val_acc
            wandb.run.summary["final_val_l_score"] = final_val_l_score
    except Exception as e:
        print(f"Error during final model evaluation: {e}")
        import traceback; traceback.print_exc()
    return best_model_path


