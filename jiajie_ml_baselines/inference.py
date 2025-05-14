import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import json
import math

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from peft import PeftModel

import config
import data_loader as dl
import preprocessing as pp
import dataset as ds
import models as custom_models


def load_inference_config(config_json_path):
    """Loads inference configuration from a JSON file."""
    if config_json_path and os.path.exists(config_json_path):
        try:
            with open(config_json_path, 'r') as f: 
                inf_config = json.load(f)
            print(f"Loaded inference config from {config_json_path}")
            return inf_config
        except Exception as e: print(f"Warning: Could not load config file {config_json_path}: {e}")
    print(f"Warning: Inference config file not found or not provided at {config_json_path}.")
    return None

def run_local_model_inference(args):
    """Runs inference using a locally trained custom model (MLP, CNN, BiLSTM)."""
    model_path = args.model_path
    config_json_path = os.path.join(os.path.dirname(model_path), f"{os.path.splitext(os.path.basename(model_path))[0]}_config.json")
    print(f"Starting inference for locally trained model: {model_path}")
    inf_config = load_inference_config(config_json_path)
    if not inf_config: 
        print(f"Error: Could not load config for local model {model_path}. Exiting.")
        return

    try:
        model_type, emb_model_name, vec_strat = inf_config['model_type'], inf_config['embedding_model'], inf_config['vectorization_strategy']
        max_seq, num_cls, emb_dim = inf_config.get('max_seq_len_used'), inf_config['num_classes'], inf_config['embedding_dim']
        le = pp.LabelEncoder()
        le.classes_ = np.array(inf_config['label_encoder_classes'])
    except KeyError as e: 
        print(f"Error: Missing key {e} in local model config.")
        return

    emb_model_gensim = pp.load_embedding_model(emb_model_name)
    if emb_model_gensim.vector_size != emb_dim: 
        emb_dim = emb_model_gensim.vector_size
    test_df = dl.load_test_data()
    test_sentences = test_df['sentence'].astype(str).tolist()
    X_test_proc, _ = pp.preprocess_data_non_hf(test_sentences, vec_strat, emb_model_name, max_seq_len_non_hf=max_seq)
    X_test_tensor = torch.tensor(X_test_proc, dtype=torch.float32)
    model_instance = custom_models.get_model(model_type, num_cls, inf_config, embedding_dim=emb_dim)
    try:
        model_instance.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model_instance.to(config.DEVICE); model_instance.eval()
    except Exception as e: 
        print(f"Error loading local model state_dict: {e}")
        return

    all_preds = []
    test_ds = ds.SentimentDataset(X_test_tensor.numpy(), np.zeros(len(X_test_tensor)))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    with torch.no_grad():
        for inputs_b, _ in test_loader:
            outputs = model_instance(inputs_b.to(config.DEVICE))
            _, pred_b = torch.max(outputs.data, 1)
            all_preds.append(pred_b.cpu().numpy())
    preds_indices = np.concatenate(all_preds)
    try: 
        pred_labels = le.inverse_transform(preds_indices)
    except Exception as e: 
        print(f"Inverse transform error: {e}")
        pred_labels = [f"err_{i}" for i in preds_indices]
    sub_df = pd.DataFrame({'id': test_df.index, 'label': pred_labels})
    sub_fname = f"submission_local_{os.path.splitext(os.path.basename(model_path))[0]}.csv"
    sub_path = os.path.join(config.SUBMISSION_DIR, sub_fname)
    try: 
        sub_df.to_csv(sub_path, index=False)
        print(f"Local model predictions saved to {sub_path}")
    except Exception as e: 
        print(f"Error saving local model submission: {e}")

def run_huggingface_direct_inference(args):
    """Runs inference using a direct pre-trained Hugging Face sentiment model (zero-shot)."""
    hf_model_name_to_load = args.hf_model_name if args.hf_model_name else config.HF_PRETRAINED_SENTIMENT_MODEL_NAME
    print(f"Starting inference for direct HF model: {hf_model_name_to_load}")
    test_df = dl.load_test_data()
    test_sentences = test_df['sentence'].astype(str).tolist()
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name_to_load)
        model = AutoModelForSequenceClassification.from_pretrained(hf_model_name_to_load)
        model.to(config.DEVICE); model.eval()
    except Exception as e: 
        print(f"Error loading direct HF model/tokenizer: {e}")
        return
    hf_map = {0:"Very Negative",1:"Negative",2:"Neutral",3:"Positive",4:"Very Positive"}
    five2three = {"Very Negative":"negative","Negative":"negative","Neutral":"neutral","Positive":"positive","Very Positive":"positive"}
    preds_3cls = []
    num_batches = math.ceil(len(test_sentences) / args.batch_size)
    with torch.no_grad():
        for i in range(0, len(test_sentences), args.batch_size):
            batch_texts = test_sentences[i : i+args.batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(config.DEVICE)
            outputs = model(**inputs); idx_hf = torch.argmax(outputs.logits, dim=-1).tolist()
            for idx_val in idx_hf: preds_3cls.append(five2three.get(hf_map.get(idx_val, "Neutral"), "neutral"))
            if ((i // args.batch_size) + 1) % 10 == 0 or ((i // args.batch_size) + 1) == num_batches: 
                print(f" Processed direct HF batch {(i // args.batch_size) + 1}/{num_batches}")
    sub_df = pd.DataFrame({'id': test_df.index, 'label': preds_3cls})
    sub_fname = f"submission_hf_direct_{hf_model_name_to_load.split('/')[-1]}.csv"
    sub_path = os.path.join(config.SUBMISSION_DIR, sub_fname)
    try: 
        sub_df.to_csv(sub_path, index=False)
        print(f"Direct HF predictions saved to {sub_path}")
    except Exception as e: 
        print(f"Error saving direct HF submission: {e}")

def run_peft_model_inference(args):
    """Runs inference using a PEFT fine-tuned model (e.g., LoRA + MLP head)."""
    adapter_path, config_json_path, head_path = args.peft_adapter_path, args.peft_config_path, args.peft_head_path
    print(f"Starting PEFT inference. Adapters: {adapter_path}, Config: {config_json_path}, Head: {head_path}")
    inf_config = load_inference_config(config_json_path) # Pass direct path for PEFT config
    if not inf_config: 
        print(f"Error: Could not load PEFT config from {config_json_path}.")
        return
    try:
        hf_base_model_name, max_seq_len, num_classes = inf_config['hf_base_model_name'], inf_config['max_seq_len'], inf_config['num_classes']
        mlp_hidden_dims, mlp_dropout = inf_config.get('classifier_mlp_hidden_dims', [128,64]), inf_config.get('classifier_mlp_dropout', 0.5)
        le = pp.LabelEncoder(); le.classes_ = np.array(inf_config['label_encoder_classes'])
    except KeyError as e: 
        print(f"Error: Missing key {e} in PEFT config.")
        return

    try:
        base_model = AutoModel.from_pretrained(hf_base_model_name)
        peft_model_base = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e: 
        print(f"Error loading PEFT base/adapters: {e}") 
        return
    
    full_model = custom_models.PeftWithMLPHeadClassifier(peft_model_base, num_classes, mlp_hidden_dims, mlp_dropout)
    try:
        full_model.mlp_head.load_state_dict(torch.load(head_path, map_location=config.DEVICE))
        full_model.to(config.DEVICE); full_model.eval()
    except Exception as e: 
        print(f"Error loading PEFT MLP head: {e}")
        return

    test_df = dl.load_test_data(); test_sentences = test_df['sentence'].astype(str).tolist()
    hf_tokenizer = pp.load_hf_tokenizer(hf_base_model_name)
    all_preds = []
    num_batches = math.ceil(len(test_sentences) / args.batch_size)
    print(f"Making PEFT predictions in {num_batches} batches (max_len: {max_seq_len})...")
    with torch.no_grad():
        for i in range(0, len(test_sentences), args.batch_size):
            batch_texts = test_sentences[i : i + args.batch_size]
            inputs = hf_tokenizer(batch_texts, return_tensors="pt", truncation=True, padding='max_length', max_length=max_seq_len).to(config.DEVICE)
            outputs = full_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu().numpy())
            if ((i // args.batch_size) + 1) % 10 == 0 or ((i // args.batch_size) + 1) == num_batches: 
                print(f"  Processed PEFT batch {(i // args.batch_size) + 1}/{num_batches}")
    preds_indices = np.concatenate(all_preds)
    try: 
        pred_labels = le.inverse_transform(preds_indices)
    except Exception as e: 
        print(f"PEFT inverse transform error: {e}")
        pred_labels = [f"err_{i}" for i in preds_indices]
    
    sub_df = pd.DataFrame({'id': test_df.index, 'label': pred_labels})
    run_name = inf_config.get('run_name', 'peft_run')
    adapter_suffix = inf_config.get('hf_base_model_name', 'unknown').split("/")[-1] + f"_R{inf_config.get('lora_rank','na')}_A{inf_config.get('lora_alpha','na')}"
    sub_fname = f"submission_peft_{run_name}_{adapter_suffix}.csv"
    sub_path = os.path.join(config.SUBMISSION_DIR, sub_fname)
    try: sub_df.to_csv(sub_path, index=False); print(f"PEFT predictions saved to {sub_path}")
    except Exception as e: print(f"Error saving PEFT submission: {e}")

def main(cli_args):
    """Main function to route inference to the appropriate model-specific function."""
    if cli_args.model_type_for_inference == "local_custom":
        if not cli_args.model_path: parser.error("--model_path is required for 'local_custom'.")
        run_local_model_inference(cli_args)
    elif cli_args.model_type_for_inference == "hf_direct":
        if cli_args.hf_model_name is None: 
            cli_args.hf_model_name = config.HF_PRETRAINED_SENTIMENT_MODEL_NAME
        run_huggingface_direct_inference(cli_args)
    elif cli_args.model_type_for_inference == "peft_bert_mlp":
        if not all([cli_args.peft_adapter_path, cli_args.peft_config_path, cli_args.peft_head_path]):
            parser.error("--peft_adapter_path, --peft_config_path, and --peft_head_path are required for 'peft_bert_mlp' inference.")
        run_peft_model_inference(cli_args)
    else:
        parser.print_help()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for sentiment analysis models.")
    parser.add_argument("--model_type_for_inference", type=str, required=True,
                        choices=['local_custom', 'hf_direct', 'peft_bert_mlp'],
                        help="Specify the type of model for inference.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to .pth model file (for 'local_custom').")
    parser.add_argument("--hf_model_name", type=str, default=None, help=f"HF model name (for 'hf_direct', e.g., '{config.HF_PRETRAINED_SENTIMENT_MODEL_NAME}').")
    parser.add_argument("--peft_adapter_path", type=str, default=None, help="Path to PEFT adapter directory (for 'peft_bert_mlp').")
    parser.add_argument("--peft_config_path", type=str, default=None, help="Path to PEFT model's inference config JSON (for 'peft_bert_mlp').")
    parser.add_argument("--peft_head_path", type=str, default=None, help="Path to PEFT model's MLP head .pth (for 'peft_bert_mlp').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    args = parser.parse_args()
    main(args)



