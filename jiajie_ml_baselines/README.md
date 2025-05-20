# Deep Learning Models

## Training Commands

### MLP
```
python train.py mlp --run_name refactored_MLP_Word2Vec
```

### 1DCNN

```
python train.py cnn --run_name refactored_CNN_FastText
```

### Bi-LSTM
```
python train.py bilstm --run_name refactored_BiLSTM_FastText
```

### PEFT Fine-tuned MPNet with MLP Head (LoRA)
```
python train.py peft_mpnet_mlp --run_name refactored_PEFT_MPNet_LoRA
```

## Inference Commands

### MLP
```
python inference.py --model_type_for_inference local_custom --model_path ./testing_checkpoints/best_mlp_refactored_MLP_Word2Vec.pth
```

### 1DCNN

```
python inference.py --model_type_for_inference local_custom --model_path ./testing_checkpoints/best_cnn_refactored_CNN_FastText.pth
```

### Bi-LSTM
```
python inference.py --model_type_for_inference local_custom --model_path ./testing_checkpoints/best_bilstm_refactored_BiLSTM_FastText.pth
```

### PEFT Fine-tuned MPNet with MLP Head (LoRA)
```
python inference.py \
        --model_type_for_inference peft_mpnet_mlp \
        --peft_adapter_path ./testing_checkpoints/best_lora_adapters_refactored_PEFT_MPNet_LoRA_all-mpnet-base-v2_R64_A128 \
        --peft_config_path ./testing_checkpoints/best_peft_config_refactored_PEFT_MPNet_LoRA_all-mpnet-base-v2_R64_A128_config.json \
        --peft_head_path ./testing_checkpoints/best_mlp_head_refactored_PEFT_MPNet_LoRA_all-mpnet-base-v2_R64_A128.pth
```

### Pre-trained Multilingual Sentiment Analysis Model (Zero-Shot)
```
python inference.py --model_type_for_inference hf_direct --hf_model_name NameOfAnotherHFSentimentModel
```

# Classic ML Methods
```
# Train and infer in one script
python TF_IDF_ML_methods.py
```