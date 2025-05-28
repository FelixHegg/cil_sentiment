# Next-Token Prediction Baseline

This baseline implements a standard **next-token prediction** setup for sentiment classification, using a pretrained language model without inference-time reasoning. It utilizes the [SFT pipeline of Hugginface](https://huggingface.co/docs/trl/en/sft_trainer)

---

## Configuration

The training is configured with the following parameters using the config file:

- **Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Data path**: `data/`
- **Max sequence length**: `3072`
- **Epochs**: `1`
- **Learning rate**: `5e-5`
- **Optimizer**: `adamw_torch`
- **Batch size per device**: `4`
- **Gradient accumulation steps**: `16`
- **Save steps**: `50`
- **Output directory**: `outputs/`

### LoRA & Quantization

- **4-bit quantization**: Enabled
- **LoRA rank**: `64`
- **LoRA alpha**: `32`
- **LoRA dropout**: `0.1`
- **Target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

---

## Training and Inference

Run the training script from the base directory:

```bash
python -m baselines_next_token_pred.train
```

```bash
python -m baselines_next_token_pred.inference
```
