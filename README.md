# Sentiment Classification through Inference-Time Reasoning

This repository contains a training script for fine-tuning large language models using **GRPO** (Generative Reinforcement Policy Optimization). It supports LoRA-based fine-tuning and uses the Hugging Face Transformers library.

## 🚀 Features

- Supports any Hugging Face model (default: `Qwen/Qwen2.5-3B-Instruct`)
- LoRA integration for efficient fine-tuning
- 4-bit quantization (optional)
- Custom training hyperparameters
- Generation-based optimization with GRPO
- Logging, checkpointing, and resume support

---

## 🧠 Usage

### 🛠️ Command Line Arguments

| Argument                        | Type      | Default                           | Description |
|--------------------------------|-----------|-----------------------------------|-------------|
| `--results-dir`                | `str`     | -                                 | Directory to save results. |
| `--data-path`                  | `str`     | -                                 | Path to the dataset (must be compatible with your model). |
| `--seed`                       | `int`     | `42`                              | Random seed for reproducibility. |
| `--verbose`                    | `int`     | `1`                               | Verbosity level (0: silent, 1: info, 2: debug). |
| `--model`                      | `str`     | `Qwen/Qwen2.5-3B-Instruct`        | Model name or path. |
| `--save_steps`                 | `int`     | `200`                             | Number of steps between checkpoints. |
| `--load_in_4bit`               | `flag`    | `False`                           | Load model in 4-bit quantized format. |
| `--lora_rank`                  | `int`     | `16`                              | Rank for LoRA adapters. |
| `--target_modules`             | `list`    | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Target modules for LoRA adaptation. |
| `--lora_alpha`                 | `int`     | `16`                              | Alpha parameter for LoRA. |
| `--lora_dropout`               | `float`   | `0.1`                             | Dropout rate for LoRA layers. |

### 🧪 Training Parameters

| Argument                        | Type      | Default  | Description |
|--------------------------------|-----------|----------|-------------|
| `--learning_rate`              | `float`   | `2e-4`   | Learning rate. |
| `--adam_beta1`                | `float`   | `0.9`    | Adam optimizer beta1. |
| `--adam_beta2`                | `float`   | `0.999`  | Adam optimizer beta2. |
| `--weight_decay`              | `float`   | `0.01`   | Weight decay. |
| `--warmup_ratio`              | `float`   | `0.0`    | Warmup ratio for LR scheduler. |
| `--max_grad_norm`             | `float`   | `1.0`    | Max gradient clipping norm. |
| `--lr_scheduler_type`         | `str`     | `constant` | LR scheduler type (e.g., linear, cosine, constant). |
| `--optim`                     | `str`     | `adamw_torch` | Optimizer type. |
| `--logging_steps`             | `int`     | `1`      | Logging frequency in steps. |
| `--gradient_accumulation_steps` | `int`   | `1`      | Gradient accumulation steps. |
| `--per_device_train_batch_size` | `int`   | `2`      | Training batch size per device. |

### 🌀 GRPO Parameters

| Argument                        | Type      | Default | Description |
|--------------------------------|-----------|---------|-------------|
| `--num_generations`           | `int`     | `2`     | Number of generations per prompt (≥ 2). |
| `--max_prompt_length`         | `int`     | `256`   | Maximum length of prompts. |
| `--max_completion_length`     | `int`     | `256`   | Maximum length of completions. |
| `--max_steps`                 | `int`     | `1000`  | Maximum number of training steps. |

### 🔁 Resume Training

| Argument         | Type   | Default | Description |
|------------------|--------|---------|-------------|
| `--continue-from` | `str` | `None`  | Path to checkpoint to resume training from. |

---

## 📦 Example

```bash
python train.py \
  --data-path ./data/train.json \
  --results-dir ./results \
  --model Qwen/Qwen2.5-3B-Instruct \
  --load_in_4bit \
  --lora_rank 16 \
  --learning_rate 2e-4 \
  --num_generations 4 \
  --max_steps 1000
