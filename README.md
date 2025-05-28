# Sentiment Classification through Inference-Time Reasoning

This repository contains a training script for fine-tuning large language models using **GRPO** (Group Relative Policy Optimization). It supports LoRA-based fine-tuning and uses the Hugging Face Transformers library.


## 🚀 Key Features

- ✅ Works with any Hugging Face-compatible language model (default: `Qwen/Qwen2.5-3B-Instruct`)
- ⚡ Efficient fine-tuning using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685)
- 🧠 GRPO: Group Relative Policy Optimization for generation-based learning
- 🧊 Optional 4-bit quantization for memory-efficient training
- 🎯 Fully configurable training hyperparameters
- 💾 Checkpointing and resume training support
- 📊 Verbose logging for debugging and tracking

## 🧪 Baselines

We provide several baselines, each located in its own subfolder with a dedicated `README.md` for usage instructions:

- **Deep Learning** (`baselines_deep_learning/`): Includes `MPNet + MLP + LoRA`, `Multilingual BERT`, `Bi-LSTM`, `1D CNN`, `MLP (Word2Vec)`, and `SVM`.
- **Next-Token Prediction** (`baselines_next_token_pred/`): Standard language modeling without reasoning.
- **Rule-Based** (`baselines_rule_based/`): Includes VADER, TextBlob, and a script for generating the human baseline.

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
  --data-path /data/train.json \
  --results-dir /results \
  --model Qwen/Qwen2.5-3B-Instruct \
  --load_in_4bit \
  --lora_rank 16 \
  --learning_rate 2e-4 \
  --num_generations 4 \
  --max_steps 1000

## 🔍 Inference

After training, you can run inference using the provided script to generate predictions from a saved model checkpoint.

### 🛠️ Arguments

| Argument         | Type   | Default             | Description                                      |
|------------------|--------|---------------------|--------------------------------------------------|
| `--result-dir`   | `str`  | _required_          | Directory containing the model and configuration. |
| `--ckpt-name`    | `str`  | _required_          | Name of the checkpoint to load.                  |
| `--batch-size`   | `int`  | `16`                | Batch size for inference.                        |
| `--output-file`  | `str`  | `predictions.csv`   | File to save the generated predictions.          |
| `--seed`         | `int`  | `42`                | Random seed for reproducibility.                 |

### 🧪 Example

```bash
python inference.py \
  --result-dir /results/004-Qwen-Qwen2.5-3B-Instruct \
  --ckpt-name checkpoint-1000 \
  --batch-size 32 \
  --output-file final_predictions.csv
