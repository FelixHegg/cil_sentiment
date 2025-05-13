import os
from glob import glob

import yaml
import argparse
from functools import partial

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.utils import Logger, prepare_data
from src.reward import create_format_reward_fn, create_correctness_reward_fn

def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup experiment directory
    exp_dir = setup_experiment(args.model, args.results_dir)
    logger = Logger(exp_dir, verbose=args.verbose)
    logger.info(f"using device {device}")
    logger.info(f"experiment directory created at {exp_dir}")

    # Save config
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    data_train, data_eval = prepare_data(tokenizer, args.data_path)
    logger.info(f"train dataset size: {len(data_train)}")
    logger.info(f"eval dataset size: {len(data_eval)}")

    # Load model
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    
    # Apply LoRA
    peft_config = peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_rank,
        target_modules=args.target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )  

    # Load model with LoRA
    model = get_peft_model(base_model, peft_config)
    model.train()

    # Train model
    training_config = GRPOConfig(
        use_vllm=False,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        logging_steps=args.logging_steps,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        report_to="wandb",
        output_dir=exp_dir,
        log_completions=False,
    )

    reward_format = create_format_reward_fn(log_fn=logger.debug)
    reward_correctness = create_correctness_reward_fn(log_fn=logger.debug)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_format, reward_correctness],
        args=training_config,
        train_dataset=data_train,
    )
    trainer.add_callback(logger)

    logger.info("Starting training...")
    trainer.train()

def setup_experiment(model_name: str, results_dir: os.PathLike):
    """Create an experiment directory for the current run."""

    # Make results directory
    os.makedirs(results_dir, exist_ok=True)

    experiment_index = len(glob(os.path.join(results_dir, "*")))
    model_string_name = model_name.replace("/", "-")
    experiment_dir = os.path.join(results_dir, f"{experiment_index:03d}-{model_string_name}")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    # Make experiment directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    return experiment_dir

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a model with GRPO.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0: silent, 1: info, 2: debug).")

    # Model parameters
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model name or path.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save steps.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit.")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--target_modules", type=list, default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], help="Target modules for LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer type.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device for training.")

    # GRPO parameters
    parser.add_argument("--num_generations", type=int, default=2, help="Number of generations (>= 2).")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Max prompt length.")
    parser.add_argument("--max_completion_length", type=int, default=256, help="Max completion length.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max training steps.")

    args = parser.parse_args()

    main(args)