import os

import argparse
from tqdm import tqdm
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.utils import prepare_data

from src.reward import ANSWER_PATTERN


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)

    # Load config
    with open(args.result_dir + "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=True)

    # Create model config
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if config["load_in_4bit"]:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model"],
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config["model"],
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    # 2) Load your LoRA adapters *also* on CPU
    model = PeftModel.from_pretrained(
        base_model,
        os.path.join(args.result_dir, args.ckpt_name),
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    _, data_eval = prepare_data(tokenizer, config["data_path"])
    print(f"Eval dataset size: {len(data_eval)}")

    
    # Create output file
    output_file = os.path.join(args.result_dir, args.output_file)
    with open(output_file, "w", encoding="utf-8") as f, torch.no_grad():
        f.write("id,label\n")

        counter = 0
        for i in tqdm(range(0, len(data_eval), args.batch_size), desc="Inference"):
            batch = data_eval[i : i + args.batch_size]
            enc = tokenizer(
                batch["prompt"],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out_ids = model.generate(
                **enc,
                max_new_tokens=config["max_completion_length"],
                pad_token_id=tokenizer.eos_token_id,
            )

            # remove prompt tokens
            prompt_len = enc["input_ids"].shape[1]
            gen_ids = out_ids[:, prompt_len:]

            # decode and extract answers
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for text in decoded:
                answer = extract_answer(text)
                f.write(f"{counter},{answer}\n")
                counter += 1

def extract_answer(text: str) -> str:
    """Get the content inside <answer>…</answer>, or 'neutral' if missing."""
    m = ANSWER_PATTERN.search(text)
    answer = m.group(1).strip().lower() if m else "neutral"

    if answer not in ["positive", "negative", "neutral"]:
        answer = "neutral"
    
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Inference")
    parser.add_argument("--result-dir", type=str, required=True, help="Directory with the model and config.")
    parser.add_argument("--ckpt-name", type=str, required=True, help="Checkpoint name.")

    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--output-file", type=str, default="predictions.csv", help="Output file for predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    args = parser.parse_args()
    main(args)