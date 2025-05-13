from typing import Optional
import os

import logging

from transformers import AutoTokenizer, TrainerCallback
from datasets import load_dataset


class Logger(TrainerCallback):
    def __init__(self, logging_dir: Optional[str]=None, verbose: int=1):
        verbose_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

        handlers = [logging.StreamHandler()]
        if logging_dir is not None:
            handlers.append(logging.FileHandler(os.path.join(logging_dir, "log.txt")))

        logging.basicConfig(
            level=verbose_map.get(verbose, logging.INFO),
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )
        self.logger = logging.getLogger(__name__)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)

    # def on_step_end(self, args, state, control, **kwargs):#TODO: print more info
    #     self.info(f"----------------------- step {state.global_step} --------------------------")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = logs.get("loss") or 0.0
        format = logs.get("rewards/format") or 0.0
        correctness = logs.get("rewards/correctness") or 0.0
        reward_std = logs.get("reward_std")
        grad_norm = logs.get("grad_norm") or 0.0
            
        self.info(f"Step {state.global_step}: loss={loss:.5f}, format={format:.1f}, correctness={correctness:.1f}, reward_std={reward_std:.2f}, grad_norm={grad_norm:.5f}")


def generate_reasoning_prompt(tokenizer: AutoTokenizer, text: str) -> dict:
    """Generate individual prompt with reasoning."""
    r1_prefix = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. You first thinks about the reasoning "
                "process and then provides the user with the answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Determine the sentiment of the following text:\n{text}"
                "\nShow your analysis in <think></think> tags, and put the final answer "
                "(positive/negative/neutral) in <answer></answer> tags."
            ),
        },
        {
            "role": "assistant",
            "content": "Let me think step by step.\n<think>",
        },
    ]

    return tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True)


def prepare_data(tokenizer, data_path: str):
    """Load and format dataset."""
    dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(data_path, "training.csv"),
            "test": os.path.join(data_path, "test.csv"),
        },
    )

    def map_fn(x):
        prompt = generate_reasoning_prompt(tokenizer, x["sentence"])
        return {"prompt": prompt, "target": x["label"]}

    train_ds = dataset["train"].map(map_fn, remove_columns=dataset["train"].column_names)
    eval_ds = dataset["test"].map(map_fn, remove_columns=dataset["test"].column_names)
    return train_ds, eval_ds