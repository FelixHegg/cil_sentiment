import os
from transformers import TrainerCallback
import yaml
from transformers import AutoTokenizer

with open("rl_grpo/config.yaml", "r") as f:
    config = yaml.safe_load(f)

ALL_FILE = "completions.txt"
CORRECT_FILE = "correct_completions.txt"
log_dir = config["log_dir"]


def write(path: os.path, completion: str):
    with open(path, "a") as f:
        f.write(f"\n\n==============\n")
        f.write(completion)


def write_log(completion: str, correct: bool = False):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, ALL_FILE)
    path_correct = os.path.join(log_dir, CORRECT_FILE)
    if correct:
        write(path_correct, completion)
    else:
        write(path, completion)


class LoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        log = f"""\n\n\n\n 
      ----------------------- step {state.global_step} --------------------------
      \n\n\n\n"""
        write_log(log, False)


def generate_reasoning_prompt(text: str, label: str, tokenizer: AutoTokenizer) -> dict:
    """Generate individual prompt with reasoning."""
    r1_prefix = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. You first thinks about the reasoning "
                "process in the mind and then provides the user with the answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"You have to determine the sentiment of the following text: \n{text}"
                f"\nShow your analysis in <think> </think> tags. The final answer "
                f"goes inside <answer> </answer> tags. The answer can only be positive,"
                f" negative or neutral, but nothing else. Think step by step inside "
                f"<think> tags."
            ),
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>",
        },
    ]
    return {
        "prompt": tokenizer.apply_chat_template(
            r1_prefix, tokenize=False, continue_final_message=True
        ),
        "target": label,
    }
