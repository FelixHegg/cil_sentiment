import os
from transformers import TrainerCallback
import yaml

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
    os.makedirs(log_dir, exist_ok = True)
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