from typing import List, Callable, Optional

import re

# Since we determine the start of the answer, we don't need to check for the start tag
TAG_PATTERN = re.compile(r"(.*?)</think>.*?<answer>(.*?)</answer>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL)

LABEL_TO_VALUE = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

def create_format_reward_fn(log_fn: Optional[Callable[[str], None]] = None) -> Callable:
    def format(
        completions: List[str],
        target: List[str],
        prompts: List[str],
    ) -> List[float]:
        return format_reward(completions, target, prompts, log_fn=log_fn)
    
    return format

def create_correctness_reward_fn(log_fn: Optional[Callable[[str], None]] = None) -> Callable:
    def correctness(
        completions: List[str],
        target: List[str],
        prompts: List[str],
    ) -> List[float]:
        return correctness_reward(completions, target, prompts, log_fn=log_fn)
    
    return correctness

def format_reward(
    completions: List[str],
    target: List[str],
    prompts: List[str],
    *,
    log_fn: Optional[Callable[[str], None]] = None
) -> List[float]:
    """
    Assign a binary reward for each completion:
      1.0 if it matches the <think>…</think><answer>…</answer> format,
      0.0 otherwise.

    Args:
        completions: Generated outputs (one per prompt).
        labels:      Ground-truth labels (not used here, but could be for more complex logic).
        log_fn:      Optional callback to record debugging info (e.g. wandb.log).

    Returns:
        A list of float rewards (0.0 or 1.0), one per completion.
    """
    rewards: List[float] = []
    for comp, trgt, prmpt in zip(completions, target, prompts):

        match = TAG_PATTERN.search(comp)
        rewards.append(1.0 if match else 0.0)

        if log_fn:
            log_fn(f"FORMAT REWARD\nPROMPT:\n{prmpt})\nLABEL: {trgt}\nOUTPUT:\n{comp}\nREWARD: {rewards[-1]}")
    return rewards

def correctness_reward(
    completions: List[str],
    target: List[str],
    prompts: List[str],
    *,
    log_fn: Optional[Callable[[str], None]] = None
) -> List[float]:
    """
    Compute rewards based on the correctness of the <answer> tag in each completion.

    Reward formula:
      reward = max(0, 2 - abs(pred_value - true_value))
    which gives:
      - 2 if prediction exactly matches target
      - 1 if off by one category
      - 0 if off by two categories or tag missing/invalid

    Args:
        completions: Model-generated strings containing an <answer>…</answer> section.
        target:     Ground-truth labels matching the order of completions.
        prompts:     Original prompts (used only if log_fn is provided).
        log_fn:      Optional function for debug logging (receives a single string).

    Returns:
        List[float]: Numeric rewards, one per completion.
    """
    rewards: List[float] = []

    for comp, trgt, prompt in zip(completions, target, prompts):

        # Extract the answer text
        match = ANSWER_PATTERN.search(comp)
        if not match:
            rewards.append(0.0)
            continue

        answer_text = match.group(1).strip().lower()
        if answer_text not in LABEL_TO_VALUE:
            rewards.append(0.0)
            continue

        # Compute reward and append
        pred_val = LABEL_TO_VALUE[answer_text]
        true_val = LABEL_TO_VALUE[trgt]
        rewards.append(2.0 - abs(pred_val - true_val))
        
        # Optional debug log
        if log_fn:
            log_fn(f"CORRECTNESS REWARD\nPROMPT:\n{prompt}\nTARGET: {trgt}\nOUTPUT:\n{comp}\nAnser: {answer_text}\nREWARD: {rewards[-1]}")

    return rewards