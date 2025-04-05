import re
from rl_grpo.utils import write_log

def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>.

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
    Returns:
        list[float]: Reward scores
    """
    rewards = []

    for completion, label, prompt in zip(completions, target, kwargs["prompts"]):
      
      log = "prompt: \n" + prompt + "\n label: \n" + str(label) + "\n" + completion
      write_log(log, False)

      try:
        completion = "<think>" + completion        
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        match = re.search(regex, completion, re.DOTALL) 

        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)

    return rewards

def translate_label(label: str) -> int:
   """Translate label to numerical value."""
   if label == "positive":
      return 2
   elif label == "neutral":
      return 1
   elif label == "negative":
      return 0
   else:
      msg = f"Label: {label} is not valid"
      raise ValueError("")

def correctness_reward(completions, target, **kwargs):
    """
    Evaluate if the classified sentiment is correct.

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, label, prompt in zip(completions, target, kwargs["prompts"]):
      try:
        completion = "<think>" + completion
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards.append(0.0)
            continue

        answer = match.group(1).strip() # answer part

        if answer not in ["positive", "negative", "neutral"]:
           rewards.append(0)
           continue

        value_answer = translate_label(answer)
        value_label = translate_label(label)

        rewards.append(2 - abs(value_answer-value_label))
        
        if answer == label:
           log = "prompt: \n" + prompt + "\n answer: \n" + completion
           write_log(log, True)


      except Exception:
            rewards.append(0.0) 
    return rewards