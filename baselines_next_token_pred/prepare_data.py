from transformers import AutoTokenizer, TrainerCallback
from datasets import load_dataset
import os


def generate_sft_prompt(tokenizer: AutoTokenizer, text: str, label: str) -> dict:
    """Generate prompt with label for SFT."""
    prompt = [
        {
            "role": "system",
            "content": ("You are a helpful assistant."),
        },
        {
            "role": "user",
            "content": (
                f"Determine the sentiment of the following text:\n{text}"
                "\n Answer in one word. The answer can only be positive, negative, or neutral."
            ),
        },
    ]
    completion = [
        {
            "role": "assistant",
            "content": f"{label}",
        }
    ]

    return {
        "prompt": tokenizer.apply_chat_template(prompt, tokenize=False),
        "completion": tokenizer.apply_chat_template(completion, tokenize=False),
    }


def generate_eval_prompt(tokenizer: AutoTokenizer, text: str) -> dict:
    """Generate prompt without label for evaluation."""
    sample = [
        {
            "role": "system",
            "content": ("You are a helpful assistant."),
        },
        {
            "role": "user",
            "content": (
                f"Determine the sentiment of the following text:\n{text}"
                "\n Answer in one word. The answer can only be positive, negative, or neutral."
            ),
        },
        {
            "role": "assistant",
            "content": "",
        },
    ]

    return {
        "text": tokenizer.apply_chat_template(
            sample, tokenize=False, continue_final_message=True
        )
    }


def prepare_data(tokenizer, data_path: str):
    """Load and format dataset."""
    dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(data_path, "training.csv"),
            "test": os.path.join(data_path, "test.csv"),
        },
    )

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    train_ds = dataset["train"].map(
        lambda x: generate_sft_prompt(tokenizer, x["sentence"], x["label"]),
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
    )
    eval_ds = dataset["test"].map(
        lambda x: generate_eval_prompt(tokenizer, x["sentence"]),
        remove_columns=dataset["test"].column_names,
        load_from_cache_file=False,
    )
    # train_ds = train_ds.map(tokenize_function, batched=True)
    # eval_ds = eval_ds.map(tokenize_function, batched=True)
    print("Train Datasetttt")
    print(len(train_ds))
    return train_ds, eval_ds
