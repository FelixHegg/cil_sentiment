from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from rl_grpo.utils import generate_reasoning_prompt
from unsloth import FastLanguageModel
import os
import yaml
import torch
from rl_grpo.reward_functions import extract_answer
import random


class Inference:
    """Load checkpoint and make predictions on test data."""

    def __init__(self, config: dict) -> None:
        """Load data and checkpoint."""
        self._config = config
        self._tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        self._data = self._load_dataset(config["path_data"])
        self._model = self._load_model()  # TODO: Have to load from a checkpoint !!!!!

    def _load_dataset(self, data_path: str) -> DatasetDict:
        """Load and format dataset."""
        dataset = load_dataset(
            "csv",
            data_files={"test": os.path.join(data_path, "test.csv")},
        )

        dataset = dataset["test"].map(
            lambda x: generate_reasoning_prompt(
                x["sentence"], x["label"], self._tokenizer
            )
        )
        return dataset

    def _load_model(self):
        """Load the fine-tuned model checkpoint."""
        model_cfg = self._config["model"]
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_cfg["name"],
            max_seq_length=model_cfg["max_seq_length"],
            load_in_4bit=model_cfg["load_in_4bit"],
            fast_inference=model_cfg["fast_inference"],
            max_lora_rank=model_cfg["lora_rank"],
            gpu_memory_utilization=model_cfg["gpu_memory_utilization"],
        )

        # Load PEFT adapter weights from the output_dir
        output_dir = self._config["training"]["output_dir"]
        model = FastLanguageModel.load_peft_model(model, output_dir)

        model.eval()
        return model

    def _inference_one_sample(self, prompt: str) -> dict:
        """Run inference on one sample and extract result."""
        tokenized = self._tokenizer(
            prompt,
            return_tensor="pt",
            truncation=True,
            max_length=self._config["model"]["max_seq_length"],
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **tokenized,
                max_new_tokens=self._config["training"]["max_completion_length"],
                do_sample=True,
            )
            decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer = extract_answer(decoded)
        if answer == "error":
            answer = "neutral"

        return {"label": answer}

    def run(self, submission_name: str = None) -> None:
        """Run inference on test data and save as submission file."""
        self._data["test"].map(lambda x: self._inference_one_sample(x["sentence"]))
        submission_dataset = self._data["test"].remove_columns(
            [
                col
                for col in self._data["test"].column_names
                if col not in ["id", "label"]
            ]
        )

        if not submission_name:
            submission_name = f"unnamed_submission_{random.randint(1, 10000)}.csv"
        submission_path = os.join(self._config["path_data"], submission_name)

        submission_dataset.to_csv(submission_path)


if __name__ == "__main__":
    with open("rl_grpo/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    inference = Inference(config=config)
    inference.run()
