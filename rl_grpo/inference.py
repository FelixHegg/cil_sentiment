from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from rl_grpo.utils import generate_reasoning_prompt
from unsloth import FastLanguageModel
import os
import yaml
import torch
from rl_grpo.reward_functions import extract_answer
import random
import time
import unsloth, importlib.metadata as im

print("Unsloth version:", im.version("unsloth"))


class Inference:
    """Load checkpoint and make predictions on test data."""

    def __init__(self, config: dict) -> None:
        """Load data and checkpoint."""
        self._config = config
        self._tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["name"],
            padding_side="left",
        )
        self._data = self._load_dataset(config["path_data"])
        self._model = self._load_model()

    def _load_dataset(self, data_path: str) -> DatasetDict:
        """Load and format dataset."""
        dataset = load_dataset(
            "csv",
            data_files={"test": os.path.join(data_path, "test.csv")},
        )

        dataset = dataset["test"].map(
            lambda x: generate_reasoning_prompt(x["sentence"], "", self._tokenizer)
        )
        return dataset

    def _load_model(self):
        """Load the fine-tuned model checkpoint."""
        model_cfg = self._config["model"]

        model, _ = FastLanguageModel.from_pretrained(
            model_name=os.path.join(
                self._config["training"]["output_dir"],
                self._config["inference"]["checkpoint_name"],
            ),
            max_seq_length=model_cfg["max_seq_length"],
            load_in_4bit=model_cfg["load_in_4bit"],
            fast_inference=model_cfg["fast_inference"],
            gpu_memory_utilization=self._config["inference"]["gpu_memory_utilization"],
        )

        model = FastLanguageModel.for_inference(model)
        model.eval()
        return model

    def _inference_one_sample(self, prompt: str) -> dict:
        """Run inference on one sample and extract result."""
        tokenized = self._tokenizer(
            prompt,
            return_tensors="pt",
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

        return {"pred": answer}

    def _inference_batch(self, prompts: list) -> dict:
        """Run inference on one sample and extract result."""
        tokenized = self._tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=self._config["model"]["max_seq_length"],
            padding=True,
        ).to(self._model.device)
        tokenized = {k: v.to(self._model.device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs: torch.Tensor = self._model.generate(
                **tokenized,
                max_new_tokens=self._config["training"]["max_completion_length"],
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
            )  # shape (BS, SEQ_LEN)
            decoded: list[str] = self._tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            print(f"decoded: {decoded}")

        answers = []
        for i in decoded:
            answer = extract_answer(i)
            if answer == "error":
                answer = "neutral"
            answers.append(answer)

        return {"pred": answers}

    def run(self, submission_name: str = None) -> None:
        """Run inference on test data and save as submission file."""
        # self._data = self._data.map(lambda x: self._inference_one_sample(x["prompt"]))
        self._data = self._data.map(
            lambda x: self._inference_batch(x["prompt"]),
            batched=True,
            batch_size=self._config["inference"]["batch_size"],
        )

        submission_dataset = self._data.remove_columns(
            [col for col in self._data.column_names if col not in ["id", "pred"]]
        )

        if not submission_name:
            submission_name = f"unnamed_submission_{time.time()}.csv"
        submission_path = os.path.join(self._config["path_data"], submission_name)

        submission_dataset.to_csv(submission_path)


if __name__ == "__main__":
    with open("rl_grpo/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    inference = Inference(config=config)
    inference.run()
