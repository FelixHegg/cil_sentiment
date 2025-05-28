import torch
from datasets import load_dataset, DatasetDict
from peft import prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import yaml
import os
import time


class Inference:
    """Load checkpoint and make predictions on test data."""

    def __init__(self, config: dict) -> None:
        """Load data and checkpoint."""
        self._config = config
        self._tokenizer = AutoTokenizer.from_pretrained(config["model"])
        self._data = self._load_dataset(config["path_data"])
        self._model = self._load_model()

    def _load_dataset(self, data_path: str) -> DatasetDict:
        """Load and format dataset."""
        dataset = load_dataset(
            "csv",
            data_files={"test": os.path.join(data_path, "test.csv")},
        )

        dataset = dataset["test"].map(lambda x: self._generate_prompt(x["sentence"]))
        return dataset

    def _generate_prompt(self, text: str) -> dict:
        """Generate prompt with proper format."""
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

        return {"prompt": self._tokenizer.apply_chat_template(prompt, tokenize=False)}

    def _load_model(self) -> AutoModelForCausalLM:
        """Load model."""
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if self._config["load_in_4bit"]:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self._config["model"],
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            base_model = prepare_model_for_kbit_training(base_model)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self._config["model"],
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable()

        model = PeftModel.from_pretrained(base_model, self._config["lora_path"])
        model.eval()
        return model

    def _inference_batch(self, prompts: list) -> dict:
        """Run inference on one sample and extract result."""
        tokenized = self._tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=self._config["max_seq_length"],
            padding=True,
        ).to(self._model.device)
        tokenized = {k: v.to(self._model.device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs: torch.Tensor = self._model.generate(
                **tokenized,
                max_new_tokens=self._config["max_completion_length"],
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
            )  # shape (BS, SEQ_LEN)
            decoded: list[str] = self._tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        answers = []
        for i in decoded:
            answer = self._extract_answer(i)
            if answer == "error":
                answer = "neutral"
            answers.append(answer)

        return {"pred": answers}

    def _extract_answer(self, result: str) -> str:
        """Get predicted answer from completion - positive, neutral, negative or error."""
        try:
            answer = result.rsplit("\n", 1)[-1].strip()

            if answer not in ["positive", "negative", "neutral"]:
                return "error"

            return answer

        except Exception:
            print("exeption")
            return "error"

    def run(self, submission_name: str = None) -> None:
        """Run inference on test data and save as submission file."""
        # self._data = self._data.map(lambda x: self._inference_one_sample(x["prompt"]))
        self._data = self._data.map(
            lambda x: self._inference_batch(x["prompt"]),
            batched=True,
            batch_size=self._config["batch_size"],
        )

        submission_dataset = self._data.remove_columns(
            [col for col in self._data.column_names if col not in ["id", "pred"]]
        )

        if not submission_name:
            submission_name = f"unnamed_submission_{time.time()}.csv"
        submission_path = os.path.join(self._config["path_data"], submission_name)

        submission_dataset.to_csv(submission_path)


if __name__ == "__main__":
    with open("sft_baseline/config_inference.yaml", "r") as f:
        config = yaml.safe_load(f)

    inference = Inference(config=config)
    inference.run()
