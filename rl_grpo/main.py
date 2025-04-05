import argparse
import os
import yaml
import wandb

from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
from rl_grpo.reward_functions import correctness_reward, format_reward_func
from rl_grpo.utils import LoggingCallback
from trl import GRPOConfig, GRPOTrainer

PatchFastRL("GRPO", FastLanguageModel)


class Trainer:
    def __init__(self, config: dict):
        """Initialize tokenizer, datasets, model, config."""
        self.config = config
        self._tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        self._dataset_train, self._dataset_test = self._generate_dataset(
            config["path_data"]
        )
        self._model = self._load_model()
        self._grpo_config = self._load_grpo_config()

    def _generate_dataset(self, data_path: str) -> tuple[DatasetDict, DatasetDict]:
        """Load and format dataset."""
        dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(data_path, "training.csv"),
                "test": os.path.join(data_path, "test.csv"),
            },
        )

        train_dataset = dataset["train"].map(
            lambda x: self._generate_reasoning_prompt(x["sentence"], x["label"])
        )
        test_dataset = dataset["test"].map(
            lambda x: self._generate_reasoning_prompt(x["sentence"], x["label"])
        )

        return train_dataset, test_dataset

    def _generate_reasoning_prompt(self, text: str, label: str) -> dict:
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
            "prompt": self._tokenizer.apply_chat_template(
                r1_prefix, tokenize=False, continue_final_message=True
            ),
            "target": label,
        }

    def _load_model(self) -> FastLanguageModel:
        """Load and prepare the model with LoRA and quantization."""
        model_cfg = self.config["model"]
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_cfg["name"],
            max_seq_length=model_cfg["max_seq_length"],
            load_in_4bit=model_cfg["load_in_4bit"],
            fast_inference=model_cfg["fast_inference"],
            max_lora_rank=model_cfg["lora_rank"],
            gpu_memory_utilization=model_cfg["gpu_memory_utilization"],
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=model_cfg["lora_rank"],
            target_modules=model_cfg["target_modules"],
            lora_alpha=model_cfg["lora_alpha"],
            use_gradient_checkpointing=model_cfg["use_gradient_checkpointing"],
            random_state=model_cfg["random_state"],
        )
        return model

    def _load_grpo_config(self) -> GRPOConfig:
        """Load GRPOConfig with values from config."""
        train_cfg = self.config["training"]
        print(train_cfg["learning_rate"])
        return GRPOConfig(
            use_vllm=True,
            learning_rate=train_cfg["learning_rate"],
            adam_beta1=train_cfg["adam_beta1"],
            adam_beta2=train_cfg["adam_beta2"],
            weight_decay=train_cfg["weight_decay"],
            warmup_ratio=train_cfg["warmup_ratio"],
            max_grad_norm=train_cfg["max_grad_norm"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            optim=train_cfg["optim"],
            logging_steps=train_cfg["logging_steps"],
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            num_generations=train_cfg["num_generations"],
            max_prompt_length=256,
            max_completion_length=train_cfg["max_completion_length"],
            max_steps=train_cfg["max_steps"],
            save_steps=train_cfg["save_steps"],
            report_to="wandb",
            output_dir=train_cfg["output_dir"],
        )

    def train(self):
        """Start training loop using GRPOTrainer."""
        trainer = GRPOTrainer(
            model=self._model,
            reward_funcs=[format_reward_func, correctness_reward],
            args=self._grpo_config,
            train_dataset=self._dataset_train,
            eval_dataset=self._dataset_test,
        )

        trainer.add_callback(LoggingCallback)
        trainer.train()


if __name__ == "__main__":
    with open("rl_grpo/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    wandb.init(
        project="sentiment_classification_rl",
        config=config["training"],
    )

    trainer = Trainer(config=config)
    trainer.train()
