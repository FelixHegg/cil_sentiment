import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
import yaml
from sft_baseline.prepare_data import prepare_data
from transformers import Trainer, TrainingArguments


class SFTTrainerWrapper:
    """Directly predict label as SFT baseline."""

    def __init__(self, config: dict):
        """Load model and tokenizer."""
        self._config = config
        self._model = self._load_model()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config["model"], use_fast=True
        )
        self._train_dataset, self._eval_dataset = prepare_data(
            self._tokenizer, self._config["path_data"]
        )
        self._sft_config = self._get_sft_config()
        self._trainer = self._get_SFT_trainer()

    def train(self):
        """Train model."""
        self._trainer.train()

    def _get_sft_config(self) -> SFTConfig:
        """Get SFT config."""
        return SFTConfig(
            gradient_checkpointing=self._config["gradient_checkpointing"],
            gradient_checkpointing_kwargs={
                "use_reentrant": self._config["use_reentrant"]
            },
            gradient_accumulation_steps=self._config["gradient_accumulation_steps"],
            per_device_train_batch_size=self._config["per_device_train_batch_size"],
            # max_seq_length=self._config["max_seq_length"],
            num_train_epochs=self._config["num_train_epochs"],
            learning_rate=self._config["learning_rate"],
            optim=self._config["optim"],
            save_steps=self._config["save_steps"],
            report_to="wandb",
            logging_steps=1,
            output_dir=self._config["exp_dir"],
            remove_unused_columns=False,
        )

    def _get_SFT_trainer(self) -> SFTTrainer:
        """Get SFT trainer."""
        return SFTTrainer(
            model=self._model,
            train_dataset=self._train_dataset,
            args=self._sft_config,
            # tokenizer=self._tokenizer,
            processing_class=self._tokenizer,
        )

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

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=self._config["lora_rank"],
            target_modules=self._config["target_modules"],
            lora_alpha=self._config["lora_alpha"],
            lora_dropout=self._config["lora_dropout"],
        )

        model = get_peft_model(base_model, peft_config)
        model.train()
        return model


if __name__ == "__main__":
    with open("sft_baseline/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    trainer = SFTTrainerWrapper(config)
    trainer.train()
