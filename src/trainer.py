class Trainer:
    def __init__(self, config: dict):
        """Initialize tokenizer, datasets, model, config."""
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], use_fast=True)
        self.train_dataset, self.eval_dataset = self._generate_dataset(config["path_data"])
        self.model = self._load_model()
        self.grpo_config = self._load_grpo_config()

    def _generate_dataset(self, data_path: str) -> tuple[DatasetDict, DatasetDict]:
        """Load and format dataset."""
        dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(data_path, "training.csv"),
                "test": os.path.join(data_path, "test.csv"),
            },
        )
        def map_fn(x):
            prompt = self._generate_reasoning_prompt(x["sentence"])
            return {"prompt": prompt, "target": x["label"]}

        train_ds = dataset["train"].map(map_fn, remove_columns=dataset["train"].column_names)
        eval_ds = dataset["test"].map(map_fn, remove_columns=dataset["test"].column_names)
        return train_ds, eval_ds

    def _generate_reasoning_prompt(self, text: str) -> str:
        r1_prefix = [
            {"role": "system",  "content": (
                "You are a helpful assistant. You first think about the reasoning "
                "process and then provide the answer."
            )},
            {"role": "user", "content": (
                f"Determine the sentiment of the following text:\n{text}"    
                "\nShow your analysis in <think></think> tags, and put the final answer "
                "(positive/negative/neutral) in <answer></answer> tags."
            )},
            {"role": "assistant", "content": "Let me think step by step.\n<think>"},
        ]
        return self.tokenizer.apply_chat_template(
            r1_prefix, tokenize=False, continue_final_message=True
        )

    def _load_model(self):
        """Load and prepare the model with optional quantization and LoRA."""
        model_cfg = self.config["model"]
        bf16 = torch.cuda.is_bf16_supported()

        # Load base model
        if model_cfg.get("load_in_4bit", False):
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_cfg["name"],
                quantization_config=quant_config,
                low_cpu_mem_usage=True,
            )
        else:
            dtype = torch.bfloat16 if bf16 else torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                model_cfg["name"],
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )

        # Apply LoRA
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=model_cfg["lora_rank"],
            target_modules=model_cfg["target_modules"],
            lora_alpha=model_cfg.get("lora_alpha", 16),
            lora_dropout=model_cfg.get("lora_dropout", 0.0),
        )
        model = get_peft_model(model, peft_config)
        model.eval()

        # Move entire model to GPU to avoid meta tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    def _load_grpo_config(self) -> GRPOConfig:
        """Load GRPOConfig with values from config."""
        train_cfg = self.config["training"]
        return GRPOConfig(
            use_vllm=False,
            learning_rate=train_cfg["learning_rate"],
            adam_beta1=train_cfg["adam_beta1"],
            adam_beta2=train_cfg["adam_beta2"],
            weight_decay=train_cfg["weight_decay"],
            warmup_ratio=train_cfg["warmup_ratio"],
            max_grad_norm=train_cfg["max_grad_norm"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            optim=train_cfg["optim"],
            logging_steps=train_cfg["logging_steps"],
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
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
            model=self.model,
            reward_funcs=[format_reward_func, correctness_reward],
            args=self.grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        trainer.add_callback(LoggingCallback)
        trainer.train()