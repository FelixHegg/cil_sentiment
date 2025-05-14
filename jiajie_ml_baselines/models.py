import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType

# --- MLP Model ---
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[128, 256, 128, 64], dropouts=[0.5, 0.3, 0.2, 0.1]):
        super(MLP, self).__init__()
        assert len(hidden_dims) == len(dropouts), "Number of hidden layers and dropouts must match"
        layers = []
        current_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropouts[i]))
            current_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, num_classes)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        return x

# --- 1D CNN Model ---
class CNN1DReg(nn.Module):
    def __init__(self, embedding_dim, num_classes, num_filters=80, filter_sizes=[3, 4, 5], dropout=0.6):
        super(CNN1DReg, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        conved = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        x = self.dropout(cat)
        x = self.fc(x)
        return x

# --- BiLSTM Model ---
class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.dropout(hidden_cat)
        out = self.fc(out)
        return out

# --- PEFT BERT + MLP Head Model ---
class PeftWithMLPHeadClassifier(nn.Module):
    def __init__(self, peft_base_model, num_classes, mlp_hidden_dims=[128, 64], mlp_dropout=0.5):
        super(PeftWithMLPHeadClassifier, self).__init__()
        self.bert_peft = peft_base_model
        self.bert_hidden_size = self.bert_peft.config.hidden_size if hasattr(self.bert_peft.config, 'hidden_size') else self.bert_peft.model.config.hidden_size
        
        self.mlp_head = nn.Sequential(
            nn.Linear(self.bert_hidden_size, mlp_hidden_dims[0]), nn.ReLU(), nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]), nn.ReLU(), nn.Dropout(mlp_dropout * 0.6),
            nn.Linear(mlp_hidden_dims[1], num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_peft(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.mlp_head(cls_output)
        return logits

# Function to get model instance
def get_model(model_type, num_classes, config_dict, embedding_dim=None, hf_tokenizer=None):
    model_type_lower = model_type.lower()
    print(f"Initializing model: {model_type_lower}, Num classes: {num_classes}")

    if model_type_lower == 'mlp':
        if embedding_dim is None: raise ValueError("embedding_dim required for MLP model.")
        model = MLP(
            input_dim=embedding_dim, num_classes=num_classes,
            hidden_dims=config_dict.get('hidden_dims', [128, 256, 128, 64]),
            dropouts=config_dict.get('dropouts', [0.5, 0.3, 0.2, 0.1])
        )
    elif model_type_lower == 'cnn':
        if embedding_dim is None: raise ValueError("embedding_dim required for CNN model.")
        model = CNN1DReg(
            embedding_dim=embedding_dim, num_classes=num_classes,
            num_filters=config_dict.get('num_filters', 80),
            filter_sizes=config_dict.get('filter_sizes', [3, 4, 5]),
            dropout=config_dict.get('dropout', 0.6)
        )
    elif model_type_lower == 'bilstm':
        if embedding_dim is None: raise ValueError("embedding_dim required for BiLSTM model.")
        model = BiLSTMClassifier(
            embedding_dim=embedding_dim, num_classes=num_classes,
            hidden_dim=config_dict.get('hidden_dim', 128),
            num_layers=config_dict.get('num_layers', 1),
            dropout=config_dict.get('dropout', 0.5)
        )
    elif model_type_lower == 'peft_bert_mlp':
        print(f"Loading base model '{config_dict['hf_base_model_name']}' for PEFT...")
        try:
            base_model = AutoModel.from_pretrained(config_dict['hf_base_model_name'])
            print("Base model loaded.")
        except Exception as e:
            print(f"Error loading base model for PEFT: {e}"); raise

        print("Applying LoRA configuration...")
        lora_config_peft = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config_dict.get('lora_rank', 64),
            lora_alpha=config_dict.get('lora_alpha', 128),
            target_modules=config_dict.get('lora_target_modules', ["q_lin", "v_lin"]),
            lora_dropout=config_dict.get('lora_dropout', 0.1),
            bias=config_dict.get('lora_bias', "none")
        )
        peft_model = get_peft_model(base_model, lora_config_peft)
        print("LoRA applied. Trainable parameters of PEFT base model:")
        peft_model.print_trainable_parameters()

        model = PeftWithMLPHeadClassifier(
            peft_base_model=peft_model,
            num_classes=num_classes,
            mlp_hidden_dims=config_dict.get('classifier_mlp_hidden_dims', [128, 64]),
            mlp_dropout=config_dict.get('classifier_mlp_dropout', 0.5)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type_lower}")

    print(f"\n{model_type_lower.upper()} Model Architecture:")
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters in Combined Model: {num_params:,}")
    return model
