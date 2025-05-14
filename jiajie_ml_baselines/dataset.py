import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    """
    PyTorch Dataset for self-defined Pytorch model inputs.
    Handles both average vectors and sequences of word embeddings.
    """
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.feature_shape = self.features.shape[1:]
        print(f"SentimentDataset created. Feature shape (per sample): {self.feature_shape}, Num samples: {len(self.features)}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class TransformerSentimentDataset(Dataset):
    """
    PyTorch Dataset for Hugging Face transformer model inputs.
    Expects pre-tokenized input_ids and attention_masks.
    """
    def __init__(self, input_ids, attention_masks, labels_encoded):
        # input_ids and attention_masks should already be tensors
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = torch.tensor(labels_encoded, dtype=torch.long)
        print(f"TransformerSentimentDataset created. Num samples: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx].clone().detach(),
            'attention_mask': self.attention_masks[idx].clone().detach(),
            'labels': self.labels[idx].clone().detach()
        }
        return item