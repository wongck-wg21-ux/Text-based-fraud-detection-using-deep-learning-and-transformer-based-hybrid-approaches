import torch
from torch.utils.data import Dataset

class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # Return total number of samples in the dataset.
        return len(self.texts)

    def __getitem__(self, idx):
        # Get a single sample by index.
        # Returns a dict with:
        # - input_ids: token IDs (tensor, shape [max_len])
        # - attention_mask: mask for non-padding tokens
        # - label: ground truth label (0.0 or 1.0)

        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and encode the text
        encoded = self.tokenizer(
            text,
            padding='max_length',    # pad to fixed max_len
            truncation=True,         # truncate longer texts
            max_length=self.max_len,
            return_tensors="pt"      # return PyTorch tensors
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }
