import torch
import torch.nn as nn
from src.utils.config import *

class HybridModel(nn.Module):
    def __init__(self, transformer_model, hidden_dim=128, dropout=0.3, mode=MODEL_MODE):
        
        super(HybridModel, self).__init__()

        # Pretrained transformer encoder (BERT or compatible)
        self.transformer = transformer_model
        transformer_dim = transformer_model.config.hidden_size
        self.mode = mode

        # ---- CNN branch ----
        # Applies 1D convolution across the token dimension to capture local n-gram features.
        # Only included if mode includes "cnn".
        if mode in ["bert_cnn_lstm", "bert_cnn"]:
            self.conv1d = nn.Conv1d(in_channels=transformer_dim, out_channels=128, kernel_size=3, padding=1)
            self.relu = nn.ReLU()

        # ---- LSTM branch ----
        # Applies BiLSTM to capture long-range sequential dependencies in the text.
        # If CNN precedes LSTM, we feed conv features into LSTM.
        if mode in ["bert_cnn_lstm", "bert_lstm"]:
            lstm_input_dim = 128 if mode == "bert_cnn_lstm" else transformer_dim
            self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # ---- Final classifier ----
        # For LSTM modes: output is bidirectional (hidden_dim * 2).
        # For CNN-only: output dimension is fixed at 128.
        final_dim = hidden_dim * 2 if mode in ["bert_cnn_lstm", "bert_lstm"] else 128
        self.fc = nn.Linear(final_dim, 1) # map to single fraud/legit score
        self.sigmoid = nn.Sigmoid()       # convert to probability [0,1]

    def forward(self, input_ids, attention_mask):

        # ---- Transformer encoding ----
        # Output: last hidden state (batch_size, seq_len, hidden_dim)
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        if self.mode == "bert_cnn_lstm":
            # Transformer → CNN → BiLSTM
            x = x.permute(0, 2, 1)         # (batch, hidden_dim, seq_len) for Conv1d
            x = self.relu(self.conv1d(x))  # apply convolution + ReLU
            x = x.permute(0, 2, 1)         # back to (batch, seq_len, channels)
            lstm_out, _ = self.lstm(x)     # run through BiLSTM
            x_out = lstm_out[:, -1, :]     # take last timestep representation

        elif self.mode == "bert_cnn":
            x = x.permute(0, 2, 1)
            x = self.relu(self.conv1d(x))
            x = x.permute(0, 2, 1)
            x_out = x.mean(dim=1)  # Global average pooling

        elif self.mode == "bert_lstm":
            lstm_out, _ = self.lstm(x)
            x_out = lstm_out[:, -1, :]  # take last timestep representation

        x_out = self.dropout(x_out)     # apply dropout for regularization
        x_out = self.fc(x_out)          # linear classifier
        return self.sigmoid(x_out)      # output probability (fraud vs legit)
