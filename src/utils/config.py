# config.py
import os
import torch

# Base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model mode options:
# - "bert_cnn_lstm"
# - "bert_cnn"
# - "bert_lstm"
MODEL_MODE = "bert_cnn_lstm"

# Paths
NORMALIZED_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "normalized_dataset.csv")
TRANSFORMER_MODEL = "bert-base-uncased"

# Best hyperparameters from Optuna
MAX_LEN = 128 # Initial: 128
HIDDEN_DIM = 128 # Initial: 256
DROPOUT = 0.4784 # Initial: 0.3
EPOCHS = 10 # Initial: 10
BATCH_SIZE = 16 # Initial: 32
LR = 2.35e-5 # Initial: 2e-5
WEIGHT_DECAY = 0.000809 # Initial: 0.01