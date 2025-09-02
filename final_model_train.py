# main_train_final.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from src.model.fraud_dataset import FraudDataset
from src.data.data_merger import merge_datasets
from src.preprocessing.clean_text import clean_text
from src.model.build_model import HybridModel
from src.utils.config import *
from src.utils.logging import get_logger

logger = get_logger()

# Training loop
def train(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    logger.info("Loading and cleaning dataset for final retraining...")

    df = merge_datasets(
        dataset_list=[
            ("CEAS_08.csv", "subject_body"),
            ("Enron.csv", "subject_body"),
            ("Ling.csv", "subject_body"),
            ("Nazario.csv", "subject_body"),
            ("Nigerian_Fraud.csv", "subject_body"),
            ("phishing_email.csv", "text_combined"),
            ("SpamAssasin.csv", "subject_body"),
            ("fraud_call.csv", "no_header"),
            ("fraud_calls_data.csv", "no_header"),
            ("spam.csv", "spam_csv")
        ],
        input_dir="data/raw",
        output_path=NORMALIZED_DATASET_PATH
    )

    df["clean_text"] = df["text"].apply(clean_text)
    texts = df["clean_text"].tolist()
    labels = df["label"].tolist()

    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL)

    # No train/val split for final training
    X_train = texts
    y_train = labels

    train_ds = FraudDataset(X_train, y_train, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    bert_model = BertModel.from_pretrained(TRANSFORMER_MODEL)
    model = HybridModel(bert_model, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, mode=MODEL_MODE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCELoss()

    logger.info(f"Starting final retrain with MODEL_MODE={MODEL_MODE}, Batch={BATCH_SIZE}, Dropout={DROPOUT}, LR={LR}, WeightDecay={WEIGHT_DECAY}")

    # Train for fixed epochs
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train(model, train_loader, optimizer, loss_fn)
        logger.info(f"Train Loss: {train_loss:.4f}")

    # Save final model
    model_save_path = f"models/final_model_{MODEL_MODE}.pt"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Final model saved to {model_save_path}")
