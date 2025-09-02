import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

# Validation loss (for plotting)
def evaluate_loss_only(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].unsqueeze(1).to(DEVICE)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Final evaluation (metrics)
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            preds = (outputs.squeeze() >= 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_preds, digits=4))


if __name__ == "__main__":
    logger.info("Loading and cleaning dataset...")
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

    # 80/20 split
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    logger.info(f"Total dataset size: {len(df)} records")
    logger.info(f"Training set size: {len(X_train)} records")
    logger.info(f"Validation set size: {len(X_val)} records")

    train_ds = FraudDataset(X_train, y_train, tokenizer)
    val_ds = FraudDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    bert_model = BertModel.from_pretrained(TRANSFORMER_MODEL)
    model = HybridModel(bert_model, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, mode=MODEL_MODE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    # Track losses
    train_losses = []
    val_losses = []

    # Early Stopping setup
    best_val_loss = float('inf')
    patience = 2
    counter = 0

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss = train(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate_loss_only(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info("Validation:")
        evaluate(model, val_loader)

        # Early Stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            logger.info(f"Validation loss improved to {val_loss:.4f}")
        else:
            counter += 1
            logger.info(f"No improvement in validation loss. Patience counter: {counter}/{patience}")

            if counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # Define dynamic file names
    model_save_path = f"models/hybrid_model_{MODEL_MODE}.pt"
    loss_save_path = f"models/loss_tracking_{MODEL_MODE}.pt"

    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Training complete. Model saved to {model_save_path}")

    torch.save({
        "train_losses": train_losses,
        "val_losses": val_losses
    }, loss_save_path)
    logger.info(f"Loss tracking data saved to {loss_save_path}")
