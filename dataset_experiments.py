import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from src.data.data_merger import merge_datasets
from src.preprocessing.clean_text import clean_text
from src.model.fraud_dataset import FraudDataset
from src.model.build_model import HybridModel
from src.utils.config import DEVICE, TRANSFORMER_MODEL, NORMALIZED_DATASET_PATH

# Define dataset subsets for progressive experiments
datasets = [
    [("spam.csv", "spam_csv")],
    [("spam.csv", "spam_csv"), ("phishing_email.csv", "text_combined")],
    [("spam.csv", "spam_csv"), ("phishing_email.csv", "text_combined"), ("CEAS_08.csv", "subject_body")],
    [("spam.csv", "spam_csv"), ("phishing_email.csv", "text_combined"), ("CEAS_08.csv", "subject_body"), ("Enron.csv", "subject_body")],
    [("spam.csv", "spam_csv"), ("phishing_email.csv", "text_combined"), ("CEAS_08.csv", "subject_body"),
     ("Enron.csv", "subject_body"), ("fraud_call.csv", "no_header")]
]

# Function for training and validation
def train_and_evaluate(train_loader, val_loader, model, optimizer, loss_fn, epochs=10, patience=2):
    best_val_loss = float("inf")
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].unsqueeze(1).to(DEVICE)
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                running_val_loss += loss.item()
                preds = (outputs >= 0.5).int().cpu().numpy().flatten()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds, zero_division=0)
        rec = recall_score(val_labels, val_preds, zero_division=0)
        f1 = f1_score(val_labels, val_preds, zero_division=0)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_model.pt"))
    return train_losses, val_losses, acc, prec, rec, f1

results = []

for i, subset in enumerate(datasets):
    print(f"\n===== Experiment {i+1}: Using {len(subset)} dataset(s) =====")
    # Merge and preprocess
    df = merge_datasets(subset, input_dir="data/raw", output_path=NORMALIZED_DATASET_PATH)
    df["clean_text"] = df["text"].apply(clean_text)

    # Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL)

    # Datasets and loaders
    train_ds = FraudDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    val_ds = FraudDataset(X_val.tolist(), y_val.tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    # Model setup
    bert_model = BertModel.from_pretrained(TRANSFORMER_MODEL)
    model = HybridModel(bert_model, hidden_dim=128, dropout=0.48, mode="bert_cnn_lstm").to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=2.3e-5)
    loss_fn = nn.BCELoss()

    # Train and evaluate
    train_losses, val_losses, acc, prec, rec, f1 = train_and_evaluate(train_loader, val_loader, model, optimizer, loss_fn, epochs=10, patience=2)
    results.append((len(subset), acc, prec, rec, f1))

    # Plot losses for each experiment
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.title(f"Experiment {i+1} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"loss_curve_exp_{i+1}.png")
    plt.close()

# Summary of all experiments
print("\n===== Summary of Progressive Experiments =====")
for r in results:
    print(f"{r[0]} dataset(s): Acc={r[1]:.4f}, Prec={r[2]:.4f}, Rec={r[3]:.4f}, F1={r[4]:.4f}")

# Save results as CSV
df_results = pd.DataFrame(results, columns=["Num_Datasets", "Accuracy", "Precision", "Recall", "F1"])
df_results.to_csv("progressive_results.csv", index=False)
