import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.model.fraud_dataset import FraudDataset
from src.data.data_merger import merge_datasets
from src.preprocessing.clean_text import clean_text
from src.model.build_model import HybridModel
from src.utils.config import *

import optuna.visualization as vis
import pandas as pd

def train(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in dataloader:
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

def validate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].unsqueeze(1).to(DEVICE)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            batch_preds = (outputs >= 0.5).int().cpu().numpy().flatten()
            preds.extend(batch_preds)
            true_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(true_labels, preds)
    return avg_loss, f1

def objective(trial):
    # Define hyperparameter search space for this trial
    # (e.g., lr, batch_size, hidden_dim, dropout, weight_decay)
    
    # Initialize model with sampled parameters
    
    # Train for a fixed number of epochs with early stopping
    
    # Evaluate on validation set and return F1-score (or other metric)

    learning_rate = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.01)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])

    # Load data
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

    # print(f"Total number of records in merged dataset: {len(df)}")
    # print(df['label'].value_counts())  # To see class distribution (fraud / non-fraud)

    df["clean_text"] = df["text"].apply(clean_text)
    texts = df["clean_text"].tolist()
    labels = df["label"].tolist()

    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL)
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_ds = FraudDataset(X_train, y_train, tokenizer)
    val_ds = FraudDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    bert_model = BertModel.from_pretrained(TRANSFORMER_MODEL)
    model = HybridModel(bert_model, hidden_dim=hidden_dim, dropout=dropout_rate).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    # Training loop with early stopping
    best_f1 = 0
    patience = 2
    counter = 0

    for epoch in range(EPOCHS):
        train(model, train_loader, optimizer, loss_fn)
        val_loss, val_f1 = validate(model, val_loader, loss_fn)

        trial.report(val_f1, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return -best_f1

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print(f"Using device: {DEVICE}")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1 Score: {-study.best_trial.value:.4f}")
    print(f"Best Hyperparameters: {study.best_trial.params}")

    df_trials = study.trials_dataframe()
    df_trials.to_csv("optuna_full_search.csv", index=False)

    try:
        vis.plot_optimization_history(study).show()
        vis.plot_param_importances(study).show()
    except:
        print("Visualization skipped (non-interactive environment).")
