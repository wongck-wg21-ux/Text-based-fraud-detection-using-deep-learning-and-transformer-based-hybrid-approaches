# kfold_evaluation_balanced.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import collections
import csv

from imblearn.over_sampling import RandomOverSampler

from src.model.build_model import HybridModel
from src.model.fraud_dataset import FraudDataset
from src.data.data_merger import merge_datasets
from src.preprocessing.clean_text import clean_text
from src.utils.config import *

def run_kfold(model_class, texts, labels, tokenizer, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracies, precisions, recalls, f1s = [], [], [], []

    dataset = FraudDataset(texts, labels, tokenizer)

    # Create CSV to save results
    with open("kfold_results_balanced.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Fold", "Accuracy", "Precision", "Recall", "F1-Score"])

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n=== Fold {fold+1}/{k} ===")

        # Show label distribution
        train_labels_raw = [labels[i] for i in train_idx]
        val_labels_raw = [labels[i] for i in val_idx]

        print(f"Original Train labels: {collections.Counter(train_labels_raw)}")
        print(f"Validation labels: {collections.Counter(val_labels_raw)}")

        # Apply oversampling to balance training set
        ros = RandomOverSampler(random_state=42)
        train_idx_resampled, _ = ros.fit_resample(np.array(train_idx).reshape(-1, 1), train_labels_raw)
        train_idx_resampled = train_idx_resampled.flatten()

        train_subset = Subset(dataset, train_idx_resampled)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        # Fresh model per fold
        bert_model = BertModel.from_pretrained(TRANSFORMER_MODEL)
        model = model_class(bert_model, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, mode=MODEL_MODE).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.BCELoss()

        # Early stopping setup
        best_f1 = 0
        patience = 2
        counter = 0

        # Training loop
        for epoch in range(EPOCHS):
            model.train()  # ✅ Ensure training mode
            total_loss = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels_batch = batch["label"].unsqueeze(1).to(DEVICE)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")

            # Validation
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    labels_batch = batch["label"].to(DEVICE)

                    outputs = model(input_ids, attention_mask)
                    preds = (outputs >= 0.5).int().cpu().numpy().flatten()

                    val_preds.extend(preds)
                    val_labels.extend(labels_batch.cpu().numpy())

            val_f1 = f1_score(val_labels, val_preds)
            print(f"Validation F1: {val_f1:.4f}")

            # Early stopping check
            if val_f1 > best_f1:
                best_f1 = val_f1
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Final Evaluation on validation set
        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds, zero_division=0)
        rec = recall_score(val_labels, val_preds)
        f1 = val_f1

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        print(f"Fold {fold+1} Results:")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")

        # Save to CSV
        with open("kfold_results_balanced.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([fold+1, acc, prec, rec, f1])

    # Summary
    print("\n=== K-Fold Cross Validation Summary (With Oversampling & Early Stopping) ===")
    print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall: {np.mean(recalls):.4f}")
    print(f"Average F1-Score: {np.mean(f1s):.4f}")

if __name__ == "__main__":
    print("Loading and merging datasets...")

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

    print("Preprocessing texts...")
    df["clean_text"] = df["text"].apply(clean_text)
    texts = df["clean_text"].tolist()
    labels = df["label"].tolist()

    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL)

    print(f"Running K-Fold Cross Validation with MODEL_MODE={MODEL_MODE} (Oversampling + Early Stopping)")
    run_kfold(HybridModel, texts, labels, tokenizer, k=5)
