import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils.config import *
from src.model.build_model import HybridModel
from src.preprocessing.clean_text import clean_text
from src.data.data_merger import merge_datasets
from src.model.fraud_dataset import FraudDataset
from src.utils.analysis_utils import plot_loss_curves, evaluate_and_report

# Load model
tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL)
bert = BertModel.from_pretrained(TRANSFORMER_MODEL)
model = HybridModel(bert, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, mode=MODEL_MODE)
model_save_path = f"models/final_model_{MODEL_MODE}.pt"
model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Prepare validation set
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
# class_counts = df['label'].value_counts()
# print("Dataset Distribution (Training + Validation):")
# print(class_counts)
# print(f"Fraudulent class percentage: {class_counts[1] / len(df) * 100:.2f}%")
# print(f"Legitimate class percentage: {class_counts[0] / len(df) * 100:.2f}%")

df["clean_text"] = df["text"].apply(clean_text)
X_train, X_val, y_train, y_val = train_test_split(df["clean_text"], df["label"], test_size=0.2, random_state=42)

# Check class distribution in the validation set
# val_class_counts = y_val.value_counts()
# print("Validation Set Distribution:")
# print(val_class_counts)
# print(f"Fraudulent class percentage in validation: {val_class_counts[1] / len(y_val) * 100:.2f}%")
# print(f"Legitimate class percentage in validation: {val_class_counts[0] / len(y_val) * 100:.2f}%")

val_dataset = FraudDataset(X_val.tolist(), y_val.tolist(), tokenizer)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Evaluate
print(f"Evaluating model combination: {MODEL_MODE}")
evaluate_and_report(model, val_loader, DEVICE)

# # Plot loss curve
loss_save_path = f"models/loss_tracking_{MODEL_MODE}.pt"
loss_data = torch.load(loss_save_path)
plot_loss_curves(loss_data["train_losses"], loss_data["val_losses"])
