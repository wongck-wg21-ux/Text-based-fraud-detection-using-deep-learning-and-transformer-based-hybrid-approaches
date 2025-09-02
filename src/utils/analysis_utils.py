import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def plot_loss_curves(train_losses, val_losses):
    # Plot loss curves and RETURN the matplotlib Figure (no blocking show).
    # This makes it easy to save the figure or embed it in Streamlit/Docs.

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_and_report(model, dataloader, device):

    # Run evaluation on a dataloader and RETURN:
    #   - metrics dict (accuracy, precision, recall, f1)
    #   - classification report as DataFrame
    #   - confusion-matrix Figure/Axes (heatmap)
    # Notes:
    #   * Assumes model outputs sigmoid probabilities in [0,1].
    #   * If your model returns logits, apply torch.sigmoid before thresholding.
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = (outputs >= 0.5).int().cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
