import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import BertTokenizer, BertModel
import streamlit as st

# Project imports (from your repo)
from src.model.build_model import HybridModel
from src.utils.config import (
    TRANSFORMER_MODEL, MODEL_MODE, MAX_LEN, HIDDEN_DIM, DROPOUT, DEVICE,
    NORMALIZED_DATASET_PATH, LR, WEIGHT_DECAY, BATCH_SIZE, EPOCHS
)
from src.preprocessing.clean_text import clean_text

# --------------------------
# Utilities
# --------------------------

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    # Load the pretrained tokenizer, BERT backbone, and final trained hybrid model.
    # Uses Streamlit's cache_resource to avoid reloading every time the app refreshes.
    
    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL)
    bert = BertModel.from_pretrained(TRANSFORMER_MODEL)
    model = HybridModel(
        transformer_model=bert,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        mode=MODEL_MODE
    )
    ckpt_path = f"models/final_model_{MODEL_MODE}.pt"
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return tokenizer, model

@st.cache_data(show_spinner=False)
def load_provenance_table():
    # Load the normalized dataset (if available) and build a provenance summary:
    # - Count of legitimate vs fraudulent samples per dataset source.
    # Returns the summary table and total records.

    if os.path.exists(NORMALIZED_DATASET_PATH):
        df = pd.read_csv(NORMALIZED_DATASET_PATH)
        if "source" in df.columns:
            counts = df.groupby(["source", "label"], dropna=False).size().unstack(fill_value=0)
            counts = counts.rename(columns={0: "legitimate", 1: "fraudulent"}).reset_index()
            counts["total"] = counts.get("legitimate", 0) + counts.get("fraudulent", 0)
            return counts, len(df)
    return None, 0

@torch.inference_mode()
def predict_proba(texts, tokenizer, model):
    # Preprocess input text(s), tokenize with BERT, and run model inference.
    # Returns fraud probabilities for each input and cleaned text for reference.
    
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [clean_text(t) for t in texts]
    enc = tokenizer(
        cleaned,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    probs = model(enc["input_ids"], enc["attention_mask"]).squeeze(1).detach().cpu().numpy()
    return probs, cleaned

def badge(label: str):
    # Generate a colored badge (HTML span) for fraud/legitimate predictions.
    # Fraudulent = red, Legitimate = green
    
    color = "#ef4444" if label.lower().startswith("fraud") else "#10b981"
    return f"<span style='background:{color};color:white;padding:6px 10px;border-radius:12px;font-weight:600'>{label}</span>"

# --------------------------
# Streamlit App
# --------------------------

# Page configuration
st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")

# Sidebar navigation and controls
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection System")
    st.caption("Hybrid BERT + CNN + BiLSTM")

    # Radio navigation: switch between app pages
    page = st.radio("Navigate", ["Classify", "Batch Evaluate", "Explain", "Dataset", "About"], index=0)

    st.markdown("---")
    # Decision threshold slider
    threshold = st.slider("Decision threshold (Fraud)", 0.35, 0.55, 0.50, 0.01)
    st.toggle("Show cleaned text", value=False, key="show_clean")

    st.markdown("---")
    st.caption("Model checkpoint:")
    st.code(f"models/final_model_{MODEL_MODE}.pt", language="text")

# Load tokenizer and model once
try:
    tokenizer, model = load_model_and_tokenizer()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --------------------
# Page: Classify
# --------------------
if page == "Classify":
    st.title("Fraud Detection System")
    st.write("Enter a message to classify it as fraudulent or legitimate:")

    cols = st.columns([3,1])
    with cols[0]:
        user_input = st.text_area("Message", height=160, placeholder="Paste email/SMS/call transcript snippet here…")

    if st.button("Classify", type="primary"):
        if not user_input or not user_input.strip():
            st.warning("Please enter a message to classify.")
        else:
            # Predict fraud probability
            probs, cleaned = predict_proba(user_input, tokenizer, model)
            p_fraud = float(probs[0])
            label = "Fraudulent" if p_fraud >= threshold else "Legitimate"
            
            # Display results
            conf = p_fraud if label == "Fraudulent" else (1.0 - p_fraud)
            st.markdown(badge(label), unsafe_allow_html=True)
            st.progress(conf, text=f"Confidence: {conf:.2%}")
            if st.session_state.get("show_clean"):
                with st.expander("Preprocessing preview"):
                    st.code(cleaned[0])

# --------------------
# Page: Explain
# --------------------
if page == "Explain":
    st.title("Explain Prediction")
    st.caption("Local and global explanations using SHAP (if installed).")

    text = st.text_area(
        "Message to explain",
        value=user_input if 'user_input' in locals() else "",
        height=160
    )

    if st.button("Generate explanation"):
        if not text.strip():
            st.warning("Please enter a message first.")
        else:
            # --- 1) Run model to classify using current threshold ---
            probs, cleaned = predict_proba(text, tokenizer, model)
            p_fraud = float(probs[0])
            label = "Fraudulent" if p_fraud >= threshold else "Legitimate"
            conf = p_fraud if label == "Fraudulent" else (1.0 - p_fraud)

            # Show result + confidence
            st.markdown(badge(label), unsafe_allow_html=True)
            st.progress(conf, text=f"Confidence: {conf:.2%}")
            st.caption(f"Fraud probability: {p_fraud:.2%} • Threshold: {threshold:.2f}")

            if st.session_state.get("show_clean"):
                with st.expander("Preprocessing preview"):
                    st.code(cleaned[0])

            # --- 2) SHAP explanation (best-effort) ---
            try:
                import shap
                st.info("Computing SHAP values…")

                # small wrapper returning probabilities for SHAP
                def _predict(batch_texts):
                    p, _ = predict_proba(batch_texts, tokenizer, model)
                    return p

                explainer = shap.Explainer(_predict, shap.maskers.Text(tokenizer))
                sv = explainer([text])

                # Display local explanation
                st.subheader("Local explanation")
                html = shap.plots.text(sv[0], display=False)
                st.components.v1.html(html, height=320, scrolling=True)

            except Exception as e:
                st.error(f"SHAP explanation unavailable: {e}")


# --------------------
# Page: Dataset
# --------------------
if page == "Dataset":
    st.title("Training Dataset Overviewe")
    tbl, total = load_provenance_table()
    if tbl is not None:
        st.dataframe(tbl)
        st.metric("Total records", total)
    else:
        st.info("Normalized dataset not found.")
    with st.expander("Preprocessing summary"):
        st.markdown(f"- Max sequence length: **{MAX_LEN}**")
        st.markdown("- Lowercasing, URL stripping, tokenization, stopword & disfluency removal.")

# --------------------
# Page: Evaluate
# --------------------
if page == "Batch Evaluate":
    st.title("Batch Evaluate / Predict")
    file = st.file_uploader("Upload CSV with 'text' (and optional 'label')", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            # Predict on all uploaded texts
            probs, cleaned = predict_proba(df["text"].tolist(), tokenizer, model)
            df["prob_fraud"] = probs
            df["prediction"] = (df["prob_fraud"] >= threshold).astype(int)
            
            # Show sample results
            st.dataframe(df.head(50))

            # If ground truth provided, compute metrics
            if "label" in df.columns:
                y_true = df["label"].astype(int)
                y_pred = df["prediction"]
                st.write({
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred),
                    "f1": f1_score(y_true, y_pred),
                })

                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
            
            # Allow download of predictions
            st.download_button("Download predictions.csv", df.to_csv(index=False).encode("utf-8"))

# --------------------
# Page: About
# --------------------
if page == "About":
    st.title("About")

    # Original model card for completeness
    st.subheader("Model Card (concise)")
    st.json({
        "architecture": "BERT encoder + 1D CNN + BiLSTM + FC (sigmoid)",
        "transformer": TRANSFORMER_MODEL,
        "mode": MODEL_MODE,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "max_len": MAX_LEN,
        "optimizer": "Adam/AdamW (training)",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "device": str(DEVICE),
    })

    # ---- Helpers: K-fold results loader and summary (2) ----
    @st.cache_data(show_spinner=False)
    def load_kfold_results(path: str = "kfold_results_balanced.csv"):
        if os.path.exists(path):
            df = pd.read_csv(path)
            # expect columns: Fold, Accuracy, Precision, Recall, F1-Score
            means = df[["Accuracy","Precision","Recall","F1-Score"]].mean()
            stds  = df[["Accuracy","Precision","Recall","F1-Score"]].std()
            return df, means, stds
        return None, None, None

    st.subheader("K‑fold Cross Validation summary, K = 5")
    kf_df, kf_mean, kf_std = load_kfold_results()
    if kf_df is None:
        st.info("No kfold_results_balanced.csv found. Run kfold_evaluation.py to generate it.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{kf_mean['Accuracy']:.3f} ± {kf_std['Accuracy']:.3f}")
        c2.metric("Precision", f"{kf_mean['Precision']:.3f} ± {kf_std['Precision']:.3f}")
        c3.metric("Recall",    f"{kf_mean['Recall']:.3f} ± {kf_std['Recall']:.3f}")
        c4.metric("F1-Score",  f"{kf_mean['F1-Score']:.3f} ± {kf_std['F1-Score']:.3f}")
        with st.expander("View per-fold table"):
            st.dataframe(kf_df, use_container_width=True)

    # ---- (1) Exportable evaluation report (markdown) ----
    st.subheader("Generate evaluation report")
    st.caption("Creates a concise markdown report you can attach to your FYP. It includes config, threshold, and K‑fold summary.")

    def build_report_md():
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines = []
        lines.append(f"# Fraud Detection System — Evaluation Report\n")
        lines.append(f"Generated: {ts}\n")
        lines.append("\n## Configuration\n")
        lines.append(f"- Architecture: BERT encoder + 1D CNN + BiLSTM + FC (sigmoid)")
        lines.append(f"\n- Transformer: {TRANSFORMER_MODEL}")
        lines.append(f"\n- Mode: {MODEL_MODE}")
        lines.append(f"\n- Max length: {MAX_LEN}")
        lines.append(f"\n- Hidden dim: {HIDDEN_DIM}")
        lines.append(f"\n- Dropout: {DROPOUT}")
        lines.append(f"\n- Threshold (demo): {threshold:.2f}")
        lines.append(f"\n- Training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, weight_decay={WEIGHT_DECAY}")

        if kf_df is not None:
            lines.append("\n## K‑fold results (oversampling + early stopping)\n")
            lines.append(kf_df.to_csv(index=False))
            lines.append("\n**Summary**\\\n")
            lines.append(f"Accuracy: {kf_mean['Accuracy']:.3f} ± {kf_std['Accuracy']:.3f}\\\n")
            lines.append(f"Precision: {kf_mean['Precision']:.3f} ± {kf_std['Precision']:.3f}\\\n")
            lines.append(f"Recall: {kf_mean['Recall']:.3f} ± {kf_std['Recall']:.3f}\\\n")
            lines.append(f"F1-Score: {kf_mean['F1-Score']:.3f} ± {kf_std['F1-Score']:.3f}\\\n")
        else:
            lines.append("\n## K‑fold results\n")
            lines.append("kfold_results_balanced.csv not found. Run kfold_evaluation.py to generate.\n")

        lines.append("\n## Notes & Limitations\n- English-focused model; non-English inputs may reduce accuracy.\n- Use the threshold slider to trade recall vs precision.\n- Low-confidence cases should be reviewed by an analyst.\n")
        return "\n".join(lines)

    if st.button("Build report (Markdown)"):
        md = build_report_md()
        st.download_button(
            "Download evaluation_report.md",
            data=md.encode("utf-8"),
            file_name="evaluation_report.md",
            mime="text/markdown",
        )

    # ---- (4) Who benefits? panel ----
    with st.container():
        st.subheader("Who benefits?")
        st.markdown("""
**Target users:** Trust & Safety analysts, SOC analysts, and operations teams who triage *text-based* incidents (email, SMS, call transcripts).

**Primary use cases:**
- Rapid screening of inbound messages for potential fraud before escalation.
- Batch evaluation of campaigns or archives to quantify risk.
- Explainable reviews (token-level contributions) for analyst training.

**Limitations:** The model is English-focused (`bert-base-uncased`). Non-English or heavy code-switching may reduce accuracy; treat low-confidence results with human review.
        """)
