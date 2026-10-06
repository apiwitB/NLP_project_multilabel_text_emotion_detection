"""
Thai Multilabel Emotion Detection Training Script
Architecture: Stacked Ensemble (WangchanBERTa + mBERT + XLM-RoBERTa -> Random Forest)
Dataset: goemotions/data/go_emotions_test_with_translation.csv (text_th column)
"""

import os
import ast
import json
import joblib
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, hamming_loss, f1_score
from tqdm import tqdm
from src.eval_logger import save_eval_report

# ---------------------------------------------------------
# 0. Global Setup & Device Configuration
# ---------------------------------------------------------
DATA_PATH = "goemotions/data/go_emotions_test_with_translation.csv"
SAVE_DIR  = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running training pipeline on device: {device}")

# GoEmotions 28 Taxonomy Classes
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]
NUM_LABELS = len(EMOTION_LABELS)  # 28

# ---------------------------------------------------------
# 1. Load Data & Filter Thai Text
# ---------------------------------------------------------
print(f"\n📂 Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
initial_len = len(df)
df = df.dropna(subset=['text_th']).reset_index(drop=True)
print(f"✅ Filtered non-null 'text_th': {len(df)} rows (from {initial_len} total rows)")

# Decode labels column: "[0, 27]" -> [0, 27]
df['labels_list'] = df['labels'].apply(ast.literal_eval)

mlb = MultiLabelBinarizer(classes=list(range(NUM_LABELS)))
Y_bin = mlb.fit_transform(df['labels_list'])  # (N, 28)
X_texts = df['text_th'].astype(str).tolist()

# Calculate pos_weight for imbalanced classes
pos_counts = Y_bin.sum(axis=0)
neg_counts = len(df) - pos_counts
pos_weight = neg_counts / (pos_counts + 1e-6)
pos_weight_tensor = torch.FloatTensor(pos_weight).to(device)

print(f"Label matrix shape: {Y_bin.shape}")

# Split Train / Val / Test (80% / 10% / 10%)
X_train_val, X_test, Y_train_val, Y_test = train_test_split(
    X_texts, Y_bin, test_size=0.10, random_state=42
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_val, Y_train_val, test_size=0.1111, random_state=42  # 0.1111 * 0.9 = 0.1
)

print(f"📊 Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ---------------------------------------------------------
# 2. PyTorch Dataset & DataLoader Definition
# ---------------------------------------------------------
class ThaiEmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

# ---------------------------------------------------------
# 3. Fine-Tune Base Transformer Models
# ---------------------------------------------------------
model_configs = {
    "wangchan": {
        "pretrained": "airesearch/wangchanberta-base-att-spm-uncased",
        "use_fast": False
    },
    "mbert": {
        "pretrained": "bert-base-multilingual-cased",
        "use_fast": True
    },
    "xlm_roberta": {
        "pretrained": "xlm-roberta-base",
        "use_fast": True
    }
}

trained_models = {}
trained_tokenizers = {}

def train_single_model(model_key, cfg, epochs=3, batch_size=16, lr=2e-5):
    print(f"\n==================================================")
    print(f"🤖 Fine-tuning Base Model: {model_key} ({cfg['pretrained']})")
    print(f"==================================================")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg['pretrained'], use_fast=cfg['use_fast'])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg['pretrained'],
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    ).to(device)

    train_ds = ThaiEmotionDataset(X_train, Y_train, tokenizer)
    val_ds   = ThaiEmotionDataset(X_val,   Y_val,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=mask)
                loss = criterion(outputs.logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"  Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    del optimizer, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return model, tokenizer

# Train all 3 base models
for key, config in model_configs.items():
    m, tok = train_single_model(key, config, epochs=3, batch_size=16, lr=2e-5)
    trained_models[key] = m
    trained_tokenizers[key] = tok

# ---------------------------------------------------------
# 4. Extract Probability Features (84-dim Features)
# ---------------------------------------------------------
def extract_probs(model, tokenizer, texts, batch_size=32):
    model.eval()
    dummy_labels = np.zeros((len(texts), NUM_LABELS))
    ds = ThaiEmotionDataset(texts, dummy_labels, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            out = model(input_ids, attention_mask=mask)
            prob = torch.sigmoid(out.logits).cpu().numpy()
            all_probs.append(prob)

    return np.concatenate(all_probs, axis=0)

print("\n🔄 Extracting probability features from all 3 base models...")
train_features = []
test_features  = []

for key in ["wangchan", "mbert", "xlm_roberta"]:
    m = trained_models[key]
    tok = trained_tokenizers[key]
    train_p = extract_probs(m, tok, X_train)
    test_p  = extract_probs(m, tok, X_test)
    train_features.append(train_p)
    test_features.append(test_p)
    print(f"  ✅ {key}: Train shape={train_p.shape}, Test shape={test_p.shape}")

X_train_meta = np.concatenate(train_features, axis=1)  # (N_train, 84)
X_test_meta  = np.concatenate(test_features,  axis=1)  # (N_test, 84)

print(f"\nConcat Meta Feature Shape -> Train: {X_train_meta.shape} | Test: {X_test_meta.shape}")

# ---------------------------------------------------------
# 5. Train Random Forest Meta-Classifier
# ---------------------------------------------------------
print("\n🌲 Training Random Forest Meta-Classifier...")
rf_base = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_meta = MultiOutputClassifier(rf_base)
rf_meta.fit(X_train_meta, Y_train)
print("✅ Meta-Classifier trained successfully!")

# ---------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------
print("\n📊 Evaluating Ensemble Model on Test Set...")
Y_pred = rf_meta.predict(X_test_meta)

micro_f1 = f1_score(Y_test, Y_pred, average='micro')
macro_f1 = f1_score(Y_test, Y_pred, average='macro')
h_loss   = hamming_loss(Y_test, Y_pred)

print(f"🎯 Micro F1:     {micro_f1:.4f}")
print(f"🎯 Macro F1:     {macro_f1:.4f}")
print(f"🎯 Hamming Loss: {h_loss:.4f}\n")
print(classification_report(Y_test, Y_pred, target_names=EMOTION_LABELS, zero_division=0))

# Save detailed evaluation report to markdown file in results/
report_path = save_eval_report(
    y_true=Y_test,
    y_pred=Y_pred,
    target_names=EMOTION_LABELS,
    model_name="Stacked Ensemble (WangchanBERTa + mBERT + XLM-RoBERTa -> Random Forest)",
    hyperparams={
        "Epochs": 3,
        "Batch Size": 16,
        "Learning Rate": "2e-5",
        "Meta Estimators": 200,
        "Meta Class Weight": "balanced"
    },
    notes="Full training evaluation on GoEmotions Thai test set split.",
    output_dir="results"
)

# ---------------------------------------------------------
# 7. Save Models & Label Classes
# ---------------------------------------------------------
print(f"\n💾 Saving trained models and tokenizers to '{SAVE_DIR}/'...")
for key in ["wangchan", "mbert", "xlm_roberta"]:
    mod_path = os.path.join(SAVE_DIR, f"{key}_model")
    tok_path = os.path.join(SAVE_DIR, f"{key}_tokenizer")
    trained_models[key].save_pretrained(mod_path)
    trained_tokenizers[key].save_pretrained(tok_path)
    print(f"  ✅ Saved {key} model & tokenizer")

rf_path = os.path.join(SAVE_DIR, "random_forest_meta.joblib")
joblib.dump(rf_meta, rf_path)
print(f"  ✅ Saved Meta-Classifier to {rf_path}")

label_path = os.path.join(SAVE_DIR, "label_classes.json")
with open(label_path, "w", encoding="utf-8") as f:
    json.dump(EMOTION_LABELS, f, ensure_ascii=False, indent=2)
print(f"  ✅ Saved {NUM_LABELS} label classes to {label_path}")

# ---------------------------------------------------------
# 8. Release GPU VRAM Memory
# ---------------------------------------------------------
print("\n🧹 Releasing GPU VRAM memory...")
del trained_models, trained_tokenizers
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
gc.collect()
print("✅ GPU VRAM released successfully!")

print(f"\n🎉 ALL COMPLETED SUCCESSFULLY! You can now start the API with:")
print(f"   uvicorn src.app:app --reload")
