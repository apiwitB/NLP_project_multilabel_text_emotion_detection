"""
Thai Multilabel Emotion Detection: Full Dataset Ekman Mapping + 2-Level Class Weighting
Architecture: Stacked Ensemble (WangchanBERTa + mBERT + XLM-RoBERTa -> Weighted Random Forest)
Dataset: Full GoEmotions (Train 43,410 | Val 5,426 | Test 5,131)
Taxonomy: Ekman 7 Emotions (Joy, Anger, Sadness, Surprise, Fear, Disgust, Neutral)
Optimization: FP16 Mixed Precision + Dynamic VRAM Management
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
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, hamming_loss, f1_score
from tqdm.auto import tqdm
import sys

# เพิ่ม root dir เพื่อให้ import โมดูลใน src ได้
if "." not in sys.path:
    sys.path.append(".")
from src.eval_logger import save_eval_report

# ---------------------------------------------------------
# 0. Configuration & Setup
# ---------------------------------------------------------
SAVE_DIR = "saved_models_ekman_weighted"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running Ekman + Class Weighting Pipeline on device: {device}")

# ---------------------------------------------------------
# 1. Dataset Loading & Ekman Mapping (28 -> 7)
# ---------------------------------------------------------
TRAIN_PATH = "goemotions/data/go_emotions_train_with_translation.csv"
VAL_PATH   = "goemotions/data/go_emotions_validation_with_translation.csv"
TEST_PATH  = "goemotions/data/go_emotions_test_with_translation.csv"

# Fallback path กรณีรันจากโฟลเดอร์อื่น
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "../goemotions/data/go_emotions_train_with_translation.csv"
    VAL_PATH   = "../goemotions/data/go_emotions_validation_with_translation.csv"
    TEST_PATH  = "../goemotions/data/go_emotions_test_with_translation.csv"

df_train = pd.read_csv(TRAIN_PATH).dropna(subset=["text_th"]).reset_index(drop=True)
df_val   = pd.read_csv(VAL_PATH).dropna(subset=["text_th"]).reset_index(drop=True)
df_test  = pd.read_csv(TEST_PATH).dropna(subset=["text_th"]).reset_index(drop=True)

print(f"📂 ข้อมูล Train      : {len(df_train):,} แถว")
print(f"📂 ข้อมูล Validation : {len(df_val):,} แถว")
print(f"📂 ข้อมูล Test       : {len(df_test):,} แถว")

ORIGINAL_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

EKMAN_MAPPING = {
    "Joy": [
        "admiration", "amusement", "approval", "caring", "desire",
        "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"
    ],
    "Anger": ["anger", "annoyance", "disapproval"],
    "Sadness": ["disappointment", "embarrassment", "grief", "remorse", "sadness"],
    "Surprise": ["confusion", "curiosity", "realization", "surprise"],
    "Fear": ["fear", "nervousness"],
    "Disgust": ["disgust"],
    "Neutral": ["neutral"]
}

EKMAN_LABELS = list(EKMAN_MAPPING.keys())  # 7 classes
NUM_LABELS = len(EKMAN_LABELS)

ekman_label_to_id = {lbl: i for i, lbl in enumerate(EKMAN_LABELS)}
orig_id_to_ekman_id = {}
for orig_id, orig_lbl in enumerate(ORIGINAL_LABELS):
    for ekman_lbl, orig_list in EKMAN_MAPPING.items():
        if orig_lbl in orig_list:
            orig_id_to_ekman_id[orig_id] = ekman_label_to_id[ekman_lbl]
            break

def map_labels_to_ekman(labels_raw):
    mapped = set()
    for lbl in labels_raw:
        if isinstance(lbl, int) and lbl in orig_id_to_ekman_id:
            mapped.add(orig_id_to_ekman_id[lbl])
        elif isinstance(lbl, str):
            for ek_lbl, orig_list in EKMAN_MAPPING.items():
                if lbl in orig_list:
                    mapped.add(ekman_label_to_id[ek_lbl])
    return list(mapped)

for df in [df_train, df_val, df_test]:
    df["labels_list"] = df["labels"].apply(ast.literal_eval)
    df["ekman_labels"] = df["labels_list"].apply(map_labels_to_ekman)

mlb = MultiLabelBinarizer(classes=list(range(NUM_LABELS)))
Y_train = mlb.fit_transform(df_train["ekman_labels"])
X_train = df_train["text_th"].astype(str).tolist()

Y_val = mlb.transform(df_val["ekman_labels"])
X_val = df_val["text_th"].astype(str).tolist()

Y_test = mlb.transform(df_test["ekman_labels"])
X_test = df_test["text_th"].astype(str).tolist()

# ---------------------------------------------------------
# 2. Level 1 Class Weighting: BCEWithLogitsLoss pos_weight
# ---------------------------------------------------------
pos_counts = Y_train.sum(axis=0)
neg_counts = len(Y_train) - pos_counts
base_weight = neg_counts / (pos_counts + 1e-6)

# Boost factor สำหรับ Minority classes:
# Joy=0, Anger=1, Sadness=2, Surprise=3, Fear=4, Disgust=5, Neutral=6
BOOST_FACTOR = np.array([
    1.0,   # Joy
    1.0,   # Anger
    2.5,   # Sadness  (boost 2.5x)
    1.0,   # Surprise
    3.0,   # Fear     (boost 3.0x)
    3.0,   # Disgust  (boost 3.0x)
    1.0,   # Neutral
])

boosted_weight = base_weight * BOOST_FACTOR
pos_weight_tensor = torch.FloatTensor(boosted_weight).to(device)

print("\n⚖️ Level 1: Boosted pos_weight (BCEWithLogitsLoss):")
for i, lbl in enumerate(EKMAN_LABELS):
    boost_txt = f">> BOOSTED x{BOOST_FACTOR[i]}" if BOOST_FACTOR[i] > 1.0 else ""
    print(f"  [{i}] {lbl:<10} | pos={int(pos_counts[i]):>5} | pos_weight={boosted_weight[i]:.2f} {boost_txt}")

# ---------------------------------------------------------
# 3. PyTorch Dataset & FP16 Training Function
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
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.FloatTensor(self.labels[idx])
        }

def train_model_fp16(model_name, use_fast=True, epochs=3, lr=2e-5, batch_size=16):
    print(f"\n==================================================")
    print(f"🤖 Fine-tuning Base Model: {model_name} (FP16)")
    print(f"==================================================")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    ).to(device)

    train_ds = ThaiEmotionDataset(X_train, Y_train, tok)
    val_ds   = ThaiEmotionDataset(X_val,   Y_val,   tok)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for b in pbar:
            optimizer.zero_grad()
            input_ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            labels = b["labels"].to(device)

            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                out = model(input_ids, attention_mask=mask)
                loss = criterion(out.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = b["input_ids"].to(device)
                mask = b["attention_mask"].to(device)
                labels = b["labels"].to(device)

                with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    out = model(input_ids, attention_mask=mask)
                    val_loss += criterion(out.logits, labels).item()

        print(f"  ✨ Epoch {epoch+1} -> Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    del optimizer, train_loader, val_loader, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return model, tok

# ---------------------------------------------------------
# 4. Train & Save 3 Base Models Sequentially (Zero VRAM Spill)
# ---------------------------------------------------------
base_configs = [
    ("wangchan", "airesearch/wangchanberta-base-att-spm-uncased", False),
    ("mbert", "bert-base-multilingual-cased", True),
    ("xlm_roberta", "xlm-roberta-base", True)
]

trained_models = {}
trained_tokenizers = {}

for key, pretrained_name, use_fast in base_configs:
    m, tok = train_model_fp16(pretrained_name, use_fast=use_fast, epochs=3, batch_size=16)
    
    # บันทึกโมเดลลงโฟลเดอร์ทันที
    m.save_pretrained(f"{SAVE_DIR}/{key}_model")
    tok.save_pretrained(f"{SAVE_DIR}/{key}_tokenizer")
    print(f"💾 บันทึก {key} ลงใน {SAVE_DIR}/ เรียบร้อย")
    
    # ย้ายโมเดลไปพักไว้ที่ CPU เพื่อเคลียร์ VRAM ก่อนตัวถัดไป
    m.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    trained_models[key] = m
    trained_tokenizers[key] = tok

# ---------------------------------------------------------
# 5. Extract Probability Features Safely (21 Dimensions)
# ---------------------------------------------------------
def extract_probs_safe(model, tok, texts, desc_name, batch_size=32):
    print(f"\n🔮 กำลังสกัด Feature จาก {desc_name}...")
    model.to(device)
    model.eval()
    
    ds = ThaiEmotionDataset(texts, np.zeros((len(texts), NUM_LABELS)), tok)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    
    with torch.no_grad():
        for b in tqdm(loader, desc=f"Extracting {desc_name}"):
            input_ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                out = model(input_ids, attention_mask=mask)
                probs.append(torch.sigmoid(out.logits).cpu().numpy())
                
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(probs, axis=0)

train_w = extract_probs_safe(trained_models["wangchan"], trained_tokenizers["wangchan"], X_train, "WangchanBERTa (Train)")
test_w  = extract_probs_safe(trained_models["wangchan"], trained_tokenizers["wangchan"], X_test,  "WangchanBERTa (Test)")

train_m = extract_probs_safe(trained_models["mbert"], trained_tokenizers["mbert"], X_train, "mBERT (Train)")
test_m  = extract_probs_safe(trained_models["mbert"], trained_tokenizers["mbert"], X_test,  "mBERT (Test)")

train_x = extract_probs_safe(trained_models["xlm_roberta"], trained_tokenizers["xlm_roberta"], X_train, "XLM-RoBERTa (Train)")
test_x  = extract_probs_safe(trained_models["xlm_roberta"], trained_tokenizers["xlm_roberta"], X_test,  "XLM-RoBERTa (Test)")

# รวม 7 + 7 + 7 = 21 dimensions
X_train_meta = np.concatenate([train_w, train_m, train_x], axis=1)
X_test_meta  = np.concatenate([test_w, test_m, test_x], axis=1)

# ---------------------------------------------------------
# 6. Level 2 Class Weighting: Random Forest Meta-Classifier
# ---------------------------------------------------------
# Custom class weights per Ekman label:
RF_CLASS_WEIGHTS = [
    "balanced",        # Joy
    "balanced",        # Anger
    {0: 1, 1: 3},      # Sadness  (boost 3x)
    "balanced",        # Surprise
    {0: 1, 1: 5},      # Fear     (boost 5x)
    {0: 1, 1: 5},      # Disgust  (boost 5x)
    "balanced",        # Neutral
]

print("\n🌲 กำลังเทรน Random Forest Meta-Classifier ด้วย Custom Class Weighting...")
fitted_estimators = []
for i in range(NUM_LABELS):
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight=RF_CLASS_WEIGHTS[i],
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_meta, Y_train[:, i])
    fitted_estimators.append(rf)
    print(f"  ✅ Fitted estimator for [{i}] {EKMAN_LABELS[i]} (weight: {RF_CLASS_WEIGHTS[i]})")

joblib.dump(fitted_estimators, f"{SAVE_DIR}/random_forest_ekman_weighted.joblib")
with open(f"{SAVE_DIR}/label_classes.json", "w", encoding="utf-8") as f:
    json.dump(EKMAN_LABELS, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------
# 7. Evaluation & Automatic Report Generation
# ---------------------------------------------------------
Y_pred_cols = [est.predict(X_test_meta) for est in fitted_estimators]
Y_pred = np.stack(Y_pred_cols, axis=1)

micro_f1 = f1_score(Y_test, Y_pred, average="micro", zero_division=0)
macro_f1 = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(Y_test, Y_pred, average="weighted", zero_division=0)
h_loss = hamming_loss(Y_test, Y_pred)

print("\n" + "=" * 60)
print("📊 ผลการประเมิน Ekman 7 + 2-Level Class Weighting (Full Dataset)")
print("=" * 60)
print(f"  Micro F1     : {micro_f1:.4f}")
print(f"  Macro F1     : {macro_f1:.4f}")
print(f"  Weighted F1  : {weighted_f1:.4f}")
print(f"  Hamming Loss : {h_loss:.4f}")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred, target_names=EKMAN_LABELS, zero_division=0))

# บันทึกผลลงโฟลเดอร์ results/ อัตโนมัติ
save_eval_report(
    y_true=Y_test,
    y_pred=Y_pred,
    target_names=EKMAN_LABELS,
    model_name="Stacked Ensemble Ekman 7 + 2-Level Class Weighting (Full Train)",
    hyperparams={
        "Dataset": f"Train {len(X_train):,} rows -> Test Official {len(X_test):,} rows",
        "Taxonomy": "Ekman 7 Emotions",
        "Base Models": "WangchanBERTa, mBERT, XLM-RoBERTa (FP16)",
        "Level 1 Weighting": "BCE pos_weight + Boost (Fear x3.0, Disgust x3.0, Sadness x2.5)",
        "Level 2 Weighting": "RandomForest (Fear/Disgust {0:1,1:5}, Sadness {0:1,1:3})"
    },
    notes="การทดลอง Full Dataset รวม Ekman 7 และ 2-Level Class Weighting เพื่อดัน F1 minority classes",
    output_dir="results"
)
print("🎉 จบ Pipeline สำเร็จเรียบร้อย บันทึกผลลง results/ และ SUMMARY.md แล้ว!")
