# 🚀 Training Roadmap: Thai Multilabel Emotion Detection (Fine-tuning on Thai Dataset)

**อ้างอิงสถาปัตยกรรม:** *A Hybrid Stacked Ensemble Learning Framework for Thai Multilabel Emotion Detection* (Paper Architecture) + *GoEmotions Taxonomy (28 Emotions)*

---

## 📊 ชุดข้อมูลที่ใช้ในการ Fine-tune

| ไฟล์ | จำนวนแถว | หมายเหตุ |
|---|---|---|
| `goemotions/data/go_emotions_test_with_translation.csv` | **5,427** (ใช้ได้ **5,131** แถวหลัง dropna) | ✅ มีคอลัมน์ `text_th` (ภาษาไทย) |
| **Input Feature** | `text_th` | ข้อความภาษาไทย |
| **Target Labels** | `labels` | Binary Matrix 28 คลาส (GoEmotions taxonomy, index 0-27) |

---

## 🗺️ ภาพรวม Pipeline การเทรน (ตาม Paper Architecture)

```
       GoEmotions Thai Data (go_emotions_test_with_translation.csv)
                                ↓
        [Step 1] Data Preprocessing & 28-Label Binarization
                                ↓
        [Step 2] Fine-tune Transformer Base Models ด้วย text_th
           ├── 1. WangchanBERTa (airesearch/wangchanberta-base-att-spm-uncased) → Probs (28)
           ├── 2. mBERT (bert-base-multilingual-cased)                           → Probs (28)
           └── 3. XLM-RoBERTa (xlm-roberta-base)                                 → Probs (28)
                                ↓
        [Step 3] Concatenate Prediction Features [28 + 28 + 28 = 84 Dim]
                                ↓
        [Step 4] Train Level-2 Meta-Classifier (Random Forest)
                                ↓
        [Step 5] Evaluation & Per-Class Threshold Optimization (F1 Micro/Macro, Hamming Loss)
                                ↓
        [Step 6] Save Models & Deploy ใน src/app.py
```

---

## 📋 รายละเอียดขั้นตอนการดำเนินงาน (Step-by-Step)

### Step 1 — การเตรียมข้อมูลภาษาไทย (Data Preparation & Label Encoding) 🔧

#### 1.1 โหลดข้อมูลและคัดกรองเฉพาะแถวที่มีภาษาไทย
```python
import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# 1. โหลดข้อมูล
df = pd.read_csv("goemotions/data/go_emotions_test_with_translation.csv")

# 2. คัดกรองแถวที่ไม่มี text_th ออก
df = df.dropna(subset=['text_th']).reset_index(drop=True)
print(f"จำนวนข้อมูลภาษาไทยที่พร้อมใช้งาน: {len(df)} แถว")
```

#### 1.2 Binarize Labels (28 Class GoEmotions Taxonomy)
```python
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]
NUM_LABELS = len(EMOTION_LABELS)  # 28

# แปลง string '[25]' เป็น list [25]
df['labels_list'] = df['labels'].apply(ast.literal_eval)

mlb = MultiLabelBinarizer(classes=list(range(28)))
Y = mlb.fit_transform(df['labels_list'])  # Shape: (N, 28)
X_texts = df['text_th'].tolist()
```

#### 1.3 คำนวณ `pos_weight` สำหรับ Imbalanced Data (BCEWithLogitsLoss)
```python
pos_counts = Y.sum(axis=0)
neg_counts = len(df) - pos_counts
pos_weight = neg_counts / (pos_counts + 1e-6)
pos_weight_tensor = torch.FloatTensor(pos_weight)
```

#### 1.4 แบ่งข้อมูล Train / Validation / Test
```python
# แบ่ง 80% Train, 10% Val, 10% Test
X_train_val, X_test, Y_train_val, Y_test = train_test_split(
    X_texts, Y, test_size=0.10, random_state=42
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_val, Y_train_val, test_size=0.1111, random_state=42  # 0.1111 * 0.9 ≈ 0.1
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
```

---

### Step 2 — Fine-tune Base Models ด้วยภาษาไทย 🤖

 Fine-tune Transformer Base Models ทั้ง 3 ตัวโดยใช้ **`text_th`** เพื่อเรียนรู้บริบทภาษาไทย:

1. **WangchanBERTa** (`airesearch/wangchanberta-base-att-spm-uncased`) — โมเดลภาษาไทยเฉพาะทาง
2. **mBERT** (`bert-base-multilingual-cased`) — Multilingual BERT
3. **XLM-RoBERTa** (`xlm-roberta-base`) — Multilingual RoBERTa

#### Hyperparameters แนะนำ
| Parameter | ค่าที่ตั้ง |
|---|---|
| Learning Rate | `2e-5` |
| Batch Size | `16` |
| Epochs | `3` - `5` |
| Max Token Length | `128` (ภาษาไทยประหยัด Token) |
| Loss Function | `BCEWithLogitsLoss(pos_weight=pos_weight)` |
| Optimizer | AdamW |

#### PyTorch Training Implementation
```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ThaiEmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
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
            'labels': torch.FloatTensor(self.labels[item])
        }

def train_base_model(model_name, use_fast=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=28,
        problem_type="multi_label_classification"
    ).to(device)
    
    train_ds = ThaiEmotionDataset(X_train, Y_train, tokenizer)
    val_ds   = ThaiEmotionDataset(X_val, Y_val, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Train Loop (3 Epochs)
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
    return model, tokenizer
```

---

### Step 3 — Extraction ของ Probability Features (84 Dimensions) 🔄

ดึง Probability outputs (ผ่าน Sigmoid) จาก Base Models ทั้ง 3 ตัว เพื่อนำมาเรียงต่อกัน (Concatenate):

```python
def extract_probabilities(model, tokenizer, texts, device):
    model.eval()
    ds = ThaiEmotionDataset(texts, np.zeros((len(texts), 28)), tokenizer)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            out = model(input_ids, attention_mask=mask)
            prob = torch.sigmoid(out.logits).cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs, axis=0)  # Shape: (N, 28)

# รวม 3 โมเดล → 84 features
prob_wangchan = extract_probabilities(model_wangchan, tok_wangchan, X_train, device)
prob_mbert    = extract_probabilities(model_mbert, tok_mbert, X_train, device)
prob_xlmr     = extract_probabilities(model_xlmr, tok_xlmr, X_train, device)

X_train_meta = np.concatenate([prob_wangchan, prob_mbert, prob_xlmr], axis=1) # Shape: (N, 84)
```

---

### Step 4 — Train Level-2 Meta-Classifier (Random Forest) 🌲

ใช้ **Random Forest** (ผ่าน `MultiOutputClassifier`) เทรนบน Feature Matrix 84 มิติ:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

rf_base = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_meta = MultiOutputClassifier(rf_base)
rf_meta.fit(X_train_meta, Y_train)
```

---

### Step 5 — Evaluation & Threshold Tuning 📊

ประเมินผลบน Test Set (10% ที่แยกไว้):

```python
from sklearn.metrics import f1_score, hamming_loss, classification_report

# ดึง probability บน Test Set
test_wangchan = extract_probabilities(model_wangchan, tok_wangchan, X_test, device)
test_mbert    = extract_probabilities(model_mbert, tok_mbert, X_test, device)
test_xlmr     = extract_probabilities(model_xlmr, tok_xlmr, X_test, device)

X_test_meta = np.concatenate([test_wangchan, test_mbert, test_xlmr], axis=1)

# Predict ด้วย Random Forest
Y_pred = rf_meta.predict(X_test_meta)

print("Micro F1 Score:", f1_score(Y_test, Y_pred, average='micro'))
print("Macro F1 Score:", f1_score(Y_test, Y_pred, average='macro'))
print("Hamming Loss:  ", hamming_loss(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred, target_names=EMOTION_LABELS, zero_division=0))
```

---

### Step 6 — Export Models & Update FastAPI Deployment 🚀

บันทึกโมเดลทั้งหมดลงโฟลเดอร์ `saved_models/`:

```python
import joblib, json, os

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. บันทึก Transformer Base Models & Tokenizers
model_wangchan.save_pretrained(f"{SAVE_DIR}/wangchan_model")
tok_wangchan.save_pretrained(f"{SAVE_DIR}/wangchan_tokenizer")

model_mbert.save_pretrained(f"{SAVE_DIR}/mbert_model")
tok_mbert.save_pretrained(f"{SAVE_DIR}/mbert_tokenizer")

model_xlmr.save_pretrained(f"{SAVE_DIR}/xlm_roberta_model")
tok_xlmr.save_pretrained(f"{SAVE_DIR}/xlm_roberta_tokenizer")

# 2. บันทึก Meta-Classifier
joblib.dump(rf_meta, f"{SAVE_DIR}/random_forest_meta.joblib")

# 3. บันทึก 28 Emotion Labels
with open(f"{SAVE_DIR}/label_classes.json", "w", encoding="utf-8") as f:
    json.dump(EMOTION_LABELS, f, ensure_ascii=False, indent=2)

print("✅ บันทึกโมเดลทั้งหมดสำเร็จ สามารถรัน uvicorn src.app:app ได้ทันที!")
```

---

## ⚡ สรุปสิ่งที่ต้องอัปเดตและรันจริง

1. **ใช้ข้อมูลภาษาไทย**: `goemotions/data/go_emotions_test_with_translation.csv` (คอลัมน์ `text_th`)
2. **ปรับปรุง Notebook**: อัปเดต `notebooks/04_base_models.ipynb` และ `notebooks/05_ensemble.ipynb` ให้ใช้ข้อมูลภาษาไทย 28 อารมณ์
3. **รันสคริปต์การเทรน**: รัน Base Models ทั้ง 3 ตัว -> สกัด 84-dim Features -> เทรน Random Forest Meta-Classifier -> บันทึกเข้า `saved_models/`
