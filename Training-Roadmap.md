# 🚀 Training Roadmap: Thai Multilabel Emotion Detection
**อ้างอิง:** *A Hybrid Stacked Ensemble Learning Framework for Thai Multilabel Emotion Detection* + *GoEmotions: A Dataset of Fine-Grained Emotions*

---

## 📊 ข้อมูลที่มี (ณ ตอนนี้)

| ไฟล์ | แถว | หมายเหตุ |
|---|---|---|
| `goemotions/data/go_emotions_train.csv` | ~43,410 | ต้องแปลเพิ่ม (ยังไม่มี `text_th`) |
| `goemotions/data/go_emotions_validation.csv` | ~5,426 | ต้องแปลเพิ่ม |
| `goemotions/data/go_emotions_test_with_translation.csv` | ~5,428 | ✅ มี `text_th` แล้ว |
| Columns | `text`, `text_th`, `labels`, `id` | labels = list เช่น `[25]` |
| Label space | **28 อารมณ์** (GoEmotions taxonomy) | Index 0-27 |

---

## 🗺️ ภาพรวม Pipeline (ตาม Paper)

```
Raw GoEmotions CSV (EN+TH)
        ↓
[Step 1] Data Preparation & Label Binarization
        ↓
[Step 2] Fine-tune Base Models แยกกัน
   ├── WangchanBERTa  → Prob vector (28-dim)
   ├── mBERT          → Prob vector (28-dim)
   └── XLM-RoBERTa    → Prob vector (28-dim)
        ↓
[Step 3] Extract OOF Predictions (Out-of-Fold)
        ↓
[Step 4] Train Meta-Classifier (Random Forest)
   Input: Concatenate [28+28+28] = 84 features
   Output: Binary multi-label prediction
        ↓
[Step 5] Evaluate (F1 Micro/Macro, Hamming Loss)
        ↓
[Step 6] Export & Deploy ใน src/app.py
```

---

## 📋 ขั้นตอนละเอียด

### Step 1 — Data Preparation 🔧

**ปัญหา:** `go_emotions_train.csv` และ `go_emotions_validation.csv` ยังไม่มีคอลัมน์ `text_th`
**ต้องทำ:**

#### 1.1 แปล train/validation (ถ้ายังไม่มี)
```python
# ใช้ script ที่มีอยู่แล้ว goemotions/scripts/translate_goemotions.py
# หรือรัน Google Translate API สำหรับ train set (~43K rows)
```
> ⚠️ **ถ้ายังไม่มีงบแปล** → ใช้ `go_emotions_test_with_translation.csv` เป็น **validation/test set**  
> แล้วแบ่ง train set จาก `go_emotions_train.csv` โดยใช้แค่ภาษาอังกฤษก่อน (mBERT และ XLM-RoBERTa รองรับ)

#### 1.2 Binarize Labels (MultiLabelBinarizer)
```python
import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv("goemotions/data/go_emotions_test_with_translation.csv")
df['labels'] = df['labels'].apply(ast.literal_eval)

mlb = MultiLabelBinarizer(classes=list(range(28)))
Y = mlb.fit_transform(df['labels'])  # shape: (N, 28)
```

#### 1.3 Label Names (GoEmotions 28 classes)
```python
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]
# Index 0=admiration, ..., 27=neutral
```

#### 1.4 ตรวจ Class Imbalance
```python
import matplotlib.pyplot as plt
label_counts = Y.sum(axis=0)
plt.bar(EMOTION_LABELS, label_counts)
plt.xticks(rotation=90)
# คาดว่า "neutral" (27) จะเยอะที่สุด, "grief"/"pride" จะน้อยมาก
```

---

### Step 2 — Fine-tune Base Models 🤖

**ใช้ notebook:** `notebooks/04_base_models.ipynb`

#### สถาปัตยกรรมแต่ละ Base Model
```
Input: text (ภาษาไทย หรือ EN ก็ได้สำหรับ mBERT/XLM)
  ↓
[Transformer Encoder]
  ↓
[CLS token] → Linear(hidden_size → 28)
  ↓
Sigmoid activation → 28 probabilities (0-1)
  ↓
BCEWithLogitsLoss (multi-label)
```

#### 2.1 WangchanBERTa (แนะนำสำหรับภาษาไทย)
```python
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
# ใช้ข้อมูล text_th เท่านั้น
# max_length = 128 (ภาษาไทยประหยัด token กว่า EN)
```

#### 2.2 mBERT (Multilingual BERT)
```python
model_name = "bert-base-multilingual-cased"
# ใช้ text_th หรือ text ก็ได้ (รองรับทั้งคู่)
```

#### 2.3 XLM-RoBERTa
```python
model_name = "xlm-roberta-base"
# รองรับภาษาไทยได้ดี — ใช้ text_th
```

#### Hyperparameters แนะนำ (ตาม Paper)
| Parameter | ค่า |
|---|---|
| Learning Rate | `2e-5` |
| Batch Size | `16` หรือ `32` |
| Epochs | `3-5` |
| Max Length | `128` |
| Optimizer | AdamW |
| Scheduler | Linear warmup (10% steps) |
| Loss | `BCEWithLogitsLoss` |
| Threshold | `0.5` (ปรับทีหลัง) |

#### Training Loop สำคัญ
```python
from torch.nn import BCEWithLogitsLoss

criterion = BCEWithLogitsLoss(pos_weight=pos_weights)  # จัดการ class imbalance
# pos_weight = (N_neg / N_pos) per label
```

---

### Step 3 — Extract Out-of-Fold (OOF) Predictions 🔄

> **ทำไมต้อง OOF?**  
> ป้องกัน Data Leakage ตอน train Meta-Classifier  
> ถ้าใช้ predictions จาก train set ตรง ๆ โมเดลจะ overfit แน่นอน

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_probs_wangchan = np.zeros((len(df_train), 28))
oof_probs_mbert    = np.zeros((len(df_train), 28))
oof_probs_xlmr     = np.zeros((len(df_train), 28))

for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
    # fine-tune แต่ละ model บน train_idx
    # predict บน val_idx → เก็บใส่ oof_probs[val_idx]
    pass

# ตอน inference ใช้ full model (trained on all data)
```

---

### Step 4 — Train Meta-Classifier (Stacked Ensemble) 🌲

**ใช้ notebook:** `notebooks/05_ensemble.ipynb`

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np

# Concatenate OOF predictions → 84 features
X_meta = np.concatenate([
    oof_probs_wangchan,  # (N, 28)
    oof_probs_mbert,     # (N, 28)
    oof_probs_xlmr       # (N, 28)
], axis=1)               # → (N, 84)

# Train Random Forest
rf = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
)
rf.fit(X_meta, Y_train)  # Y_train shape: (N, 28)
```

> 📝 **Note:** RF ใน v1 เดิมใช้ `predict_proba` แยกทีละ label  
> → ต้องใช้ `MultiOutputClassifier` หรือ loop predict ทีละ class

---

### Step 5 — Evaluation 📊

```python
from sklearn.metrics import f1_score, hamming_loss, classification_report

# Predict ด้วย Ensemble
Y_pred = rf.predict(X_test_meta)  # binary (0/1)

print("Micro F1:", f1_score(Y_test, Y_pred, average='micro'))
print("Macro F1:", f1_score(Y_test, Y_pred, average='macro'))
print("Hamming Loss:", hamming_loss(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred, target_names=EMOTION_LABELS))
```

#### Threshold Tuning (สำคัญมาก!)
```python
# ทดลอง threshold ต่างกันแต่ละ class
thresholds = np.arange(0.3, 0.7, 0.05)
best_thresh = {}
for i, label in enumerate(EMOTION_LABELS):
    best_f1 = 0
    for t in thresholds:
        pred = (proba_pos[:, i] >= t).astype(int)
        f1 = f1_score(Y_test[:, i], pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh[label] = t
```

---

### Step 6 — Export & Update API 🚀

หลัง train เสร็จ save โมเดลไว้ที่ `saved_models/`

```python
import joblib
# save rf model
joblib.dump(rf, "saved_models/random_forest_meta.joblib")

# save label classes (28 GoEmotions labels)
import json
with open("saved_models/label_classes.json", "w") as f:
    json.dump(EMOTION_LABELS, f)
```

อัปเดต [`src/app.py`](file:///home/apiwit/NLP_project/src/app.py) line 27:
```python
# เดิม
SAVE_DIR = "saved_models"
# ถ้า run จาก src/ ให้เปลี่ยนเป็น
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
```

---

## ⚡ ลำดับการทำงานจริง (Quick Start)

```
1. [ตอนนี้]  ตรวจ train/validation มี text_th ไหม? 
             ถ้าไม่มี → รัน translate script ก่อน
             ถ้าไม่อยากรอ → ใช้ EN + mBERT/XLM-RoBERTa ไปก่อน

2. [Step 1]  เปิด notebooks/04_base_models.ipynb
             → แก้ให้รองรับ 28 classes (v1 เดิมใช้ 7)
             → เพิ่ม BCEWithLogitsLoss + sigmoid

3. [Step 2]  Fine-tune ทีละโมเดล บันทึก checkpoint

4. [Step 3]  Extract OOF predictions ทั้ง 3 โมเดล

5. [Step 4]  เปิด notebooks/05_ensemble.ipynb
             → train Random Forest บน 84 features

6. [Step 5]  Evaluate + Threshold tuning

7. [Step 6]  Save models → อัปเดต src/app.py
```

---

## ⚠️ สิ่งสำคัญที่ต้องระวัง

> [!WARNING]
> **v1 เดิมใช้ 7 labels** — ตอนนี้จะเปลี่ยนเป็น **28 labels (GoEmotions)**  
> ต้องแก้ทั้ง base model output layer, RF, และ `saved_models/label_classes.json`

> [!TIP]
> Train ทีละโมเดลแล้ว save checkpoint — GPU/CPU อาจใช้เวลานาน  
> WangchanBERTa เหมาะกับข้อความไทยที่สุด ให้ผลดีที่สุด

> [!NOTE]
> ถ้ายังไม่มี `text_th` สำหรับ train set → ใช้ `text` (EN) กับ mBERT + XLM-RoBERTa ได้เลย  
> WangchanBERTa ต้องการ TH เท่านั้น
