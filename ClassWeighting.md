# 🏋️ Class Weighting — อธิบายอย่างละเอียด

**Notebook ที่เพิ่ม:** [`notebooks/05_class_weighting.ipynb`](file:///home/apiwit/NLP_project/notebooks/05_class_weighting.ipynb)

---

## 🔍 ปัญหาที่ต้องแก้

จากผลล่าสุด (`04_base_models_map_emo.ipynb` — Ekman Baseline):

| Class | F1 | Precision | Recall | Support |
|-------|----|-----------|--------|---------|
| **Fear** | **0.00** | 0.00 | 0.00 | 11 |
| **Disgust** | **0.00** | 0.00 | 0.00 | 10 |
| **Sadness** | **0.23** | 0.56 | 0.15 | 34 |

**สาเหตุหลัก:** Class Imbalance  
Fear และ Disgust มีตัวอย่างน้อยมาก เมื่อเทรนปกติ Loss function "มองข้าม" minority class ไปเพราะ:
- ถ้าโมเดล predict "ไม่มี Fear" ทุกครั้ง → ก็ผิดแค่ 11/514 ตัวอย่าง = 2.1% error เท่านั้น
- โมเดลจึง "ขี้เกียจ" ไม่ยอม learn Fear/Disgust

---

## 🔧 เทคนิคที่ทำ: Class Weighting 2 ระดับ

### ระดับที่ 1 — `pos_weight` ใน BCEWithLogitsLoss (Transformer Level)

**ทำงานตรงไหน:** ทุก Epoch ทุก Batch ขณะ Fine-tune WangchanBERTa / mBERT / XLM-RoBERTa

**สูตร Loss ปกติ (Binary Cross-Entropy):**
```
loss = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]
```

**สูตร Loss พร้อม pos_weight:**
```
loss = -[w * y * log(σ(x)) + (1-y) * log(1-σ(x))]
       ↑
       pos_weight คือตัวคูณที่ทำให้ gradient ของ positive sample หนักขึ้น
```

**ผลลัพธ์:** เมื่อโมเดลพลาดไม่ detect Fear/Disgust → ถูกลงโทษหนักกว่าปกติ 3 เท่า

#### Base Weight (อัตโนมัติจาก data ratio):
```python
base_weight[i] = neg_count[i] / pos_count[i]
```
คำนวณจาก training set เพื่อรับ imbalance ขั้นต้น

#### Manual Boost Multiplier:
```python
BOOST_FACTOR = np.array([
    1.0,   # Joy      — F1 ดีอยู่แล้ว (0.57)
    1.0,   # Anger    — F1 พอใช้ (0.25)
    2.5,   # Sadness  — F1 ต่ำ (0.23)  ← boost 2.5x
    1.0,   # Surprise — F1 พอใช้ (0.32)
    3.0,   # Fear     — F1 = 0.00 ← boost 3.0x
    3.0,   # Disgust  — F1 = 0.00 ← boost 3.0x
    1.0,   # Neutral  — F1 ดีอยู่แล้ว (0.44)
])

# Final weight = base_weight × BOOST_FACTOR
boosted_weight = base_weight * BOOST_FACTOR
pos_weight_tensor = torch.FloatTensor(boosted_weight).to(device)
```

#### ใส่เข้า Loss Function:
```python
criterion = BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
# ← เปลี่ยนจาก BCEWithLogitsLoss() ล้วนๆ ใน Notebook 04
```

---

### ระดับที่ 2 — `class_weight` ใน Random Forest (Meta-Classifier Level)

**ทำงานตรงไหน:** เมื่อ RandomForestClassifier สร้าง Decision Trees

**แนวคิด:** สร้าง estimator **แยกต่างหาก** สำหรับแต่ละ Ekman class เพื่อให้กำหนด `class_weight` ต่างกันได้อิสระ

```python
RF_CLASS_WEIGHTS = [
    'balanced',        # Joy      — auto balance
    'balanced',        # Anger    — auto balance
    {0: 1, 1: 3},     # Sadness  — negative เป็น 1x, positive เป็น 3x
    'balanced',        # Surprise — auto balance
    {0: 1, 1: 5},     # Fear     — negative เป็น 1x, positive เป็น 5x ← ลงโทษหนักมาก
    {0: 1, 1: 5},     # Disgust  — negative เป็น 1x, positive เป็น 5x ← ลงโทษหนักมาก
    'balanced',        # Neutral  — auto balance
]
```

**ผลลัพธ์:** เมื่อ Random Forest สร้าง Decision Tree node ใดก็ตาม มันจะคิด impurity โดยให้ Fear sample หนัก 5 เท่าของ non-Fear sample → ต้นไม้จะพยายาม split ให้แยก Fear ออกมาได้

#### วิธี Fit แยกต่างหาก:
```python
fitted_estimators = []
for i, est in enumerate(estimators):
    est.fit(X_train_meta, Y_train[:, i])  # fit binary classification ทีละคลาส
    fitted_estimators.append(est)
```

#### Predict:
```python
Y_pred_cols = [est.predict(X_test_meta) for est in fitted_estimators]
Y_pred = np.stack(Y_pred_cols, axis=1)   # รวมกลับเป็น (N_test, 7)
```

---

## 📊 สรุปทุกอย่างที่เปลี่ยนจาก Notebook 04

| สิ่งที่เปลี่ยน | Notebook 04 (Baseline) | Notebook 05 (Class Weighting) |
|----------------|------------------------|-------------------------------|
| `pos_weight` Transformer | `neg/pos` ratio เท่านั้น | `(neg/pos) × BOOST_FACTOR` |
| Fear `pos_weight` | ~30x base | ~90x (30 × 3.0) |
| Disgust `pos_weight` | ~40x base | ~120x (40 × 3.0) |
| Sadness `pos_weight` | ~8x base | ~20x (8 × 2.5) |
| RF Estimator | ตัวเดียว `class_weight='balanced'` | แยก 7 ตัว per class |
| RF Fear weight | balanced | `{0:1, 1:5}` |
| RF Disgust weight | balanced | `{0:1, 1:5}` |
| RF Sadness weight | balanced | `{0:1, 1:3}` |
| Save dir | `saved_models_ekman/` | `saved_models_class_weight/` |

---

## 📁 ไฟล์ที่สร้างใหม่

| ไฟล์ | วัตถุประสงค์ |
|------|-------------|
| [`notebooks/05_class_weighting.ipynb`](file:///home/apiwit/NLP_project/notebooks/05_class_weighting.ipynb) | Notebook หลัก (12 cells) |
| `results/class_weighting_comparison.png` | กราฟ Before/After F1 (สร้างหลัง run) |
| `results/eval_YYYY-MM-DD_HH-MM-SS.md` | Eval report (สร้างหลัง run) |
| `saved_models_class_weight/` | โมเดลที่เทรนด้วย class weighting |
| `saved_models_class_weight/class_weight_config.json` | บันทึก config ที่ใช้ |

---

## ⚠️ Trade-off ที่ต้องระวัง

> **Precision-Recall Trade-off:** การ boost ทำให้ Recall ของ Fear/Disgust/Sadness สูงขึ้น แต่ Precision อาจต่ำลง (โมเดล predict minority class มากขึ้น รวมทั้ง False Positive ด้วย)

หาก Macro F1 ขึ้นแต่ Micro F1 ลง → แปลว่า minority class ดีขึ้น แต่ majority class (Joy/Neutral) เสียไปบ้าง ซึ่งยอมรับได้เพราะเป้าหมายคือ balance overall performance

---

## 🚀 วิธี Run

เปิด [`notebooks/05_class_weighting.ipynb`](file:///home/apiwit/NLP_project/notebooks/05_class_weighting.ipynb) แล้ว `Run All Cells`  
เวลาประมาณ เท่ากับ Notebook 04 (~15-30 นาทีขึ้นกับ GPU)
