# โครงงาน: A Hybrid Stacked Ensemble Framework for Multilabel Emotion Detection in Thai Text

**เป้าหมาย:** พัฒนาระบบจำแนกอารมณ์แบบหลายป้ายกำกับ (Multi-label) สำหรับข้อความภาษาไทย  
โดยใช้เทคนิคการรวมโมเดล Transformer หลายตัว (Ensemble) เข้ากับ Meta-classifier

---

## 🏗️ สถาปัตยกรรมระบบ (System Architecture)

ระบบเป็นแบบ **Hybrid Stacked Ensemble** ประกอบด้วย 3 ชั้น:

```
[ข้อความภาษาไทย]
        │
        ▼
┌───────────────────────────────────────────┐
│         LAYER 1: Base Transformers        │
│  ┌─────────────┐  ┌───────┐  ┌─────────┐ │
│  │WangchanBERTa│  │ mBERT │  │XLM-RoBERTa│ │
│  └─────────────┘  └───────┘  └─────────┘ │
│  (Fine-tune แยกกันบน Ekman 7 Classes)     │
└───────────────────────────────────────────┘
        │
        ▼ Probability Vectors (7-dim each → concat 21-dim)
        │
┌───────────────────────────────────────────┐
│      LAYER 2: Meta-Classifier             │
│         Random Forest (7 estimators)      │
│      (1 Binary Classifier ต่อ 1 Class)    │
└───────────────────────────────────────────┘
        │
        ▼
[ผลการทำนายอารมณ์ (Ekman 7 Classes)]
```

### Base Transformer Models

| Model | Pretrained | ภาษาที่ถนัด |
|-------|-----------|-------------|
| **WangchanBERTa** | `airesearch/wangchanberta-base-att-spm-uncased` | ไทย |
| **mBERT** | `bert-base-multilingual-cased` | หลายภาษา |
| **XLM-RoBERTa** | `xlm-roberta-base` | หลายภาษา |

### Ekman 7 Emotion Classes (Label Mapping จาก GoEmotions 28 → 7)

| Ekman Class | GoEmotions ดั้งเดิม (ที่ map เข้ามา) |
|-------------|--------------------------------------|
| **Joy** | admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief |
| **Anger** | anger, annoyance, disapproval |
| **Sadness** | disappointment, embarrassment, grief, remorse, sadness |
| **Surprise** | confusion, curiosity, realization, surprise |
| **Fear** | fear, nervousness |
| **Disgust** | disgust |
| **Neutral** | neutral |

---

## 📂 โครงสร้างไฟล์ (Project Structure)

```
NLP_project/
├── goemotions/data/
│   └── go_emotions_test_with_translation.csv   ← Dataset หลัก (Thai Translation)
├── notebooks/
│   ├── 01_dataset.ipynb                        ← โหลดและสำรวจ Dataset
│   ├── 02_data_cleaning.ipynb                  ← ทำความสะอาดข้อมูล
│   ├── 03_ai_labeling.ipynb                    ← AI-assisted Annotation
│   ├── 04_base_models.ipynb                    ← Run 1: Baseline 28 Classes
│   ├── 04_base_models_map_emo.ipynb            ← Run 2: Ekman Label Mapping
│   └── 05_class_weighting.ipynb                ← Run 3: Class Weighting (ล่าสุด)
├── src/
│   ├── train_thai_emotions.py                  ← Training Script หลัก
│   ├── eval_logger.py                          ← Auto Evaluation Logger
│   └── app.py                                  ← FastAPI REST API
├── results/
│   ├── SUMMARY.md                              ← Experiment Leaderboard
│   └── eval_*.md                               ← Evaluation Reports รายครั้ง
├── saved_models_ekman/                         ← โมเดล Run 2
├── saved_models_class_weight/                  ← โมเดล Run 3 (ล่าสุด)
│   ├── wangchan_model/
│   ├── mbert_model/
│   ├── xlm_roberta_model/
│   ├── rf_estimators_per_class.joblib
│   ├── label_classes.json
│   └── class_weight_config.json
├── ClassWeighting.md                           ← อธิบาย Class Weighting อย่างละเอียด
└── Training-Roadmap.md                         ← Roadmap การทดลอง
```

---

## 📊 ประวัติการทดลองและผลลัพธ์ (Experiment History)

### 🎯 Benchmark Target
> **Macro F1 = 0.4600** จาก GoEmotions Paper (ACL 2020) ใช้ BERT-base เป็น Baseline

---

### Run 1 — Baseline 28 Classes (`04_base_models.ipynb`)

**สิ่งที่ทำ:**
- ใช้ GoEmotions ดั้งเดิม **28 อารมณ์** โดยตรง ไม่มี Label Mapping
- Fine-tune 3 Transformer ด้วย `BCEWithLogitsLoss` + `pos_weight` แบบ auto (neg/pos ratio)
- เทรน Random Forest เดียว ด้วย `class_weight='balanced'`

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| RF n_estimators | 200 |
| RF class_weight | balanced |

**ผลลัพธ์:**
| Metric | Value |
|--------|-------|
| Micro F1 | 0.3177 |
| Macro F1 | 0.2498 |
| Hamming Loss | 0.1492 |

---

### Run 2 — Ekman Label Mapping (`04_base_models_map_emo.ipynb`)

**สิ่งที่เปลี่ยนจาก Run 1:**
- **ลด Label จาก 28 → 7 Classes** ด้วย Ekman Mapping (ดูตาราง Label Mapping ด้านบน)
- ที่เหลือทุกอย่างเหมือน Run 1

**ผลลัพธ์ (Per-Class F1):**
| Class | F1 | Support |
|-------|-----|---------|
| Joy | 0.57 | 186 |
| Anger | 0.25 | 73 |
| Sadness | 0.23 | 34 |
| Surprise | 0.32 | 63 |
| Fear | **0.00** ❌ | 11 |
| Disgust | **0.00** ❌ | 10 |
| Neutral | 0.44 | 177 |

| Metric | Value |
|--------|-------|
| Micro F1 | 0.4332 |
| Macro F1 | 0.2592 |
| Hamming Loss | 0.1520 |

**ปัญหาที่พบ:** Fear และ Disgust มีตัวอย่างน้อยมาก (Fear ≈ 1.7%, Disgust ≈ 2.3% ของ Dataset)
โมเดลจึง "เลือก" ไม่ทำนาย Fear/Disgust เลย เพราะ Loss ต่ำกว่าเมื่อ predict 0 ทุกครั้ง

---

### Run 3 — Class Weighting (`05_class_weighting.ipynb`) ⭐ ล่าสุด

**สิ่งที่เปลี่ยนจาก Run 2:**

**ระดับ 1 — BCEWithLogitsLoss pos_weight (Transformer Level):**

```python
# สูตร: loss = -[w * y * log(σ(x)) + (1-y) * log(1-σ(x))]
#               ↑ pos_weight คูณทับ gradient ของ positive sample

base_weight = neg_count / pos_count   # คำนวณจาก training data

BOOST_FACTOR = {
    'Joy':     1.0,   # F1 ดีอยู่แล้ว
    'Anger':   1.0,   # F1 พอใช้
    'Sadness': 2.5,   # F1 ต่ำ → boost 2.5x
    'Surprise':1.0,   # F1 พอใช้
    'Fear':    3.0,   # F1 = 0.00 → boost 3.0x
    'Disgust': 3.0,   # F1 = 0.00 → boost 3.0x
    'Neutral': 1.0,   # F1 ดีอยู่แล้ว
}

# Final pos_weight ของ Fear  ≈ 59.35 × 3.0 = 178.06
# Final pos_weight ของ Disgust ≈ 43.61 × 3.0 = 130.83
# Final pos_weight ของ Sadness ≈ 12.82 × 2.5 = 32.05
```

**ระดับ 2 — Random Forest class_weight (Meta-Classifier Level):**

```python
# สร้าง estimator แยกต่างหาก 7 ตัว (1 ตัวต่อ 1 Ekman class)
RF_CLASS_WEIGHTS = {
    'Joy':     'balanced',
    'Anger':   'balanced',
    'Sadness': {0: 1, 1: 3},   # positive sample หนัก 3x
    'Surprise':'balanced',
    'Fear':    {0: 1, 1: 5},   # positive sample หนัก 5x
    'Disgust': {0: 1, 1: 5},   # positive sample หนัก 5x
    'Neutral': 'balanced',
}
```

**ผลลัพธ์ (Per-Class F1 เปรียบเทียบ):**
| Class | Run 2 (Baseline) | Run 3 (Class Weighting) | Change |
|-------|-----------------|------------------------|--------|
| Joy | 0.57 | 0.66 | ▲ +0.09 |
| Anger | 0.25 | 0.43 | ▲ +0.18 |
| Sadness | 0.23 | **0.43** | ▲ **+0.20** ⭐ |
| Surprise | 0.32 | 0.43 | ▲ +0.11 |
| Fear | 0.00 | **0.19** | ▲ **+0.19** ⭐ |
| Disgust | 0.00 | **0.12** | ▲ **+0.12** ⭐ |
| Neutral | 0.44 | 0.56 | ▲ +0.12 |

| Metric | Run 1 (28 Class) | Run 2 (Ekman) | Run 3 (Class Weight) | 🎯 Target |
|--------|-----------------|--------------|---------------------|-----------|
| Micro F1 | 0.3177 | 0.4332 | **0.5434** | — |
| Macro F1 | 0.2498 | 0.2592 | **0.4019** | **0.4600** |
| Hamming Loss | 0.1492 | 0.1520 | **0.1373** | — |

> **Macro F1 ปัจจุบัน = 0.4019** คิดเป็น **87% ของ Benchmark Target (0.4600)**

---

## 🚀 วิธีรัน (Quick Start)

### ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### รัน Training (Notebook)
เปิด Jupyter และรัน Notebook ตามลำดับ:
```
notebooks/04_base_models_map_emo.ipynb   ← Run 2: Ekman Baseline
notebooks/05_class_weighting.ipynb       ← Run 3: Class Weighting (แนะนำ)
```

### รัน API
```bash
uvicorn src.app:app --reload
```

**ตัวอย่าง Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "วันนี้อากาศดีมากๆ เลย"}'
```

---

## 📅 แผนการปรับปรุงถัดไป (Next Steps)

| ลำดับ | เทคนิค | เป้าหมาย |
|-------|--------|---------|
| 1 | **Threshold Tuning** | ปรับ Decision Threshold รายคลาส ให้สมดุล Precision/Recall |
| 2 | **Data Augmentation** | เพิ่ม Fear/Disgust samples ด้วย Back-translation หรือ Paraphrase |
| 3 | **Focal Loss** | แทน BCEWithLogitsLoss ด้วย Focal Loss เพื่อ focus minority class อัตโนมัติ |
| 4 | **Epochs Tuning** | เพิ่ม Epochs จาก 3 เป็น 5-10 พร้อม Early Stopping |
| 5 | **Ensemble Weights** | กำหนดน้ำหนักต่างกันสำหรับแต่ละ Base Model แทน concat เท่ากัน |

---

## 📝 อ้างอิง (References)

- Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). **GoEmotions: A Dataset of Fine-Grained Emotions**. ACL 2020.
- WangchanBERTa: [airesearch/wangchanberta-base-att-spm-uncased](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased)
- mBERT: [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)
- XLM-RoBERTa: [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)