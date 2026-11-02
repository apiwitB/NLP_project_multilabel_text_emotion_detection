# โครงงาน: A Hybrid Stacked Ensemble Framework for Multilabel Emotion Detection in Thai Text

**เป้าหมาย:** พัฒนาระบบจำแนกอารมณ์แบบหลายป้ายกำกับ (Multi-label) สำหรับข้อความภาษาไทย  
โดยใช้เทคนิคการรวมโมเดล Transformer หลายตัว (Ensemble) เข้ากับ Meta-classifier พร้อมรับมือกับปัญหาความไม่สมดุลของข้อมูลอย่างรุนแรง (Severe Class Imbalance)

---

## 🏆 ผลลัพธ์สุดท้าย (Final Benchmark Achievement)

> **🎯 Benchmark Target (GoEmotions Paper ACL 2020):** `Macro F1 = 0.4600`  
> **🚀 ผลลัพธ์รอบล่าสุดของเรา (Final Run):** **`Macro F1 = 0.4988`** และ **`Micro F1 = 0.6004`**  
> *(สามารถเอาชนะเกณฑ์มาตรฐานของ GoEmotions Paper ได้สำเร็จด้วยเทคนิค Two-Level Class Weighting + Modular Fine-tuning)*

---

## 🏗️ สถาปัตยกรรมระบบ (System Architecture)

ระบบเป็นแบบ **Hybrid Stacked Ensemble** ประกอบด้วย 2 ระดับหลัก:

```
[ข้อความภาษาไทย]
        │
        ▼
┌────────────────────────────────────────────────────────┐
│         LAYER 1: Base Transformers (Fine-tuned)        │
│  ┌─────────────┐       ┌───────┐       ┌─────────────┐ │
│  │WangchanBERTa│       │ mBERT │       │ XLM-RoBERTa │ │
│  └─────────────┘       └───────┘       └─────────────┘ │
│  (Fine-tune แยกกันบน Ekman 7 Classes + BCE pos_weight) │
└────────────────────────────────────────────────────────┘
        │
        ▼ Probability Vectors (7-dim each → Concat เป็น 21-dim)
        │
┌────────────────────────────────────────────────────────┐
│      LAYER 2: Meta-Classifier (Decision Level)         │
│         Random Forest (7 Binary Estimators)            │
│  (ปรับ class_weight พิเศษสำหรับ Minority Classes)      │
└────────────────────────────────────────────────────────┘
        │
        ▼
[ผลการทำนายอารมณ์ภาษาไทย (Ekman 7 Classes)]
```

### Base Transformer Models
| Model | Pretrained Identifier | ภาษาที่ถนัด |
|-------|----------------------|-------------|
| **WangchanBERTa** | `airesearch/wangchanberta-base-att-spm-uncased` | ไทย (Monolingual RoBERTa-based) |
| **mBERT** | `bert-base-multilingual-cased` | หลายภาษา (Multilingual) |
| **XLM-RoBERTa** | `xlm-roberta-base` | หลายภาษา (Cross-lingual RoBERTa) |

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

## 📂 โครงสร้างไฟล์และไฟล์ที่ใช้ในแต่ละรอบ (File Guide & Pipeline)

```
NLP_project/
├── goemotions/data/
│   ├── go_emotions_train_with_translation.csv      ← Full Train Set ภาษาไทย (~43k แถว)
│   ├── go_emotions_validation_with_translation.csv ← Validation Set ภาษาไทย (~5.4k แถว)
│   └── go_emotions_test_with_translation.csv       ← Test Set ภาษาไทย (~5.1k แถว)
│
├── notebooks/
│   ├── 01_dataset.ipynb                            ← สำรวจ Dataset
│   ├── 02_data_cleaning.ipynb                      ← ทำความสะอาดข้อมูล (PyThaiNLP normalize + Regex)
│   ├── 04_base_models_full_train.ipynb             ← [Run 1 & Run 4] โมเดล 28 Classes บน Full Train Set
│   ├── 06_ekman_class_weighting_full.ipynb         ← [Run 5] สคริปต์ Ekman Full Train + 2-Level Weighting
│   └── fine_tune_ekman_class_weight.ipynb          ← ⭐ [Run 6 - Final Run] โมดูลาร์สมบูรณ์แบบ
│
├── saved_models_ekman_finetune/                    ← 📁 โฟลเดอร์เก็บโมเดลที่ดีที่สุดรอบสุดท้าย (Final Best Models)
│   ├── wangchan_model/                             ← Best Checkpoint WangchanBERTa
│   ├── mbert_model/                                ← Best Checkpoint mBERT
│   ├── xlm_roberta_model/                          ← Best Checkpoint XLM-RoBERTa
│   └── random_forest_ekman_weighted.joblib         ← Fitted RF Meta-learner
│
├── results/
│   ├── SUMMARY.md                                  ← Leaderboard สรุปการทดลองทุกรอบ
│   └── eval_2026-10-25_21-45-41.md                 ← รายงานผลการทดสอบรอบสุดท้ายอย่างละเอียด
└── README.md
```

---

## 📊 ประวัติและวิวัฒนาการของการทดลอง (Experiment History)

| รอบ (Run) | ชื่อการทดลอง / ไฟล์หลักที่ใช้ | Micro F1 | Macro F1 | สถานะ / สรุปสาระสำคัญ |
| :---: | :--- | :---: | :---: | :--- |
| **Target** | [**GoEmotions Paper Benchmark (ACL 2020)**](results/baseline_goemotions_paper.md) | **-** | **`0.4600`** | **เกณฑ์มาตรฐานอ้างอิงของชุดข้อมูลนี้** |
| **Run 1** | `04_base_models.ipynb` (Baseline 28 Classes) | 0.3177 | 0.2498 | ทดสอบ Stacked Ensemble เริ่มต้นบน 28 อารมณ์โดยตรง |
| **Run 2** | `04_base_models_map_emo.ipynb` (Ekman 7 Classes) | 0.4332 | 0.2592 | ลด Label เหลือ 7 คลาส แต่ติดปัญหา **Fear และ Disgust ได้ F1 = 0.00** |
| **Run 3** | `05_class_weighting.ipynb` (Class Weighting Pilot) | 0.5434 | 0.4019 | เริ่มนำ Two-Level Weighting มา Boost minority classes ดัน F1 Fear/Disgust ขึ้นมาได้ |
| **Run 4** | `04_base_models_full_train.ipynb` (Full Train 43k) | 0.4629 | 0.3331 | ขยายชุดข้อมูลเป็น Full Dataset (แปลครบทั้ง 43,410 แถว) |
| **Run 5** | `06_ekman_class_weighting_full.ipynb` | 0.5835 | 0.4833 | รัน Ekman + Two-Level Weighting บน Full Train ชนะ Benchmark ได้ครั้งแรก |
| **Run 6 ⭐** | `fine_tune_ekman_class_weight.ipynb` **(รอบสุดท้าย)** | **`0.6004`** | **`0.4988`** | **สถาปัตยกรรมสมบูรณ์แบบ + ระบบ Best Checkpoint + LR Scheduler** |

---

## 🔬 เจาะลึกรอบสุดท้าย (Final Run: Run 6)

### 1. ไฟล์หลักที่ใช้:
- **Notebook:** [`notebooks/fine_tune_ekman_class_weight.ipynb`](notebooks/fine_tune_ekman_class_weight.ipynb)
- **Model Checkpoints:** เก็บไว้ที่ `saved_models_ekman_finetune/`
- **Full Report:** [`results/eval_2026-10-25_21-45-41.md`](results/eval_2026-10-25_21-45-41.md)

### 2. สิ่งที่ปรับปรุงในรอบสุดท้าย:
1. **Modular Architecture & Independent Training:** แยกการจูน Hyperparameters ของแต่ละ Base Transformer อย่างอิสระ
2. **Learning Rate Scheduler:** ใช้ Cosine Annealing พร้อม Warmup ป้องกันโมเดลแกว่ง
3. **Best Checkpoint & Rollback System:** ติดตามค่า Validation Loss อย่างใกล้ชิดเพื่อบันทึกเฉพาะโมเดลใน Epoch ที่ดีที่สุด (ป้องกันปัญหา Overfitting)
4. **Two-Level Class Weighting:**
   - **Level 1 (Transformer Loss):** ปรับแต่ง `pos_weight` ใน `BCEWithLogitsLoss` โดยเพิ่ม Manual Boost Factor: **Fear $\times 3.0$**, **Disgust $\times 3.0$**, **Sadness $\times 2.5$**
   - **Level 2 (Meta-Classifier):** กำหนด `class_weight` ใน Random Forest ให้กับคลาสส่วนน้อยโดยเฉพาะ: Fear `{0:1, 1:5}`, Disgust `{0:1, 1:5}`, Sadness `{0:1, 1:3}`

### 3. ผลลัพธ์สุดท้ายรายคลาส (Per-Class F1 Score บน Unseen Test Set 5,131 ตัวอย่าง):
| Class | Precision | Recall | **F1-Score** | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Joy** | 0.7449 | 0.7286 | **0.7367** | 2008 |
| **Neutral** | 0.5354 | 0.6335 | **0.5804** | 1670 |
| **Surprise** | 0.4882 | 0.5457 | **0.5154** | 645 |
| **Sadness** | 0.5275 | 0.4540 | **0.4880** | 359 |
| **Anger** | 0.4500 | 0.4716 | **0.4606** | 687 |
| **Fear** | 0.4500 | 0.4091 | **0.4286** | 88 |
| **Disgust** | 0.5000 | 0.1966 | **0.2822** | 117 |
| **เฉลี่ยรวม (Overall)** | | | | |
| **Micro F1** | | | **`0.6004`** | |
| **Macro F1** | | | **`0.4988`** | *(Benchmark Paper: 0.4600)* |
| **Hamming Loss** | | | **`0.1267`** | *(ทายผิดเฉลี่ยเพียง 12.67%)* |

---

## 🚀 วิธีรันโมเดลรอบสุดท้าย (Quick Start)

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. รันเทรนและประเมินผลรอบสมบูรณ์
เปิด Jupyter Notebook และรัน:
```
notebooks/fine_tune_ekman_class_weight.ipynb
```
*(สมุดงานมีระบบแคชและบันทึกโมเดลไว้ที่ `saved_models_ekman_finetune/` โดยอัตโนมัติ)*

---

## 📝 อ้างอิง (References)
- Demszky, D., et al. (2020). **GoEmotions: A Dataset of Fine-Grained Emotions**. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*.
- Saleh, S. N. (2025). **Enhancing multilabel classification for unbalanced COVID-19 vaccination hesitancy tweets using ensemble learning**. *Computers in Biology and Medicine*.
- Khan, B., Usman, M., & Binsawad, M. (2026). **A novel hybrid model for emotion detection in text through sequential and transformer-based approaches: LSTM enhanced RoBERTa (LER)**. *Scientific Reports*.
- WangchanBERTa: [airesearch/wangchanberta-base-att-spm-uncased](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased)
- mBERT: [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)
- XLM-RoBERTa: [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
