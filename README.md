# โครงงาน: A Hybrid Stacked Ensemble Framework for Multilabel Emotion Detection in Thai Text (Version 1 - v1)
**เป้าหมาย:** พัฒนาระบบจำแนกอารมณ์แบบหลายป้ายกำกับ (Multi-label) สำหรับข้อความภาษาไทย โดยใช้เทคนิคการรวมโมเดล Transformer หลายตัว (Ensemble) เข้ากับ Meta-classifier 

---

## 📅 ตารางสรุปแผนการดำเนินงาน (Project Timeline)

| ระยะที่ (Phase) | รายละเอียดงาน | ผลลัพธ์ที่คาดหวัง (Deliverables) |
| :--- | :--- | :--- |
| **Phase 1** | ศึกษาและกำหนดขอบเขต | กำหนดรายการอารมณ์ และเลือกสถาปัตยกรรมโมเดล |
| **Phase 2** | รวบรวมและจัดการข้อมูล | Dataset ภาษาไทยที่ผ่านการทำความสะอาดและติดป้ายกำกับ |
| **Phase 3** | พัฒนา Base Models | โมเดล Transformer ย่อย (เช่น WangchanBERTa, mBERT) |
| **Phase 4** | พัฒนาระบบ Ensemble | ระบบรวมผลลัพธ์ผ่าน Meta-classifier (เช่น Random Forest) |
| **Phase 5** | วัดผลและปรับปรุงโมเดล | รายงานผล F1-score และกราฟเปรียบเทียบประสิทธิภาพ |
| **Phase 6** | การนำไปใช้งานและสรุปผล | API สำหรับวิเคราะห์ข้อความ และเล่มปริญญานิพนธ์/รายงาน |

---

## 🛠️ ขั้นตอนการดำเนินงานอย่างละเอียด (Step-by-Step)

### Phase 1: การเตรียมตัวและวางแผน (Preparation & Scoping)
* **กำหนดหมวดหมู่อารมณ์:** เลือกอารมณ์พื้นฐานที่จะให้ระบบตรวจจับ (เช่น ดีใจ, เศร้า, โกรธ, กลัว, ประหลาดใจ, ขยะแขยง, เป็นกลาง)
* **เลือก Base Models:** กำหนดโมเดล Transformer ที่จะนำมาทำ Ensemble (แนะนำให้มีอย่างน้อย 1 ตัวที่เป็นโมเดลภาษาไทยเฉพาะ เช่น WangchanBERTa คู่กับ mBERT หรือ XLM-RoBERTa)
``` เลือก WangchanBERTa + XLM-RoBERTa + mBERT ```
* **กำหนดเครื่องมือ:** เลือก Framework (เช่น PyTorch หรือ Hugging Face `transformers`) และไลบรารีวิเคราะห์ข้อมูล (เช่น `scikit-learn`)
```PyTorch + Hugging Face ในการโหลดและเทรนโมเดลภาษาไทย จากนั้นส่งต่อผลลัพธ์ให้ Scikit-learn```

### Phase 2: การรวบรวมและจัดการข้อมูล (Data Pipeline & Annotation)
* **การดึงข้อมูล (Web Scraping):** รวบรวมข้อความภาษาไทยจาก X (Twitter), YouTube Comments หรือ Pantip 
* **การทำความสะอาด (Preprocessing):** 
  * ใช้ `PyThaiNLP` ตัดคำ ลบเว้นวรรค และกรองอักขระขยะ
  * จัดการ Emoji (คงไว้หรือแปลงเป็นข้อความ)
* **การติดป้ายกำกับ (Data Annotation):** 
  * ใช้เครื่องมืออย่าง **Label Studio** เพื่อให้ทีมงานช่วยกันกำหนดอารมณ์
  * *ข้อควรระวัง:* 1 ข้อความต้องสามารถเลือกได้หลายอารมณ์ (Multi-label)
* **แบ่งสัดส่วนข้อมูล:** แบ่งเป็น Train (70%), Validation (15%) และ Test (15%)

### Phase 3: การพัฒนาโมเดลพื้นฐาน (Base Models Training)
* **Fine-Tuning:** นำข้อมูลไปสอนโมเดล Transformer แต่ละตัวให้จดจำอารมณ์ภาษาไทย (ฝึกแยกกันทีละโมเดล)
* **Extract Predictions:** สกัดผลการทำนายของแต่ละโมเดล (ค่าความน่าจะเป็น/Probabilities) ออกมา เพื่อเตรียมส่งเป็น Input ให้ชั้นต่อไป

### Phase 4: การพัฒนาระบบรวมพลัง (Hybrid Stacked Ensemble)
* **สร้าง Decision Layer:** นำผลลัพธ์ (Probabilities) จาก Phase 3 มารวมกัน (Concatenation)
* **Train Meta-Classifier:** นำชุดข้อมูลที่รวมกันแล้ว ไปสอนอัลกอริทึม Machine Learning ดั้งเดิม (เช่น Random Forest, XGBoost หรือ SVM) เพื่อให้เรียนรู้ว่าควร "เชื่อ" โมเดล Transformer ตัวไหนในบริบทไหนมากที่สุด

### Phase 5: การวัดผลและปรับปรุง (Evaluation & Optimization)
* **วัดประสิทธิภาพด้วย Metrics สำหรับ Multi-label:**
  * **F1-Score (Micro / Macro):** เป็นตัวชี้วัดหลัก
  * **Hamming Loss:** วัดสัดส่วนการทายผิดแบบรายป้ายกำกับ
* **จัดการช่องโหว่:** ปรับแต่ง Hyperparameters เพื่อแก้ปัญหา Overfitting หรือความล่าช้าในการประมวลผล (Inference Time)

### Phase 6: การนำไปใช้งาน (Deployment & Reporting)
* **สร้าง API:** ห่อระบบด้วย FastAPI เพื่อให้สามารถส่งข้อความผ่าน HTTP Request แล้วรับผลลัพธ์อารมณ์เป็น JSON คืนกลับมา
* **วิเคราะห์ผล (Error Analysis):** ดึงประโยคที่โมเดลทายผิดมาวิเคราะห์สาเหตุ
* **จัดทำเอกสาร:** สรุปผลการทดลอง อภิปรายข้อค้นพบ และเขียนรายงานโครงงาน

---

## 📌 สรุปสถานะและรายงานผลการทดลอง (Version 1 - v1)

> **สถานะโครงงานปัจจุบัน:** สำเร็จในระดับ **v1 Prototype** (พัฒนาโมเดล Ensemble + Meta-Classifier + FastAPI สำเร็จครบทุก Phase)

### 📊 รายงานผลการดำเนินงาน (v1 Summary)
1. **สถาปัตยกรรม:** พัฒนาโครงสร้าง **Hybrid Stacked Ensemble** โดยใช้ 3 Base Models (`WangchanBERTa`, `mBERT`, `XLM-RoBERTa`) ส่งออก Probability Vector (7 อารมณ์) Concatenate รวมเป็น 21 Features แล้วใช้ **`Random Forest`** เป็น Meta-Classifier ตัวตัดสินใจหลัก
2. **การนำไปใช้งาน (Deployment):** พัฒนา REST API ด้วย **FastAPI** (`app.py`) และมีฟังก์ชัน `predict_emotion()` พร้อมใช้งาน

### ⚠️ ข้อค้นพบและวิเคราะห์ปัญหาใน v1 (Error Analysis & Limitations)
จากการทดสอบและวิเคราะห์ผลการทำนาย พบว่า **โมเดลยังทำนายอารมณ์ได้ไม่ดีเท่าที่ควร** โดยมีข้อค้นพบสำคัญดังนี้:

* **สาเหตุหลักมาจาก คุณภาพของชุดข้อมูล (Dataset Quality Issue / Label Noise):**
  * ข้อมูลที่นำมาใช้อยู่ในสภาวะที่มี **Label Noise (การติดป้ายกำกับไม่ตรงกับความหมายจริง)** สูง
  * **ตัวอย่างข้อผิดพลาดที่พบ:** ข้อความประเภทคำอุทานหรือประโยคก้ำกวม เช่นคำว่า *"อืมม..."* หรือ *"โอเค"* ซึ่งโดยธรรมชาติควรติดป้ายกำเป็น **`Neutral (เฉยๆ)`** แต่ใน Dataset กลับถูกบันทึกป้ายกำกับเป็น **`Joy (ดีใจ)`**
  * ทำให้ Transformer Base Models และ Random Forest สับสนในการเรียนรู้ Pattern ความสัมพันธ์ระหว่างข้อความกับอารมณ์

---

## 🚀 แผนการปรับปรุงสำหรับเวอร์ชันถัดไป (v2 Roadmap)
1. **Data Cleaning & Re-annotation:** ทำการตรวจสอบและแก้ไข Label Noise ในชุดข้อมูล โดยเน้นการจัดหมวดหมู่ข้อความก้ำกวมและคำอุทานใหม่ให้ถูกต้องตามบริบทจริง
2. **Class Imbalance & Threshold Tuning:** ปรับปรุงน้ำหนัก `pos_weight` และปรับ Decision Threshold รายอารมณ์แยกตามความยากง่ายของแต่ละ Class
3. **Data Augmentation:** เพิ่มชุดข้อมูลตัวอย่างสำหรับกลุ่มอารมณ์ที่มีจำนวนน้อย (Minority Classes) เพื่อลดการทำนายเอนเอียง (Bias)