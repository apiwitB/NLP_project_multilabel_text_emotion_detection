"""
split_train_for_translation.py
──────────────────────────────
แยกคอลัมน์ 'text' จาก go_emotions_train.csv ออกเป็นไฟล์ย่อย ๆ
ไฟล์ละ 2,000 แถว เพื่อนำไปแปล พร้อมคอลัมน์ "row_index", "text"
บันทึกไว้ในโฟลเดอร์ data/train_chunks/

Output files:
  data/train_chunks/chunk_001.csv  (row index 0–1999)
  data/train_chunks/chunk_002.csv  (row index 2000–3999)
  ...
  data/train_chunks/chunk_022.csv  (row index 42000–43409)

คอลัมน์:
  - row_index : index เดิมของแถวใน go_emotions_train.csv (ใช้สำหรับ merge กลับ)
  - text      : ข้อความภาษาอังกฤษ
"""

import os
import csv
from pathlib import Path
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
# อ้างอิง path สัมพัทธ์จากตำแหน่งของโฟลเดอร์ goemotions เสมอ
BASE_DIR    = Path(__file__).resolve().parent.parent
INPUT_CSV   = BASE_DIR / "data" / "go_emotions_train.csv"
OUTPUT_DIR  = BASE_DIR / "data" / "train_chunks"
CHUNK_SIZE  = 2000
# ────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📂 กำลังอ่านไฟล์: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        print(f"❌ ไม่พบไฟล์: {INPUT_CSV}")
        return

    # อ่านเฉพาะคอลัมน์ text หรืออ่านทั้งหมดแล้วเลือก text
    df = pd.read_csv(INPUT_CSV, usecols=["text"])

    total_rows = len(df)
    print(f"✅ จำนวนแถวทั้งหมด: {total_rows:,} แถว")

    # ใส่ row_index นำหน้า
    df.insert(0, "row_index", df.index)

    # คำนวณจำนวนไฟล์ chunk
    num_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"📦 จะแยกเป็น {num_chunks} ไฟล์ (ไฟล์ละ {CHUNK_SIZE:,} แถว)")
    print(f"📁 โฟลเดอร์ปลายทาง: {OUTPUT_DIR}\n")

    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end   = min(start + CHUNK_SIZE, total_rows)
        chunk = df.iloc[start:end]

        filename = OUTPUT_DIR / f"chunk_{i+1:03d}.csv"
        # ใช้ quoting=csv.QUOTE_ALL เพื่อให้ครอบ "row_index","text" และป้องกันข้อความที่มี comma/newline ปน
        chunk.to_csv(filename, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")
        print(f"  ✍️  {filename.name}  ({len(chunk):,} แถว | row_index {start} ถึง {end-1})")

    print(f"\n🎉 แยกไฟล์เรียบร้อยทั้งหมด {num_chunks} ไฟล์!")
    print(f"📂 ไฟล์ทั้งหมดถูกบันทึกไว้ใน: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
