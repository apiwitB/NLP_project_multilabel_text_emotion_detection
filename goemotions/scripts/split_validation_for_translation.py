"""
split_validation_for_translation.py
───────────────────────────────────
แยกคอลัมน์ 'text' จาก go_emotions_validation.csv ออกเป็นไฟล์ย่อย ๆ
ไฟล์ละ 1,000 แถว เพื่อนำไปแปล พร้อมคอลัมน์ "row_index", "text"
บันทึกไว้ในโฟลเดอร์ data/validation_chunks/

Output files:
  data/validation_chunks/chunk_001.csv  (row index 0–999)
  data/validation_chunks/chunk_002.csv  (row index 1000–1999)
  ...

คอลัมน์:
  - row_index : index เดิมของแถวใน go_emotions_validation.csv (ใช้สำหรับ merge กลับ)
  - text      : ข้อความภาษาอังกฤษ
"""

import os
import csv
from pathlib import Path
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
INPUT_CSV   = BASE_DIR / "data" / "go_emotions_validation.csv"
OUTPUT_DIR  = BASE_DIR / "data" / "validation_chunks"
CHUNK_SIZE  = 1000
# ────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📂 กำลังตรวจสอบไฟล์: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        print(f"❌ ไม่พบไฟล์: {INPUT_CSV}")
        return

    # อ่านเฉพาะคอลัมน์ text เป็น chunk ขนาด 1,000 แถว (สตรีมมิ่งเพื่อประหยัด RAM และรวดเร็ว)
    print(f"📦 กำลังอ่านและแยกไฟล์ละ {CHUNK_SIZE:,} แถว...")
    print(f"📁 โฟลเดอร์ปลายทาง: {OUTPUT_DIR}\n")

    chunk_idx = 1
    total_rows = 0

    reader = pd.read_csv(INPUT_CSV, usecols=["text"], chunksize=CHUNK_SIZE)
    for chunk in reader:
        # กำหนด row_index ให้ตรงกับตำแหน่งแถวเดิมในไฟล์ validation
        chunk.insert(0, "row_index", range(total_rows, total_rows + len(chunk)))

        filename = OUTPUT_DIR / f"chunk_{chunk_idx:03d}.csv"
        chunk.to_csv(filename, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")

        start = total_rows
        end = total_rows + len(chunk)
        print(f"  ✍️  {filename.name}  ({len(chunk):,} แถว | row_index {start} ถึง {end-1})")

        total_rows += len(chunk)
        chunk_idx += 1

    print(f"\n🎉 แยกไฟล์เรียบร้อยทั้งหมด {chunk_idx - 1} ไฟล์ (รวม {total_rows:,} แถว)!")
    print(f"📂 ไฟล์ทั้งหมดถูกบันทึกไว้ใน: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
