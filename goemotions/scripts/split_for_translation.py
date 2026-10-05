"""
split_for_translation.py
────────────────────────
แยกคอลัมน์ 'text' จาก go_emotions_test.csv ออกเป็นไฟล์ย่อย ๆ
ไฟล์ละ 500 แถว เพื่อส่งไปแปล จากนั้นบันทึกไว้ใน data/path/

Output files:
  data/path/chunk_001.csv  (row index 0–499)
  data/path/chunk_002.csv  (row index 500–999)
  ...

แต่ละไฟล์มี 2 คอลัมน์:
  - row_index : index เดิมของแถวใน CSV (ใช้ตอน merge)
  - text      : ข้อความภาษาอังกฤษที่จะส่งแปล
"""

import os
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_CSV   = "data/go_emotions_test.csv"
OUTPUT_DIR  = "data/path"
CHUNK_SIZE  = 500
# ────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📂 อ่านไฟล์: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    total_rows = len(df)
    print(f"✅ จำนวนแถวทั้งหมด: {total_rows:,} แถว")

    # เก็บเฉพาะ text + เพิ่ม row_index เพื่อใช้ตอน merge
    text_df = df[["text"]].copy()
    text_df.insert(0, "row_index", text_df.index)

    # แยกเป็น chunk
    num_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"📦 จะแยกเป็น {num_chunks} ไฟล์ (ไฟล์ละ {CHUNK_SIZE} แถว)\n")

    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end   = min(start + CHUNK_SIZE, total_rows)
        chunk = text_df.iloc[start:end]

        filename = os.path.join(OUTPUT_DIR, f"chunk_{i+1:03d}.csv")
        chunk.to_csv(filename, index=False, encoding="utf-8")
        print(f"  ✍️  {filename}  ({len(chunk)} แถว, index {start}–{end-1})")

    print(f"\n🎉 เสร็จแล้ว! ไฟล์ทั้งหมดอยู่ใน: {OUTPUT_DIR}/")
    print("─" * 60)
    print("หลังแปลเสร็จ ให้เปลี่ยนชื่อไฟล์แปลเป็น:")
    print("  chunk_001_translated.csv, chunk_002_translated.csv, ...")
    print("แล้วรัน  merge_translations.py  เพื่อรวมข้อมูล")


if __name__ == "__main__":
    main()
