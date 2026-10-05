"""
merge_translations.py
─────────────────────
รวมไฟล์แปลทั้งหมด (full_text_translate.txt) กลับเข้ากับ
ข้อมูลเดิม (go_emotions_test.csv) โดยใช้ row_index เป็นตัวจับคู่

โครงสร้างของ TRANSLATED_FILE:
  row_index, text, text_th

Output:
  data/go_emotions_test_with_translation.csv
  คอลัมน์: text (eng), text_th (thai), labels, id
"""

import os
from pathlib import Path
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
# ใช้ directory ของ script เป็น base เสมอ (ไม่ขึ้นกับ working directory)
BASE_DIR         = Path(__file__).parent
INPUT_CSV        = BASE_DIR / "data/go_emotions_test.csv"
TRANSLATED_FILE  = BASE_DIR / "data/full_text_translate_fixed.txt"   # ไฟล์แปลที่ fix แล้ว
OUTPUT_CSV       = BASE_DIR / "data/go_emotions_test_with_translation.csv"

# ชื่อคอลัมน์ในไฟล์แปลที่เก็บข้อความภาษาไทย
TRANSLATED_COL   = "text_th"
# ────────────────────────────────────────────────────────────────────────────

def main():
    # 1) อ่านไฟล์ต้นฉบับ
    print(f"📂 อ่านไฟล์ต้นฉบับ: {INPUT_CSV}")
    df_original = pd.read_csv(INPUT_CSV)
    # เพิ่ม row_index เฉพาะเมื่อยังไม่มี (ป้องกัน column conflict ตอน merge)
    if "row_index" not in df_original.columns:
        df_original["row_index"] = df_original.index
    print(f"✅ แถวทั้งหมด: {len(df_original):,}")

    # 2) อ่านไฟล์แปลไฟล์เดียว
    if not os.path.exists(TRANSLATED_FILE):
        print(f"\n❌ ไม่พบไฟล์แปล: {TRANSLATED_FILE}")
        return

    print(f"\n📖 อ่านไฟล์แปล: {TRANSLATED_FILE}")
    df_translated = pd.read_csv(TRANSLATED_FILE)
    print(f"   พบ {len(df_translated):,} แถว, คอลัมน์: {list(df_translated.columns)}")

    # ตรวจว่ามีคอลัมน์ที่ต้องการครบ
    if "row_index" not in df_translated.columns:
        print("❌ ไม่พบคอลัมน์ 'row_index' ในไฟล์แปล")
        return
    if TRANSLATED_COL not in df_translated.columns:
        # พยายาม rename คอลัมน์อื่นที่ไม่ใช่ row_index / text
        other_cols = [c for c in df_translated.columns if c not in ("row_index", "text")]
        if other_cols:
            df_translated = df_translated.rename(columns={other_cols[0]: TRANSLATED_COL})
            print(f"   ✍️  rename '{other_cols[0]}' → '{TRANSLATED_COL}'")
        else:
            print(f"❌ ไม่พบคอลัมน์ '{TRANSLATED_COL}' ในไฟล์แปล")
            return

    # เก็บเฉพาะคอลัมน์ที่จำเป็น
    df_translated = df_translated[["row_index", TRANSLATED_COL]].drop_duplicates(subset="row_index")

    # 3) Merge กับข้อมูลเดิมโดยใช้ row_index
    df_merged = df_original.merge(df_translated, on="row_index", how="left")

    # 4) จัดเรียงคอลัมน์: text (eng), text_th, labels, id
    desired_cols = ["text", TRANSLATED_COL, "labels", "id"]
    cols = [c for c in desired_cols if c in df_merged.columns]
    df_merged = df_merged[cols]

    # 5) บันทึก
    df_merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    translated_count = df_merged[TRANSLATED_COL].notna().sum()
    print(f"\n🎉 รวมข้อมูลเสร็จแล้ว!")
    print(f"   แถวที่มีคำแปล : {translated_count:,} / {len(df_merged):,}")
    print(f"   บันทึกไว้ที่   : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
