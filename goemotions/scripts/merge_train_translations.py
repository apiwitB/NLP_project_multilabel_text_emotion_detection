"""
merge_train_translations.py
────────────────────────────
รวมไฟล์แปลภาษาไทยทั้งหมดจาก data/train_chunks-thai/ (1.txt ถึง 22.txt)
เข้ากับข้อมูลชุดสอนเดิม (go_emotions_train.csv)

Output:
  1. data/go_emotions_train_with_translation.csv
     คอลัมน์: text, text_th, labels, id
  2. data/full_train_text_translate.csv
     คอลัมน์: row_index, text, text_th
  3. data/missing_train_translations.csv (ถ้ามีแถวที่ขาด)
     คอลัมน์: row_index, text
"""

import os
import sys
import re
import csv
import io
from pathlib import Path
import pandas as pd

# เพิ่มขนาด field limit สำหรับข้อความยาว/ASCII art
csv.field_size_limit(sys.maxsize)

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parent.parent
INPUT_CSV         = BASE_DIR / "data" / "go_emotions_train.csv"
CHUNKS_DIR        = BASE_DIR / "data" / "train_chunks-thai"
OUTPUT_MERGED_CSV = BASE_DIR / "data" / "go_emotions_train_with_translation.csv"
OUTPUT_TEXT_CSV   = BASE_DIR / "data" / "full_train_text_translate.csv"
OUTPUT_MISSING    = BASE_DIR / "data" / "missing_train_translations.csv"
FIX_FILE          = BASE_DIR / "data" / "missing_train_translations_fix.txt"

# Pattern เริ่มต้นเรคอร์ด: ตัวเลข row_index ตามด้วย delimiter (, หรือ ; หรือ |)
REC_START = re.compile(r'^\s*"?(\d[\d;]*)"?\s*([,;|])\s*(.*)', re.DOTALL)


def parse_record_body(body: str):
    """แยก (text_en, text_th) จาก body ของแต่ละแถวอย่างรัดกุม"""
    # 1) ลองใช้ csv.reader
    try:
        reader = list(csv.reader(io.StringIO(body)))
        if reader and len(reader[0]) >= 2:
            return reader[0][0].strip(), reader[0][-1].strip()
    except Exception:
        pass

    # 2) กรณีใช้ pipe separator
    if "|" in body:
        parts = [p.strip().strip('"') for p in body.split("|")]
        if len(parts) >= 2:
            return parts[0], parts[-1]

    # 3) Regex กรณี "text","text_th"
    m = re.match(r'^"?(.*?)"?\s*,\s*"?([^"]*)"?$', body, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return "", body.strip()


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print(f"📂 1. อ่านไฟล์ต้นฉบับ: {INPUT_CSV}")
    df_train = pd.read_csv(INPUT_CSV)
    # ตัด BOM ถ้ามี
    if df_train.columns[0].startswith('\ufeff'):
        df_train = df_train.rename(columns={df_train.columns[0]: df_train.columns[0].replace('\ufeff', '')})

    total_rows = len(df_train)
    print(f"   พบข้อมูลทั้งหมด: {total_rows:,} แถว")
    train_texts = df_train["text"].tolist()

    # Index ข้อความภาษาอังกฤษต้นฉบับ เพื่อใช้แมปกรณี row_index มี typo
    text_to_indices = {}
    for idx, text in enumerate(df_train["text"]):
        t_clean = str(text).strip()
        text_to_indices.setdefault(t_clean, []).append(idx)

    # 2. อ่านและ parse ทุก chunk ใน train_chunks-thai/
    print(f"\n📂 2. กำลังรวมไฟล์จาก: {CHUNKS_DIR}")
    chunk_files = sorted(CHUNKS_DIR.glob("*.txt"), key=lambda p: int(re.search(r'(\d+)', p.name).group(1)))
    print(f"   พบ chunk files ทั้งหมด: {len(chunk_files)} ไฟล์")

    final_th = [None] * total_rows
    recovered_count = 0

    for f in chunk_files:
        c_num = int(re.search(r'(\d+)', f.name).group(1))
        expected_start = (c_num - 1) * 2000
        expected_end = min(c_num * 2000, total_rows)

        with open(f, "r", encoding="utf-8", errors="replace") as fp:
            lines = fp.read().splitlines()

        records = []
        curr_idx_str = None
        curr_body = []

        for line in lines:
            line_s = line.strip()
            if not line_s:
                continue
            # ข้าม header
            if line_s.lower().startswith('row_index') or line_s.lower().startswith('"row_index"'):
                continue

            m = REC_START.match(line)
            if m:
                if curr_idx_str is not None:
                    records.append((curr_idx_str, "\n".join(curr_body)))
                curr_idx_str = m.group(1)
                curr_body = [m.group(3)]
            else:
                if curr_idx_str is not None:
                    curr_body.append(line)

        if curr_idx_str is not None:
            records.append((curr_idx_str, "\n".join(curr_body)))

        chunk_recovered = 0
        for idx_str, body in records:
            raw_idx = int(re.sub(r'\D', '', idx_str))
            text_en, text_th = parse_record_body(body)

            assigned_idx = None
            if text_en and text_en in text_to_indices:
                candidates = text_to_indices[text_en]
                matching_in_chunk = [c for c in candidates if expected_start <= c < expected_end]
                if matching_in_chunk:
                    assigned_idx = matching_in_chunk[0]
                elif raw_idx in candidates:
                    assigned_idx = raw_idx
                else:
                    assigned_idx = candidates[0]
            else:
                if 0 <= raw_idx < total_rows:
                    assigned_idx = raw_idx

            if assigned_idx is not None and 0 <= assigned_idx < total_rows:
                if final_th[assigned_idx] is None:
                    chunk_recovered += 1
                final_th[assigned_idx] = text_th

        recovered_count += chunk_recovered
        print(f"   - {f.name:8s}: ดึงได้ {chunk_recovered:4d} แถว (ช่วงเป้าหมาย: {expected_start}..{expected_end-1})")

    # 2.1 อ่านไฟล์แก้คำแปลเสริมถ้ามี (FIX_FILE)
    if FIX_FILE.exists():
        print(f"\n📂 2.1 กำลังนำคำแปลเสริมจาก: {FIX_FILE.name} มาเติม")
        with open(FIX_FILE, "r", encoding="utf-8", errors="replace") as fp:
            reader = csv.reader(fp)
            fix_count = 0
            for row in reader:
                if not row or row[0].strip().lower().startswith("row_index"):
                    continue
                raw_idx_str = row[0].strip()
                raw_idx = int(re.sub(r'\D', '', raw_idx_str)) if re.sub(r'\D', '', raw_idx_str) else None
                text_en = row[1].strip() if len(row) > 1 else ""
                text_th = row[2].strip() if len(row) > 2 else ""
                # ทำความสะอาด trailing punctuation / quote ที่อาจหลงมาตอนท้าย
                text_th = re.sub(r'[\."\s]+$', '', text_th)

                assigned_idx = None
                if raw_idx is not None and 0 <= raw_idx < total_rows:
                    if train_texts[raw_idx].strip() == text_en.strip():
                        assigned_idx = raw_idx

                if assigned_idx is None:
                    if text_en and text_en in text_to_indices:
                        candidates = text_to_indices[text_en]
                        if raw_idx is not None:
                            assigned_idx = min(candidates, key=lambda c: abs(c - raw_idx))
                        else:
                            assigned_idx = candidates[0]
                    elif raw_idx is not None and 0 <= raw_idx < total_rows:
                        assigned_idx = raw_idx

                if assigned_idx is not None and 0 <= assigned_idx < total_rows:
                    if final_th[assigned_idx] is None:
                        fix_count += 1
                    final_th[assigned_idx] = text_th
        print(f"   เติมคำแปลเสริมสำเร็จ: {fix_count} แถว")

    valid_translations = sum(1 for t in final_th if t is not None)
    print(f"\n✅ ดึงคำแปลสำเร็จทั้งหมด: {valid_translations:,} / {total_rows:,} แถว ({valid_translations/total_rows*100:.2f}%)")

    # 3. สร้างและบันทึก full_train_text_translate.csv
    print(f"\n💾 3. บันทึกไฟล์รวมคำแปล: {OUTPUT_TEXT_CSV}", flush=True)
    train_texts = df_train["text"].tolist()
    with open(OUTPUT_TEXT_CSV, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, quoting=csv.QUOTE_ALL)
        writer.writerow(["row_index", "text", "text_th"])
        for idx in range(total_rows):
            writer.writerow([idx, train_texts[idx], final_th[idx] if final_th[idx] is not None else ""])

    # 4. Merge เข้ากับ go_emotions_train.csv
    print(f"💾 4. บันทึกไฟล์พร้อมเทรน: {OUTPUT_MERGED_CSV}")
    df_merged = df_train.copy()
    df_merged.insert(1, "text_th", final_th)

    # จัดเรียงคอลัมน์ให้ตรงกับ go_emotions_test_with_translation.csv
    desired_cols = ["text", "text_th", "labels", "id"]
    cols = [c for c in desired_cols if c in df_merged.columns] + [c for c in df_merged.columns if c not in desired_cols]
    df_merged = df_merged[cols]
    df_merged.to_csv(OUTPUT_MERGED_CSV, index=False, encoding="utf-8")

    # 5. สรุปแถวที่ขาด (ถ้ามี)
    missing_indices = [i for i, t in enumerate(final_th) if t is None]
    if missing_indices:
        print(f"\n⚠️  พบแถวที่ไม่มีคำแปล: {len(missing_indices)} แถว ({len(missing_indices)/total_rows*100:.2f}%)")
        df_missing = df_train.iloc[missing_indices][["text"]].copy()
        df_missing.insert(0, "row_index", missing_indices)
        df_missing.to_csv(OUTPUT_MISSING, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")
        print(f"   บันทึกรายการที่ขาดไว้ที่: {OUTPUT_MISSING}")
    else:
        if OUTPUT_MISSING.exists():
            OUTPUT_MISSING.unlink()
        print(f"\n🎉 ข้อมูลครบ 100% สมบูรณ์ทั้งหมด {total_rows:,} แถว ไม่มีแถวใดขาด!")

    print(f"\n🎉 รวมข้อมูลเสร็จสมบูรณ์!")


if __name__ == "__main__":
    main()
