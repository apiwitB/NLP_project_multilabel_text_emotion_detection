"""
fix_translate_txt.py
────────────────────
ซ่อมไฟล์ full_text_translate.txt ที่มี malformed lines 2 ประเภท:
  1. Unmatched quote  → ',' expected after '"'
  2. Extra comma      → Expected 3 fields, saw 4/5

กลยุทธ์:
  - อ่านทีละบรรทัด group เป็น logical records
    (record ใหม่ = บรรทัดที่ขึ้นต้นด้วย integer แล้วตามด้วย comma)
  - แต่ละ record → ลอง parse ด้วย csv.reader ก่อน
  - ถ้า fail → ใช้ regex fallback แยก row_index / text / text_th
  - เขียน output ด้วย QUOTE_ALL เพื่อให้อ่านได้ปลอดภัย
  - Reindex row_index ใหม่ตามลำดับที่ถูกต้อง (0-based)
"""

import csv
import re
import io
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
INPUT_TXT  = BASE_DIR / "data/full_text_translate.txt"
OUTPUT_TXT = BASE_DIR / "data/full_text_translate_fixed.txt"

RECORD_START = re.compile(r"^\d+,")


def parse_record(raw: str):
    """ลอง parse record → (row_index, text, text_th) หรือ None"""
    # วิธีที่ 1: csv.reader strict (single line)
    try:
        rows = list(csv.reader(io.StringIO(raw)))
        if rows and len(rows[0]) == 3:
            return tuple(rows[0])
    except csv.Error:
        pass

    # วิธีที่ 2: regex fallback
    m = re.match(r"^(\d+),(.*)", raw, re.DOTALL)
    if not m:
        return None
    row_idx = m.group(1)
    rest    = m.group(2)

    # ลอง extract English text ที่ quote ไว้
    quoted = re.match(r'^"((?:[^"\\]|\\.)*)"\s*,\s*(.*)', rest, re.DOTALL)
    if quoted:
        text_eng = quoted.group(1).replace('\\"', '"')
        text_th  = quoted.group(2).strip().strip('"')
        return (row_idx, text_eng, text_th)

    # fallback: แยกด้วย comma แรก
    parts = rest.split(",", 1)
    if len(parts) == 2:
        return (row_idx, parts[0].strip().strip('"'), parts[1].strip().strip('"'))

    return None


def main():
    print(f"📂 อ่านไฟล์: {INPUT_TXT}")

    parsed  = []
    failed  = []
    current = ""
    record_count = 0

    with INPUT_TXT.open(encoding="utf-8") as f:
        header = f.readline()  # skip header row

        for line in f:
            line = line.rstrip("\n")
            if RECORD_START.match(line) and current:
                # flush record ก่อนหน้า
                record_count += 1
                result = parse_record(current.strip())
                if result:
                    parsed.append(result)
                else:
                    failed.append(current[:80])
                current = line
            else:
                current = current + "\n" + line if current else line

        # flush record สุดท้าย
        if current:
            result = parse_record(current.strip())
            if result:
                parsed.append(result)
            else:
                failed.append(current[:80])

    print(f"   ✅ parse สำเร็จ : {len(parsed):,} แถว")
    print(f"   ❌ parse ไม่ได้ : {len(failed)} แถว")
    if failed:
        for snippet in failed:
            print(f"     → {snippet!r}")

    # เรียงตาม row_index เดิม
    try:
        parsed_sorted = sorted(parsed, key=lambda x: int(x[0]))
    except ValueError:
        parsed_sorted = parsed

    # เขียน output
    print(f"\n💾 เขียนไฟล์: {OUTPUT_TXT}")
    with OUTPUT_TXT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["row_index", "text", "text_th"])
        for row_idx, text_eng, text_th in parsed_sorted:
            writer.writerow([int(row_idx), text_eng, text_th])

    total = len(parsed_sorted)
    print(f"🎉 เสร็จ! {total:,} แถว (คงค่า row_index เดิม)")
    print(f"   → {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
