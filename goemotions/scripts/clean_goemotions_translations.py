import pandas as pd
import re
import os

def fix_placeholders(orig_text, thai_text):
    if not isinstance(thai_text, str) or not isinstance(orig_text, str):
        return thai_text
    
    orig_tags = re.findall(r'\[[A-Z_]+\]', orig_text)
    if not orig_tags:
        return thai_text
    
    thai_tags = re.findall(r'\[[^\]]+\]', thai_text)
    
    if len(orig_tags) == len(thai_tags):
        res = thai_text
        for ot, tt in zip(orig_tags, thai_tags):
            res = res.replace(tt, ot, 1)
        return res
    else:
        res = thai_text
        res = re.sub(r'\[\s*(?:นาม|ชื่อ|ชื่อของฉัน|ชื่อ\s*)\s*\]', '[NAME]', res)
        res = re.sub(r'\[\s*(?:ศาสนา|ศาสนา\s*)\s*\]', '[RELIGION]', res)
        res = re.sub(r'\[\s*(?:สถานที่)\s*\]', '[LOCATION]', res)
        return res

def clean_common_mistranslations(orig_text, thai_text):
    if not isinstance(thai_text, str) or not isinstance(orig_text, str):
        return thai_text
    
    res = thai_text

    # PTSD mistranslated as โรคสะเก็ดเงิน
    if 'PTSD' in orig_text and 'สะเก็ดเงิน' in res:
        res = res.replace('ผู้ป่วยจากโรคสะเก็ดเงินบาดเจ็บ', 'ผู้ป่วย PTSD')
        res = res.replace('โรคสะเก็ดเงิน', 'PTSD')

    # baptism mistranslated as รับประทานน้ำ
    if 'baptism' in orig_text.lower() and 'รับประทานน้ํา' in res:
        res = res.replace('การรับประทานน้ํา', 'พิธีศีลล้างบาป')

    # half-blood mistranslated as เลือดครึ่ง
    if 'half-blood' in orig_text.lower() and 'เลือดครึ่ง' in res:
        res = res.replace('คนมีเลือดครึ่ง', 'ลูกครึ่ง')
        res = res.replace('เลือดครึ่ง', 'ลูกครึ่ง')

    # panic attack mistranslated as อาการตกใจ
    if 'panic attack' in orig_text.lower() and 'อาการตกใจ' in res:
        res = res.replace('อาการตกใจ', 'อาการแพนิก (ตื่นตระหนก)')

    # bring up his father mistranslated as ยกพ่อของเขา
    if 'bring up his father' in orig_text.lower() and 'ยกพ่อ' in res:
        res = res.replace('ที่จะยกพ่อของเขา', 'ที่จะพูดถึงพ่อของเขา')

    # bow mistranslated as กอง
    if 'got the bow' in orig_text.lower() and 'ได้กอง' in res:
        res = res.replace('เพิ่งได้กอง', 'เพิ่งได้คันธนู')

    # what not mistranslated as สิ่งที่ไม่
    if 'what not' in orig_text.lower() and 'สิ่งที่ไม่' in res:
        res = res.replace('และสิ่งที่ไม่', 'หรือทำนองนั้น')

    return res

def process_file(file_path):
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)
    
    placeholder_fixes = 0
    mistranslation_fixes = 0
    
    for idx in df.index:
        orig = df.loc[idx, 'text']
        th = df.loc[idx, 'thai_text']
        
        new_th = fix_placeholders(orig, th)
        if new_th != th:
            placeholder_fixes += 1
            th = new_th
            
        clean_th = clean_common_mistranslations(orig, th)
        if clean_th != th:
            mistranslation_fixes += 1
            th = clean_th
            
        df.loc[idx, 'thai_text'] = th
        
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"Done {file_path}: Restored {placeholder_fixes} placeholders, fixed {mistranslation_fixes} common mistranslations.\n")

def main():
    base_dir = '/home/apiwit/NLP_project/GoEmotions'
    files = ['go_emotions_train.csv', 'go_emotions_validation.csv', 'go_emotions_test.csv']
    for f in files:
        path = os.path.join(base_dir, f)
        if os.path.exists(path):
            process_file(path)

if __name__ == '__main__':
    main()
