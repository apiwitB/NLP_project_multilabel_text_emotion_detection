import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate_dataset(input_file, output_file, model, tokenizer, batch_size=128, checkpoint_interval=1000):
    print(f"\n==========================================")
    print(f"Processing: {input_file}")
    print(f"==========================================")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    df = pd.read_csv(input_file)
    
    # Check if existing output file or progress exists
    if os.path.exists(output_file):
        try:
            out_df = pd.read_csv(output_file)
            if 'thai_text' in out_df.columns and len(out_df) == len(df):
                df['thai_text'] = out_df['thai_text']
                translated_count = df['thai_text'].notna().sum()
                print(f"Found existing progress: {translated_count}/{len(df)} translated rows.")
                if translated_count == len(df):
                    print(f"File {output_file} is already fully translated! Skipping.")
                    return
            else:
                df['thai_text'] = None
        except Exception:
            df['thai_text'] = None
    else:
        df['thai_text'] = None

    untranslated_indices = df[df['thai_text'].isna()].index.tolist()
    total_untranslated = len(untranslated_indices)
    print(f"Total rows: {len(df)} | Untranslated rows remaining: {total_untranslated}")

    if total_untranslated == 0:
        print("All rows are already translated!")
        return

    target_lang_id = tokenizer.convert_tokens_to_ids('tha_Thai')
    start_time = time.time()
    processed_count = 0

    for i in range(0, total_untranslated, batch_size):
        batch_idx = untranslated_indices[i : i + batch_size]
        batch_texts = df.loc[batch_idx, 'text'].astype(str).tolist()

        try:
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to('cuda')
            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=target_lang_id,
                    max_length=128
                )
            translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            df.loc[batch_idx, 'thai_text'] = translated_texts
        except Exception as e:
            print(f"\nError during batch translation at index {i}: {e}")
            continue

        processed_count += len(batch_idx)

        # Print progress & save checkpoint
        if (i + batch_size) % checkpoint_interval < batch_size or (i + batch_size) >= total_untranslated:
            elapsed = time.time() - start_time
            speed = processed_count / elapsed if elapsed > 0 else 0
            print(f"[{processed_count}/{total_untranslated}] Translated ({speed:.1f} rows/sec) - Saving checkpoint to {output_file}...")
            df.to_csv(output_file, index=False, encoding='utf-8-sig')

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Finished {input_file} -> Saved to {output_file}\n")

def main():
    model_name = 'facebook/nllb-200-distilled-600M'
    print(f"Loading model '{model_name}' to GPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang='eng_Latn')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
    model.eval()
    print("Model loaded successfully onto GPU!")

    # Process validation first (5,426 rows), test second (5,427 rows), train last (43,410 rows)
    files_to_process = [
        ('go_emotions_validation.csv', 'go_emotions_validation.csv'),
        ('go_emotions_test.csv', 'go_emotions_test.csv'),
        ('go_emotions_train.csv', 'go_emotions_train.csv'),
    ]

    for in_f, out_f in files_to_process:
        in_path = os.path.join('/home/apiwit/NLP_project/GoEmotions', in_f)
        out_path = os.path.join('/home/apiwit/NLP_project/GoEmotions', out_f)
        translate_dataset(in_path, out_path, model, tokenizer, batch_size=128, checkpoint_interval=1000)

if __name__ == '__main__':
    main()
