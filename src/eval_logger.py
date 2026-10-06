"""
Evaluation Logger Utility for Thai Multilabel Emotion Detection
Saves model evaluation results into Markdown (.md) files timestamped by date and time in the results directory,
and maintains an experiment comparison summary in results/SUMMARY.md.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    accuracy_score
)

def update_summary_index(
    output_dir: str,
    timestamp_str: str,
    formatted_date: str,
    report_filename: str,
    model_name: str,
    micro_f1: float,
    macro_f1: float,
    weighted_f1: float,
    h_loss: float,
    notes: str = ""
):
    """
    Maintains a centralized results/SUMMARY.md experiment tracking table.
    """
    summary_path = os.path.join(output_dir, "SUMMARY.md")
    clean_notes = notes.replace("\n", " ").strip() if notes else "-"
    row_line = (
        f"| {formatted_date} | [{model_name}]({report_filename}) | "
        f"**{micro_f1:.4f}** | **{macro_f1:.4f}** | {weighted_f1:.4f} | {h_loss:.4f} | {clean_notes} |\n"
    )

    if not os.path.exists(summary_path):
        header = (
            "# 📈 Experiment Evaluation Summary & Leaderboard\n\n"
            "ตารางสรุปผลการประเมินและการปรับแต่งโมเดล (เรียงตามลำดับเวลาการทดลอง)\n\n"
            "| Date / Time | Model / Run | Micro F1 | Macro F1 | Weighted F1 | Hamming Loss | Notes |\n"
            "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |\n"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(header + row_line)
    else:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(row_line)

def save_eval_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list,
    model_name: str = "Model",
    hyperparams: dict = None,
    notes: str = "",
    output_dir: str = "results",
    filename_prefix: str = "eval"
) -> str:
    """
    Evaluates y_true vs y_pred and writes a detailed Markdown report to output_dir with datetime filename,
    and updates results/SUMMARY.md.
    
    Args:
        y_true: Ground truth binary matrix (N, num_labels)
        y_pred: Predicted binary matrix (N, num_labels)
        target_names: List of label class names
        model_name: Name of the evaluated model/architecture
        hyperparams: Dictionary of hyperparameters used in training/evaluation
        notes: Tuning notes or experiment comments
        output_dir: Folder to save markdown result files (default: 'results')
        filename_prefix: Prefix for file name (default: 'eval')
        
    Returns:
        filepath: Path to the generated Markdown file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    formatted_date = now.strftime("%d/%m/%Y %H:%M:%S")
    filename = f"{filename_prefix}_{timestamp_str}.md"
    filepath = os.path.join(output_dir, filename)
    
    # Calculate overall metrics
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    h_loss = hamming_loss(y_true, y_pred)
    subset_acc = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    
    class_df = pd.DataFrame({
        'Class': target_names,
        'Precision': p,
        'Recall': r,
        'F1-Score': f1,
        'Support': s
    })
    
    # Sort for top & bottom analysis
    top_5 = class_df.sort_values(by='F1-Score', ascending=False).head(5)
    bottom_5 = class_df.sort_values(by='F1-Score', ascending=True).head(5)
    
    # Build Markdown Content
    md = []
    md.append(f"# 📊 Evaluation Log: {model_name}")
    md.append(f"**Date/Time:** `{formatted_date}`  ")
    md.append(f"**Report File:** `{filename}`  ")
    md.append(f"**Output Directory:** `{output_dir}/`  ")
    md.append(f"**Overview Summary Table:** [SUMMARY.md](SUMMARY.md)  ")
    md.append("\n---\n")
    
    # Section 1: Summary Metrics
    md.append("## 🎯 1. Summary Metrics")
    md.append("| Metric | Value | Description |")
    md.append("| :--- | :---: | :--- |")
    md.append(f"| **Micro F1** | **`{micro_f1:.4f}`** | Global F1 considering total true positives/false negatives |")
    md.append(f"| **Macro F1** | **`{macro_f1:.4f}`** | Unweighted mean F1 across all 28 classes |")
    md.append(f"| **Weighted F1** | **`{weighted_f1:.4f}`** | Mean F1 weighted by class support |")
    md.append(f"| **Hamming Loss** | `{h_loss:.4f}` | Fraction of wrong labels to total labels (Lower is better) |")
    md.append(f"| **Subset Accuracy** | `{subset_acc:.4f}` | Exact match ratio across all labels |")
    md.append("\n")
    
    # Section 2: Hyperparameters & Configuration
    if hyperparams:
        md.append("## ⚙️ 2. Hyperparameters & Settings")
        md.append("| Parameter | Value |")
        md.append("| :--- | :--- |")
        for k, v in hyperparams.items():
            md.append(f"| `{k}` | `{v}` |")
        md.append("\n")
        
    # Section 3: Key Insights (Top & Bottom Performers)
    md.append("## 🔍 3. Key Insights (Top & Bottom Performers)")
    md.append("### 🏆 Top 5 Classes by F1-Score")
    md.append("| Class | Precision | Recall | F1-Score | Support |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for _, row in top_5.iterrows():
        md.append(f"| **{row['Class']}** | {row['Precision']:.4f} | {row['Recall']:.4f} | **{row['F1-Score']:.4f}** | {int(row['Support'])} |")
    
    md.append("\n### ⚠️ Bottom 5 Classes by F1-Score (Need Tuning/Data Augmentation)")
    md.append("| Class | Precision | Recall | F1-Score | Support |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for _, row in bottom_5.iterrows():
        md.append(f"| **{row['Class']}** | {row['Precision']:.4f} | {row['Recall']:.4f} | **{row['F1-Score']:.4f}** | {int(row['Support'])} |")
    md.append("\n")

    # Section 4: Detailed Per-Class Breakdown
    md.append("## 📋 4. Detailed Per-Class Classification Report")
    md.append("| # | Class Name | Precision | Recall | F1-Score | Support |")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: |")
    for i, row in class_df.iterrows():
        md.append(f"| {i+1} | `{row['Class']}` | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {int(row['Support'])} |")
    md.append("\n")
    
    # Section 5: Experiment Notes
    if notes:
        md.append("## 📝 5. Experiment Notes & Observations")
        md.append(f"> {notes}")
        md.append("\n")

    md_text = "\n".join(md)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_text)
        
    # Update centralized summary leaderboard
    update_summary_index(
        output_dir=output_dir,
        timestamp_str=timestamp_str,
        formatted_date=formatted_date,
        report_filename=filename,
        model_name=model_name,
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        h_loss=h_loss,
        notes=notes
    )
        
    print(f"✅ Saved evaluation report to markdown file: {filepath}")
    print(f"📊 Updated experiment leaderboard: {os.path.join(output_dir, 'SUMMARY.md')}")
    return filepath
