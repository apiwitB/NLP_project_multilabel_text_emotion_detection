import os, json, joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(
    title="Thai Emotion Detection API",
    description="Hybrid Stacked Ensemble: WangchanBERTa + mBERT + XLM-RoBERTa → Random Forest",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Load Saved Models at Startup
# ---------------------------------------------------------
SAVE_DIR = "saved_models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_CLASSES = []
tokenizers = {}
bert_models = {}
rf_model = None
model_keys = ['wangchan', 'mbert', 'xlm_roberta']

class SingleTextDataset(Dataset):
    def __init__(self, text, tokenizer, max_len=128):
        self.text = text
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return 1
    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.text),
            add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten()
        }

@app.on_event("startup")
def load_all_models():
    global LABEL_CLASSES, tokenizers, bert_models, rf_model
    print(f"Loading models on device: {device}...")
    
    label_path = os.path.join(SAVE_DIR, "label_classes.json")
    if os.path.exists(label_path):
        with open(label_path, encoding="utf-8") as f:
            LABEL_CLASSES = json.load(f)
    
    for key in model_keys:
        tok_path = os.path.join(SAVE_DIR, f"{key}_tokenizer")
        mod_path = os.path.join(SAVE_DIR, f"{key}_model")
        if os.path.exists(tok_path) and os.path.exists(mod_path):
            use_fast = False if key == 'wangchan' else True
            tokenizers[key] = AutoTokenizer.from_pretrained(tok_path, use_fast=use_fast)
            m = AutoModelForSequenceClassification.from_pretrained(mod_path).to(device)
            m.eval()
            bert_models[key] = m
            print(f" Loaded {key}")
            
    rf_path = os.path.join(SAVE_DIR, "random_forest_meta.joblib")
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        print(" Loaded Random Forest meta-classifier")
    print("All models loaded successfully!")

def get_single_prob(model, loader):
    with torch.no_grad():
        for batch in loader:
            out = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            return torch.sigmoid(out.logits).cpu().numpy()

def predict_emotion(text: str, threshold: float = 0.5) -> dict:
    if not bert_models or rf_model is None:
        raise HTTPException(status_code=500, detail="Models are not loaded yet.")
    
    feats = []
    for key in model_keys:
        loader = DataLoader(SingleTextDataset(text, tokenizers[key]), batch_size=1)
        prob = get_single_prob(bert_models[key], loader)
        feats.append(prob)
        
    X = np.concatenate(feats, axis=1)  # shape (1, 21)
    rf_proba_list = rf_model.predict_proba(X)
    prob_pos = np.array([p[0][1] for p in rf_proba_list])
    
    predicted_emotions = [
        LABEL_CLASSES[i] for i, p in enumerate(prob_pos) if p >= threshold
    ]
    if not predicted_emotions:
        predicted_emotions = [LABEL_CLASSES[np.argmax(prob_pos)]]
        
    return {
        "text": text,
        "emotions": predicted_emotions,
        "probabilities": {lbl: round(float(p), 4) for lbl, p in zip(LABEL_CLASSES, prob_pos)}
    }

class TextRequest(BaseModel):
    text: str = Field(..., example="ดีใจมากเลย ได้รับของแล้ว")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

@app.get("/")
def root():
    return {"message": "Thai Emotion Detection API is running!", "status": "online"}

@app.post("/predict")
def predict(req: TextRequest):
    return predict_emotion(req.text, threshold=req.threshold)