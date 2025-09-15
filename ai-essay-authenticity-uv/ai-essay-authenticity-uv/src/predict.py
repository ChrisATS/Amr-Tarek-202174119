import argparse, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

LABELS = ["human","ai"]

def load(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tok, model

def predict_texts(texts, tok, model, max_length=384):
    enc = tok(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**{k:v for k,v in enc.items()}).logits
        probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()
    labels = [LABELS[int(p>=0.5)] for p in probs]
    return list(zip(texts, labels, probs))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="runs/deberta/best_model")
    ap.add_argument("--text", nargs="+", required=True, help="One or more texts")
    args = ap.parse_args()
    tok, model = load(args.model_dir)
    for t, lab, p in predict_texts(args.text, tok, model):
        print(f"[{lab:5s}] p(ai)={p:.3f} :: {t[:80]}...")
