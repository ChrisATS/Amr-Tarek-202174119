import argparse, random, json
import numpy as np, pandas as pd
from pathlib import Path
import torch
from datasets import Dataset, ClassLabel
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F

from metrics import compute_binary_metrics
from utils import guess_columns

RNG = 42

def main(args):
    np.random.seed(RNG); random.seed(RNG); torch.manual_seed(RNG)

    df = pd.read_csv(args.csv)
    text_col, y_col = guess_columns(df)

    df[y_col] = df[y_col].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        df[text_col].astype(str), df[y_col], test_size=args.valid_size, stratify=df[y_col], random_state=RNG
    )

    train_ds = Dataset.from_pandas(pd.DataFrame({ "text": X_train.values, "label": y_train.values }))
    val_ds   = Dataset.from_pandas(pd.DataFrame({ "text": X_val.values,   "label": y_val.values   }))
    class_label = ClassLabel(num_classes=2, names=["human","ai"])
    train_ds = train_ds.cast_column("label", class_label)
    val_ds   = val_ds.cast_column("label", class_label)

    model_name = args.model
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def tok_fn(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)
    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_tok   = val_ds.map(tok_fn, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out),
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=max(10, args.eval_steps//5),
        seed=RNG,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tok,
        compute_metrics=compute_binary_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    metrics = trainer.evaluate()
    with open(out/"val_metrics.json","w") as f: json.dump(metrics, f, indent=2)

    logits = np.array(trainer.predict(val_tok).predictions)
    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()[:,1]
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(val_ds["label"], preds)
    plt.figure()
    plt.imshow(cm, interpolation='nearest'); plt.title('Transformer Confusion Matrix'); plt.colorbar()
    plt.xlabel('Predicted'); plt.ylabel('True')
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha='center',va='center')
    plt.tight_layout(); plt.savefig(out/"confusion_matrix.png", dpi=160)

    trainer.save_model(out/"best_model")
    tok.save_pretrained(out/"best_model")
    print("Saved best model to", out/"best_model")
    print("Validation metrics:", metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-base")  # or "distilroberta-base"
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--valid_size", type=float, default=0.2)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--outdir", default="runs/deberta")
    args = ap.parse_args()
    main(args)
