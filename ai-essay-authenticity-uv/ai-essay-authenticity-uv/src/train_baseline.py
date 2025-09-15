import argparse, json, random
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

RNG = 42

def find_label_column(df):
    for cand in ["produced", "generated", "label", "target", "is_ai", "is_ai_generated"]:
        if cand in df.columns:
            return cand
    raise ValueError(f"No label column found. Columns: {df.columns.tolist()}")

def main(args):
    np.random.seed(RNG); random.seed(RNG)
    df = pd.read_csv(args.csv)

    text_col = "text" if "text" in df.columns else next(c for c in df.columns if df[c].dtype == 'O')
    y_col = find_label_column(df)

    y = df[y_col].astype(int).values
    X = df[text_col].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RNG
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=100_000, ngram_range=(1,2))),
        ("logreg", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = float("nan")

    print(classification_report(y_test, preds))
    print({"accuracy":acc, "precision":prec, "recall":rec, "f1":f1, "roc_auc":auc})

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out/"baseline_tfidf_logreg.joblib")
    with open(out/"baseline_metrics.json","w") as f:
        json.dump({"accuracy":acc, "precision":prec, "recall":rec, "f1":f1, "roc_auc":auc}, f, indent=2)

    cm = confusion_matrix(y_test, preds)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Baseline Confusion Matrix'); plt.colorbar()
    plt.xlabel('Predicted'); plt.ylabel('True')
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha='center',va='center')
    plt.tight_layout(); plt.savefig(out/"baseline_confusion_matrix.png", dpi=160)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Kaggle CSV")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--outdir", default="runs/baseline")
    args = ap.parse_args()
    main(args)
