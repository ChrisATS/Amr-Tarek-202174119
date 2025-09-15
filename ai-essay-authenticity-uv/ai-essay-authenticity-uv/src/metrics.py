import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def _softmax_2d(logits: np.ndarray):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def compute_binary_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.array(logits)
    labels = np.array(labels).astype(int)

    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        probs = 1.0 / (1.0 + np.exp(-logits.squeeze()))
    else:
        probs = _softmax_2d(logits)[:, 1]

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
