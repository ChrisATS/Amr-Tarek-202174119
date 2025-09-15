# Model Card — AI Essay Authenticity Classifier

## Overview
Binary classifier separating human vs AI‑generated essays.

## Data
- Kaggle: Human vs AI Generated Essays.
- Minimal preprocessing; stratified split; tokenization with truncation.

## Training
- Baseline: TF‑IDF + Logistic Regression.
- Main: microsoft/deberta-v3-base (alt: distilroberta-base).
- Early stopping (patience=3), LR≈2e‑5, wd=0.01, warmup=10%.

## Evaluation
Accuracy, Precision, Recall, F1, ROC‑AUC. Confusion matrices saved in `runs/**`.

## Limitations
Distribution shift, very short texts, and adversarial paraphrases can degrade accuracy.

## Author
Amr Tarek Saif Noaman Al-Hammadi — Student ID: 202174119
