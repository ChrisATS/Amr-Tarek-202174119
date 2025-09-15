# AI Essay Authenticity — Human vs AI Essays (Binary Classification)

**Student:** Amr Tarek Saif Noaman Al-Hammadi  
**Academic Number:** 202174119

Detect whether an essay is human-written (0) or AI-generated (1). Project uses UV-managed env, a TF‑IDF baseline, and a DeBERTa‑v3 fine‑tune.

## 1) Structure
```
ai-essay-authenticity-uv/
├─ data/
├─ notebooks/
│  └─ 01_eda.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ train_baseline.py
│  ├─ train_transformer.py
│  ├─ predict.py
│  ├─ utils.py
│  └─ metrics.py
├─ runs/
├─ pyproject.toml
├─ README.md
├─ MODEL_CARD.md
└─ SUBMISSION_CHECKLIST.md
```

## 2) Setup with UV
Windows PowerShell:
```powershell
iwr -useb https://astral.sh/uv/install.ps1 | iex
$env:Path += ";$env:USERPROFILE\.local\bin"
cd "C:\path\to\ai-essay-authenticity-uv"
uv sync
```
macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/path/to/ai-essay-authenticity-uv
uv sync
```

## 3) Data
Place the Kaggle CSV into `data/` or use:
```powershell
uv run -m kaggle datasets download -d navjotkaushal/human-vs-ai-generated-essays -p data --unzip
```

## 4) Train
Baseline:
```powershell
uv run -m src.train_baseline --csv "data\YOUR_FILE.csv"
```
Transformer (stronger):
```powershell
uv run -m src.train_transformer --csv "data\YOUR_FILE.csv" --model microsoft/deberta-v3-base
# or faster:
uv run -m src.train_transformer --csv "data\YOUR_FILE.csv" --model distilroberta-base --epochs 4 --batch 16
```

## 5) Inference
```powershell
uv run -m src.predict --model_dir runs\deberta\best_model --text "Sample essay text here..."
```

## 6) Notes
- Artifacts in `runs/` (metrics JSON + confusion_matrix.png).
- For CUDA, install a matching torch wheel after `uv sync`.
