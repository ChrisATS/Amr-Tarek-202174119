import pandas as pd

def guess_columns(df: pd.DataFrame):
    text_col = "text" if "text" in df.columns else next(c for c in df.columns if df[c].dtype == 'O')
    for cand in ["produced","generated","label","target","is_ai","is_ai_generated"]:
        if cand in df.columns:
            return text_col, cand
    raise ValueError(f"Label column not found. Found columns: {df.columns.tolist()}")
