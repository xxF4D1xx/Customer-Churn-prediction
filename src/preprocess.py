"""Load and clean the Telco churn dataset."""
from pathlib import Path
import os

import pandas as pd

# Columns to drop (IDs, constants, leakage)
DROP_COLS = [
    'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
    'Lat Long', 'Latitude', 'Longitude',
    'Churn Label', 'Churn Value', 'Churn Score', 'Churn Reason', 'CLTV',
]

def _resolve_data_path() -> Path:
    try:
        import kagglehub
        path = kagglehub.dataset_download("yeanzc/telco-customer-churn-ibm-dataset")
        files = os.listdir(path)
        for f in files:
            if f.lower().endswith('.csv'):
                return Path(path) / f
        for f in files:
            if f.lower().endswith('.xlsx'):
                return Path(path) / f
        raise FileNotFoundError(f"no csv/xlsx in {path}")
    except Exception:
        for cand in [Path("data/churn.csv"), Path("data/churn.xlsx")]:
            if cand.exists():
                return cand
        raise FileNotFoundError("No dataset found. Install kagglehub or place churn.csv in data/.")

def load_raw() -> pd.DataFrame:
    p = _resolve_data_path()
    if p.suffix.lower() == '.csv':
        return pd.read_csv(p)
    return pd.read_excel(p)

def clean(df: pd.DataFrame, slim: bool = True):
    y = df['Churn Value'].astype(int)
    
    # Drop IDs, constants, and leakage
    X = df.drop(columns=DROP_COLS, errors='ignore').copy()

    # Fix Total Charges
    X['Total Charges'] = pd.to_numeric(X['Total Charges'], errors='coerce').fillna(0)

    # Drop weak features IF slim=True
    if slim:
        SLIM_DROP = ['Gender', 'Streaming Movies', 'Multiple Lines']
        X = X.drop(columns=SLIM_DROP, errors='ignore')

    # Safety checks
    assert X.isna().sum().sum() == 0
    assert not any(c in X.columns for c in ['Churn Score', 'Churn Reason', 'Churn Label'])
    assert [c for c in X.columns if X[c].nunique() <= 1] == []
    
    return X, y