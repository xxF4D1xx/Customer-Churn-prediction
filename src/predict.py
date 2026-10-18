"""Score new customers. Usage: python -m src.predict new_customers.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import load_raw, clean, DROP_COLS

RANDOM_STATE = 42

def main(path: str):
    df = load_raw()
    X, y = clean(df, slim=True)

    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns if X[c].dtype == 'object']
    pre = ColumnTransformer([
        ('num', 'passthrough', num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)])
    pipe = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE))])
    pipe.fit(X, y)

    new = pd.read_csv(path)
    X_new = new.drop(columns=DROP_COLS, errors='ignore').copy()
    X_new['Total Charges'] = pd.to_numeric(X_new['Total Charges'], errors='coerce').fillna(0)
    X_new = X_new[[c for c in X.columns if c in X_new.columns]]

    proba = pipe.predict_proba(X_new)[:, 1]
    out = pd.DataFrame({
        'churn_probability': proba,
        'risk_band': pd.cut(proba, bins=[0, 0.30, 0.60, 1.0], labels=['low', 'medium', 'high']),
    })
    Path('results').mkdir(exist_ok=True)
    out.to_csv('results/predictions.csv', index=False)
    print(f"Wrote results/predictions.csv ({len(out)} rows)")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict new_customers.csv")
        sys.exit(1)
    main(sys.argv[1])