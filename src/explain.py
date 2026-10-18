"""SHAP global + per-customer explanations and retention segmentation."""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import load_raw, clean
from src.plotting import apply_style, PALETTE

warnings.filterwarnings('ignore')
apply_style()

RANDOM_STATE = 42
FIG = Path('figures'); FIG.mkdir(exist_ok=True)
RES = Path('results'); RES.mkdir(exist_ok=True)

def _normalize_shap(raw, base):
    if isinstance(raw, list):
        vals = raw[1]
        b = base[1] if isinstance(base, (list, np.ndarray)) and np.ndim(base) > 0 else base
    elif np.ndim(raw) == 3:
        vals = raw[:, :, 1]
        b = base[1] if isinstance(base, (list, np.ndarray)) and np.ndim(base) > 0 else base
    else:
        vals = raw
        b = base[1] if isinstance(base, (list, np.ndarray)) and np.ndim(base) > 0 else base
    return np.asarray(vals), float(np.asarray(b).ravel()[-1])

def _pretty_feature(name: str) -> str:
    n = name
    if '__' in n:
        n = n.split('__', 1)[1]
    if '_' in n and n.count('_') >= 1 and ('Monthly Charges' not in n and 'Total Charges' not in n and 'Tenure Months' not in n):
        head, _, tail = n.partition('_')
        return f'{head}: {tail.replace("_", " ")}'
    return n.replace('_', ' ')

def fig_shap_bar(sv, Xt, feat_names):
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[-15:][::-1]
    labels = [_pretty_feature(feat_names[i]) for i in order]
    values = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(labels, values, color=PALETTE['primary'], edgecolor='none')
    for i, v in enumerate(values):
        ax.text(v * 1.01, i, f'{v:.3f}', va='center', fontsize=9, color='#111827')
    ax.set_xlabel('mean |SHAP value|  (impact on churn log-odds)')
    ax.set_title('SHAP feature importance — top 15')
    ax.invert_yaxis()
    ax.set_xlim(0, values.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIG / 'shap_bar.png', bbox_inches='tight')
    plt.close(fig)

def fig_shap_beeswarm(sv, Xt, feat_names):
    fig = plt.figure(figsize=(8.5, 6))
    shap.summary_plot(sv, Xt, feature_names=[_pretty_feature(f) for f in feat_names],
                     max_display=15, show=False, plot_size=None)
    plt.title('SHAP beeswarm — direction of effect', pad=14)
    plt.tight_layout()
    plt.savefig(FIG / 'shap_beeswarm.png', bbox_inches='tight')
    plt.close(fig)

def fig_shap_waterfall(sv, base, Xt, feat_names, i, proba):
    explanation = shap.Explanation(
        values=sv[i], base_values=base, data=Xt[i],
        feature_names=[_pretty_feature(f) for f in feat_names])
    fig = plt.figure(figsize=(9, 5.5))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f'Why is this customer at risk?  P(churn) = {proba[i]:.2f}', pad=14)
    plt.tight_layout()
    plt.savefig(FIG / 'shap_waterfall.png', bbox_inches='tight')
    plt.close(fig)

def fig_retention_segments(feat_names, dom, n_high_risk):
    counts = pd.Series([_pretty_feature(feat_names[i]) for i in dom])
    counts = counts.value_counts().head(6)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=PALETTE['secondary'], edgecolor='none')
    for i, v in enumerate(counts.values[::-1]):
        pct = v / n_high_risk * 100
        ax.text(v + max(counts.values) * 0.01, i, f'{v}  ({pct:.0f}%)', va='center', fontsize=10)
    ax.set_xlabel('Number of high-risk customers')
    ax.set_title(f'Retention segmentation — dominant SHAP driver ({n_high_risk} high-risk customers)')
    ax.set_xlim(0, max(counts.values) * 1.25)
    fig.tight_layout()
    fig.savefig(FIG / 'retention_segments.png', bbox_inches='tight')
    plt.close(fig)

def main():
    df = load_raw()
    X, y = clean(df, slim=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    num = X_train.select_dtypes(include=np.number).columns.tolist()
    cat = X_train.select_dtypes(include=['object', 'string', 'str']).columns.tolist()

    pre = ColumnTransformer([
        ('num', 'passthrough', num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)])
    pipe = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE))])
    pipe.fit(X_train, y_train)

    X_test_t = pipe.named_steps['pre'].transform(X_test)
    feat_names = pipe.named_steps['pre'].get_feature_names_out()
    clf = pipe.named_steps['clf']

    explainer = shap.TreeExplainer(clf)
    sv, base = _normalize_shap(explainer.shap_values(X_test_t), explainer.expected_value)
    print(f"SHAP: {sv.shape}  base: {base:.4f}")

    proba = pipe.predict_proba(X_test)[:, 1]
    top_i = int(np.argmax(proba))

    fig_shap_bar(sv, X_test_t, feat_names)
    fig_shap_beeswarm(sv, X_test_t, feat_names)
    fig_shap_waterfall(sv, base, X_test_t, feat_names, top_i, proba)

    high_idx = np.where(proba >= 0.5)[0]
    dom = np.abs(sv[high_idx]).argmax(axis=1)
    fig_retention_segments(feat_names, dom, len(high_idx))

    dom_names = pd.Series([_pretty_feature(feat_names[i]) for i in dom])
    counts = dom_names.value_counts().head(10)

    print(f"\nHigh-risk (p≥0.50): {len(high_idx)} / {len(proba)}")
    print("Dominant drivers:")
    print(counts.to_string())

    (RES / 'shap_summary.json').write_text(json.dumps({
        'n_high_risk': int(len(high_idx)), 'dominant_drivers': counts.to_dict()}, indent=2))
    print(f"\nSaved results/shap_summary.json")

if __name__ == '__main__':
    main()