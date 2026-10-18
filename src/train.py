"""Train, compare models, evaluate, and save professional figures."""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold, StratifiedKFold,
    cross_val_score, cross_val_predict,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve,
)
from lightgbm import LGBMClassifier

from src.preprocess import load_raw, clean
from src.plotting import apply_style, PALETTE

warnings.filterwarnings('ignore')
apply_style()

RANDOM_STATE = 42
FIG = Path('figures'); FIG.mkdir(exist_ok=True)
RES = Path('results'); RES.mkdir(exist_ok=True)

def _pipeline(pre, clf):
    return Pipeline([('pre', pre), ('clf', clf)])

def _cv(pipe, X, y, cv):
    return cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

def fig_model_comparison(cv_results: dict):
    names = list(cv_results.keys())
    means = [cv_results[n]['mean'] for n in names]
    stds = [cv_results[n]['std'] for n in names]
    best_i = int(np.argmax(means))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = [PALETTE['primary'] if i == best_i else PALETTE['muted']
              for i in range(len(names))]
    bars = ax.barh(names, means, xerr=stds, color=colors,
                   edgecolor='none', capsize=4)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.002, i, f'{m:.4f} ± {s:.4f}',
                va='center', fontsize=10,
                color='#111827',
                fontweight='bold' if i == best_i else 'normal')
    ax.set_xlim(0.80, max(means) + 0.03)
    ax.set_xlabel('Repeated stratified 5-fold CV ROC-AUC (3 repeats)')
    ax.set_title('Model comparison')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / 'model_comparison.png', bbox_inches='tight')
    plt.close(fig)

def fig_lift_curve(proba, y_test, model_name):
    order = np.argsort(proba)[::-1]
    y_sorted = np.asarray(y_test)[order]
    n = len(y_sorted)
    total = y_sorted.sum()
    base = total / n

    pcts = np.linspace(0.01, 1.0, 200)
    cum = np.array([y_sorted[: max(1, int(p * n))].sum() / total for p in pcts])
    rand = pcts

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(pcts, cum, color=PALETTE['primary'], linewidth=2.4,
            label=f'{model_name} (test)')
    ax.plot([0, 1], [0, 1], linestyle='--', color=PALETTE['muted'],
            linewidth=1.4, label='Random targeting')
    ax.fill_between(pcts, cum, rand, color=PALETTE['primary'], alpha=0.08)

    k30 = int(0.30 * n)
    caught30 = y_sorted[:k30].sum() / total
    ax.scatter([0.30], [caught30], s=80, color=PALETTE['secondary'],
               zorder=5, edgecolor='white', linewidth=1.5)
    ax.annotate(
        f'top 30% → {caught30:.0%} of churners\n'
        f'({caught30 / 0.30 / base:.2f}× lift)',
        xy=(0.30, caught30), xytext=(20, -30),
        textcoords='offset points',
        fontsize=10, color=PALETTE['secondary'], fontweight='bold',
        arrowprops=dict(arrowstyle='-', color=PALETTE['secondary'], lw=1.2),
    )
    ax.set_xlabel('Fraction of customers contacted (highest risk first)')
    ax.set_ylabel('Fraction of churners captured')
    ax.set_title('Cumulative lift curve — test set')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(FIG / 'lift_curve.png', bbox_inches='tight')
    plt.close(fig)

def fig_roc(proba, y_test, model_name):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE['primary'], linewidth=2.4,
            label=f'{model_name} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color=PALETTE['muted'],
            linewidth=1.2, label='Random')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve — test set')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(FIG / 'roc_curve.png', bbox_inches='tight')
    plt.close(fig)

def fig_threshold_curve(proba_train, y_train):
    ts = np.arange(0.20, 0.81, 0.01)
    prec, rec, f1, flagged = [], [], [], []
    for t in ts:
        pred = (proba_train >= t).astype(int)
        prec.append(precision_score(y_train, pred, zero_division=0))
        rec.append(recall_score(y_train, pred))
        f1.append(f1_score(y_train, pred, zero_division=0))
        flagged.append(pred.sum())

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(ts, prec, color=PALETTE['primary'], linewidth=2, label='Precision')
    ax1.plot(ts, rec, color=PALETTE['secondary'], linewidth=2, label='Recall')
    ax1.plot(ts, f1, color=PALETTE['accent'], linewidth=2, label='F1')
    ax1.axvline(0.50, linestyle=':', color='#6B7280', linewidth=1.2)
    ax1.text(0.505, 0.05, 'default 0.50', color='#6B7280',
             fontsize=9, rotation=90, va='bottom')
    ax1.set_xlabel('Decision threshold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1.02)
    ax1.legend(loc='upper right')
    ax1.set_title('Threshold sensitivity — train CV probabilities')

    ax2 = ax1.twinx()
    ax2.fill_between(ts, flagged, alpha=0.12, color=PALETTE['primary'])
    ax2.plot(ts, flagged, linestyle='--', color=PALETTE['primary'], alpha=0.7, linewidth=1)
    ax2.set_ylabel('# customers flagged', color=PALETTE['primary'])
    ax2.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / 'threshold_sensitivity.png', bbox_inches='tight')
    plt.close(fig)

def fig_confusion(y_test, proba, model_name, threshold=0.5):
    from sklearn.metrics import ConfusionMatrixDisplay
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No churn', 'Churn'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title(f'Confusion matrix — {model_name} @ {threshold:.2f}')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / 'confusion_matrix.png', bbox_inches='tight')
    plt.close(fig)

def main():
    df = load_raw()
    X, y = clean(df, slim=True)
    print(f"X: {X.shape}, churn rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    num = X_train.select_dtypes(include=np.number).columns.tolist()
    
    # THE FIX: Use select_dtypes to catch 'str', 'string', and 'object'
    cat = X_train.select_dtypes(include=['object', 'string', 'str']).columns.tolist()

    pre_scaled = ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)])
    pre_unscaled = ColumnTransformer([
        ('num', 'passthrough', num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)])

    models = {
        'LogReg': _pipeline(pre_scaled, LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE)),
        'RandomForest': _pipeline(pre_unscaled, RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)),
        'LightGBM': _pipeline(pre_unscaled, LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
            class_weight='balanced', random_state=RANDOM_STATE, verbose=-1)),
    }

    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    print("\n=== Repeated CV (3×5-fold) ROC-AUC ===")
    cv_results = {}
    for name, pipe in models.items():
        s = _cv(pipe, X_train, y_train, rcv)
        cv_results[name] = {'mean': float(s.mean()), 'std': float(s.std())}
        print(f"{name:14s}: {s.mean():.4f} ± {s.std():.4f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]['mean'])
    best_pipe = models[best_name]
    print(f"\nSelected: {best_name}")

    best_pipe.fit(X_train, y_train)
    proba_train = cross_val_predict(best_pipe, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        method='predict_proba', n_jobs=-1)[:, 1]
    proba_test = best_pipe.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, proba_test)
    pr = average_precision_score(y_test, proba_test)
    print(f"\nTest ROC-AUC: {roc:.4f}")
    print(f"Test PR-AUC : {pr:.4f}")

    pred = (proba_test >= 0.50).astype(int)
    print("\nClassification report @ 0.50:")
    print(classification_report(y_test, pred, target_names=['No churn', 'Churn']))

    order = proba_test.argsort()[::-1]
    y_sorted = y_test.iloc[order].values
    n = len(y_sorted); total = y_sorted.sum(); base = total / n

    print(f"\n=== Lift (test) — base {base:.1%} ===")
    print(f"{'Top %':>6} {'#Contact':>9} {'#Caught':>8} {'%All':>7} {'Hit':>7} {'Lift':>6}")
    lift_rows = []
    for pct in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        k = int(n * pct)
        caught = int(y_sorted[:k].sum())
        hit = caught / k
        lift = hit / base
        lift_rows.append({'top_pct': pct, 'contacted': k, 'caught': caught, 'pct_of_all': caught / total, 'hit_rate': hit, 'lift': lift})
        print(f"{pct:>5.0%} {k:>9} {caught:>8} {caught/total:>6.1%} {hit:>6.1%} {lift:>5.2f}×")

    print("\nGenerating figures…")
    fig_model_comparison(cv_results)
    fig_lift_curve(proba_test, y_test, best_name)
    fig_roc(proba_test, y_test, best_name)
    fig_threshold_curve(proba_train, y_train)
    fig_confusion(y_test, proba_test, best_name)

    metrics = {
        'model': best_name, 'cv': cv_results,
        'test': {'roc_auc': roc, 'pr_auc': pr, 'threshold': 0.50,
                 'precision': float(precision_score(y_test, pred)),
                 'recall': float(recall_score(y_test, pred)),
                 'f1': float(f1_score(y_test, pred))},
        'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
        'lift': lift_rows,
    }
    (RES / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved results/metrics.json")

if __name__ == '__main__':
    main()