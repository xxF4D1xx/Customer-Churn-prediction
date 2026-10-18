# ============================================================
# TELCO CHURN — COMPLETE PIPELINE
# 1. Load & clean      2. Train/test split    3. Preprocess
# 4. Model comparison  5. Threshold + lift    6. SHAP explanations
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, cross_val_score,
                                     cross_val_predict)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from lightgbm import LGBMClassifier

RANDOM_STATE = 42


# ============================================================
# 1. LOAD & CLEAN
# ============================================================
df = pd.read_excel("C:/Users/Fadil/.cache/kagglehub/datasets/yeanzc/telco-customer-churn-ibm-dataset/versions/1/Telco_customer_churn.xlsx") 
y = df['Churn Value'].astype(int)

# --- drop: IDs, constants, leakage, redundant ---
drop_cols = [
    'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
    'Lat Long', 'Latitude', 'Longitude',
    'Churn Label', 'Churn Value', 'Churn Score', 'Churn Reason', 'CLTV',
]
X = df.drop(columns=drop_cols, errors='ignore').copy()

# --- fix Total Charges dtype + fill tenure-0 blanks with 0 ---
X['Total Charges'] = pd.to_numeric(X['Total Charges'], errors='coerce').fillna(0)

assert X.isna().sum().sum() == 0, "Still have NaNs"
assert not any(c in X.columns for c in ['Churn Score', 'Churn Reason', 'Churn Label'])
assert [c for c in X.columns if X[c].nunique() <= 1] == []

print(f"X: {X.shape}   churn rate: {y.mean():.3f}")


# ============================================================
# 2. TRAIN / TEST SPLIT (stratified)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")


# ============================================================
# 3. PREPROCESSING PIPELINES
# ============================================================
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

pre_scaled = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
])
pre_unscaled = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
])


# ============================================================
# 4. MODEL COMPARISON (repeated stratified CV)
# ============================================================
models = {
    'LogReg': Pipeline([('pre', pre_scaled),
                        ('clf', LogisticRegression(max_iter=2000,
                                                   class_weight='balanced',
                                                   random_state=RANDOM_STATE))]),
    'RF':     Pipeline([('pre', pre_unscaled),
                        ('clf', RandomForestClassifier(n_estimators=500,
                                                       min_samples_leaf=5,
                                                       class_weight='balanced',
                                                       n_jobs=-1,
                                                       random_state=RANDOM_STATE))]),
    'LGBM':   Pipeline([('pre', pre_unscaled),
                        ('clf', LGBMClassifier(n_estimators=500, learning_rate=0.03,
                                               num_leaves=31, min_child_samples=30,
                                               class_weight='balanced',
                                               random_state=RANDOM_STATE,
                                               verbose=-1))]),
}

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
print("\n=== Repeated CV (3x5-fold) ROC-AUC ===")
cv_results = {}
for name, pipe in models.items():
    s = cross_val_score(pipe, X_train, y_train, cv=rcv,
                        scoring='roc_auc', n_jobs=-1)
    cv_results[name] = (s.mean(), s.std())
    print(f"{name:8s}: {s.mean():.4f} ± {s.std():.4f}")

best_name = max(cv_results, key=lambda k: cv_results[k][0])
best_pipe = models[best_name]
print(f"\nSelected model: {best_name}")


# ============================================================
# 5. FIT, THRESHOLD, TEST EVAL
# ============================================================
best_pipe.fit(X_train, y_train)

proba_train = cross_val_predict(best_pipe, X_train, y_train,
                                cv=StratifiedKFold(5, shuffle=True,
                                                   random_state=RANDOM_STATE),
                                method='predict_proba', n_jobs=-1)[:, 1]
proba_test = best_pipe.predict_proba(X_test)[:, 1]

print("\n=== Test set ===")
print(f"ROC-AUC : {roc_auc_score(y_test, proba_test):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, proba_test):.4f}")

# --- threshold sensitivity table (train CV probs) ---
print("\n=== Threshold sensitivity (train CV) ===")
print(f"{'Thresh':>7} {'Prec':>6} {'Recall':>7} {'F1':>6} {'#Flag':>7} {'#Catch':>7} {'#Miss':>6}")
for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    pred = (proba_train >= t).astype(int)
    p = precision_score(y_train, pred, zero_division=0)
    r = recall_score(y_train, pred)
    f = f1_score(y_train, pred)
    print(f"{t:>7.2f} {p:>6.3f} {r:>7.3f} {f:>6.3f} {pred.sum():>7} "
          f"{((pred==1)&(y_train==1)).sum():>7} {((pred==0)&(y_train==1)).sum():>6}")

# --- pick default threshold ---
THRESHOLD = 0.50
pred_test = (proba_test >= THRESHOLD).astype(int)
print(f"\n=== Test @ threshold {THRESHOLD} ===")
print(confusion_matrix(y_test, pred_test))
print(classification_report(y_test, pred_test,
                            target_names=['No churn', 'Churn']))

# --- lift table (test) — headline result ---
order = proba_test.argsort()[::-1]
y_sorted = y_test.iloc[order].values
n = len(y_sorted)
total_churners = y_sorted.sum()
base_rate = total_churners / n

print(f"\n=== Lift (test) — base churn rate {base_rate:.1%} ===")
print(f"{'Top %':>7} {'#Contact':>9} {'#Caught':>8} {'%AllChurn':>10} {'HitRate':>9} {'Lift':>6}")
for pct in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    k = int(n * pct)
    caught = y_sorted[:k].sum()
    hit = caught / k
    print(f"{pct:>6.0%} {k:>9} {caught:>8} {caught/total_churners:>9.1%} "
          f"{hit:>9.1%} {hit/base_rate:>5.2f}x")


# ============================================================
# 6. SLIM-FEATURE CHECK  (corrected)
# ============================================================
print("\n=== Slim feature check (drop noise-level features) ===")
weak = ['Gender', 'Streaming Movies', 'Multiple Lines']
weak = [c for c in weak if c in X_train.columns]
X_train_slim = X_train.drop(columns=weak)

# rebuild preprocessors for the slim column set
num_slim = X_train_slim.select_dtypes(include=np.number).columns.tolist()
cat_slim = X_train_slim.select_dtypes(include='object').columns.tolist()

pre_scaled_slim = ColumnTransformer([
    ('num', StandardScaler(), num_slim),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_slim),
])
pre_unscaled_slim = ColumnTransformer([
    ('num', 'passthrough', num_slim),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_slim),
])

models_slim = {
    'LogReg': Pipeline([('pre', pre_scaled_slim),
                        ('clf', LogisticRegression(max_iter=2000,
                                                   class_weight='balanced',
                                                   random_state=RANDOM_STATE))]),
    'RF':     Pipeline([('pre', pre_unscaled_slim),
                        ('clf', RandomForestClassifier(n_estimators=500,
                                                       min_samples_leaf=5,
                                                       class_weight='balanced',
                                                       n_jobs=-1,
                                                       random_state=RANDOM_STATE))]),
}

for name in [best_name, 'LogReg']:
    full = cross_val_score(models[name], X_train, y_train, cv=rcv,
                           scoring='roc_auc', n_jobs=-1).mean()
    slim = cross_val_score(models_slim[name], X_train_slim, y_train, cv=rcv,
                           scoring='roc_auc', n_jobs=-1).mean()
    print(f"{name:8s}: full={full:.4f}  slim={slim:.4f}  Δ={slim-full:+.4f}")
# ============================================================
# 7. PERMUTATION IMPORTANCE
# ============================================================
from sklearn.inspection import permutation_importance

print("\n=== Top 15 features (permutation importance, test) ===")
r = permutation_importance(best_pipe, X_test, y_test, n_repeats=10,
                           scoring='roc_auc', random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.Series(r.importances_mean, index=X_test.columns).sort_values(ascending=False)
print(imp.head(15).to_string())


# ============================================================
# 8. SHAP — GLOBAL + PER-CUSTOMER  (robust to all SHAP versions)
# ============================================================
try:
    import shap
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
    import shap

print("\n=== SHAP analysis ===")

# transform inputs through the fitted preprocessor
X_train_t = best_pipe.named_steps['pre'].transform(X_train)
X_test_t  = best_pipe.named_steps['pre'].transform(X_test)
feature_names = best_pipe.named_steps['pre'].get_feature_names_out()
clf = best_pipe.named_steps['clf']

# --- build explainer ---
if best_name in ('RF', 'LGBM'):
    explainer = shap.TreeExplainer(clf)
else:
    explainer = shap.LinearExplainer(clf, X_train_t)

raw = explainer.shap_values(X_test_t)
base_raw = explainer.expected_value

# --- normalize to (n_samples, n_features) for class = churn (1) ---
def to_churn_shap(raw, base):
    """
    Returns (values_2d, base_scalar) for the churn class.
    Handles:
      - list [class0, class1]           (old SHAP)
      - 3D array (n_samples, n_feat, C) (new SHAP for RF)
      - 2D array (n_samples, n_feat)    (already churn, e.g. LGBM often)
    """
    if isinstance(raw, list):
        vals = raw[1]
        b = base[1] if isinstance(base, (list, np.ndarray)) and np.ndim(base) > 0 else base
    elif np.ndim(raw) == 3:
        # some versions put classes in last axis; assume (n, f, C)
        vals = raw[:, :, 1]
        b = base[1] if isinstance(base, (list, np.ndarray)) and np.ndim(base) > 0 else base
    else:
        vals = raw
        b = base[1] if isinstance(base, (list, np.ndarray)) and np.ndim(base) > 0 else base
    return np.asarray(vals), float(np.asarray(b).ravel()[-1])

sv, base_val = to_churn_shap(raw, base_raw)
assert sv.ndim == 2, f"expected 2D SHAP, got {sv.shape}"
assert sv.shape == X_test_t.shape, f"SHAP {sv.shape} != X {X_test_t.shape}"
print(f"SHAP values shape: {sv.shape}, base value: {base_val:.4f}")

# --- global bar (magnitude) ---
plt.figure()
shap.summary_plot(sv, X_test_t, feature_names=feature_names,
                  plot_type='bar', show=False)
plt.title(f"SHAP feature importance — {best_name}")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=120, bbox_inches='tight')
plt.close()

# --- global beeswarm (direction) ---
plt.figure()
shap.summary_plot(sv, X_test_t, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=120, bbox_inches='tight')
plt.close()

# --- waterfall for the highest-risk customer ---
top_i = int(np.argmax(proba_test))
print(f"\nHighest-risk test customer: P(churn) = {proba_test[top_i]:.3f}")

shap.plots.waterfall(
    shap.Explanation(
        values=sv[top_i],
        base_values=base_val,
        data=X_test_t[top_i],
        feature_names=feature_names,
    ),
    show=False,
)
plt.tight_layout()
plt.savefig("shap_waterfall.png", dpi=120, bbox_inches='tight')
plt.close()

print("Saved: shap_bar.png, shap_beeswarm.png, shap_waterfall.png")

# --- retention playbook: cluster high-risk customers by dominant SHAP feature ---
high_risk_idx = np.where(proba_test >= 0.5)[0]
print(f"\nHigh-risk customers (p>=0.5): {len(high_risk_idx)} / {len(proba_test)}")

# dominant feature = argmax |SHAP| per high-risk customer
dom = np.abs(sv[high_risk_idx]).argmax(axis=1)
dom_names = pd.Series(feature_names[dom]).value_counts().head(10)
print("\nTop drivers among high-risk customers (dominant feature):")
print(dom_names.to_string())

# --- global bar (magnitude) ---
plt.figure()
shap.summary_plot(sv, X_test_t, feature_names=feature_names,
                  plot_type='bar', show=False)
plt.title(f"SHAP feature importance — {best_name}")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=120, bbox_inches='tight')
plt.close()

# --- global beeswarm (direction) ---
plt.figure()
shap.summary_plot(sv, X_test_t, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=120, bbox_inches='tight')
plt.close()

# --- waterfall for one highest-risk customer ---
risk_order = proba_test.argsort()[::-1]
top_i = risk_order[0]
print(f"\nHighest-risk test customer: P(churn) = {proba_test[top_i]:.3f}")

shap.plots.waterfall(
    shap.Explanation(values=sv[top_i],
                     base_values=base_val,
                     data=X_test_t[top_i],
                     feature_names=feature_names),
    show=False
)
plt.tight_layout()
plt.savefig("shap_waterfall.png", dpi=120, bbox_inches='tight')
plt.close()

print("Saved: shap_bar.png, shap_beeswarm.png, shap_waterfall.png")

# --- retention playbook: cluster high-risk customers by dominant SHAP feature ---
risk_mask = proba_test >= 0.5
high_risk_idx = np.where(risk_mask)[0]
print(f"\nHigh-risk customers (p>=0.5): {len(high_risk_idx)} / {len(proba_test)}")

# dominant feature = argmax |SHAP| per customer
dom = np.abs(sv[high_risk_idx]).argmax(axis=1)
dom_names = pd.Series(feature_names[dom]).value_counts().head(10)
print("\nTop drivers among high-risk customers (dominant feature):")
print(dom_names.to_string())