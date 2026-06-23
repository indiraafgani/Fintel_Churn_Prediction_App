"""
generate_artifacts.py
=====================
Jalankan script ini di lokal / Colab untuk membuat models/artifacts.pkl
yang kompatibel dengan app.py FINTel Churn Intelligence Dashboard.

Requirements:
    pip install pandas numpy scikit-learn catboost imbalanced-learn
                category_encoders shap

Cara pakai:
    1. Letakkan script ini di root folder project
       (sejajar dengan folder data/, models/, utils/)
    2. Pastikan data/df_clean.csv ada
    3. python generate_artifacts.py
    4. Upload models/artifacts.pkl yang dihasilkan ke GitHub
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    recall_score, precision_score, fbeta_score, roc_auc_score,
    roc_curve
)
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("📂 Loading data...")
df = pd.read_csv("data/df_clean.csv")
print(f"   Shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("🔧 Preprocessing...")

# Target column — sesuaikan jika namanya berbeda di df_clean.csv
TARGET = "Churn"

# Konversi target ke binary 0/1 jika masih string
if df[TARGET].dtype == object:
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
df[TARGET] = df[TARGET].astype(int)

# Drop kolom yang tidak diperlukan
drop_cols = [TARGET]
for c in ["CustomerID", "customerID", "customer_id"]:
    if c in df.columns:
        drop_cols.append(c)

X = df.drop(columns=drop_cols)
y = df[TARGET]

# Encode categorical dengan OrdinalEncoder / get_dummies
cat_cols = X.select_dtypes(include="object").columns.tolist()
X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
X = X.astype(float)

feature_names_all = X.columns.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("✂️  Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SCALING
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE SELECTION — SelectKBest k=20 (sesuai sidebar app)
# ─────────────────────────────────────────────────────────────────────────────
print("🔍 Feature selection (SelectKBest k=20)...")
selector = SelectKBest(f_classif, k=20)
X_train_sel = selector.fit_transform(X_train_sc, y_train)
X_test_sel  = selector.transform(X_test_sc)

selected_mask  = selector.get_support()
selected_names = [feature_names_all[i] for i, m in enumerate(selected_mask) if m]
print(f"   Selected features: {selected_names}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SMOTE (handle class imbalance)
# ─────────────────────────────────────────────────────────────────────────────
print("⚖️  Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_sel, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAIN CATBOOST
# ─────────────────────────────────────────────────────────────────────────────
print("🚀 Training CatBoost...")
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric="F1",
    random_seed=42,
    verbose=100,
)
model.fit(X_train_res, y_train_res)

# ─────────────────────────────────────────────────────────────────────────────
# 8. THRESHOLD OPTIMISATION (F2-score, β=2)
# ─────────────────────────────────────────────────────────────────────────────
print("📐 Optimising threshold for F2 (β=2)...")
y_proba = model.predict_proba(X_test_sel)[:, 1]

best_thr, best_f2 = 0.5, 0.0
for thr in np.arange(0.05, 0.95, 0.01):
    preds = (y_proba >= thr).astype(int)
    f2    = fbeta_score(y_test, preds, beta=2, zero_division=0)
    if f2 > best_f2:
        best_f2, best_thr = f2, thr

print(f"   Best threshold: {best_thr:.4f}  F2: {best_f2:.4f}")
y_pred = (y_proba >= best_thr).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 9. METRICS
# ─────────────────────────────────────────────────────────────────────────────
metrics = {
    "recall":    recall_score(y_test, y_pred,    zero_division=0),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "f2":        fbeta_score(y_test, y_pred, beta=2, zero_division=0),
    "roc_auc":   roc_auc_score(y_test, y_proba),
    "threshold": best_thr,
}
print("\n📊 Model Metrics:")
for k, v in metrics.items():
    print(f"   {k:12s}: {v:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. SHAP — Global Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
print("\n🧠 Computing SHAP values (global)...")
explainer     = shap.TreeExplainer(model)
shap_vals_mat = explainer.shap_values(X_test_sel)

# shap_values bisa 2D (binary) atau 3D (multiclass)
if isinstance(shap_vals_mat, list):
    sv = shap_vals_mat[1]   # class 1 (Churn)
else:
    sv = shap_vals_mat

mean_abs_shap = np.abs(sv).mean(axis=0)
global_shap_importance = dict(zip(selected_names, mean_abs_shap.tolist()))
print("   Top 5 SHAP features:")
for k, v in sorted(global_shap_importance.items(), key=lambda x: -x[1])[:5]:
    print(f"   {k:40s}: {v:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

artifacts = {
    "model":                   model,
    "scaler":                  scaler,
    "selector":                selector,
    "model_metrics":           metrics,
    "global_shap_importance":  global_shap_importance,
    "selected_feature_names":  selected_names,
    "feature_names_all":       feature_names_all,
}

with open("models/artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\n✅ artifacts.pkl saved to models/artifacts.pkl")
print(f"   Keys: {list(artifacts.keys())}")
