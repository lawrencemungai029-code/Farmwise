"""
training.py — Final robust version for XGBoost + SHAP
Fixes:
- Aggressively cleans stringified numerics like "[6.6125E-1]"
- Ensures SHAP always receives numeric float32 inputs
- Adds strong debug diagnostics if non-numeric cols persist
"""

import os
import re
import json
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_PATH = "./data/merged_enhanced_behavioral.csv"
MODEL_DIR = "model_checkpoint"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
print("[INFO] Loading data...")
df = pd.read_csv(DATA_PATH)

drop_cols = ['farmer_id', 'phone_number', '37203063', '37203063_behav', 'phone_number_behav']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

TARGET_COL = 'loan_class'
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in data.")

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def clean_string_value(v):
    """Clean up weird numeric and list-like strings into scalars."""
    if pd.isna(v):
        return np.nan

    if isinstance(v, (int, float, np.number)):
        return v

    if isinstance(v, str):
        v = v.strip()
        # Remove brackets and quotes
        v = re.sub(r"[\[\]'\"\s]", "", v)
        # Remove multiple commas (keep decimal structure)
        v = v.replace(",", "")
        # If scientific or numeric pattern, try float
        if re.match(r"^-?\d*\.?\d+(e[-+]?\d+)?$", v, re.IGNORECASE):
            try:
                return float(v)
            except Exception:
                return np.nan
        return np.nan
    return np.nan


# ---------------------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------------------
print("[INFO] Cleaning data...")
X = X.applymap(clean_string_value)

# Force numeric coercion globally
X = X.apply(pd.to_numeric, errors="coerce")

# Replace missing with 0
X = X.fillna(0)

# ---------------------------------------------------------------------
# ENCODE CATEGORICAL
# ---------------------------------------------------------------------
# After numeric coercion, recheck object columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    print(f"[INFO] Encoding {len(cat_cols)} categorical columns: {cat_cols}")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
else:
    print("[INFO] No categorical columns detected after cleaning.")

# ---------------------------------------------------------------------
# FINAL SAFETY: ensure all numeric
# ---------------------------------------------------------------------
non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
if non_numeric:
    print("[WARN] Non-numeric columns remain, coercing to numeric:", non_numeric)
    for c in non_numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

X = X.astype(np.float32)
X = X.fillna(0)

# ---------------------------------------------------------------------
# SCALE
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------
# TRAIN/TEST
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------
print("[INFO] Training model...")
model = XGBClassifier(
    eval_metric="logloss",
    enable_categorical=False,
    random_state=42
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# EVAL
# ---------------------------------------------------------------------
print("[INFO] Classification report:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------
joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
    json.dump(X.columns.tolist(), f)

# ---------------------------------------------------------------------
# SHAP EXPLANATION
# ---------------------------------------------------------------------
print("[INFO] Generating SHAP explanations...")

try:
    # Build sample dataframe for SHAP
    sample = pd.DataFrame(X_scaled[:50], columns=X.columns).astype(np.float32)

    # Preferred modern SHAP API
    try:
        explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample)
    except Exception:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

    # SHAP summary (bar)
    shap_bar_path = os.path.join(MODEL_DIR, "shap_bar.png")
    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(shap_bar_path)
    plt.close()

    # SHAP beeswarm
    shap_beeswarm_path = os.path.join(MODEL_DIR, "shap_beeswarm.png")
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(shap_beeswarm_path)
    plt.close()

    print(f"[INFO] SHAP plots saved to {MODEL_DIR}")

except Exception as e:
    print(f"[WARN] SHAP explanation failed: {e}")

print(f"[✅] Training complete. Model and SHAP outputs saved to '{MODEL_DIR}'")