"""
training.py — XGBoost + LIME (Final Version)
---------------------------------------------
✔ Cleans stringified numerics like "[6.6125E-1]"
✔ Ensures all features are float32
✔ Trains an XGBClassifier
✔ Generates LIME explanations for interpretability
✔ Saves model, scaler, features, and explanation plots
"""

import os
import re
import json
import joblib
import lime
import lime.lime_tabular
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
DATA_PATH = "./data/synthetic_expanded_enhanced.csv"
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
    """Convert weird numeric strings into floats."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return v
    if isinstance(v, str):
        v = v.strip()
        v = re.sub(r"[\[\]'\"\s]", "", v)
        v = v.replace(",", "")
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
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# ---------------------------------------------------------------------
# ENCODE CATEGORICAL
# ---------------------------------------------------------------------
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    print(f"[INFO] Encoding {len(cat_cols)} categorical columns: {cat_cols}")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
else:
    print("[INFO] No categorical columns detected.")

# ---------------------------------------------------------------------
# FINAL SAFETY
# ---------------------------------------------------------------------
non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
if non_numeric:
    print("[WARN] Non-numeric columns remain, coercing to numeric:", non_numeric)
    for c in non_numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

X = X.astype(np.float32).fillna(0)

# ---------------------------------------------------------------------
# SCALE
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------
# TRAIN/TEST SPLIT
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# TRAIN XGBOOST MODEL
# ---------------------------------------------------------------------
print("[INFO] Training XGBoost model...")
model = XGBClassifier(
    eval_metric="logloss",
    learning_rate=0.05,
    max_depth=5,
    n_estimators=200,
    random_state=42,
    enable_categorical=False
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------------------
print("[INFO] Model Evaluation:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------
# SAVE ARTIFACTS
# ---------------------------------------------------------------------
joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
    json.dump(X.columns.tolist(), f)

# ---------------------------------------------------------------------
# LIME EXPLANATION
# ---------------------------------------------------------------------
print("[INFO] Generating LIME explanations...")

try:
    # Rebuild sample DataFrame for LIME
    sample = pd.DataFrame(X_scaled, columns=X.columns)

    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X.columns.tolist(),
        class_names=[str(c) for c in np.unique(y)],
        mode="classification",
        discretize_continuous=True
    )

    # Choose a sample instance for explanation
    i = np.random.randint(0, X_test.shape[0])
    exp = explainer.explain_instance(
        X_test[i],
        model.predict_proba,
        num_features=10
    )

    # Save explanation as HTML and PNG
    html_path = os.path.join(MODEL_DIR, "lime_explanation.html")
    exp.save_to_file(html_path)

    png_path = os.path.join(MODEL_DIR, "lime_explanation.png")
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"[INFO] LIME explanation saved: {html_path} and {png_path}")

except Exception as e:
    print(f"[WARN] LIME explanation failed: {e}")

print(f"[✅] Training complete. Model + LIME outputs saved to '{MODEL_DIR}'")