import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
import lime
import lime.lime_tabular
import joblib
import os

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_checkpoint")
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_json("synthetic_farmers_10000.json")
print("[INFO] Loaded data:", df.shape)

# Drop name column
df = df.drop(columns=["name"])

# === Encode categorical columns ===
label_encoders = {}
categorical_cols = ["region", "loan_purpose"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=["creditworthy"])
y = df["creditworthy"]

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train XGBoost ===
print("[INFO] Training XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"[METRICS] ROC-AUC: {roc_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
print(classification_report(y_test, y_pred))

# === Save Model and Encoders ===
joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.joblib"))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, "encoders.joblib"))

# === Generate LIME explanations ===
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=['Rejected', 'Approved'],
    mode='classification'
)

sample_indices = [0, 1, 2]
for idx in sample_indices:
    exp = explainer.explain_instance(X_test.iloc[idx], model.predict_proba)
    exp.save_to_file(os.path.join(MODEL_DIR, f"lime_explanation_{idx}.html"))

print("[✅] Model training complete. LIME explanations saved to model_checkpoint/")