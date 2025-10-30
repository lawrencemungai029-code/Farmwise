import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve, classification_report
import matplotlib.pyplot as plt
import json

# --- CONFIGURABLE ---
PROC_CSV = 'farmers_data_proc_v2.csv'
REPORTS_DIR = 'reports'
MODELS_DIR = 'models'
SEED = 42
N_SPLITS = 5
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(PROC_CSV)
X = df.drop(columns=['loan_class'])
y = df['loan_class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# Models
def get_models():
    return {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=SEED),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED)
    }

results = {}
best_f1 = -1
best_model = None
best_name = None
for name, model in get_models().items():
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    f1s = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        f1 = roc_auc_score(y_train.iloc[val_idx], preds)
        f1s.append(f1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    pr, rc, _ = precision_recall_curve(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        'cv_roc_auc_mean': float(np.mean(f1s)),
        'cv_roc_auc_std': float(np.std(f1s)),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    # Save plots
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC-AUC ({roc_auc:.2f})')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.savefig(f'{REPORTS_DIR}/roc_auc_{name}.png')
    plt.close()

    plt.figure()
    plt.plot(rc, pr, label='Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.legend()
    plt.savefig(f'{REPORTS_DIR}/precision_recall_{name}.png')
    plt.close()

    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.colorbar()
    plt.savefig(f'{REPORTS_DIR}/confusion_matrix_{name}.png')
    plt.close()

    if report['weighted avg']['f1-score'] > best_f1:
        best_f1 = report['weighted avg']['f1-score']
        best_model = model
        best_name = name

# Save best model
joblib.dump(best_model, f'{MODELS_DIR}/credit_model_v2.pkl')

# Save metrics
with open(f'{REPORTS_DIR}/model_metrics.json', 'w') as f:
    json.dump({'results': results, 'best_model': best_name}, f, indent=2)
print(f"Best model: {best_name} saved to {MODELS_DIR}/credit_model_v2.pkl")
print(f"Metrics and plots saved to {REPORTS_DIR}/")
