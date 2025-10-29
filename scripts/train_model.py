import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
import os
warnings.filterwarnings('ignore')

def train_and_evaluate(X, y, results_dir, models_dir):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    metrics = {}
    best_f1 = -1
    best_model = None
    best_name = None
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        metrics[name] = {
            'cv_f1_mean': np.mean(cv_scores),
            'cv_f1_std': np.std(cv_scores),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision_score(y_test, y_pred_test, average='weighted'),
            'recall': recall_score(y_test, y_pred_test, average='weighted'),
            'f1': f1,
            'classification_report': classification_report(y_test, y_pred_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
        }
        # Overfitting warning
        if train_acc - test_acc > 0.1:
            print(f"Warning: {name} may be overfitting (train acc {train_acc:.2f} vs test acc {test_acc:.2f})")
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
    # Save best model
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, 'credit_model.pkl'))
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'metrics_summary.txt'), 'w') as f:
        for name, m in metrics.items():
            f.write(f"{name}:\n")
            for k, v in m.items():
                if k in ['classification_report', 'confusion_matrix']:
                    f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {k}: {v:.4f}\n")
            f.write('\n')
        f.write(f"Best model: {best_name}\n")
    return best_model, best_name, metrics, (X_test, y_test)
