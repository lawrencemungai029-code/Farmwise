import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def optimize_model(X, y, model_name='RandomForest'):
    from imblearn.over_sampling import SMOTE
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_bal, y_bal = X, y
    if use_smote:
        sm = SMOTE(random_state=42)
        X_bal, y_bal = sm.fit_resample(X, y)
    if model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, 7, None]}
    elif model_name == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, 7, None]}
    elif model_name == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l2', 'none']
        }
    else:
        raise ValueError('Unknown model_name')
    grid = GridSearchCV(model, param_grid, cv=skf, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_bal, y_bal)
    return grid.best_estimator_, grid.best_params_, grid.best_score_
