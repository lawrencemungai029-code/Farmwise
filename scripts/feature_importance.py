import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_feature_importance(model, X, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        return pd.DataFrame({'feature': X.columns[indices], 'importance': importances[indices]})
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        indices = np.argsort(importances)[::-1][:top_n]
        return pd.DataFrame({'feature': X.columns[indices], 'importance': importances[indices]})
    else:
        raise ValueError('Model does not support feature importance.')
