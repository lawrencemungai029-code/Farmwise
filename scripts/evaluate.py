import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def sample_prediction(model, X_test, y_test, df):
    idx = X_test.index[0]
    farmer_id = df.loc[idx, 'farmer_id'] if 'farmer_id' in df.columns else idx
    pred = model.predict([X_test.iloc[0]])[0]
    print(f"Sample Farmer ID: {farmer_id}\nPredicted Loan Class: {pred}")
    return farmer_id, pred
