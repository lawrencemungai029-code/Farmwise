import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- CONFIGURABLE ---
ENHANCED_CSV = 'farmers_data_enhanced_v2.csv'
SCALER_PATH = 'scaler_final.pkl'
FEATURES = [
    'repayment_history_score',
    'community_support_score',
    'neighbour_performance_index',
    'loan_behavior_factor',
    'climatic_index',
    'rainfall_forecast_next_90d',
    'past_crop_performance_index',
    'farm_size_acres',
    'yield_per_acre',
    'farming_years',
    'climate_risk_score',
    'disabled'
]
TARGET = 'loan_class'

# Load data
assert os.path.exists(ENHANCED_CSV), f"{ENHANCED_CSV} not found. Run enhance_v2.py first."
df = pd.read_csv(ENHANCED_CSV)

# Feature selection
X = df[FEATURES].copy()
y = df[TARGET]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}")

# Save processed features for model training
df_proc = pd.DataFrame(X_scaled, columns=FEATURES)
df_proc[TARGET] = y.values
df_proc.to_csv('farmers_data_proc_v2.csv', index=False)
print("Processed features saved to farmers_data_proc_v2.csv")
