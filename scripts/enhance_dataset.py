import pandas as pd
import numpy as np

# Load the existing dataset
file_path = "data/synthetic_expanded.csv"
df = pd.read_csv(file_path)

# Normalize relevant columns
farm_size_norm = (df['farm_size_acres'] - df['farm_size_acres'].min()) / (df['farm_size_acres'].max() - df['farm_size_acres'].min())
previous_yield_norm = (df['yield_per_acre'] - df['yield_per_acre'].min()) / (df['yield_per_acre'].max() - df['yield_per_acre'].min())

# Add new features
np.random.seed(42)
df['repayment_history_score'] = np.clip(np.random.beta(2, 1.5, len(df)), 0, 1)
df['community_support_score'] = np.clip(np.random.normal(0.7, 0.15, len(df)), 0, 1)
df['loan_behavior_factor'] = np.clip(np.random.normal(0.5, 0.2, len(df)) + 0.2 * df['repayment_history_score'], 0, 1)
df['gender_bias_correction'] = np.where(df['gender'] == 'Female', 0.1, 0.02)

# Calculate climatic_index from rainfall, humidity, temperature if available, else random
if 'rainfall_forecast_next_90d' in df.columns and 'climate_risk_score' in df.columns:
    rainfall_norm = (df['rainfall_forecast_next_90d'] - df['rainfall_forecast_next_90d'].min()) / (df['rainfall_forecast_next_90d'].max() - df['rainfall_forecast_next_90d'].min())
    climate_risk_norm = (df['climate_risk_score'] - df['climate_risk_score'].min()) / (df['climate_risk_score'].max() - df['climate_risk_score'].min())
    df['climatic_index'] = np.clip(0.5 * rainfall_norm + 0.5 * (1 - climate_risk_norm), 0, 1)
else:
    df['climatic_index'] = np.clip(np.random.normal(0.6, 0.2, len(df)), 0, 1)

# Rebalance/adjust label for demo separability
prob = 0.25*farm_size_norm + 0.25*previous_yield_norm + 0.15*df['repayment_history_score'] + 0.1*df['community_support_score'] + 0.1*df['climatic_index']
prob += 0.1*df['loan_behavior_factor'] + 0.05*df['gender_bias_correction'] + np.random.normal(0, 0.05, len(df))
df['loan_class'] = (prob > 0.45).astype(int)

# Save enhanced dataset
out_path = "data/synthetic_expanded_enhanced.csv"
df.to_csv(out_path, index=False)
print(f"Enhanced dataset saved to {out_path}")
