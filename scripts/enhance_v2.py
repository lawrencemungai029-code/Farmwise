import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
import os

# --- CONFIGURABLE PARAMETERS ---
N_NEIGHBORS_FOR_INDEX = 5
DEMO_FARMER_IDS = ['F0001', 'F0002', 'F0003']  # Example demo IDs
ENHANCED_CSV = 'farmers_data_enhanced_v2.csv'

# Load dataset (try enhanced, fallback to merged)
if os.path.exists('farmers_data_enhanced.csv'):
    df = pd.read_csv('farmers_data_enhanced.csv')
else:
    df = pd.read_csv('data/merged_enhanced_behavioral.csv')

# 1. Add 'disabled' column
if 'disabled' not in df.columns:
    df['disabled'] = df['farmer_id'].apply(lambda x: 1 if x in DEMO_FARMER_IDS else 0)

# 2. Compute neighbour_performance_index
if 'county' in df.columns:
    # Use county-based neighbor mean
    def neighbor_index(row):
        group = df[df['county'] == row['county']]
        group = group[group['farmer_id'] != row['farmer_id']]
        vals = group['past_crop_performance_index'].values
        if len(vals) == 0:
            return row['past_crop_performance_index']
        return np.mean(vals[:N_NEIGHBORS_FOR_INDEX])
    df['neighbour_performance_index'] = df.apply(neighbor_index, axis=1)
elif {'lat','lon','past_crop_performance_index'}.issubset(df.columns):
    # Use geo-coordinates
    coords = df[['lat','lon']].values
    kdt = KDTree(coords)
    indices = kdt.query(coords, k=N_NEIGHBORS_FOR_INDEX+1, return_distance=False)
    neighbor_means = []
    for i, idxs in enumerate(indices):
        idxs = idxs[idxs != i]  # Exclude self
        vals = df.iloc[idxs]['past_crop_performance_index'].values
        neighbor_means.append(np.mean(vals) if len(vals) else df.iloc[i]['past_crop_performance_index'])
    df['neighbour_performance_index'] = neighbor_means
else:
    # Fallback: use global mean
    df['neighbour_performance_index'] = df['past_crop_performance_index'].mean()

# Normalize to [0,1]
scaler = MinMaxScaler()
df['neighbour_performance_index'] = scaler.fit_transform(df[['neighbour_performance_index']])

# Save enhanced dataset
os.makedirs('data', exist_ok=True)
df.to_csv(ENHANCED_CSV, index=False)
print(f"Enhanced dataset saved to {ENHANCED_CSV}")
