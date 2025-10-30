import pandas as pd

# Load main enhanced dataset
main_df = pd.read_csv('data/synthetic_expanded_enhanced.csv')
# Load behavioral dataset
behav_df = pd.read_csv('data/farmers_data_with_target.csv')

# Merge on farmer_id (if present in both)
if 'farmer_id' in main_df.columns and 'farmer_id' in behav_df.columns:
    merged = pd.merge(main_df, behav_df, on='farmer_id', suffixes=('', '_behav'))
else:
    merged = pd.concat([main_df, behav_df], axis=0, ignore_index=True)

# Save merged dataset
merged.to_csv('data/merged_enhanced_behavioral.csv', index=False)
print('Merged dataset saved to data/merged_enhanced_behavioral.csv')
