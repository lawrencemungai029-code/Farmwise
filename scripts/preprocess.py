import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np

def preprocess_data(df, encode_type='auto', normalize=True):
    import re
    df = df.copy()
    # Sanitize column names for model compatibility
    df.columns = [re.sub(r'[^0-9a-zA-Z_]', '_', str(col)) for col in df.columns]
    # Detect categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Remove target if present
    if 'loan_class' in cat_cols:
        cat_cols.remove('loan_class')
    # Encode categorical columns
    if encode_type == 'label':
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        # Sanitize column names after label encoding
        df.columns = [re.sub(r'[^0-9a-zA-Z_]', '_', str(col)) for col in df.columns]
    else:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        # Sanitize column names after one-hot encoding
        df.columns = [re.sub(r'[^0-9a-zA-Z_]', '_', str(col)) for col in df.columns]
    # Normalize numeric columns
    if normalize:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'loan_class' in num_cols:
            num_cols.remove('loan_class')
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    return df
