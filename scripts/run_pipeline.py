
import sys
import os
import pandas as pd
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.preprocess import preprocess_data
from scripts.train_model import train_and_evaluate
from scripts.evaluate import sample_prediction


DATA_PATH = os.path.join('data', 'merged_enhanced_behavioral.csv')
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Load data
df = pd.read_csv(DATA_PATH)

# Preprocess
df_proc = preprocess_data(df)
X = df_proc.drop(columns=['loan_class'])
y = df_proc['loan_class']

# Train and evaluate
best_model, best_name, metrics, (X_test, y_test) = train_and_evaluate(X, y, RESULTS_DIR, MODELS_DIR)

# Sample prediction output
sample_prediction(best_model, X_test, y_test, df)

print(f"Best model: {best_name} saved to {MODELS_DIR}/credit_model.pkl")
