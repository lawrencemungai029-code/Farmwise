import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

print("[DEBUG] Starting server.py...")

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'credit_model.pkl')
STATIC_PLOTS = os.path.join(os.path.dirname(__file__), 'static', 'plots')

print(f"[DEBUG] MODEL_PATH: {MODEL_PATH}")
print(f"[DEBUG] STATIC_PLOTS: {STATIC_PLOTS}")

app = Flask(__name__, static_folder='static')
CORS(app)

# --- Model Loading ---
def load_model():
    print("[DEBUG] Loading model...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("[DEBUG] Model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

# --- Preprocessing ---
def preprocess_input(data):
    # Define model features
    features = [
        'age', 'region', 'farm_size', 'loan_purpose',
        'disability', 'group_membership'
    ]
    processed = {k: data.get(k, 0) for k in features}
    df = pd.DataFrame([processed])

    # Convert categorical to numeric
    categorical_cols = ['region', 'loan_purpose']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.factorize(df[col])[0]  # simple integer encoding

    # Convert boolean to int
    if 'group_membership' in df.columns:
        df['group_membership'] = df['group_membership'].astype(int)

    return df

# --- Prediction & SHAP ---
def predict_score(model, input_df, raw_data):
    try:
        score = float(model.predict_proba(input_df)[0][1])
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

    reasoning = []

    # Business rules
    if raw_data.get('disability', 0) == 1:
        score = min(score + 0.05, 1.0)
        reasoning.append('Disability bias applied')

    eligibility = 'Approved' if score >= 0.5 else 'Rejected'

    recommended_limit = 300000 if raw_data.get('group_membership', 0) == 1 else 150000
    if raw_data.get('group_membership', 0) == 1:
        reasoning.append('Group loan eligibility')

    # SHAP explanation
    shap_summary_path = os.path.join(STATIC_PLOTS, 'shap_summary.png')
    shap_waterfall_path = os.path.join(STATIC_PLOTS, 'shap_waterfall.png')
    shap_data = {}

    try:
        if hasattr(model, 'predict_proba') and hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)

        shap_values = explainer(input_df)
        top_features_idx = np.argsort(np.abs(shap_values.values[0]))[::-1][:3]
        feature_names = input_df.columns[top_features_idx].tolist()

        for feat in feature_names:
            reasoning.append(f"{feat} influenced the score")

        # Save SHAP summary bar plot
        plt.figure()
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        plt.savefig(shap_summary_path)
        plt.close()

        # Save SHAP waterfall plot
        plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        plt.savefig(shap_waterfall_path)
        plt.close()

        # Add shap_data for frontend visualization
        shap_data = {
            "feature_values": input_df.to_dict(orient='records')[0],
            "shap_values": dict(zip(
                input_df.columns,
                shap_values.values[0].tolist()
            )),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        shap_summary_path = None
        shap_waterfall_path = None
        reasoning.append(f"SHAP explanation unavailable: {e}")

    # Save ROC-AUC placeholder
    roc_auc_path = os.path.join(STATIC_PLOTS, 'roc_auc.png')
    if not os.path.exists(roc_auc_path):
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC (placeholder)')
        plt.savefig(roc_auc_path)
        plt.close()

    output = {
        "farmer_name": raw_data.get("name", "Unknown"),
        "credit_score": round(score, 4),
        "eligibility": eligibility,
        "reasoning": reasoning,
        "recommended_limit": recommended_limit,
        "plots": {
            "roc_auc": "/static/plots/roc_auc.png",
            "shap_summary": "/static/plots/shap_summary.png",
            "shap_waterfall": "/static/plots/shap_waterfall.png"
        },
        "shap_data": shap_data
    }

    return output

# --- GPT Formatter Stub ---
def format_with_gpt(score_data):
    return f"Farmer {score_data['farmer_name']} is {score_data['eligibility']} for a loan of up to {score_data['recommended_limit']}. Reasoning: {', '.join(score_data['reasoning'])}."

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        model = load_model()
        input_df = preprocess_input(data)
        result = predict_score(model, input_df, data)

        result["gpt_output"] = format_with_gpt(result)
        return jsonify(result), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- Serve Plots ---
@app.route('/static/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(STATIC_PLOTS, filename)

# --- Start Server ---
if __name__ == '__main__':
    print("[DEBUG] __main__ block starting...")
    os.makedirs(STATIC_PLOTS, exist_ok=True)
    print("[DEBUG] STATIC_PLOTS directory ensured.")
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("[DEBUG] Flask app should be running.")