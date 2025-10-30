import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'credit_model.pkl')
STATIC_PLOTS = os.path.join(os.path.dirname(__file__), 'static', 'plots')

app = Flask(__name__, static_folder='static')
CORS(app)

# --- Model Loading ---
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")

# --- Preprocessing ---
def preprocess_input(data):
    # Map frontend JSON to model input DataFrame
    features = [
        'age', 'region', 'farm_size', 'loan_purpose',
        'disability', 'group_membership'
    ]
    processed = {k: data.get(k, 0) for k in features}
    df = pd.DataFrame([processed])
    return df

# --- Prediction & SHAP ---
def predict_score(model, input_df, raw_data):
    # Predict score
    try:
        score = float(model.predict_proba(input_df)[0][1])
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

    reasoning = []
    # Business rules
    if raw_data.get('disability', 0) == 1:
        score = min(score + 0.05, 1.0)
        reasoning.append('Disability bias applied')

    # Eligibility
    eligibility = 'Approved' if score >= 0.5 else 'Rejected'

    # Loan limit
    if raw_data.get('group_membership', 0) == 1:
        recommended_limit = 300000  # Group loans can be higher
        reasoning.append('Group loan eligibility')
    else:
        recommended_limit = 150000

    # SHAP explanations
    try:
        explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.Explainer(model)
        shap_values = explainer(input_df)
        top_features_idx = np.argsort(np.abs(shap_values.values[0]))[::-1][:3]
        feature_names = input_df.columns[top_features_idx].tolist()
        for feat in feature_names:
            reasoning.append(f"{feat} influenced the score")
        # Save SHAP summary plot
        shap_summary_path = os.path.join(STATIC_PLOTS, 'shap_summary.png')
        plt.figure()
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        plt.savefig(shap_summary_path)
        plt.close()
    except Exception as e:
        shap_summary_path = None
        reasoning.append(f"SHAP explanation unavailable: {e}")

    # Save ROC-AUC plot (stub: replace with real ROC-AUC if available)
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
            "shap_summary": "/static/plots/shap_summary.png"
        }
    }
    return output

# --- GPT Formatter (stub) ---
def format_with_gpt(score_data):
    # Stub: Replace with actual GPT call if available
    # For now, just return a formatted string
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
        # Optionally add GPT-formatted output
        result["gpt_output"] = format_with_gpt(result)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Static File Serving for Plots ---
@app.route('/static/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(STATIC_PLOTS, filename)

if __name__ == '__main__':
    os.makedirs(STATIC_PLOTS, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
