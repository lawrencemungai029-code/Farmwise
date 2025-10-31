import os
import json
import joblib
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")  # Prevent GUI-related errors
import matplotlib.pyplot as plt
import warnings
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Suppress harmless LIME/matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lime")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_checkpoint")
STATIC_PLOTS = os.path.join(BASE_DIR, "static", "plots")

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

app = Flask(__name__, static_folder="static")
CORS(app)

# ---------------------------------------------------------------------
# Load Artifacts Once
# ---------------------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

os.makedirs(STATIC_PLOTS, exist_ok=True)

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def preprocess_input(data, features, scaler):
    try:
        df = pd.DataFrame([data])
        df = df.reindex(columns=features, fill_value=0)
        X_scaled = scaler.transform(df)
        return pd.DataFrame(X_scaled, columns=features)
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {e}")

def generate_lime_explanation(model, input_df, raw_data, features):
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(input_df.values),
            feature_names=features,
            class_names=["Rejected", "Approved"],
            mode="classification",
            discretize_continuous=True
        )

        exp = explainer.explain_instance(
            input_df.values[0],
            model.predict_proba,
            num_features=5
        )

        # Save LIME plot
        plot_filename = f"lime_explanation_{raw_data.get('name', 'farmer').replace(' ', '_')}.png"
        plot_path = os.path.join(STATIC_PLOTS, plot_filename)

        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        explanation_tuples = exp.as_list()
        reasoning = [
            f"{feat} {'positively' if w > 0 else 'negatively'} influenced the score"
            for feat, w in explanation_tuples[:3]
        ]

        return {
            "features": [f for f, _ in explanation_tuples],
            "weights": [float(w) for _, w in explanation_tuples],
            "reasoning": reasoning,
            "plot_filename": plot_filename
        }
    except Exception as e:
        return {
            "features": [],
            "weights": [],
            "reasoning": [f"LIME explanation unavailable: {e}"],
            "plot_filename": None
        }

# ---------------------------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate incoming data
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format. Expected JSON object."}), 400

        input_df = preprocess_input(data, features, scaler)

        # Safe model inference
        try:
            score = float(model.predict_proba(input_df)[0][1])
        except AttributeError:
            return jsonify({"error": "Loaded model does not support predict_proba."}), 500
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {e}"}), 500

        eligibility = "Approved" if score >= 0.5 else "Rejected"
        recommended_limit = 300000 if data.get("group_membership") else 150000

        # LIME explanation
        explanation = generate_lime_explanation(model, input_df, data, features)

        result = {
            "farmer_name": data.get("name", "Unknown"),
            "credit_score": round(score, 4),
            "eligibility": eligibility,
            "recommended_limit": recommended_limit,
            "reasoning": explanation["reasoning"],
            "plots": {
                "lime_explanation": f"/static/plots/{explanation['plot_filename']}"
                if explanation["plot_filename"]
                else None
            },
            "explanation_data": {
                "features": explanation["features"],
                "weights": explanation["weights"],
            },
            "gpt_output": (
                f"Farmer {data.get('name', 'Unknown')} is {eligibility} "
                f"for a loan up to {recommended_limit}. "
                f"Reasoning: {', '.join(explanation['reasoning'])}."
            ),
        }

        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500

# ---------------------------------------------------------------------
# Serve Static Plots
# ---------------------------------------------------------------------
@app.route("/static/plots/<path:filename>")
def serve_plot(filename):
    try:
        return send_from_directory(STATIC_PLOTS, filename)
    except FileNotFoundError:
        return jsonify({"error": "Plot not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to serve plot: {e}"}), 500

# ---------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)