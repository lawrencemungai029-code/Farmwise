# Farmwise: AI-Powered Agri-Finance Credit Scoring for Smallholder Farmers in Kenya

## Overview
Farmwise is a prototype machine learning pipeline and API for credit scoring of smallholder farmers in Kenya. It leverages synthetic and real-world-inspired data, advanced feature engineering, fairness checks, and explainability to support equitable, data-driven lending decisions in agri-finance.

## Features
- **Data Expansion & Preprocessing:** Synthetic dataset of 1000 farmers, with composite features for productivity, stability, and climate resilience.
- **Model Training & Tuning:** Logistic Regression, Random Forest, and XGBoost models, with hyperparameter tuning via GridSearchCV.
- **Fairness & Explainability:** Demographic parity checks, feature importance, and SHAP explainability (with workarounds for XGBoost/SHAP bugs).
- **Modular Pipeline:** Scripts for preprocessing, training, evaluation, fairness, feature importance, and optimization.

## Project Structure
```
.
├── app/                  # FastAPI backend
├── data/                 # Raw, processed, and synthetic datasets
├── models/               # Saved model artifacts
├── notebooks/            # Jupyter notebooks for exploration
├── plots/                # Visualizations (e.g., SHAP summary)
├── results/              # Model evaluation summaries
├── scripts/              # Modular pipeline scripts
├── requirements.txt      # Python dependencies
├── fairness_rules.md     # Fairness and credit scoring rules
├── README.md             # Project documentation
```

## How to Run
1. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```
2. **Run the pipeline:**
	```bash
	PYTHONPATH=. python3 scripts/run_pipeline.py
	```
3. **Start the API:**
	```bash
	uvicorn app.main:app --reload
	```
4. **View results:**
	- Model summaries: `results/model_summary.json`, `results/model_summary_tuned.json`
	- SHAP plots: `plots/shap_summary.png`

## Key Files
- `data/synthetic_expanded.csv`: Main dataset (1000 rows, engineered features)
- `models/best_model_tuned.pkl`: Best tuned model
- `results/model_summary_tuned.json`: Final evaluation metrics
- `plots/shap_summary.png`: SHAP feature importance plot
- `scripts/`: All pipeline logic (preprocessing, training, evaluation, fairness, feature importance, optimization)
- `app/main.py`: FastAPI backend

## Fairness & Explainability
- **Fairness:** The pipeline checks for demographic parity and equal opportunity across gender, region, and income.
- **Explainability:** Feature importance is computed using model coefficients/importances and SHAP values (with workarounds for XGBoost/SHAP bugs).

## Limitations & Next Steps
- The dataset is synthetic; real-world deployment requires integration with actual farmer and loan data.
- SHAP explainability for XGBoost may require further environment fixes due to upstream bugs.
- Additional behavioral, climate, and productivity features can be engineered for improved accuracy and fairness.

## Authors
- Built by lawrencemungai029-code with GitHub Copilot assistance.

## License
MIT License
AI powered Credit scoring
