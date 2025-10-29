import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/credit_model.pkl')

app = FastAPI(title="Agri-Finance Credit Scoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FarmerData(BaseModel):
    data: Dict[str, Any]

# Load model at startup
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")

model = None
@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(farmer: FarmerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        X_input = np.array([list(farmer.data.values())])
        pred = model.predict(X_input)[0]
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_input).max()
        else:
            proba = None
        return {"loan_class": pred, "probability": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
