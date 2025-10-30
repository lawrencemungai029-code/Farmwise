from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.models.farmer import Farmer
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import random
import os

router = APIRouter()

# Load climate data once
CLIMATE_PATH = os.path.join(os.path.dirname(__file__), '../../dataset/synthetic_climate_kenya.csv')
climate_df = pd.read_csv(CLIMATE_PATH)

# Pydantic models
class FarmerRegister(BaseModel):
    name: str
    national_id: str
    phone: str
    location: str
    age: int
    farm_size: float
    crop_type: str
    soil_type: str
    rainfall_pattern: str
    previous_yield: Optional[float] = None
    gender: Optional[str] = None
    mpesa_monthly_transactions: Optional[int] = 10
    utility_payment_timeliness: Optional[float] = 1.0
    dependents_count: Optional[int] = 1
    loan_amount_requested: Optional[float] = 10000
    past_crop_performance_index: Optional[float] = 0.7
    soil_quality_index: Optional[float] = 0.7
    historical_drought_risk: Optional[float] = 0.1

class StageUpdate(BaseModel):
    national_id: str
    stage: str

# Dependency

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper: Credit scoring engine

def score_farmer(data: FarmerRegister) -> dict:
    scores = []
    explain_steps = []
    for _ in range(20):
        # Perturb inputs
        perturbed = data.copy()
        perturbed.farm_size += random.uniform(-0.1, 0.1)
        perturbed.previous_yield = (perturbed.previous_yield or 1.0) * random.uniform(0.95, 1.05)
        perturbed.soil_quality_index = (perturbed.soil_quality_index or 0.7) * random.uniform(0.95, 1.05)
        perturbed.past_crop_performance_index = (perturbed.past_crop_performance_index or 0.7) * random.uniform(0.95, 1.05)
        # Simple scoring logic (replace with ML model in production)
        base_score = 0.5
        base_score += 0.1 * (perturbed.farm_size / 5)
        base_score += 0.1 * (perturbed.previous_yield / 10)
        base_score += 0.1 * (perturbed.soil_quality_index)
        base_score += 0.1 * (perturbed.past_crop_performance_index)
        # Behavioral
        payment_consistency = perturbed.utility_payment_timeliness * (perturbed.mpesa_monthly_transactions / max(perturbed.dependents_count,1))
        base_score += 0.05 * min(payment_consistency, 2)
        # Resilience
        resilience_score = perturbed.soil_quality_index * (1 - perturbed.historical_drought_risk)
        base_score += 0.05 * resilience_score
        # Gender fairness
        if getattr(perturbed, 'gender', None) == 'Female':
            base_score *= 1.1
            explain_steps.append('Empowerment: Female gender incentive applied (score x1.1)')
        # Clamp
        base_score = min(max(base_score, 0), 1)
        scores.append(base_score)
    avg_score = float(np.mean(scores))
    # Risk category
    if avg_score > 0.8:
        risk = 'Low'
        support = 'Direct support.'
    elif avg_score > 0.5:
        risk = 'Medium'
        support = 'Gradual trust-building and data monitoring.'
    else:
        risk = 'High'
        support = 'Data review and mentorship.'
    # What-if simulation
    predicted_yield_gain = data.loan_amount_requested * ((data.soil_quality_index or 0.7) + (data.past_crop_performance_index or 0.7)) / 2
    # Explainability
    explanation = f"Based on your {'consistent M-Pesa activity and ' if data.mpesa_monthly_transactions > 8 else ''}healthy soil quality, your farm shows good repayment potential. "
    if risk == 'Medium':
        explanation += "You could improve your loan class by diversifying crops or joining a cooperative. "
    if explain_steps:
        explanation += ' '.join(explain_steps)
    explanation += f" Predicted yield gain if loan approved: {predicted_yield_gain:.2f} units."
    return {
        'credit_score': round(avg_score, 2),
        'risk_category': risk,
        'support_message': support,
        'explanation': explanation,
        'predicted_yield_gain': round(predicted_yield_gain, 2)
    }

# Register endpoint
@router.post("/register")
def register_farmer(farmer: FarmerRegister, db: Session = Depends(get_db)):
    # Score
    scoring = score_farmer(farmer)
    # Save
    db_farmer = Farmer(
        name=farmer.name,
        national_id=farmer.national_id,
        phone=farmer.phone,
        location=farmer.location,
        age=farmer.age,
        farm_size=farmer.farm_size,
        crop_type=farmer.crop_type,
        soil_type=farmer.soil_type,
        rainfall_pattern=farmer.rainfall_pattern,
        previous_yield=farmer.previous_yield,
        credit_score=scoring['credit_score'],
        stage='registration'
    )
    db.add(db_farmer)
    db.commit()
    db.refresh(db_farmer)
    return scoring

# Score endpoint
@router.get("/score")
def get_score(national_id: str, db: Session = Depends(get_db)):
    farmer = db.query(Farmer).filter(Farmer.national_id == national_id).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")
    return {"credit_score": farmer.credit_score}

# Journey endpoint
@router.get("/journey")
def get_journey(national_id: str, db: Session = Depends(get_db)):
    farmer = db.query(Farmer).filter(Farmer.national_id == national_id).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")
    stages = [
        "Registration", "Land Preparation", "Planting", "Monitoring", "Harvest", "Feedback"
    ]
    current_stage = farmer.stage or "Registration"
    next_stage = stages[min(stages.index(current_stage) + 1, len(stages)-1)] if current_stage in stages else "Land Preparation"
    guidance = f"Stage: {current_stage}. Next Action: {next_stage}. Tip: Based on recent rainfall patterns, maize planting is ideal this month."
    return {
        "current_stage": current_stage,
        "next_stage": next_stage,
        "guidance": guidance
    }

# Update stage endpoint
@router.post("/update_stage")
def update_stage(update: StageUpdate, db: Session = Depends(get_db)):
    farmer = db.query(Farmer).filter(Farmer.national_id == update.national_id).first()
    if not farmer:
        raise HTTPException(status_code=404, detail="Farmer not found")
    farmer.stage = update.stage
    db.commit()
    return {"message": f"Stage updated to {update.stage}"}
