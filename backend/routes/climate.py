from fastapi import APIRouter
import pandas as pd
import os

router = APIRouter()

CLIMATE_PATH = os.path.join(os.path.dirname(__file__), '../../dataset/synthetic_climate_kenya.csv')
climate_df = pd.read_csv(CLIMATE_PATH)

@router.get("/climate_data")
def get_climate_data(county: str = None, month: int = None):
    df = climate_df.copy()
    if county:
        df = df[df['county'] == county]
    if month:
        df = df[df['month'] == month]
    return df.to_dict(orient='records')
