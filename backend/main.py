from fastapi import FastAPI
from backend.database import engine, Base
from backend.routes import farmer, climate

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Farmwise Inclusive Credit Scoring API")

# Include routers
app.include_router(farmer.router)
app.include_router(climate.router)
