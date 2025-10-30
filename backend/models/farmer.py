from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, text
from ..database import Base

class Farmer(Base):
    __tablename__ = "farmers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    national_id = Column(String, unique=True, nullable=False)
    phone = Column(String, unique=True, nullable=False)
    location = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    farm_size = Column(Float, nullable=False)
    crop_type = Column(String, nullable=False)
    soil_type = Column(String, nullable=False)
    rainfall_pattern = Column(String, nullable=False)
    previous_yield = Column(Float, nullable=True)
    credit_score = Column(Float, nullable=True)
    stage = Column(String, default='registration')
    created_at = Column(TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'))
