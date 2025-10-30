import random
from sqlalchemy.orm import Session
from backend.models.farmer import Farmer

def preload_farmers(db: Session, n: int = 50):
    names = [f"Demo Farmer {i+1}" for i in range(n)]
    for i, name in enumerate(names):
        national_id = f"DF{str(i+1).zfill(4)}"
        phone = f"0700{str(100000+i).zfill(6)}"
        location = random.choice(["Nairobi", "Kisumu", "Eldoret", "Mombasa", "Meru"])
        age = random.randint(22, 65)
        farm_size = round(random.uniform(1, 10), 2)
        crop_type = random.choice(["Maize", "Beans", "Tea", "Coffee", "Wheat"])
        soil_type = random.choice(["Loam", "Clay", "Sandy", "Silty"])
        rainfall_pattern = random.choice(["Bimodal", "Unimodal"])
        previous_yield = round(random.uniform(0.5, 3.0), 2)
        credit_score = round(random.uniform(0.3, 0.95), 2)
        stage = random.choice(["registration", "land preparation", "planting", "monitoring", "harvest", "feedback"])
        db_farmer = Farmer(
            name=name,
            national_id=national_id,
            phone=phone,
            location=location,
            age=age,
            farm_size=farm_size,
            crop_type=crop_type,
            soil_type=soil_type,
            rainfall_pattern=rainfall_pattern,
            previous_yield=previous_yield,
            credit_score=credit_score,
            stage=stage
        )
        db.merge(db_farmer)
    db.commit()
    print(f"Preloaded {n} farmers into the database.")
