import pandas as pd
import numpy as np

np.random.seed(42)
counties = ["Kiambu", "Machakos", "Nakuru", "Bungoma", "Kericho", "Embu", "Meru", "Nyeri"]
synthetic_data = pd.DataFrame({
    "county": np.repeat(counties, 12),
    "month": list(range(1,13)) * len(counties),
    "rainfall_mm": np.random.normal(120, 40, len(counties)*12).clip(0),
    "temperature_c": np.random.normal(24, 3, len(counties)*12),
    "humidity": np.random.uniform(45, 85, len(counties)*12),
    "soil_moisture_index": np.random.uniform(0.2, 0.8, len(counties)*12)
})
synthetic_data.to_csv("/workspaces/Farmwise/dataset/synthetic_climate_kenya.csv", index=False)
print("Synthetic climate data generated at dataset/synthetic_climate_kenya.csv")
