import pandas as pd
import json
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# Load base data
with open("sample_data.json", "r") as f:
    base_data = json.load(f)

df = pd.DataFrame(base_data)

# Drop name column for modeling
df_model = df.drop(columns=["name"])

# Detect metadata automatically
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_model)

# Train CTGAN model
print("🧠 Training CTGAN synthesizer...")
synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(df_model)

# Generate synthetic data
print("✨ Generating 10,000 synthetic samples...")
synthetic_df = synthesizer.sample(10000)

# Add synthetic names
synthetic_df["name"] = [f"Farmer_{i}" for i in range(1, 10001)]

# Save to JSON
synthetic_df.to_json("synthetic_farmers_10000.json", orient="records", indent=2)
print("✅ Saved synthetic dataset → synthetic_farmers_10000.json")