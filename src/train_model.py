import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "delhi_heat_aqi_satellite.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "heat_aqi_model.pkl")

data = pd.read_csv(DATA_PATH)

features = [
    "day_temp",
    "night_temp",
    "humidity",
    "wind_speed",
    "built_up",
    "green_cover",
    "aqi",
    "ndvi",
    "lst"
]

X = data[features]
y = data["heat_class"]

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X, y)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("✅ Heat prediction model trained & saved")
