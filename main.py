import joblib
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# =========================
# LAZY MODEL LOADER (CLOUD SAFE)
# =========================
preprocessor = None
ann_model = None
xgb_model = None

def load_models():
    global preprocessor, ann_model, xgb_model

    if preprocessor is None:
        preprocessor = joblib.load("preprocessor_locked.pkl")

    if ann_model is None:
        ann_model = load_model("ann_locked_model.h5", compile=False)

    if xgb_model is None:
        booster = xgb.Booster()
        booster.load_model("xgb_locked_model.json")
        xgb_model = booster


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Malnutrition Risk Predictor (Hybrid)")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# FEATURE ORDER (LOCKED)
# =========================
FEATURE_COLUMNS = [
    "Age_Months",
    "Weight_kg",
    "Height_cm",
    "MUAC",
    "Hemoglobin_gdl",
    "Sex",
    "Mother_Education",
    "Water_Source",
    "Toilet_Facility",
    "HAZ_Zscore",
    "WAZ_Zscore",
    "WHZ_Zscore",
    "BAZ_Zscore"
]

# =========================
# CATEGORICAL MAPS
# =========================
SEX_MAP = {"Male": 1, "Female": 0}
EDU_MAP = {"No education": 0, "Primary": 1, "Secondary": 2, "Higher": 3}
WATER_MAP = {"Unsafe": 0, "Protected well": 1, "Tap": 2}
TOILET_MAP = {"Open": 0, "Shared": 1, "Private": 2}

# =========================
# Z-SCORE REFERENCE
# =========================
REF = {
    "height_mean": 85.0, "height_std": 6.0,
    "weight_mean": 12.0, "weight_std": 2.0,
    "wh_mean": 0.14, "wh_std": 0.03,
    "bmi_mean": 15.0, "bmi_std": 1.5
}

# =========================
# LOAD MODELS & CONFIG
# =========================
#preprocessor = joblib.load("preprocessor_locked.pkl")

#ann_model = load_model("ann_locked_model.h5", compile=False)

#xgb_model = xgb.Booster()
#xgb_model.load_model("xgb_locked_model.json")

with open("ann_calibration.json") as f:
    ANN_CALIB = json.load(f)

with open("hybrid_config.json") as f:
    HYBRID_CFG = json.load(f)

with open("label_map.json") as f:
    LABEL_MAP = json.load(f)

TEMPERATURE = ANN_CALIB.get("temperature", 1.0)
ANN_WEIGHT = HYBRID_CFG["ann_weight"]
XGB_WEIGHT = HYBRID_CFG["xgb_weight"]

# =========================
# INPUT SCHEMA
# =========================
class PatientInput(BaseModel):
    age: int
    weight: float
    height: float
    muac: float
    hemoglobin: float
    sex: str
    mother_edu: str
    water_source: str
    toilet: str

# =========================
# UTILITIES
# =========================
def zscore(v, m, s):
    return 0.0 if s == 0 else (v - m) / s

def calibrate_probs(probs, temperature):
    probs = probs ** (1 / temperature)
    return probs / probs.sum()

# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"status": "Hybrid backend running successfully"}

@app.post("/predict")
def predict(data: PatientInput):

    load_models()   # 🔑 CRITICAL LINE

    # -------------------------
    # CATEGORICAL → NUMERIC
    # -------------------------
    sex = SEX_MAP[data.sex]
    edu = EDU_MAP[data.mother_edu]
    water = WATER_MAP[data.water_source]
    toilet = TOILET_MAP[data.toilet]

    # -------------------------
    # Z-SCORE COMPUTATION
    # -------------------------
    height_m = data.height / 100
    bmi = data.weight / (height_m ** 2)

    haz = zscore(data.height, REF["height_mean"], REF["height_std"])
    waz = zscore(data.weight, REF["weight_mean"], REF["weight_std"])
    whz = zscore((data.weight / data.height), REF["wh_mean"], REF["wh_std"])
    baz = zscore(bmi, REF["bmi_mean"], REF["bmi_std"])

    # -------------------------
    # CLINICAL PRIORITY RULES
    # -------------------------
    forced_risk = None
    forced_confidence = None

    if data.muac < 11.5 or data.hemoglobin < 9.0:
        forced_risk = "High"
        forced_confidence = 95.0

    elif (11.5 <= data.muac < 12.5) or (data.hemoglobin < 10.5):
        forced_risk = "Medium"
        forced_confidence = 85.0

    # -------------------------
    # FEATURE VECTOR
    # -------------------------
    features = [[
        data.age,
        data.weight,
        data.height,
        data.muac,
        data.hemoglobin,
        sex,
        edu,
        water,
        toilet,
        haz,
        waz,
        whz,
        baz
    ]]

    df = pd.DataFrame(features, columns=FEATURE_COLUMNS)

    X = preprocessor.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()

    # -------------------------
    # ANN + CALIBRATION
    # -------------------------
    ann_raw = ann_model.predict(X)[0]
    ann_probs = calibrate_probs(ann_raw, TEMPERATURE)

    # -------------------------
    # XGBOOST
    # -------------------------
    dmat = xgb.DMatrix(X)
    xgb_probs = xgb_model.predict(dmat)[0]

    # -------------------------
    # HYBRID SOFT VOTING
    # -------------------------
    final_probs = (ANN_WEIGHT * ann_probs) + (XGB_WEIGHT * xgb_probs)

    idx = int(np.argmax(final_probs))
    risk_label = LABEL_MAP[str(idx)]
    confidence = round(float(final_probs[idx]) * 100, 2)

    # APPLY CLINICAL OVERRIDE
    if forced_risk is not None:
        risk_label = forced_risk
        confidence = forced_confidence

    # -------------------------
    # DIET & RECOMMENDATION
    # -------------------------
    if risk_label == "High":
        diet_plan_id = "HIGH_01"
        diet_message = "Immediate medical attention required. Provide therapeutic feeding."

    elif risk_label == "Medium":
        diet_plan_id = "MED_01"
        diet_message = "Increase protein-rich foods and monitor growth monthly."

    elif risk_label == "Low":
        diet_plan_id = "LOW_01"
        diet_message = "Maintain balanced diet and monitor growth."

    else:
        diet_plan_id = "NOR_01"
        diet_message = "Child is healthy. Continue balanced diet."

    return {
        "status": "success",
        "risk": risk_label,
        "confidence": confidence,
        "diet_plan_id": diet_plan_id,
        "diet_message": diet_message
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
