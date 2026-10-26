
from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
CONFIG_PATH = BASE_DIR / "models" / "model_config.json"

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r") as f:
    model_config = json.load(f)

THRESHOLD = float(model_config["threshold"])

app = FastAPI(
    title="SentinelAI",
    description="Predictive Maintenance and Machine Failure Detection",
    version="1.1.0",
)


class MachineData(BaseModel):
    machine_type: str = Field("M", pattern="^[LMH]$")
    air_temperature: float = Field(..., ge=250, le=350)
    process_temperature: float = Field(..., ge=250, le=400)
    rotational_speed: float = Field(..., gt=0, le=10000)
    torque: float = Field(..., ge=0, le=500)
    tool_wear: float = Field(..., ge=0, le=1000)


@app.get("/")
def home():
    return {
        "message": "SentinelAI Predictive Maintenance API is running",
        "model": model_config["model_name"],
        "threshold": THRESHOLD,
        "docs": "/docs",
    }


@app.post("/predict")
def predict(data: MachineData):
    features = pd.DataFrame([{
        "Type": data.machine_type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear,
    }])

    probability = float(model.predict_proba(features)[0][1])
    failure = probability >= THRESHOLD

    return {
        "project": "SentinelAI",
        "model": model_config["model_name"],
        "machine_failure": bool(failure),
        "risk_probability": round(probability, 4),
        "risk_percentage": round(probability * 100, 2),
        "threshold": THRESHOLD,
        "status": "HIGH RISK - INSPECT MACHINE"
        if failure else "LOWER RISK - CONTINUE MONITORING",
    }
