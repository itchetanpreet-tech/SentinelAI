from pathlib import Path
import json
import logging
import sqlite3
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
)

from src.database import DB_PATH, init_db, save_prediction

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
CONFIG_PATH = BASE_DIR / "models" / "model_config.json"

logger = logging.getLogger("sentinelai.api")

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r") as f:
    model_config = json.load(f)

THRESHOLD = float(model_config["threshold"])

app = FastAPI(
    title="SentinelAI",
    description="Predictive Maintenance and Machine Failure Detection",
    version="1.1.0",
)

# Initialize prediction database
init_db()

REQUEST_COUNT = Counter(
    "sentinelai_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "sentinelai_request_duration_seconds",
    "API request latency in seconds",
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = time.perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        if request.url.path != "/metrics":
            REQUEST_COUNT.labels(
                request.method,
                request.url.path,
                str(status),
            ).inc()
            REQUEST_LATENCY.observe(time.perf_counter() - start)


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


class MachineData(BaseModel):
    machine_type: str = Field("M", pattern="^[LMH]$")
    air_temperature: float = Field(..., ge=250, le=350)
    process_temperature: float = Field(..., ge=250, le=400)
    rotational_speed: float = Field(..., gt=0, le=10000)
    torque: float = Field(..., ge=0, le=500)
    tool_wear: float = Field(..., ge=0, le=1000)


@app.get("/health")
def health():
    """Liveness probe: process is up and model is loaded."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "threshold": THRESHOLD,
    }


@app.get("/health/ready")
def health_ready():
    """Readiness probe: DB reachable and predictions table exists."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("SELECT 1 FROM predictions LIMIT 1").fetchone()
        return {"status": "ready"}
    except Exception as exc:
        logger.warning("Readiness check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": str(exc)},
        )


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

    result = {
        "project": "SentinelAI",
        "model": model_config["model_name"],
        "machine_failure": bool(failure),
        "risk_probability": round(probability, 4),
        "risk_percentage": round(probability * 100, 2),
        "threshold": THRESHOLD,
        "status": (
            "HIGH RISK - INSPECT MACHINE"
            if failure
            else "LOWER RISK - CONTINUE MONITORING"
        ),
    }

    # Save successful prediction to SQLite; DB failure must not break inference.
    try:
        save_prediction(data.model_dump(), result)
    except Exception as exc:
        logger.warning("Failed to persist prediction: %s", exc)

    return result
