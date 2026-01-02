"""FastAPI app: single and batch house-price prediction.

Run: .venv/bin/python fastapi_app.py  (serves on 0.0.0.0:8000)
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

import data as d

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "outputs" / "models"
METRICS = ROOT / "outputs" / "metrics.json"

app = FastAPI(title="Jakarta House Price Prediction API",
              description="Predict house prices with Linear Regression, Random Forest, Gradient Boosting.",
              version="1.0.0")

_loaded: dict = {"model": None, "name": None, "metrics": None}


def default_model_name() -> str:
    if METRICS.exists():
        return json.loads(METRICS.read_text())["winner"] + "_tuned"
    return "random_forest_tuned"


def ensure_model(model_name: str | None = None):
    if model_name is None:
        if _loaded["model"] is not None:
            return _loaded["model"], _loaded["name"]
        model_name = default_model_name()
    path = MODELS / f"{model_name}.joblib"
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail=f"Model '{model_name}' tidak ada. Jalankan train.py lalu restart.")
    return joblib.load(path), model_name


class HouseFeatures(BaseModel):
    district: str = Field(..., examples=["Kelapa Gading"])
    city: str = Field(..., examples=["Jakarta Utara"])
    bed_rooms: int = Field(..., ge=0, le=30, description="Kamar tidur; 0 diperlakukan sebagai missing")
    bath_rooms: int = Field(..., ge=0, le=30)
    carport: int = Field(..., ge=0, le=30)
    land_area: float = Field(..., gt=0, le=1_000_000, description="Luas tanah m²")
    building_area: float = Field(..., gt=0, le=1_000_000, description="Luas bangunan m²")


class BatchRequest(BaseModel):
    rows: list[HouseFeatures]
    model: str | None = None


class PredictionOut(BaseModel):
    model: str
    predicted_price_idr: float


@app.post("/predict", response_model=PredictionOut)
def predict(row: HouseFeatures, model: str | None = None):
    mdl, name = ensure_model(model)
    df = pd.DataFrame([row.model_dump()])
    from inference import predict_prices
    price = float(predict_prices(df, mdl)[0])
    return PredictionOut(model=name, predicted_price_idr=float(round(price)))


@app.post("/batch/predict")
def batch(req: BatchRequest):
    if not req.rows:
        raise HTTPException(status_code=422, detail="rows kosong")
    mdl, name = ensure_model(req.model)
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    from inference import predict_prices
    preds = predict_prices(df, mdl)
    return {"model": name, "n": len(preds),
            "predictions": [round(float(p)) for p in preds]}


@app.get("/models")
def list_models():
    if not MODELS.exists():
        return {"models": []}
    files = sorted(p.stem for p in MODELS.glob("*.joblib"))
    winner = None
    if METRICS.exists():
        m = json.loads(METRICS.read_text())
        winner = m.get("winner")
    return {"models": files, "default": winner}


@app.get("/metrics")
def get_metrics():
    if not METRICS.exists():
        raise HTTPException(status_code=404, detail="metrics.json belum ada. Jalankan train.py.")
    return json.loads(METRICS.read_text())


@app.get("/health")
def health():
    ready = (MODELS / f"{default_model_name()}.joblib").exists()
    return {"status": "ok" if ready else "no_model", "ready": ready}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
