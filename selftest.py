"""Self-check cepat pada data asli, tanpa framework test.

Cek: cleanup, feature engineering, metrics, prediksi, dan konsistensi
CLI vs API vs UI helper (semuanya pakai inference.predict_prices).
"""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import data as d
from inference import predict_prices

csv = ROOT / "data" / "jakarta_house.csv"
assert csv.exists() and csv.stat().st_size > 500_000, "CSV asli tidak ada (568 kB)"
raw = pd.read_csv(csv)
assert raw.shape == (10000, 9), f"shape unexpected: {raw.shape}"
assert list(raw.columns) == ["index", "price", "district", "city", "bed_rooms",
                             "bath_rooms", "carport", "land_area", "building_area"]

cleaned = d.feature_engineer(d.clean(raw))
assert len(cleaned) > 9000, "terlalu banyak baris hilang saat cleaning"
assert not cleaned[["price", "district", "city"]].isna().any().any()
assert "building_to_land_ratio" in cleaned.columns

metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())
assert set(metrics["baseline"]) == {"linear_regression", "random_forest", "gradient_boosting"}
assert metrics["tuned"]["gradient_boosting"]["r2"] > 0.7, "model tidak lebih baik dari baseline"

model = joblib.load(ROOT / "outputs" / "models" / f"{metrics['winner']}_tuned.joblib")
sample = pd.read_csv(ROOT / "data" / "sample_input.csv")
out = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--file",
                      str(ROOT / "data" / "sample_input.csv"), "-o", "outputs/_selftest.csv"],
                     cwd=ROOT, capture_output=True, text=True)
assert out.returncode == 0, out.stderr
preds = pd.read_csv(ROOT / "outputs" / "_selftest.csv")
assert len(preds) == len(sample) and (preds["predicted_price"] > 0).all()

from fastapi_app import HouseFeatures, predict as api_predict

row = sample.iloc[0].to_dict()
cli_val = float(preds.iloc[0]["predicted_price"])
api_val = float(api_predict(HouseFeatures(**row)).predicted_price_idr)
helper_val = float(predict_prices(pd.DataFrame([row]), model)[0])
assert abs(cli_val - api_val) < 1 and abs(cli_val - helper_val) < 1, (cli_val, api_val, helper_val)

print(f"OK — {len(cleaned)} clean rows, winner={metrics['winner']}, "
      f"R²={metrics['tuned'][metrics['winner']]['r2']:.4f}, {len(preds)} prediksi valid; "
      f"CLI/API/helper identik ({cli_val:,.0f})")
