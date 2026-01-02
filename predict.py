"""Batch prediction from CSV using a saved pipeline.

Usage:
  .venv/bin/python predict.py --file data/sample_input.csv -o outputs/predictions.csv
  .venv/bin/python predict.py --file data/sample_input.csv --model random_forest_tuned
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import data as d

ROOT = Path(__file__).resolve().parent


def load_model(name: str | None = None):
    metrics_file = ROOT / "outputs" / "metrics.json"
    if name is None:
        if not metrics_file.exists():
            raise SystemExit("No metrics.json; run train.py first or pass --model.")
        name = __import__("json").loads(metrics_file.read_text())["winner"] + "_tuned"
    path = ROOT / "outputs" / "models" / f"{name}.joblib"
    if not path.exists():
        raise SystemExit(f"Model not found: {path}")
    return joblib.load(path), name


def predict_frame(df_raw: pd.DataFrame, model) -> pd.DataFrame:
    """Apply identical cleaning/feature engineering as training, then predict."""
    from inference import predict_prices
    cleaned = d.feature_engineer(d.clean(df_raw, require_target=False))
    preds = predict_prices(df_raw, model)
    out = cleaned.copy()
    out["predicted_price"] = preds.round(0)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="CSV with raw house columns")
    p.add_argument("--model", default=None, help="Model filename stem; default = tuned winner")
    p.add_argument("-o", "--output", default="outputs/predictions.csv")
    args = p.parse_args()

    model, name = load_model(args.model)
    df = pd.read_csv(args.file)
    out = predict_frame(df, model)
    dest = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"Model: {name} | rows: {len(out)} | {dest}")
    print(out[[d.TARGET, "predicted_price"]].head().to_string(index=False) if d.TARGET in out
          else out["predicted_price"].head().to_string(index=False))


if __name__ == "__main__":
    main()
