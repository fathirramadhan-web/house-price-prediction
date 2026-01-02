"""Single inference entrypoint for CLI, FastAPI and Streamlit.

Keeps cleaning/feature engineering identical across every path so a single
input row produces the same prediction in all three.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data as d


def feature_columns() -> list[str]:
    return d.NUMERIC + d.DERIVED_NUM + d.CATEGORICAL


def predict_prices(df_raw: pd.DataFrame, model, require_target: bool = False) -> np.ndarray:
    """Apply the same cleaning/FE as training, return prices in Rupiah."""
    cleaned = d.feature_engineer(d.clean(df_raw, require_target=require_target))
    X = cleaned[feature_columns()]
    return np.expm1(model.predict(X)).clip(0)
