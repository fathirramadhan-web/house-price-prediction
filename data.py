"""Shared cleaning + feature engineering for every path (EDA, train, predict, apps).

Raw schema (Kaggle abiyyurasyiq/jakarta-house-price-dataset):
  index, price, district, city, bed_rooms, bath_rooms, carport, land_area, building_area
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "price"
DROP_COLS = ["index"]
NUMERIC = ["bed_rooms", "bath_rooms", "carport", "land_area", "building_area"]
CATEGORICAL = ["district", "city"]
RAW_COLUMNS = ["index", TARGET, *CATEGORICAL, *NUMERIC]

DERIVED_NUM = [
    "total_rooms", "bath_per_bed", "building_to_land_ratio",
    "room_density", "land_per_room", "log_land_area", "log_building_area",
]


def clean(raw: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Coerce types, treat impossible 0 counts as missing, drop bad rows.

    require_target=False lets prediction paths score rows that have no price.
    """
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if TARGET not in df.columns and require_target:
        raise ValueError(f"Missing target column '{TARGET}'. Got: {list(df.columns)}")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    for col in NUMERIC:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        if col in ("bed_rooms", "bath_rooms"):
            df.loc[df[col] <= 0, col] = np.nan
        df.loc[df[col] < 0, col] = np.nan

    for col in CATEGORICAL:
        if col not in df.columns:
            df[col] = "Unknown"
        s = df[col].astype("string").str.strip().str.title()
        df[col] = s.mask(s.isin(["", "?", "Nan", "None", "Unknown", "-"]), "Unknown")

    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(
            df[TARGET].astype(str).str.replace(r"[^0-9.eE-]", "", regex=True), errors="coerce"
        )
        df = df[df[TARGET].notna() & (df[TARGET] > 0)]
    return df.reset_index(drop=True)


def cap_price(df: pd.DataFrame, low: float = 0.001, high: float = 0.999) -> pd.DataFrame:
    out = df.copy()
    lo = out[TARGET].quantile(low)
    hi = out[TARGET].quantile(high)
    out[TARGET] = out[TARGET].clip(lower=lo, upper=hi)
    return out


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row derived features. No target information used."""
    out = df.copy()
    bed = out["bed_rooms"].fillna(0)
    bath = out["bath_rooms"].fillna(0)
    land = out["land_area"].fillna(0)
    bldg = out["building_area"].fillna(0)
    out["total_rooms"] = bed + bath
    out["bath_per_bed"] = bath / bed.clip(lower=1)
    out["building_to_land_ratio"] = bldg / land.clip(lower=1)
    out["room_density"] = (bed + bath) / bldg.clip(lower=1) * 100
    out["land_per_room"] = land / (bed + bath).clip(lower=1)
    out["log_land_area"] = np.log1p(land)
    out["log_building_area"] = np.log1p(bldg)
    return out


def district_target_encode(
    df: pd.DataFrame, target: pd.Series, unseen_value: float | None = None
) -> tuple[pd.Series, dict, float]:
    """Out-of-fold district median price map (for fitting on TRAIN only).

    Returns (encoded_series, mapping_dict, fallback). Fallback is the global
    median; unseen districts map to fallback at inference time. Used by callers
    that prefer ordinal district encoding over the one-hot path in train.py.
    """
    frame = pd.DataFrame({"district": df["district"].values, "y": np.asarray(target)})
    fallback = float(frame["y"].median())
    mapping = frame.groupby("district")["y"].median().to_dict()
    counts = frame.groupby("district")["y"].transform("count")
    sums = frame.groupby("district")["y"].transform("sum")
    encoded = (sums - frame["y"]) / (counts - 1).clip(lower=1)
    encoded = encoded.where(counts > 1, fallback)
    if unseen_value is not None:
        encoded = encoded.where(frame["district"].isin(mapping), unseen_value)
    return encoded.rename("district_median_price"), mapping, fallback


def load(path: str = "data/jakarta_house.csv") -> pd.DataFrame:
    """Read raw CSV, clean and cap target outliers. Training path only."""
    df = pd.read_csv(path)
    return feature_engineer(cap_price(clean(df)))
