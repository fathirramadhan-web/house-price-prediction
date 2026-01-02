"""Train baseline + tuned LinearRegression / RandomForest / GradientBoosting.

All preprocessing (impute, scale, one-hot) lives INSIDE each Pipeline so CV
and inference use identical transformations. Target = log1p(price).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data as d

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODELS = OUT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
SEED = 42

NUM_FEATURES = d.NUMERIC + d.DERIVED_NUM
CAT_FEATURES = d.CATEGORICAL


def preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM_FEATURES), ("cat", cat, CAT_FEATURES)])


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", preprocessor()), ("model", estimator)])


def candidates() -> dict:
    return {
        "linear_regression": make_pipeline(LinearRegression()),
        "random_forest": make_pipeline(RandomForestRegressor(random_state=SEED, n_jobs=-1)),
        "gradient_boosting": make_pipeline(GradientBoostingRegressor(random_state=SEED)),
    }


def param_grids() -> dict:
    return {
        "linear_regression": {},
        "random_forest": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, "sqrt"],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__loss": ["huber"],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__min_samples_leaf": [1, 3, 5],
        },
    }


def scores(y_true_rp: np.ndarray, y_pred_log: np.ndarray) -> dict:
    pred = np.expm1(y_pred_log).clip(0)
    err = y_true_rp - pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true_rp - y_true_rp.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1 - ss_res / ss_tot if ss_tot else 0.0,
    }


def main() -> None:
    df = d.load(str(ROOT / "data" / "jakarta_house.csv"))
    y_log = np.log1p(df[d.TARGET].values)
    X = df[NUM_FEATURES + CAT_FEATURES]

    X_tr, X_te, y_tr_log, y_te_log = train_test_split(
        X, y_log, test_size=0.2, random_state=SEED
    )
    y_te_rp = np.expm1(y_te_log)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {"n_rows": len(df), "train_rows": len(X_tr), "test_rows": len(X_te), "baseline": {}, "tuned": {}}

    for name, pipe in candidates().items():
        pipe.fit(X_tr, y_tr_log)
        s = scores(y_te_rp, pipe.predict(X_te))
        results["baseline"][name] = s
        joblib.dump(pipe, MODELS / f"{name}_baseline.joblib")
        print(f"BASELINE {name:20s} RMSE={s['rmse']:,.0f} MAE={s['mae']:,.0f} R2={s['r2']:.4f}")

    grids = param_grids()
    for name, pipe in candidates().items():
        grid = grids[name]
        if not grid:
            joblib.dump(pipe, MODELS / f"{name}_tuned.joblib")
            results["tuned"][name] = results["baseline"][name]
            results.setdefault("best_params", {})[name] = {}
            print(f"TUNED    {name:20s} (no hyperparameters) reusing baseline")
            continue
        search = RandomizedSearchCV(
            pipe, grid, n_iter=12, scoring="neg_root_mean_squared_error",
            cv=cv, random_state=SEED, n_jobs=-1, verbose=0,
        )
        search.fit(X_tr, y_tr_log)
        best = search.best_estimator_
        s = scores(y_te_rp, best.predict(X_te))
        results["tuned"][name] = s
        results.setdefault("best_params", {})[name] = {
            k: (v.item() if isinstance(v, np.generic) else v) for k, v in search.best_params_.items()
        }
        results.setdefault("cv_log_rmse", {})[name] = float(-search.best_score_)
        joblib.dump(best, MODELS / f"{name}_tuned.joblib")
        print(f"TUNED    {name:20s} RMSE={s['rmse']:,.0f} MAE={s['mae']:,.0f} R2={s['r2']:.4f} "
              f"CV_logRMSE={-search.best_score_:.4f}")

    ranked = sorted(results["tuned"], key=lambda k: results["tuned"][k]["rmse"])
    winner, runner = ranked[0], ranked[1]
    gap = results["tuned"][runner]["rmse"] - results["tuned"][winner]["rmse"]
    results["winner"] = winner
    results["runner_up"] = runner
    results["rmse_gap"] = gap
    results["note"] = ("Perbedaan RMSE kecil (within noise); pilih model paling sederhana" if gap < 0.02 * results["tuned"][winner]["rmse"]
                       else f"{winner} lebih baik berdasarkan test RMSE")
    results["features"] = {"numeric": NUM_FEATURES, "categorical": CAT_FEATURES}

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWINNER: {winner} — {results['note']}")
    print(f"Artifacts -> {MODELS}, metrics -> {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
