"""
Model layer: an explainable cost-of-living predictor.

Wraps an XGBoost regressor with a SHAP explainer and a nearest-neighbour index.
The public surface is the `CostModel` class, loaded once by the API. Training lives
in `scripts/train.py`, which calls `CostModel.train(...)` and persists artifacts.

Design notes
------------
* The model predicts `cost_of_living_index` from the seven category indices plus
  income, density, tourism and a one-hot region. SHAP then attributes each
  prediction back to those inputs — the heart of the "understanding" feature.
* Similarity uses standardized category features so "similar cities" means similar
  cost *structure*, not just similar overall price.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "models")
DATA_CSV = os.path.join(HERE, "data", "cities.csv")
ARTIFACT = os.path.join(MODEL_DIR, "cost_model.joblib")

CATEGORY_COLS = [
    "housing_index", "groceries_index", "transport_index", "utilities_index",
    "restaurant_index", "healthcare_index", "childcare_index",
]
NUMERIC_COLS = CATEGORY_COLS + [
    "median_income_usd", "population_density", "tourism_intensity",
]
TARGET = "cost_of_living_index"

# Human-friendly labels for the frontend.
FEATURE_LABELS = {
    "housing_index": "Housing & rent",
    "groceries_index": "Groceries",
    "transport_index": "Transport",
    "utilities_index": "Utilities",
    "restaurant_index": "Restaurants",
    "healthcare_index": "Healthcare",
    "childcare_index": "Childcare",
    "median_income_usd": "Local income",
    "population_density": "Population density",
    "tourism_intensity": "Tourism intensity",
}


@dataclass
class CostModel:
    booster: Any                 # xgboost.XGBRegressor
    explainer: Any               # shap.TreeExplainer
    scaler: Any                  # sklearn StandardScaler (similarity space)
    nn: Any                      # sklearn NearestNeighbors
    feature_names: list[str]     # full model input columns (incl. one-hot regions)
    regions: list[str]
    df: pd.DataFrame             # the reference dataset (with predictions cached)
    metrics: dict

    # ---- feature engineering ------------------------------------------------
    @staticmethod
    def _design_matrix(df: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
        """Build the model input matrix: numeric cols + one-hot region."""
        X = df[NUMERIC_COLS].copy()
        for r in regions:
            X[f"region_{r}"] = (df["region"] == r).astype(float)
        return X

    # ---- training -----------------------------------------------------------
    @classmethod
    def train(cls, df: pd.DataFrame) -> "CostModel":
        import shap
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        regions = sorted(df["region"].unique().tolist())
        X = cls._design_matrix(df, regions)
        y = df[TARGET].values
        feature_names = X.columns.tolist()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=7
        )
        booster = xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=7, n_jobs=4,
        )
        booster.fit(X_tr, y_tr)

        pred_te = booster.predict(X_te)
        metrics = {
            "r2": round(float(r2_score(y_te, pred_te)), 4),
            "mae": round(float(mean_absolute_error(y_te, pred_te)), 3),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
        }

        explainer = shap.TreeExplainer(booster)

        # Similarity index on standardized category features.
        scaler = StandardScaler().fit(df[CATEGORY_COLS])
        nn = NearestNeighbors(n_neighbors=min(8, len(df))).fit(
            scaler.transform(df[CATEGORY_COLS])
        )

        # Cache model predictions on the full reference set.
        df = df.copy().reset_index(drop=True)
        df["predicted_index"] = booster.predict(X).round(2)

        return cls(booster, explainer, scaler, nn, feature_names, regions, df, metrics)

    # ---- persistence --------------------------------------------------------
    def save(self, path: str = ARTIFACT) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = ARTIFACT) -> "CostModel":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artifact not found at {path}. Run `python -m scripts.train` first."
            )
        return joblib.load(path)

    # ---- inference ----------------------------------------------------------
    def _row_to_matrix(self, features: dict) -> pd.DataFrame:
        row = {c: float(features[c]) for c in NUMERIC_COLS}
        region = features.get("region", self.regions[0])
        df = pd.DataFrame([row])
        for r in self.regions:
            df[f"region_{r}"] = 1.0 if r == region else 0.0
        return df[self.feature_names]

    def predict_and_explain(self, features: dict) -> dict:
        """Predict the cost index for one feature dict and attribute it via SHAP."""
        X = self._row_to_matrix(features)
        pred = float(self.booster.predict(X)[0])

        shap_vals = self.explainer.shap_values(X)
        base = float(np.array(self.explainer.expected_value).ravel()[0])
        contribs = np.asarray(shap_vals).ravel()

        # Aggregate the one-hot region contributions into a single "region" driver.
        named: dict[str, float] = {}
        region_contrib = 0.0
        for name, val in zip(self.feature_names, contribs):
            if name.startswith("region_"):
                region_contrib += float(val)
            else:
                named[name] = float(val)
        named["region"] = region_contrib

        drivers = [
            {
                "feature": k,
                "label": FEATURE_LABELS.get(k, "Region" if k == "region" else k),
                "value": round(float(features.get(k, 0.0)), 2)
                         if k in NUMERIC_COLS else features.get("region"),
                "contribution": round(v, 3),
            }
            for k, v in named.items()
        ]
        drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)

        return {
            "predicted_index": round(pred, 2),
            "baseline_index": round(base, 2),
            "drivers": drivers,
        }

    def similar_cities(self, city: str, k: int = 5) -> list[dict]:
        rows = self.df.index[self.df["city"] == city].tolist()
        if not rows:
            raise KeyError(city)
        i = rows[0]
        vec = self.scaler.transform(self.df.loc[[i], CATEGORY_COLS])
        dist, idx = self.nn.kneighbors(vec, n_neighbors=min(k + 1, len(self.df)))
        out = []
        for d, j in zip(dist[0], idx[0]):
            if j == i:
                continue
            r = self.df.iloc[j]
            out.append({
                "city": r["city"],
                "region": r["region"],
                "cost_of_living_index": float(r["cost_of_living_index"]),
                "median_income_usd": float(r["median_income_usd"]),
                "distance": round(float(d), 3),
            })
            if len(out) >= k:
                break
        return out

    def global_importance(self) -> list[dict]:
        """Mean |SHAP| across the reference set — what drives cost overall."""
        X = self._design_matrix(self.df, self.regions)[self.feature_names]
        vals = np.abs(self.explainer.shap_values(X))
        mean_abs = vals.mean(axis=0)

        agg: dict[str, float] = {}
        for name, m in zip(self.feature_names, mean_abs):
            key = "region" if name.startswith("region_") else name
            agg[key] = agg.get(key, 0.0) + float(m)
        items = [
            {"feature": k,
             "label": FEATURE_LABELS.get(k, "Region" if k == "region" else k),
             "importance": round(v, 3)}
            for k, v in agg.items()
        ]
        items.sort(key=lambda d: d["importance"], reverse=True)
        return items
