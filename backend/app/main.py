"""
FastAPI app: serves the Cost of Living Explainer API and the static frontend.

Endpoints
---------
GET  /api/health                 service + model status
GET  /api/cities                 list cities (with cost + affordability)
GET  /api/city/{name}            one city: data + SHAP explanation + similar cities
GET  /api/compare?a=&b=          driver-by-driver gap between two cities
GET  /api/insights               global cost drivers + regional summary
POST /api/predict                predict + explain a custom cost profile
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model import CATEGORY_COLS, NUMERIC_COLS, CostModel, FEATURE_LABELS
from app.schemas import Explanation, PredictRequest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.normpath(os.path.join(HERE, "..", "frontend"))

app = FastAPI(title="Cost of Living Explainer", version="1.0.0")

_MODEL: CostModel | None = None


def get_model() -> CostModel:
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = CostModel.load()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _MODEL


@app.on_event("startup")
def _warm() -> None:
    try:
        get_model()
        print("Model loaded.")
    except HTTPException as e:
        print(f"[warn] {e.detail}")


# ---- API ------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    loaded = False
    metrics = None
    try:
        m = get_model()
        loaded, metrics = True, m.metrics
    except HTTPException:
        pass
    return {"status": "ok", "model_loaded": loaded, "metrics": metrics}


@app.get("/api/cities")
def cities() -> dict:
    m = get_model()
    df = m.df.sort_values("cost_of_living_index", ascending=False)
    items = [
        {
            "city": r["city"],
            "region": r["region"],
            "cost_of_living_index": float(r["cost_of_living_index"]),
            "predicted_index": float(r["predicted_index"]),
            "median_income_usd": float(r["median_income_usd"]),
            "affordability_burden": float(r["affordability_burden"]),
        }
        for _, r in df.iterrows()
    ]
    return {"count": len(items), "cities": items,
            "regions": m.regions, "feature_labels": FEATURE_LABELS}


def _city_features(m: CostModel, name: str) -> dict:
    rows = m.df[m.df["city"].str.lower() == name.lower()]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"City '{name}' not found")
    r = rows.iloc[0]
    feats = {c: float(r[c]) for c in NUMERIC_COLS}
    feats["region"] = r["region"]
    return {"row": r, "features": feats}


@app.get("/api/city/{name}")
def city_detail(name: str) -> dict:
    m = get_model()
    ctx = _city_features(m, name)
    r, feats = ctx["row"], ctx["features"]
    explanation = m.predict_and_explain(feats)
    similar = m.similar_cities(r["city"], k=5)
    return {
        "city": r["city"],
        "region": r["region"],
        "actual_index": float(r["cost_of_living_index"]),
        "median_income_usd": float(r["median_income_usd"]),
        "affordability_burden": float(r["affordability_burden"]),
        "features": {c: float(r[c]) for c in NUMERIC_COLS},
        "explanation": explanation,
        "similar_cities": similar,
    }


@app.get("/api/compare")
def compare(a: str = Query(...), b: str = Query(...)) -> dict:
    m = get_model()
    ca = _city_features(m, a)
    cb = _city_features(m, b)
    ea = m.predict_and_explain(ca["features"])
    eb = m.predict_and_explain(cb["features"])

    # Category-level gap (raw index difference) for intuitive comparison.
    gaps = []
    for col in CATEGORY_COLS:
        va, vb = float(ca["row"][col]), float(cb["row"][col])
        gaps.append({
            "feature": col,
            "label": FEATURE_LABELS.get(col, col),
            "a": round(va, 1),
            "b": round(vb, 1),
            "gap": round(va - vb, 1),
        })
    gaps.sort(key=lambda g: abs(g["gap"]), reverse=True)

    return {
        "a": {"city": ca["row"]["city"], "index": ea["predicted_index"],
              "income": float(ca["row"]["median_income_usd"])},
        "b": {"city": cb["row"]["city"], "index": eb["predicted_index"],
              "income": float(cb["row"]["median_income_usd"])},
        "category_gaps": gaps,
    }


@app.get("/api/insights")
def insights() -> dict:
    m = get_model()
    importance = m.global_importance()
    region_summary = (
        m.df.groupby("region")
        .agg(avg_index=("cost_of_living_index", "mean"),
             avg_income=("median_income_usd", "mean"),
             avg_burden=("affordability_burden", "mean"),
             n=("city", "count"))
        .round(1).reset_index()
        .sort_values("avg_index", ascending=False)
        .to_dict(orient="records")
    )
    return {"global_importance": importance, "regions": region_summary,
            "metrics": m.metrics}


@app.post("/api/predict", response_model=Explanation)
def predict(req: PredictRequest) -> Explanation:
    m = get_model()
    if req.region not in m.regions:
        raise HTTPException(status_code=400,
                            detail=f"Unknown region. Valid: {m.regions}")
    result = m.predict_and_explain(req.model_dump())
    return Explanation(**result)


# ---- static frontend ------------------------------------------------------
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
