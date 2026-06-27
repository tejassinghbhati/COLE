"""
FastAPI app: serves the Cost of Living Explainer API and the static frontend.

Real Numbeo data, re-based to **New Delhi = 100**. Each city carries a personalized,
LLM-or-template overview (see app/overview.py).

Endpoints
---------
GET  /api/health                 service + model status
GET  /api/cities                 list cities (cost + purchasing power)
GET  /api/city?id=City|Country   one city: overview + SHAP explanation + similar cities
GET  /api/compare?a=&b=          driver-by-driver gap between two cities
GET  /api/insights               global cost drivers + regional summary
POST /api/predict                predict + explain a custom cost profile
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model import (COST_CATEGORY_COLS, NUMERIC_COLS, CostModel, FEATURE_LABELS)
from app.overview import city_key, load_cache, template_overview
from app.schemas import Explanation, PredictRequest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.normpath(os.path.join(HERE, "..", "frontend"))
BASELINE = {"city": "New Delhi", "value": 100}

app = FastAPI(title="Cost of Living Explainer", version="2.0.0")

_MODEL: CostModel | None = None
_OVERVIEWS: dict = {}


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
    global _OVERVIEWS
    _OVERVIEWS = load_cache()
    try:
        get_model()
        print(f"Model loaded. {len(_OVERVIEWS)} overviews cached.")
    except HTTPException as e:
        print(f"[warn] {e.detail}")


def _overview_for(row: dict) -> str:
    return _OVERVIEWS.get(city_key(row["city"], row["country"])) or template_overview(row)


# ---- API ------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    loaded, metrics = False, None
    try:
        m = get_model()
        loaded, metrics = True, m.metrics
    except HTTPException:
        pass
    return {"status": "ok", "model_loaded": loaded, "metrics": metrics, "baseline": BASELINE}


@app.get("/api/cities")
def cities() -> dict:
    m = get_model()
    df = m.df.sort_values("total_cost_index", ascending=False)
    items = [
        {
            "id": city_key(r["city"], r["country"]),
            "city": r["city"], "country": r["country"], "region": r["region"],
            "total_cost_index": float(r["total_cost_index"]),
            "predicted_index": float(r["predicted_index"]),
            "purchasing_power_index": float(r["purchasing_power_index"]),
        }
        for _, r in df.iterrows()
    ]
    return {"count": len(items), "baseline": BASELINE, "cities": items,
            "regions": m.regions, "feature_labels": FEATURE_LABELS,
            "cost_categories": COST_CATEGORY_COLS}


def _find(m: CostModel, cid: str):
    """Resolve a 'City|Country' id (or a bare city name) to a dataframe row."""
    if "|" in cid:
        city, country = cid.split("|", 1)
        rows = m.df[(m.df["city"].str.lower() == city.lower())
                    & (m.df["country"].str.lower() == country.lower())]
    else:
        rows = m.df[m.df["city"].str.lower() == cid.lower()]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"City '{cid}' not found")
    return rows.iloc[0]


def _features(row) -> dict:
    feats = {c: float(row[c]) for c in NUMERIC_COLS}
    feats["region"] = row["region"]
    return feats


@app.get("/api/city")
def city_detail(id: str = Query(..., description="City|Country id")) -> dict:
    m = get_model()
    r = _find(m, id)
    explanation = m.predict_and_explain(_features(r))
    return {
        "id": city_key(r["city"], r["country"]),
        "city": r["city"], "country": r["country"], "region": r["region"],
        "overview": _overview_for(r.to_dict()),
        "total_cost_index": float(r["total_cost_index"]),
        "rent_index": float(r["rent_index"]),
        "groceries_index": float(r["groceries_index"]),
        "restaurant_index": float(r["restaurant_index"]),
        "purchasing_power_index": float(r["purchasing_power_index"]),
        "explanation": explanation,
        "similar_cities": m.similar_cities(r["city"], k=5),
    }


@app.get("/api/compare")
def compare(a: str = Query(...), b: str = Query(...)) -> dict:
    m = get_model()
    ra, rb = _find(m, a), _find(m, b)
    ea = m.predict_and_explain(_features(ra))
    eb = m.predict_and_explain(_features(rb))
    gaps = []
    for col in COST_CATEGORY_COLS:
        va, vb = float(ra[col]), float(rb[col])
        gaps.append({"feature": col, "label": FEATURE_LABELS.get(col, col),
                     "a": round(va, 1), "b": round(vb, 1), "gap": round(va - vb, 1)})
    gaps.sort(key=lambda g: abs(g["gap"]), reverse=True)
    return {
        "a": {"city": ra["city"], "country": ra["country"], "index": ea["predicted_index"],
              "purchasing_power": float(ra["purchasing_power_index"])},
        "b": {"city": rb["city"], "country": rb["country"], "index": eb["predicted_index"],
              "purchasing_power": float(rb["purchasing_power_index"])},
        "category_gaps": gaps,
    }


@app.get("/api/insights")
def insights() -> dict:
    m = get_model()
    region_summary = (
        m.df.groupby("region")
        .agg(avg_total=("total_cost_index", "mean"),
             avg_purchasing_power=("purchasing_power_index", "mean"),
             avg_rent=("rent_index", "mean"),
             n=("city", "count"))
        .round(1).reset_index()
        .sort_values("avg_total", ascending=False)
        .to_dict(orient="records")
    )
    return {"global_importance": m.global_importance(), "regions": region_summary,
            "metrics": m.metrics, "baseline": BASELINE}


@app.post("/api/predict", response_model=Explanation)
def predict(req: PredictRequest) -> Explanation:
    m = get_model()
    if req.region not in m.regions:
        raise HTTPException(status_code=400, detail=f"Unknown region. Valid: {m.regions}")
    return Explanation(**m.predict_and_explain(req.model_dump()))


# ---- static frontend ------------------------------------------------------
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
