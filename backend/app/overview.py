"""
Per-city natural-language overviews.

Each city gets a short, personalized paragraph explaining its cost-of-living profile
in plain English, grounded in the city's real (re-based, Delhi=100) numbers.

Two backends, same interface:
  * LLM (preferred): Anthropic Claude (Haiku 4.5) writes a fluent 2-3 sentence summary
    from the city's facts. Enabled automatically when ANTHROPIC_API_KEY is set.
  * Deterministic fallback: a data-driven template that always works offline, with no
    API key, so the app is fully functional out of the box.

Overviews are generated in a batch by `scripts/generate_overviews.py` and cached to
`data/overviews.json`. The API reads the cache and falls back to an on-the-fly template
for any city not present.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERVIEWS_JSON = os.path.join(HERE, "data", "overviews.json")

# Claude model used when an API key is available. See the claude-api skill for IDs.
LLM_MODEL = "claude-haiku-4-5-20251001"


def city_key(city: str, country: str) -> str:
    return f"{city}|{country}"


# ---- fact extraction -------------------------------------------------------
def facts(row: dict) -> dict:
    """Turn a city row into the human-meaningful facts both backends rely on."""
    total = float(row["total_cost_index"])
    pp = float(row["purchasing_power_index"])
    cats = {
        "rent": float(row["rent_index"]),
        "groceries": float(row["groceries_index"]),
        "restaurants": float(row["restaurant_index"]),
    }
    top = max(cats, key=cats.get)
    return {
        "city": row["city"], "country": row["country"], "region": row["region"],
        "total": total, "pp": pp, "cats": cats, "top_driver": top,
        "ratio": total / 100.0,           # vs New Delhi
        "pp_ratio": pp / 100.0,           # wages vs New Delhi
    }


def _x(mult: float) -> str:
    """Format a multiplier like 6.63 -> '6.6x' or 0.82 -> '0.82x'."""
    if mult >= 1.0:
        return f"{mult:.1f}×"
    return f"{mult:.2f}×"


# ---- deterministic template ------------------------------------------------
def template_overview(row: dict) -> str:
    f = facts(row)
    city, country = f["city"], f["country"]

    if city == "Delhi" and country == "India":
        return ("New Delhi is the baseline for this analysis: every index is set to 100 "
                "here, so each other city is described relative to it. Rent, groceries and "
                "dining in Delhi anchor the scale at which all 545 cities are compared.")

    r = f["ratio"]
    if r >= 1.15:
        head = f"{city}, {country} is markedly more expensive than New Delhi — overall about {_x(r)} the total cost of living"
    elif r >= 0.9:
        head = f"{city}, {country} sits close to New Delhi on overall cost — roughly {_x(r)} the total"
    else:
        head = f"{city}, {country} is cheaper than New Delhi overall — about {_x(r)} the total cost of living"

    drv = f["top_driver"]
    drv_mult = f["cats"][drv] / 100.0
    driver_word = {"rent": "housing", "groceries": "groceries", "restaurants": "eating out"}[drv]
    mid = (f"its biggest cost pressure is {driver_word}, running about {_x(drv_mult)} "
           f"New Delhi's level")

    pp = f["pp_ratio"]
    if pp >= 1.1:
        tail = (f"Local salaries stretch further too — purchasing power is about {_x(pp)} "
                f"Delhi's, softening the higher prices.")
    elif pp >= 0.9:
        tail = (f"Local purchasing power is similar to Delhi's (about {_x(pp)}), so prices "
                f"and wages are broadly in step.")
    else:
        tail = (f"Local purchasing power is lower than Delhi's (about {_x(pp)}), so those "
                f"prices bite harder on local wages.")

    return f"{head}, in the {f['region']} region. Here {mid}. {tail}"


# ---- LLM backend -----------------------------------------------------------
def _llm_client():
    """Return an Anthropic client if a key is configured, else None."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic()


def llm_overview(row: dict, client) -> str:
    f = facts(row)
    prompt = (
        "Write a concise, neutral, factual 2-3 sentence overview of this city's cost of "
        "living for a data dashboard. All numbers are indices where New Delhi = 100 "
        "(higher = more expensive; purchasing power higher = local salaries stretch "
        "further). Do not invent facts beyond the data. Do not use bullet points.\n\n"
        f"City: {f['city']}, {f['country']} ({f['region']})\n"
        f"Total cost of living index: {f['total']:.0f} (New Delhi = 100)\n"
        f"Rent index: {f['cats']['rent']:.0f}\n"
        f"Groceries index: {f['cats']['groceries']:.0f}\n"
        f"Restaurants index: {f['cats']['restaurants']:.0f}\n"
        f"Local purchasing power index: {f['pp']:.0f} (New Delhi = 100)\n"
    )
    msg = client.messages.create(
        model=LLM_MODEL, max_tokens=200, temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()


# ---- batch generation + cache ----------------------------------------------
def generate_all(df, use_llm: bool = True, limit: int | None = None) -> dict:
    client = _llm_client() if use_llm else None
    mode = "LLM (Claude)" if client else "template (no API key)"
    print(f"Generating overviews via: {mode}")
    out: dict[str, str] = {}
    rows = df.to_dict("records")
    if limit:
        rows = rows[:limit]
    for n, row in enumerate(rows, 1):
        key = city_key(row["city"], row["country"])
        try:
            out[key] = llm_overview(row, client) if client else template_overview(row)
        except Exception as e:  # network/rate errors -> fall back, keep going
            print(f"  [warn] {key}: {e} -> template")
            out[key] = template_overview(row)
        if n % 50 == 0:
            print(f"  {n}/{len(rows)}")
    with open(OVERVIEWS_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=0)
    print(f"Wrote {len(out)} overviews -> {OVERVIEWS_JSON}")
    return out


def load_cache() -> dict:
    if os.path.exists(OVERVIEWS_JSON):
        with open(OVERVIEWS_JSON, encoding="utf-8") as fh:
            return json.load(fh)
    return {}
