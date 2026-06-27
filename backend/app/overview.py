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


def _level(mult: float) -> str:
    """Describe a multiplier vs the Delhi baseline in words."""
    if mult >= 4:
        return "far above"
    if mult >= 2:
        return "well above"
    if mult >= 1.25:
        return "above"
    if mult >= 0.9:
        return "close to"
    if mult >= 0.6:
        return "below"
    return "well below"


_WORD = {"rent": "housing", "groceries": "groceries", "restaurants": "dining out"}


# ---- deterministic template ------------------------------------------------
def template_overview(row: dict) -> str:
    f = facts(row)
    city, country = f["city"], f["country"]

    cats = f["cats"]

    if city == "Delhi" and country == "India":
        return ("New Delhi is the baseline for this entire analysis: every index — rent, "
                "groceries, dining out and the overall total — is fixed at 100 here, and all "
                "545 other cities are expressed relative to it. That makes Delhi a natural "
                "anchor for South Asia, where living costs sit well below most of the wealthy "
                "world. A city scoring 300, for example, is roughly three times as expensive "
                "as Delhi, while one scoring 60 is around 40% cheaper. Local purchasing power "
                "is likewise pinned at 100, serving as the yardstick against which every other "
                "city's wages are measured — so figures above 100 elsewhere mean salaries "
                "stretch further than they do here, and figures below 100 mean they stretch "
                "less.")

    r = f["ratio"]
    rent_m, groc_m, rest_m = cats["rent"] / 100.0, cats["groceries"] / 100.0, cats["restaurants"] / 100.0

    # 1. Headline — overall position and region.
    if r >= 2.5:
        head = (f"{city}, {country} is one of the more expensive cities in the dataset: its "
                f"total cost of living is about {_x(r)} that of New Delhi")
    elif r >= 1.15:
        head = (f"{city}, {country} is markedly more expensive than New Delhi, with a total "
                f"cost of living roughly {_x(r)} Delhi's")
    elif r >= 0.9:
        head = (f"{city}, {country} sits close to New Delhi on overall cost, at about {_x(r)} "
                f"the total")
    else:
        head = (f"{city}, {country} is cheaper than New Delhi overall — around {_x(r)} the "
                f"total cost of living")
    head += f", which places it in the {f['region']} region."

    # 2. Housing — usually the dominant line item.
    house = (f"Housing is {_level(rent_m)} the Delhi baseline, with rents running about "
             f"{_x(rent_m)} those in New Delhi.")

    # 3. Everyday spending.
    daily = (f"Day-to-day costs follow a similar pattern: groceries are about {_x(groc_m)} and "
             f"eating out about {_x(rest_m)} Delhi's level.")

    # 4. Which category bites most / least.
    drv, low = f["top_driver"], min(cats, key=cats.get)
    driver = (f"The single biggest pressure on a household budget here is {_WORD[drv]}, while "
              f"{_WORD[low]} is comparatively the gentlest of the three categories.")

    # 5. Affordability — wages vs prices.
    pp = f["pp_ratio"]
    if pp >= 1.1:
        power = (f"Crucially, local salaries stretch further than in Delhi — purchasing power "
                 f"is about {_x(pp)} the baseline — so these higher prices are partly offset by "
                 f"higher pay.")
    elif pp >= 0.9:
        power = (f"Local purchasing power is broadly in line with Delhi's (about {_x(pp)}), so "
                 f"wages and prices move roughly in step.")
    else:
        power = (f"Local purchasing power, however, is lower than Delhi's (about {_x(pp)}), "
                 f"which means those prices bite harder on what residents actually earn.")

    return f"{head} {house} {daily} {driver} {power}"


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
        "Write a neutral, factual overview of this city's cost of living for a data "
        "dashboard — a single paragraph of about 5-6 sentences. Cover: how the overall cost "
        "compares to New Delhi; housing/rent specifically; groceries and dining out; which "
        "category is the biggest pressure and which is the gentlest; and what local "
        "purchasing power implies about whether wages keep up with prices. All numbers are "
        "indices where New Delhi = 100 (higher = more expensive; purchasing power higher = "
        "local salaries stretch further). Do not invent facts beyond the data given. Do not "
        "use bullet points or headings — flowing prose only.\n\n"
        f"City: {f['city']}, {f['country']} ({f['region']})\n"
        f"Total cost of living index: {f['total']:.0f} (New Delhi = 100)\n"
        f"Rent index: {f['cats']['rent']:.0f}\n"
        f"Groceries index: {f['cats']['groceries']:.0f}\n"
        f"Restaurants index: {f['cats']['restaurants']:.0f}\n"
        f"Local purchasing power index: {f['pp']:.0f} (New Delhi = 100)\n"
    )
    msg = client.messages.create(
        model=LLM_MODEL, max_tokens=500, temperature=0.5,
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
