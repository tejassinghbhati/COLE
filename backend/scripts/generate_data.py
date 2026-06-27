"""
Generate a realistic cost-of-living dataset.

The schema mirrors Numbeo's category indices (New York City = 100 baseline) so that
real data can be dropped in later with no code changes. Values are produced with an
explicit, plausible *causal* structure — region wage levels and housing scarcity drive
the category indices, and the overall index is a noisy non-linear blend of them. This
gives the ML model something genuinely learnable (interactions + region effects), so
the SHAP explanations are meaningful rather than echoing a fixed formula.

Run:  python -m scripts.generate_data
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
OUT_CSV = os.path.join(DATA_DIR, "cities.csv")

# region: (wage_level, housing_pressure, base_cost) — relative multipliers.
REGIONS = {
    "North America": dict(wage=1.15, housing=1.10, base=1.05),
    "Western Europe": dict(wage=1.05, housing=1.08, base=1.04),
    "Eastern Europe": dict(wage=0.62, housing=0.70, base=0.68),
    "East Asia": dict(wage=0.95, housing=1.05, base=0.92),
    "South Asia": dict(wage=0.40, housing=0.55, base=0.45),
    "Latin America": dict(wage=0.55, housing=0.62, base=0.58),
    "Middle East": dict(wage=0.85, housing=0.80, base=0.78),
    "Oceania": dict(wage=1.08, housing=1.12, base=1.06),
    "Africa": dict(wage=0.38, housing=0.50, base=0.44),
}

# A pool of real city names per region (purely cosmetic labels for the rows).
CITY_POOL = {
    "North America": ["New York", "San Francisco", "Toronto", "Chicago", "Austin",
                       "Vancouver", "Seattle", "Denver", "Montreal", "Atlanta",
                       "Boston", "Miami", "Phoenix", "Dallas", "Minneapolis"],
    "Western Europe": ["London", "Zurich", "Paris", "Amsterdam", "Munich", "Dublin",
                        "Madrid", "Lisbon", "Vienna", "Milan", "Copenhagen", "Oslo",
                        "Barcelona", "Brussels", "Stockholm"],
    "Eastern Europe": ["Warsaw", "Prague", "Budapest", "Bucharest", "Sofia", "Krakow",
                        "Belgrade", "Zagreb", "Vilnius", "Bratislava", "Tallinn"],
    "East Asia": ["Tokyo", "Seoul", "Singapore", "Hong Kong", "Osaka", "Taipei",
                   "Shanghai", "Beijing", "Shenzhen", "Busan", "Nagoya"],
    "South Asia": ["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Chennai", "Pune",
                    "Colombo", "Dhaka", "Karachi", "Kathmandu"],
    "Latin America": ["Mexico City", "Sao Paulo", "Buenos Aires", "Santiago", "Lima",
                       "Bogota", "Montevideo", "Quito", "Guadalajara", "Medellin"],
    "Middle East": ["Dubai", "Tel Aviv", "Doha", "Abu Dhabi", "Riyadh", "Amman",
                     "Istanbul", "Kuwait City", "Manama"],
    "Oceania": ["Sydney", "Melbourne", "Auckland", "Brisbane", "Perth", "Wellington",
                 "Adelaide", "Canberra"],
    "Africa": ["Cape Town", "Johannesburg", "Nairobi", "Lagos", "Cairo", "Accra",
                "Casablanca", "Tunis", "Kampala"],
}

CATEGORY_COLS = [
    "housing_index", "groceries_index", "transport_index", "utilities_index",
    "restaurant_index", "healthcare_index", "childcare_index",
]


def _clip(x: np.ndarray, lo: float = 5.0, hi: float = 230.0) -> np.ndarray:
    return np.clip(x, lo, hi)


def generate() -> pd.DataFrame:
    rows = []
    for region, cities in CITY_POOL.items():
        r = REGIONS[region]
        for city in cities:
            # Latent drivers ---------------------------------------------------
            # Local wage level (monthly net income, USD) with within-region spread.
            income = RNG.normal(3500 * r["wage"], 700 * r["wage"])
            income = float(max(350, income))

            # Housing pressure: scarcity + desirability. The dominant cost driver.
            housing_pressure = RNG.normal(r["housing"], 0.18)
            density = float(max(300, RNG.normal(6000 * r["housing"], 2500)))

            # tourism intensity nudges restaurants + housing up.
            tourism = float(np.clip(RNG.normal(0.5, 0.25), 0.05, 1.0))

            base = r["base"]

            # Category indices (NYC ~ 100 baseline), each a noisy function of drivers
            housing = 100 * base * housing_pressure * (0.85 + 0.5 * tourism) \
                * RNG.normal(1.0, 0.07)
            groceries = 100 * base * (0.9 + 0.25 * (income / 4000)) * RNG.normal(1.0, 0.06)
            transport = 100 * base * (0.8 + 0.3 * (density / 6000)) * RNG.normal(1.0, 0.08)
            utilities = 100 * base * RNG.normal(1.0, 0.12)
            restaurant = 100 * base * (0.85 + 0.4 * tourism) * RNG.normal(1.0, 0.08)
            healthcare = 100 * base * (0.8 + 0.4 * (income / 4000)) * RNG.normal(1.0, 0.10)
            childcare = 100 * base * (0.7 + 0.6 * (income / 4000)) * RNG.normal(1.0, 0.12)

            cats = dict(
                housing_index=housing, groceries_index=groceries,
                transport_index=transport, utilities_index=utilities,
                restaurant_index=restaurant, healthcare_index=healthcare,
                childcare_index=childcare,
            )
            for k in cats:
                cats[k] = float(_clip(np.array(cats[k])))

            # Overall cost-of-living index: a weighted, slightly non-linear blend.
            # Housing is weighted highest (the real-world reality this app teaches),
            # with a mild interaction: dense + expensive housing compounds.
            w = dict(housing_index=0.34, groceries_index=0.16, transport_index=0.10,
                     utilities_index=0.08, restaurant_index=0.12,
                     healthcare_index=0.10, childcare_index=0.10)
            col = sum(w[k] * cats[k] for k in w)
            col += 0.00018 * cats["housing_index"] * (density / 1000.0)  # interaction
            col *= RNG.normal(1.0, 0.025)  # irreducible noise
            col = float(_clip(np.array(col), 10.0, 220.0))

            rows.append(dict(
                city=city, region=region,
                **cats,
                median_income_usd=round(income, 0),
                population_density=round(density, 0),
                tourism_intensity=round(tourism, 3),
                cost_of_living_index=round(col, 2),
            ))

    df = pd.DataFrame(rows)
    # Affordability: cost burden relative to local wages (higher = harsher).
    # Normalised so ~100 means "index and income are in typical balance".
    df["affordability_burden"] = (
        df["cost_of_living_index"] / (df["median_income_usd"] / 3000.0)
    ).round(2)
    return df


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate()
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} cities -> {OUT_CSV}")
    print(df[["city", "region", "cost_of_living_index",
              "median_income_usd", "affordability_burden"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
