"""
Build the cost-of-living dataset from REAL Numbeo data.

Source: Numbeo "Cost of Living Index by City" (current rankings), the most widely
cited cross-city cost-of-living reference. All Numbeo indices are published relative
to **New York City = 100**. This script re-bases every index to **New Delhi = 100**,
so the whole app reads "how does this city compare to New Delhi?".

Data flow:
  data/numbeo_raw.html  (cached snapshot; fetched live if absent)
        │  parse the rankings table (server-rendered <tr> rows)
        ▼
  city, country, region, rent/groceries/restaurant/cost/total/purchasing-power
        │  re-base each column so New Delhi = 100
        ▼
  data/cities.csv

Run:  python -m scripts.build_dataset
"""
from __future__ import annotations

import html as ihtml
import os
import re
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
RAW_HTML = os.path.join(DATA_DIR, "numbeo_raw.html")
OUT_CSV = os.path.join(DATA_DIR, "cities.csv")
SOURCE_URL = "https://www.numbeo.com/cost-of-living/rankings_current.jsp"
BASELINE_CITY = "Delhi"  # New Delhi — the re-basing anchor (index = 100)

# Country -> region. Covers every country present in the Numbeo table.
REGION_MAP = {
    # South Asia
    "India": "South Asia", "Pakistan": "South Asia", "Bangladesh": "South Asia",
    "Nepal": "South Asia", "Sri Lanka": "South Asia",
    # East Asia
    "China": "East Asia", "Hong Kong (China)": "East Asia", "Japan": "East Asia",
    "South Korea": "East Asia", "Taiwan": "East Asia", "Mongolia": "East Asia",
    # Southeast Asia
    "Indonesia": "Southeast Asia", "Malaysia": "Southeast Asia", "Philippines": "Southeast Asia",
    "Singapore": "Southeast Asia", "Thailand": "Southeast Asia", "Vietnam": "Southeast Asia",
    "Cambodia": "Southeast Asia",
    # Central Asia & Caucasus
    "Kazakhstan": "Central Asia", "Kyrgyzstan": "Central Asia", "Tajikistan": "Central Asia",
    "Uzbekistan": "Central Asia", "Azerbaijan": "Central Asia", "Armenia": "Central Asia",
    "Georgia": "Central Asia",
    # Middle East
    "Bahrain": "Middle East", "Iran": "Middle East", "Iraq": "Middle East",
    "Israel": "Middle East", "Jordan": "Middle East", "Kuwait": "Middle East",
    "Lebanon": "Middle East", "Oman": "Middle East", "Qatar": "Middle East",
    "Saudi Arabia": "Middle East", "Syria": "Middle East",
    "United Arab Emirates": "Middle East", "Turkey": "Middle East",
    # Europe
    "Austria": "Western Europe", "Belgium": "Western Europe", "France": "Western Europe",
    "Germany": "Western Europe", "Ireland": "Western Europe", "Luxembourg": "Western Europe",
    "Netherlands": "Western Europe", "Switzerland": "Western Europe",
    "United Kingdom": "Western Europe",
    "Denmark": "Northern Europe", "Finland": "Northern Europe", "Iceland": "Northern Europe",
    "Norway": "Northern Europe", "Sweden": "Northern Europe", "Estonia": "Northern Europe",
    "Latvia": "Northern Europe", "Lithuania": "Northern Europe",
    "Italy": "Southern Europe", "Spain": "Southern Europe", "Portugal": "Southern Europe",
    "Greece": "Southern Europe", "Malta": "Southern Europe", "Cyprus": "Southern Europe",
    "Croatia": "Southern Europe", "Slovenia": "Southern Europe", "Albania": "Southern Europe",
    "Bosnia And Herzegovina": "Southern Europe", "Montenegro": "Southern Europe",
    "North Macedonia": "Southern Europe", "Kosovo (Disputed Territory)": "Southern Europe",
    "Serbia": "Southern Europe",
    "Bulgaria": "Eastern Europe", "Czech Republic": "Eastern Europe", "Hungary": "Eastern Europe",
    "Poland": "Eastern Europe", "Romania": "Eastern Europe", "Slovakia": "Eastern Europe",
    "Ukraine": "Eastern Europe", "Belarus": "Eastern Europe", "Moldova": "Eastern Europe",
    "Russia": "Eastern Europe",
    # Americas
    "Canada": "North America", "United States": "North America",
    "Argentina": "Latin America", "Bolivia": "Latin America", "Brazil": "Latin America",
    "Chile": "Latin America", "Colombia": "Latin America", "Costa Rica": "Latin America",
    "Dominican Republic": "Latin America", "Ecuador": "Latin America", "El Salvador": "Latin America",
    "Jamaica": "Latin America", "Mexico": "Latin America", "Panama": "Latin America",
    "Paraguay": "Latin America", "Peru": "Latin America", "Puerto Rico": "Latin America",
    "Uruguay": "Latin America", "Venezuela": "Latin America", "Cayman Islands": "Latin America",
    # Africa
    "Algeria": "Africa", "Egypt": "Africa", "Ethiopia": "Africa", "Ghana": "Africa",
    "Ivory Coast": "Africa", "Kenya": "Africa", "Morocco": "Africa", "Namibia": "Africa",
    "Nigeria": "Africa", "Rwanda": "Africa", "South Africa": "Africa", "Tanzania": "Africa",
    "Tunisia": "Africa", "Uganda": "Africa", "Zimbabwe": "Africa",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania",
}

# Numbeo rankings table column order (after the rank cell + city cell).
# 0: Cost of Living Index (excl. rent)   1: Rent Index
# 2: Cost of Living Plus Rent Index      3: Groceries Index
# 4: Restaurant Price Index              5: Local Purchasing Power Index


def _strip(s: str) -> str:
    return ihtml.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def fetch_html() -> str:
    if os.path.exists(RAW_HTML):
        return open(RAW_HTML, encoding="utf-8", errors="replace").read()
    print(f"Fetching live from {SOURCE_URL} ...")
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    doc = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RAW_HTML, "w", encoding="utf-8") as f:
        f.write(doc)
    return doc


def parse(doc: str) -> pd.DataFrame:
    recs = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", doc, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(tds) < 8:
            continue
        city_country = _strip(tds[1])
        if "," not in city_country:
            continue
        try:
            nums = [float(_strip(tds[k]).replace(",", "")) for k in range(2, 8)]
        except ValueError:
            continue
        city, country = (s.strip() for s in city_country.rsplit(",", 1))
        recs.append({
            "city": city,
            "country": country,
            "region": REGION_MAP.get(country, "Other"),
            "cost_of_living_index": nums[0],   # excl. rent
            "rent_index": nums[1],
            "total_cost_index": nums[2],       # cost of living + rent (the target)
            "groceries_index": nums[3],
            "restaurant_index": nums[4],
            "purchasing_power_index": nums[5],
        })
    return pd.DataFrame(recs)


REBASE_COLS = [
    "cost_of_living_index", "rent_index", "total_cost_index",
    "groceries_index", "restaurant_index", "purchasing_power_index",
]


def rebase_to_delhi(df: pd.DataFrame) -> pd.DataFrame:
    base = df.loc[df["city"] == BASELINE_CITY]
    if base.empty:
        raise SystemExit(f"Baseline city '{BASELINE_CITY}' not found in source data.")
    base = base.iloc[0]
    for col in REBASE_COLS:
        b = float(base[col])
        if b <= 0:
            continue
        df[col] = (df[col] / b * 100.0).round(2)
    return df


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = parse(fetch_html())
    # Drop rows with implausible zeros and de-duplicate city names within a country.
    df = df[(df["total_cost_index"] > 0) & (df["purchasing_power_index"] > 0)]
    df = df.drop_duplicates(subset=["city", "country"]).reset_index(drop=True)
    df = rebase_to_delhi(df)
    df = df.sort_values("total_cost_index", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"Wrote {len(df)} real cities (New Delhi = 100 baseline) -> {OUT_CSV}")
    print(f"Regions: {df['region'].nunique()} | Countries: {df['country'].nunique()}")
    cols = ["city", "country", "region", "total_cost_index", "rent_index",
            "purchasing_power_index"]
    print("\nMost expensive (vs New Delhi):")
    print(df[cols].head(6).to_string(index=False))
    print(f"\nNew Delhi baseline row:")
    print(df.loc[df['city'] == 'Delhi', cols].to_string(index=False))


if __name__ == "__main__":
    main()
