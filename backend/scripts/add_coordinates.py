"""
Geocode the dataset's cities -> data/coords.json ("City|Country" -> [lat, lng]),
used to plot the interactive map.

Two reference sources, both cached under data/:
  * worldcities.csv          (SimpleMaps, CC-BY) — clean country->ISO2 mapping
  * geonames_cities15000.txt (GeoNames, CC-BY)   — ~33k cities incl. English
                                                    alternate names (better hit rate)

Matching: Numbeo country -> ISO2 (via SimpleMaps), then look up the city by
normalized name within that country in GeoNames (primary + alternate names),
falling back to SimpleMaps and finally to a globally-unique city name.

Run:  python -m scripts.add_coordinates
"""
from __future__ import annotations

import csv
import json
import os
import re
import unicodedata

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
CITIES_CSV = os.path.join(DATA_DIR, "cities.csv")
WORLD_CSV = os.path.join(DATA_DIR, "worldcities.csv")
GEONAMES = os.path.join(DATA_DIR, "geonames_cities15000.txt")
OUT_JSON = os.path.join(DATA_DIR, "coords.json")

# Numbeo country name -> SimpleMaps country name (only those that differ).
COUNTRY_ALIAS = {
    "united states": "united states of america",
    "north macedonia": "macedonia",
    "hong kong (china)": "hong kong s.a.r.",
    "kosovo (disputed territory)": "kosovo",
}
CITY_ALIAS = {  # normalized Numbeo -> normalized reference spelling
    "bengaluru": "bangalore", "gurugram": "gurgaon",
}
# Final stragglers (spelling/island names the automatic join can't reach).
MANUAL_COORDS = {
    "Lucerne|Switzerland": [47.0502, 8.3093],
    "St. Gallen|Switzerland": [47.4245, 9.3767],
    "Palma de Mallorca|Spain": [39.5696, 2.6502],
    "Quebec City|Canada": [46.8139, -71.2080],
    "Heraklion|Greece": [35.3387, 25.1442],
    "Marrakech|Morocco": [31.6295, -7.9811],
    "Penang|Malaysia": [5.4141, 100.3288],
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"\(.*?\)", "", s)   # drop "(Lakhnau)" notes
    s = re.sub(r",.*$", "", s)       # drop ", NY" / ", CA" suffixes
    return re.sub(r"[^a-z0-9]", "", s.lower())


def country_to_iso2(world: pd.DataFrame) -> dict:
    m = {}
    for _, r in world.iterrows():
        m.setdefault(str(r["country"]).strip().lower(), str(r["iso2"]).strip().upper())
    return m


def world_index(world: pd.DataFrame):
    by_pair, by_city = {}, {}
    for _, r in world.iterrows():
        cc, cn = str(r["country"]).strip().lower(), norm(str(r["city_ascii"]))
        if not cn:
            continue
        pop = float(r["pop"]) if pd.notna(r["pop"]) else 0.0
        rec = (float(r["lat"]), float(r["lng"]), pop)
        if (cc, cn) not in by_pair or pop > by_pair[(cc, cn)][2]:
            by_pair[(cc, cn)] = rec
        by_city.setdefault(cn, []).append(rec)
    return by_pair, by_city


def ensure_geonames() -> None:
    """Download + unzip GeoNames cities15000 if the cached file is missing."""
    if os.path.exists(GEONAMES):
        return
    import io
    import urllib.request
    import zipfile
    url = "https://download.geonames.org/export/dump/cities15000.zip"
    print(f"Fetching GeoNames from {url} ...")
    data = urllib.request.urlopen(url, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open("cities15000.txt") as src, open(GEONAMES, "wb") as dst:
            dst.write(src.read())


def geonames_index():
    """(iso2, citynorm) -> (lat,lng,pop) using name + alternate names."""
    ensure_geonames()
    by_pair, by_city = {}, {}
    with open(GEONAMES, encoding="utf-8") as fh:
        for f in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(f) < 15:
                continue
            iso2 = f[8].strip().upper()
            lat, lng, pop = float(f[4]), float(f[5]), float(f[14] or 0)
            names = {norm(f[2])}                       # asciiname
            names |= {norm(a) for a in f[3].split(",")[:6]}  # a few alt names
            for cn in names:
                if not cn:
                    continue
                key = (iso2, cn)
                if key not in by_pair or pop > by_pair[key][2]:
                    by_pair[key] = (lat, lng, pop)
                by_city.setdefault(cn, []).append((lat, lng, pop))
    return by_pair, by_city


def main() -> None:
    cities = pd.read_csv(CITIES_CSV)
    world = pd.read_csv(WORLD_CSV)
    iso2_of = country_to_iso2(world)
    w_pair, w_city = world_index(world)
    g_pair, g_city = geonames_index()

    coords, misses = {}, []
    for _, row in cities.iterrows():
        city, country = row["city"], row["country"]
        cn = CITY_ALIAS.get(norm(city), norm(city))
        cc = COUNTRY_ALIAS.get(country.strip().lower(), country.strip().lower())
        iso2 = iso2_of.get(cc)

        hit = None
        if iso2:
            hit = g_pair.get((iso2, cn))
        if hit is None:
            hit = w_pair.get((cc, cn))
        if hit is None:                       # globally-unique fallbacks
            for pool in (g_city.get(cn), w_city.get(cn)):
                if pool and len({(round(p[0], 2), round(p[1], 2)) for p in pool}) == 1:
                    hit = pool[0]
                    break
            else:
                if g_city.get(cn):            # last resort: most populous
                    hit = max(g_city[cn], key=lambda p: p[2])
        cid = f"{city}|{country}"
        if hit is None:
            if cid in MANUAL_COORDS:
                coords[cid] = MANUAL_COORDS[cid]
            else:
                misses.append(f"{city}, {country}")
            continue
        coords[cid] = [round(hit[0], 4), round(hit[1], 4)]

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(coords, f, ensure_ascii=False, indent=0)

    n = len(cities)
    print(f"Matched {len(coords)}/{n} cities ({len(coords) * 100 // n}%) -> {OUT_JSON}")
    if misses:
        print(f"Unmatched ({len(misses)}): {', '.join(misses)}")


if __name__ == "__main__":
    main()
