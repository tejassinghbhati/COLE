"""
Generate per-city natural-language overviews and cache them to data/overviews.json.

Uses Anthropic Claude when ANTHROPIC_API_KEY is set (fluent, LLM-written), otherwise a
deterministic data-driven template (always works offline). Re-run after adding a key to
upgrade every overview to LLM-written text — no other changes needed.

Run:  python -m scripts.generate_overviews
      python -m scripts.generate_overviews --no-llm     (force templates)
      python -m scripts.generate_overviews --limit 20   (quick test)
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from app.model import DATA_CSV
from app.overview import generate_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="force the template backend")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cities")
    args = ap.parse_args()

    if not os.path.exists(DATA_CSV):
        raise SystemExit("Dataset not found. Run `python -m scripts.build_dataset` first.")
    df = pd.read_csv(DATA_CSV)
    generate_all(df, use_llm=not args.no_llm, limit=args.limit)


if __name__ == "__main__":
    main()
