"""
Train the cost-of-living model and persist artifacts.

Run:  python -m scripts.train
"""
from __future__ import annotations

import os

import pandas as pd

from app.model import CostModel, DATA_CSV, ARTIFACT


def main() -> None:
    if not os.path.exists(DATA_CSV):
        raise SystemExit(
            f"Dataset not found at {DATA_CSV}. Run `python -m scripts.generate_data` first."
        )
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} cities from {DATA_CSV}")

    model = CostModel.train(df)
    model.save(ARTIFACT)

    print(f"Saved model -> {ARTIFACT}")
    print(f"Hold-out metrics: R^2 = {model.metrics['r2']}, "
          f"MAE = {model.metrics['mae']} index points "
          f"(test n = {model.metrics['n_test']})")
    print("\nGlobal cost drivers (mean |SHAP|):")
    for d in model.global_importance():
        print(f"  {d['label']:<20} {d['importance']:>7.3f}")


if __name__ == "__main__":
    main()
