#!/usr/bin/env python
"""Export a stratified CSV sample from the full xlsx for CI and offline clones."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claim_agent.config import DATA_PATH_XLSX  # noqa: E402

OUT = ROOT / "data" / "claims_sample.csv"


def main() -> None:
    if not DATA_PATH_XLSX.exists():
        raise SystemExit(f"Full dataset not found: {DATA_PATH_XLSX}")

    from sklearn.model_selection import train_test_split

    df = pd.read_excel(DATA_PATH_XLSX)
    sample, _ = train_test_split(
        df,
        test_size=0.62,
        random_state=42,
        stratify=df["status"],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT, index=False)
    print(f"Wrote {len(sample)} rows to {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
