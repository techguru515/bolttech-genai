from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from claim_agent.config import APPROVED_LABEL, DECLINED_LABEL, MODEL_PATH
from claim_agent.data import FEATURE_COLUMNS, NARRATIVE_COLUMN
from claim_agent.features import extract_feature_row, top_contributing_factors

_pipeline = None


def load_pipeline(path: Path = MODEL_PATH):
    global _pipeline
    if _pipeline is None:
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run: python -m claim_agent.train"
            )
        _pipeline = joblib.load(path)
    return _pipeline


def predict_claim(claim: dict[str, Any]) -> dict[str, Any]:
    pipe = load_pipeline()
    X = extract_feature_row(claim)
    proba = float(pipe.predict_proba(X)[0, 1])
    pred_label = APPROVED_LABEL if proba >= 0.5 else DECLINED_LABEL

    factors = top_contributing_factors(
        claim,
        prediction=pred_label,
        probability=proba,
    )

    return {
        "prediction": pred_label,
        "approved": pred_label == APPROVED_LABEL,
        "probability_approved": proba,
        "probability_declined": 1.0 - proba,
        "contributing_factors": factors,
        "issue_description": claim.get(NARRATIVE_COLUMN) or claim.get("issueDesc", ""),
    }


def claim_from_dataframe_row(row: pd.Series) -> dict[str, Any]:
    claim = {col: row.get(col) for col in FEATURE_COLUMNS}
    claim[NARRATIVE_COLUMN] = row.get(NARRATIVE_COLUMN, "")
    return claim
