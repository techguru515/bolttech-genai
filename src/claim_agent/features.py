from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from claim_agent.data import FEATURE_COLUMNS

NUMERIC_COLUMNS = [
    "excessFee",
    "rrp",
    "balanceRRP",
    "oldBalanceRRP",
    "turnOnOff",
    "touchScreen",
    "smashed",
    "frontCamera",
    "backCamera",
    "frontOrBackCamera",
    "audio",
    "mic",
    "buttons",
    "connection",
    "charging",
]

CATEGORICAL_COLUMNS = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS]


def coerce_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns are numeric; invalid values become NaN for imputation."""
    out = df[FEATURE_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype(str).replace({"nan": None, "None": None})
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric_cols = NUMERIC_COLUMNS
    categorical_cols = CATEGORICAL_COLUMNS

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def extract_feature_row(claim: dict[str, Any]) -> pd.DataFrame:
    row = {col: claim.get(col) for col in FEATURE_COLUMNS}
    return coerce_feature_types(pd.DataFrame([row]))


def top_contributing_factors(
    claim: dict[str, Any],
    prediction: str,
    probability: float,
    feature_importances: dict[str, float] | None = None,
) -> list[str]:
    """Heuristic factors for GenAI context when SHAP is unavailable."""
    factors: list[str] = []
    claim_type = str(claim.get("claimType", ""))
    coverage = str(claim.get("coverage", ""))
    policy = str(claim.get("policyStatus", ""))
    rrp = float(claim.get("rrp") or 0)
    excess = float(claim.get("excessFee") or 0)

    if policy != "Active":
        factors.append(f"Policy status is '{policy}' (non-active policies increase decline risk).")
    if claim_type == "Theft" and "THEFT" not in coverage.upper():
        factors.append("Theft claim filed but coverage may not include theft protection.")
    if claim_type == "Liquid Damage":
        factors.append("Liquid damage claims require alignment between incident and ADLD coverage.")
    if rrp > 15000:
        factors.append(f"High device RRP ({rrp:.0f}) triggers additional financial review.")
    if excess > 1000:
        factors.append(f"Elevated excess fee ({excess:.0f}) relative to typical claims.")

    damage_flags = [
        "touchScreen",
        "smashed",
        "frontCamera",
        "backCamera",
        "charging",
    ]
    reported = [f for f in damage_flags if str(claim.get(f, "")) in ("1", "1.0", "True", "true")]
    if reported:
        factors.append(f"Reported damage components: {', '.join(reported)}.")

    if feature_importances:
        ranked = sorted(feature_importances.items(), key=lambda x: -abs(x[1]))[:5]
        for name, imp in ranked:
            factors.append(f"Model feature '{name}' (importance {imp:.3f}).")

    if not factors:
        factors.append(
            "Claim profile is broadly consistent with historical approved claims."
            if prediction == "Completed"
            else "Several structured fields resemble historically declined claims."
        )

    if probability < 0.55:
        factors.append(
            f"Model confidence is borderline ({probability:.0%}); manual review recommended."
        )

    return factors[:8]
