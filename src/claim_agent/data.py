from __future__ import annotations

from pathlib import Path

import pandas as pd

from claim_agent.config import APPROVED_LABEL, TARGET_COL, resolve_data_path

# Columns used for modelling (structured features only; narrative used by GenAI)
FEATURE_COLUMNS = [
    "excessFee",
    "rrp",
    "balanceRRP",
    "oldBalanceRRP",
    "coverage",
    "policyStatus",
    "retailerName",
    "deviceType",
    "make",
    "channel",
    "claimType",
    "country",
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
    "other",
]

NARRATIVE_COLUMN = "issueDesc"


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    path = path or resolve_data_path()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset")
    return df


def prepare_labels(df: pd.DataFrame) -> pd.Series:
    """Binary label: 1 = approved (Completed), 0 = declined."""
    valid = df[TARGET_COL].isin([APPROVED_LABEL, "Declined"])
    labels = (df.loc[valid, TARGET_COL] == APPROVED_LABEL).astype(int)
    return labels


def train_test_split_df(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    labels = prepare_labels(df)
    idx = labels.index
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=labels
    )
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()
