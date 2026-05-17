from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH_XLSX = PROJECT_ROOT / "claim_use_case_dataset.xlsx"
DATA_PATH_SAMPLE = PROJECT_ROOT / "data" / "claims_sample.csv"


def resolve_data_path() -> Path:
    """Prefer explicit env, then full dataset, then committed CI sample."""
    override = os.getenv("CLAIM_DATA_PATH")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"CLAIM_DATA_PATH not found: {path}")
        return path
    if DATA_PATH_XLSX.exists():
        return DATA_PATH_XLSX
    if DATA_PATH_SAMPLE.exists():
        return DATA_PATH_SAMPLE
    raise FileNotFoundError(
        "No dataset found. Place claim_use_case_dataset.xlsx in the project root "
        "or commit data/claims_sample.csv (see scripts/export_sample_data.py)."
    )


DATA_PATH = resolve_data_path()
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "claim_approval_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

TARGET_COL = "status"
APPROVED_LABEL = "Completed"
DECLINED_LABEL = "Declined"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GENAI_MODE = os.getenv("GENAI_MODE", "auto")  # auto | mock | openai
_default_mlruns = (PROJECT_ROOT / "mlruns").as_uri()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", _default_mlruns)


def genai_uses_live_llm() -> bool:
    if GENAI_MODE == "mock":
        return False
    if GENAI_MODE == "openai":
        return bool(OPENAI_API_KEY)
    return bool(OPENAI_API_KEY)
