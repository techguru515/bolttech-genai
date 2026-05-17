from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "claim_use_case_dataset.xlsx"
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
