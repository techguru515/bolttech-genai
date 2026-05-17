from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from claim_agent.config import MODEL_DIR

LOG_PATH = MODEL_DIR / "inference_log.jsonl"
_MAX_MEMORY = 500


@dataclass
class InferenceRecord:
    timestamp: float
    prediction: str
    probability_approved: float
    latency_ms: float
    genai_mode: str
    personas: list[str] = field(default_factory=list)


class MetricsStore:
    def __init__(self) -> None:
        self._recent: deque[InferenceRecord] = deque(maxlen=_MAX_MEMORY)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def log_inference(
        self,
        prediction: str,
        probability_approved: float,
        latency_ms: float,
        genai_mode: str,
        personas: list[str] | None = None,
    ) -> None:
        record = InferenceRecord(
            timestamp=time.time(),
            prediction=prediction,
            probability_approved=probability_approved,
            latency_ms=latency_ms,
            genai_mode=genai_mode,
            personas=personas or [],
        )
        self._recent.append(record)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def summary(self) -> dict[str, Any]:
        if not self._recent:
            return {"count": 0}

        probs = [r.probability_approved for r in self._recent]
        latencies = [r.latency_ms for r in self._recent]
        declined = sum(1 for r in self._recent if r.prediction == "Declined")

        return {
            "count": len(self._recent),
            "decline_rate": declined / len(self._recent),
            "avg_probability_approved": sum(probs) / len(probs),
            "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1],
            "avg_latency_ms": sum(latencies) / len(latencies),
        }


metrics_store = MetricsStore()
