from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException

from claim_agent.api.schemas import (
    ClaimRequest,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictionResponse,
    SyntheticRequest,
    SyntheticResponse,
)
from claim_agent.config import METRICS_PATH, MODEL_PATH, genai_uses_live_llm
from claim_agent.genai.explainer import explain_decision
from claim_agent.genai.synthetic import generate_synthetic_claims
from claim_agent.monitoring.metrics import metrics_store
from claim_agent.predict import predict_claim

app = FastAPI(
    title="Claim Approval Agent",
    description="ML claim approval prediction with multi-persona GenAI explanations",
    version="0.1.0",
)


def _genai_mode_label() -> str:
    return "openai" if genai_uses_live_llm() else "mock"


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL_PATH.exists(),
        genai_mode=_genai_mode_label(),
    )


@app.get("/metrics")
def metrics():
    summary = metrics_store.summary()
    if METRICS_PATH.exists():
        summary["training_metrics"] = json.loads(METRICS_PATH.read_text())
    return summary


@app.post("/predict", response_model=PredictionResponse)
def predict(claim: ClaimRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model not trained. Run python -m claim_agent.train")

    start = time.perf_counter()
    result = predict_claim(claim.to_dict())
    latency_ms = (time.perf_counter() - start) * 1000

    metrics_store.log_inference(
        prediction=result["prediction"],
        probability_approved=result["probability_approved"],
        latency_ms=latency_ms,
        genai_mode=_genai_mode_label(),
    )

    return PredictionResponse(**result)


@app.post("/explain", response_model=ExplainResponse)
def explain(body: ExplainRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model not trained. Run python -m claim_agent.train")

    claim_dict = body.claim.to_dict()
    start = time.perf_counter()
    prediction = predict_claim(claim_dict)
    explanations = explain_decision(claim_dict, prediction, body.personas)
    latency_ms = (time.perf_counter() - start) * 1000

    metrics_store.log_inference(
        prediction=prediction["prediction"],
        probability_approved=prediction["probability_approved"],
        latency_ms=latency_ms,
        genai_mode=_genai_mode_label(),
        personas=body.personas,
    )

    return ExplainResponse(
        prediction=PredictionResponse(**prediction),
        explanations=explanations,
        genai_mode=_genai_mode_label(),
    )


@app.post("/synthetic", response_model=SyntheticResponse)
def synthetic(body: SyntheticRequest):
    scenarios = generate_synthetic_claims(count=body.count, focus=body.focus)
    return SyntheticResponse(scenarios=scenarios, genai_mode=_genai_mode_label())


def main():
    import uvicorn

    uvicorn.run(
        "claim_agent.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
