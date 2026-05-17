from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claim_agent.api.main import app
from claim_agent.config import MODEL_PATH
from claim_agent.train import train

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def trained_model():
    if not MODEL_PATH.exists():
        train(tune_hyperparameters=False)
    yield


client = TestClient(app)

SAMPLE_CLAIM = {
    "excessFee": 619,
    "rrp": 15490,
    "balanceRRP": 15490,
    "oldBalanceRRP": 15490,
    "coverage": "ADLD",
    "policyStatus": "Active",
    "retailerName": "WUAWEI eStore",
    "deviceType": "SMARTPHONES",
    "make": "WUAWEI",
    "channel": "Online Portal",
    "claimType": "Accidental Damage",
    "country": "SE",
    "turnOnOff": 1,
    "touchScreen": 1,
    "smashed": 0,
    "frontCamera": 1,
    "backCamera": 1,
    "frontOrBackCamera": 0,
    "audio": 1,
    "mic": 1,
    "buttons": 1,
    "connection": 1,
    "charging": 0,
    "other": 0,
    "issueDesc": "Phone slipped from pocket; screen cracked on pavement.",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict():
    r = client.post("/predict", json=SAMPLE_CLAIM)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ("Completed", "Declined")
    assert 0 <= body["probability_approved"] <= 1


def test_explain_multi_persona():
    r = client.post(
        "/explain",
        json={"claim": SAMPLE_CLAIM, "personas": ["customer", "claims_adjuster"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert "customer" in body["explanations"]
    assert "claims_adjuster" in body["explanations"]
    assert len(body["explanations"]["customer"]) > 50


def test_synthetic():
    r = client.post("/synthetic", json={"count": 2, "focus": "denial_patterns"})
    assert r.status_code == 200
    scenarios = r.json()["scenarios"]
    assert len(scenarios) == 2
    assert "issueDesc" in scenarios[0]
