#!/usr/bin/env python
"""Quick CLI demo: train (if needed), predict, and explain a sample claim."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claim_agent.config import MODEL_PATH
from claim_agent.genai.explainer import explain_decision
from claim_agent.predict import predict_claim
from claim_agent.train import train

SAMPLE = {
    "excessFee": 2509,
    "rrp": 19490,
    "balanceRRP": 19490,
    "oldBalanceRRP": 19490,
    "coverage": "ADLD",
    "policyStatus": "Active",
    "retailerName": "WUAWEI eStore",
    "deviceType": "SMARTPHONES",
    "make": "WUAWEI",
    "channel": "Online Portal",
    "claimType": "Theft",
    "country": "SE",
    "turnOnOff": 0,
    "touchScreen": 0,
    "issueDesc": "Device stolen from gym locker; police report pending.",
}


def main():
    if not MODEL_PATH.exists():
        print("Training model...")
        metrics = train(tune_hyperparameters=False)
        print(json.dumps(metrics, indent=2))

    pred = predict_claim(SAMPLE)
    print("\n=== Prediction ===")
    print(json.dumps(pred, indent=2))

    explanations = explain_decision(
        SAMPLE, pred, personas=["customer", "claims_adjuster", "compliance_officer"]
    )
    print("\n=== Multi-persona explanations ===")
    for persona, text in explanations.items():
        print(f"\n--- {persona} ---\n{text}")


if __name__ == "__main__":
    main()
