from __future__ import annotations

from typing import Any

from claim_agent.config import genai_uses_live_llm
from claim_agent.genai.llm_client import chat_completion
from claim_agent.genai.prompts import PERSONAS, build_explanation_messages


def _mock_explanation(
    persona: str,
    prediction_result: dict[str, Any],
) -> str:
    pred = prediction_result["prediction"]
    prob = prediction_result["probability_approved"]
    factors = prediction_result.get("contributing_factors", [])
    bullets = "\n".join(f"• {f}" for f in factors[:4])

    if persona == "customer":
        if pred == "Completed":
            return (
                f"Our automated review suggests your claim is likely to be approved "
                f"(confidence {prob:.0%}). Key points:\n{bullets}\n\n"
                "Next step: ensure photos and proof of purchase are uploaded. "
                "A claims specialist may still perform a brief verification."
            )
        return (
            f"Our review indicates your claim may be declined (approval likelihood {prob:.0%}). "
            f"Main reasons:\n{bullets}\n\n"
            "Next step: reply with any missing documents (police report for theft, "
            "proof of active policy). You can request a manual review within 14 days."
        )

    if persona == "claims_adjuster":
        return (
            f"ML outcome: {pred} (P(approve)={prob:.2f}). Priority factors:\n{bullets}\n\n"
            "Recommended actions: validate coverage vs claimType, confirm policyStatus, "
            "cross-check damage flags with issueDesc. Escalate if probability 0.45–0.55."
        )

    return (
        f"Decision support output: {pred} at {prob:.0%} approval probability. "
        f"Documented factors:\n{bullets}\n\n"
        "Fairness note: verify decline reasons are policy-grounded and consistently applied. "
        "Flag borderline scores for human adjudication."
    )


def explain_decision(
    claim: dict[str, Any],
    prediction_result: dict[str, Any],
    personas: list[str] | None = None,
) -> dict[str, str]:
    personas = personas or ["customer", "claims_adjuster"]
    explanations: dict[str, str] = {}

    for persona in personas:
        if persona not in PERSONAS and persona.replace(" ", "_") not in PERSONAS:
            raise ValueError(f"Unsupported persona: {persona}")

        if genai_uses_live_llm():
            messages = build_explanation_messages(claim, prediction_result, persona)
            explanations[persona] = chat_completion(messages)
        else:
            explanations[persona] = _mock_explanation(persona, prediction_result)

    return explanations
