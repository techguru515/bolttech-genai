from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claim_agent.config import PROMPTS_DIR

PERSONAS = {
    "customer": (
        "Policyholder who filed the claim. Use empathetic, non-technical language. "
        "Explain what the decision means for them and what they should do next."
    ),
    "claims_adjuster": (
        "Internal claims adjuster. Use operational language. "
        "Reference policy/coverage checks, fraud/red-flag cues, and recommended review actions."
    ),
    "compliance_officer": (
        "Compliance and fairness reviewer. Emphasize transparency, consistency with policy, "
        "and whether manual review is warranted for edge cases."
    ),
}


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def build_explanation_messages(
    claim: dict[str, Any],
    prediction_result: dict[str, Any],
    persona: str,
) -> list[dict[str, str]]:
    persona_key = persona.lower().replace(" ", "_")
    if persona_key not in PERSONAS:
        raise ValueError(f"Unknown persona '{persona}'. Choose from: {list(PERSONAS)}")

    factors = prediction_result.get("contributing_factors", [])
    factors_text = "\n".join(f"- {f}" for f in factors)

    user_template = load_prompt("explanation_user.txt")
    user_content = user_template.format(
        claim_json=json.dumps(claim, default=str),
        prediction=prediction_result["prediction"],
        probability_approved=prediction_result["probability_approved"],
        factors=factors_text,
        issue_description=prediction_result.get("issue_description", ""),
        persona=persona_key,
        persona_guidance=PERSONAS[persona_key],
    )

    return [
        {"role": "system", "content": load_prompt("explanation_system.txt")},
        {"role": "user", "content": user_content},
    ]


def build_synthetic_prompt(
    count: int,
    focus: str,
) -> str:
    focus_map = {
        "denial_patterns": (
            "underrepresented denial patterns: theft without theft coverage, "
            "inactive policy, inconsistent damage flags vs narrative"
        ),
        "borderline": (
            "borderline approval cases where small changes in coverage or excess "
            "could flip the decision"
        ),
    }
    focus_description = focus_map.get(focus, focus)
    return load_prompt("synthetic_user.txt").format(
        count=count,
        focus_description=focus_description,
    )
