from __future__ import annotations

import random
from typing import Any

from claim_agent.config import genai_uses_live_llm
from claim_agent.genai.llm_client import chat_completion, parse_json_array
from claim_agent.genai.prompts import build_synthetic_prompt

MOCK_SCENARIOS = [
    {
        "excessFee": 619,
        "rrp": 15490,
        "balanceRRP": 15490,
        "oldBalanceRRP": 15490,
        "coverage": "ADLD",
        "policyStatus": "InActive",
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
        "issueDesc": (
            "Dropped phone on vacation; screen cracked. Policy had already expired "
            "two days before the incident."
        ),
        "expected_status": "Declined",
    },
    {
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
        "country": "NL",
        "turnOnOff": 0,
        "touchScreen": 0,
        "smashed": 0,
        "frontCamera": 0,
        "backCamera": 0,
        "frontOrBackCamera": 0,
        "audio": 0,
        "mic": 0,
        "buttons": 0,
        "connection": 0,
        "charging": 0,
        "other": 0,
        "issueDesc": (
            "Phone stolen from gym locker. No police report attached yet. "
            "Coverage is ADLD only without theft rider."
        ),
        "expected_status": "Declined",
    },
    {
        "excessFee": 1989,
        "rrp": 11990,
        "balanceRRP": 11990,
        "oldBalanceRRP": 11990,
        "coverage": "ADLD/THEFT",
        "policyStatus": "Active",
        "retailerName": "WUAWEI eStore",
        "deviceType": "SMARTPHONES",
        "make": "WUAWEI",
        "channel": "Online Portal",
        "claimType": "Theft",
        "country": "SE",
        "turnOnOff": 0,
        "touchScreen": 0,
        "smashed": 0,
        "frontCamera": 0,
        "backCamera": 0,
        "frontOrBackCamera": 0,
        "audio": 0,
        "mic": 0,
        "buttons": 0,
        "connection": 0,
        "charging": 0,
        "other": 0,
        "issueDesc": (
            "Bag snatched on commuter train; device taken. Police report filed same day. "
            "IMEI block requested."
        ),
        "expected_status": "Completed",
    },
]


def generate_synthetic_claims(
    count: int = 3,
    focus: str = "denial_patterns",
) -> list[dict[str, Any]]:
    if genai_uses_live_llm():
        prompt = build_synthetic_prompt(count, focus)
        messages = [
            {
                "role": "system",
                "content": "You generate realistic structured insurance claim data.",
            },
            {"role": "user", "content": prompt},
        ]
        raw = chat_completion(messages, temperature=0.7)
        return parse_json_array(raw)

    pool = MOCK_SCENARIOS.copy()
    if focus == "borderline":
        pool.append(
            {
                **MOCK_SCENARIOS[2],
                "excessFee": 3500,
                "issueDesc": (
                    "Theft reported with partial documentation; policy active but "
                    "high excess and delayed notification (9 days)."
                ),
                "expected_status": "Completed",
            }
        )
    random.shuffle(pool)
    return pool[:count]
