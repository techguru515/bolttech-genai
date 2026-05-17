from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ClaimRequest(BaseModel):
    excessFee: Optional[float] = None
    rrp: Optional[float] = None
    balanceRRP: Optional[float] = None
    oldBalanceRRP: Optional[float] = None
    coverage: Optional[str] = None
    policyStatus: Optional[str] = "Active"
    retailerName: Optional[str] = None
    deviceType: Optional[str] = "SMARTPHONES"
    make: Optional[str] = "WUAWEI"
    channel: Optional[str] = "Online Portal"
    claimType: Optional[str] = "Accidental Damage"
    country: Optional[str] = "NL"
    turnOnOff: Optional[float] = None
    touchScreen: Optional[float] = None
    smashed: Optional[float] = None
    frontCamera: Optional[float] = None
    backCamera: Optional[float] = None
    frontOrBackCamera: Optional[float] = None
    audio: Optional[float] = None
    mic: Optional[float] = None
    buttons: Optional[float] = None
    connection: Optional[float] = None
    charging: Optional[float] = None
    other: Optional[float] = None
    issueDesc: Optional[str] = Field(default="", description="Customer narrative")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=False)


class PredictionResponse(BaseModel):
    prediction: str
    approved: bool
    probability_approved: float
    probability_declined: float
    contributing_factors: list[str]


class ExplainRequest(BaseModel):
    claim: ClaimRequest
    personas: list[str] = Field(
        default=["customer", "claims_adjuster"],
        description="customer | claims_adjuster | compliance_officer",
    )


class ExplainResponse(BaseModel):
    prediction: PredictionResponse
    explanations: dict[str, str]
    genai_mode: str


class SyntheticRequest(BaseModel):
    count: int = Field(default=3, ge=1, le=10)
    focus: str = Field(default="denial_patterns", description="denial_patterns | borderline")


class SyntheticResponse(BaseModel):
    scenarios: list[dict[str, Any]]
    genai_mode: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    genai_mode: str
