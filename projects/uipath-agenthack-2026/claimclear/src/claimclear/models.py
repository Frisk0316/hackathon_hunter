from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    policy_id: str
    claimant_name: str
    claimant_email: str
    claim_type: str
    amount: float
    incident_date: str
    reported_date: str
    loss_location: str
    narrative: str
    documents: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Claim:
        return cls(
            claim_id=str(data["claim_id"]),
            policy_id=str(data["policy_id"]),
            claimant_name=str(data["claimant_name"]),
            claimant_email=str(data["claimant_email"]),
            claim_type=str(data["claim_type"]),
            amount=float(data["amount"]),
            incident_date=str(data["incident_date"]),
            reported_date=str(data["reported_date"]),
            loss_location=str(data["loss_location"]),
            narrative=str(data["narrative"]),
            documents=list(data.get("documents", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Policy:
    policy_id: str
    holder_name: str
    status: str
    effective_date: str
    expiration_date: str
    coverage_limits: dict[str, float]
    deductible: float
    prior_claims_last_24_months: int = 0
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        return cls(
            policy_id=str(data["policy_id"]),
            holder_name=str(data["holder_name"]),
            status=str(data["status"]),
            effective_date=str(data["effective_date"]),
            expiration_date=str(data["expiration_date"]),
            coverage_limits={key: float(value) for key, value in data["coverage_limits"].items()},
            deductible=float(data["deductible"]),
            prior_claims_last_24_months=int(data.get("prior_claims_last_24_months", 0)),
            notes=list(data.get("notes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IntakeResult:
    structured_claim: Claim
    extraction_confidence: float
    missing_fields: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["structured_claim"] = self.structured_claim.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class PolicyCheckResult:
    passed: bool
    reasons: list[str]
    coverage_limit: float
    deductible: float
    covered_amount: float
    policy_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RiskScoreResult:
    confidence: float
    risk_score: float
    risk_flags: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecisionResult:
    outcome: str
    status: str
    route: str
    payout_estimate: float
    rationale: str
    next_owner: str
    human_task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AuditEntry:
    timestamp: str
    case_id: str
    claim_id: str
    actor: str
    action: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    case_id: str
    claim: Claim
    intake: IntakeResult
    policy_check: PolicyCheckResult
    risk_score: RiskScoreResult
    initial_decision: DecisionResult
    final_decision: DecisionResult
    audit_entries: list[AuditEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "claim": self.claim.to_dict(),
            "intake": self.intake.to_dict(),
            "policy_check": self.policy_check.to_dict(),
            "risk_score": self.risk_score.to_dict(),
            "initial_decision": self.initial_decision.to_dict(),
            "final_decision": self.final_decision.to_dict(),
            "audit_entries": [entry.to_dict() for entry in self.audit_entries],
        }

