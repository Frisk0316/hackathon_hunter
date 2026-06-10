from __future__ import annotations

from typing import Any

from claimclear.decision_router import route_decision
from claimclear.intake_agent import extract_claim
from claimclear.models import Claim, Policy, PolicyCheckResult, RiskScoreResult
from claimclear.policy_check import check_policy
from claimclear.risk_scoring_agent import score_risk


def intake_agent(payload: dict[str, Any] | str) -> dict[str, Any]:
    """UiPath Coded Agent entrypoint for claim intake."""
    return extract_claim(payload).to_dict()


def policy_check_bot(payload: dict[str, Any]) -> dict[str, Any]:
    """UiPath Coded Agent/RPA shim for policy validation."""
    claim = Claim.from_dict(payload["claim"])
    policy_data = payload.get("policy")
    policy = Policy.from_dict(policy_data) if policy_data else None
    return check_policy(claim, policy).to_dict()


def risk_scoring_agent(payload: dict[str, Any]) -> dict[str, Any]:
    """UiPath Coded Agent entrypoint for risk scoring and rationale."""
    claim = Claim.from_dict(payload["claim"])
    policy_data = payload.get("policy")
    policy = Policy.from_dict(policy_data) if policy_data else None
    policy_check = PolicyCheckResult(**payload["policy_check"])
    return score_risk(claim, policy, policy_check).to_dict()


def decision_router(payload: dict[str, Any]) -> dict[str, Any]:
    """UiPath Coded Agent entrypoint for final routing."""
    claim = Claim.from_dict(payload["claim"])
    policy_check = PolicyCheckResult(**payload["policy_check"])
    risk_score = RiskScoreResult(**payload["risk_score"])
    return route_decision(claim, policy_check, risk_score).to_dict()

