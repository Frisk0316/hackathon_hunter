from __future__ import annotations

from dataclasses import dataclass

from claimclear.models import Claim, DecisionResult, PolicyCheckResult, RiskScoreResult


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    auto_approve_min_confidence: float = 0.72
    auto_approve_max_risk: float = 0.28


def route_decision(
    claim: Claim,
    policy_check: PolicyCheckResult,
    risk_score: RiskScoreResult,
    config: DecisionConfig | None = None,
) -> DecisionResult:
    config = config or DecisionConfig()
    payout_estimate = policy_check.covered_amount if policy_check.passed else 0.0

    can_auto_approve = (
        policy_check.passed
        and risk_score.confidence >= config.auto_approve_min_confidence
        and risk_score.risk_score <= config.auto_approve_max_risk
    )

    if can_auto_approve:
        return DecisionResult(
            outcome="auto_approved",
            status="resolved",
            route="straight_through_processing",
            payout_estimate=payout_estimate,
            rationale=(
                "Coverage is valid and the risk score is below the auto-clear threshold; "
                "claim can be cleared without human intervention."
            ),
            next_owner="maestro_case",
        )

    task_id = f"TASK-{claim.claim_id}-REVIEW"
    return DecisionResult(
        outcome="escalated",
        status="pending_human",
        route="human_approval_gate",
        payout_estimate=payout_estimate,
        rationale=(
            "Claim needs human review because confidence is below threshold or policy/risk "
            "signals require governance."
        ),
        next_owner="claims_reviewer",
        human_task_id=task_id,
    )


def resolve_human_decision(
    decision: DecisionResult,
    claim: Claim,
    outcome: str,
) -> DecisionResult:
    normalized = outcome.lower()
    if normalized not in {"approve", "reject"}:
        raise ValueError("human outcome must be 'approve' or 'reject'")

    approved = normalized == "approve"
    return DecisionResult(
        outcome="human_approved" if approved else "human_rejected",
        status="resolved",
        route=decision.route,
        payout_estimate=decision.payout_estimate if approved else 0.0,
        rationale=(
            f"Human reviewer {normalized}d synthetic claim {claim.claim_id} after reviewing "
            "the agent rationale and audit trail."
        ),
        next_owner="maestro_case",
        human_task_id=decision.human_task_id,
    )

