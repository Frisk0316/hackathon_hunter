from __future__ import annotations

from pathlib import Path
from typing import Any

from claimclear.audit import AuditLogger
from claimclear.decision_router import resolve_human_decision, route_decision
from claimclear.intake_agent import extract_claim
from claimclear.models import Claim, PipelineResult, Policy
from claimclear.policy_check import check_policy
from claimclear.risk_scoring_agent import score_risk
from claimclear.synthetic_data import load_claims, load_policies


class ClaimClearPipeline:
    def __init__(
        self,
        *,
        policies: dict[str, Policy],
        audit_logger: AuditLogger,
    ) -> None:
        self.policies = policies
        self.audit_logger = audit_logger

    @classmethod
    def from_fixtures(cls, audit_path: str | Path | None = None) -> ClaimClearPipeline:
        return cls(
            policies=load_policies(),
            audit_logger=AuditLogger(audit_path),
        )

    def run(
        self,
        *,
        claim_payload: Claim | dict[str, Any] | str,
        resolve_human: str | None = None,
    ) -> PipelineResult:
        intake = extract_claim(claim_payload)
        claim = intake.structured_claim
        case_id = f"CASE-{claim.claim_id}"

        self.audit_logger.record(
            case_id=case_id,
            claim_id=claim.claim_id,
            actor="maestro_case",
            action="case_started",
            details={"source": "local_demo", "synthetic": True},
        )
        self.audit_logger.record(
            case_id=case_id,
            claim_id=claim.claim_id,
            actor="intake_agent",
            action="completed",
            details=intake.to_dict(),
        )

        policy = self.policies.get(claim.policy_id)
        policy_check = check_policy(claim, policy)
        self.audit_logger.record(
            case_id=case_id,
            claim_id=claim.claim_id,
            actor="policy_check_bot",
            action="completed",
            details=policy_check.to_dict(),
        )

        risk_score = score_risk(claim, policy, policy_check)
        self.audit_logger.record(
            case_id=case_id,
            claim_id=claim.claim_id,
            actor="risk_scoring_agent",
            action="completed",
            details=risk_score.to_dict(),
        )

        initial_decision = route_decision(claim, policy_check, risk_score)
        self.audit_logger.record(
            case_id=case_id,
            claim_id=claim.claim_id,
            actor="decision_router",
            action="completed",
            details=initial_decision.to_dict(),
        )

        final_decision = initial_decision
        if initial_decision.human_task_id:
            self.audit_logger.record(
                case_id=case_id,
                claim_id=claim.claim_id,
                actor="human_approval_gate",
                action="task_created",
                details={
                    "human_task_id": initial_decision.human_task_id,
                    "queue": "claims_reviewer",
                    "risk_flags": risk_score.risk_flags,
                },
            )
            if resolve_human:
                final_decision = resolve_human_decision(initial_decision, claim, resolve_human)
                self.audit_logger.record(
                    case_id=case_id,
                    claim_id=claim.claim_id,
                    actor="human_reviewer",
                    action="task_completed",
                    details=final_decision.to_dict(),
                )

        if final_decision.status == "resolved":
            self.audit_logger.record(
                case_id=case_id,
                claim_id=claim.claim_id,
                actor="maestro_case",
                action="case_resolved",
                details=final_decision.to_dict(),
            )

        return PipelineResult(
            case_id=case_id,
            claim=claim,
            intake=intake,
            policy_check=policy_check,
            risk_score=risk_score,
            initial_decision=initial_decision,
            final_decision=final_decision,
            audit_entries=self.audit_logger.entries_for_case(case_id),
        )


def run_fixture_claim(
    claim_id: str,
    *,
    audit_path: str | Path | None = None,
    resolve_human: str | None = None,
) -> PipelineResult:
    claims = load_claims()
    if claim_id not in claims:
        known_ids = ", ".join(sorted(claims))
        raise KeyError(f"unknown claim_id '{claim_id}'. Known claims: {known_ids}")
    pipeline = ClaimClearPipeline.from_fixtures(audit_path)
    return pipeline.run(claim_payload=claims[claim_id], resolve_human=resolve_human)

