from __future__ import annotations

from claimclear.models import Claim, Policy, PolicyCheckResult, parse_date


def check_policy(claim: Claim, policy: Policy | None) -> PolicyCheckResult:
    if policy is None:
        return PolicyCheckResult(
            passed=False,
            reasons=["policy_not_found"],
            coverage_limit=0.0,
            deductible=0.0,
            covered_amount=0.0,
            policy_status="missing",
        )

    reasons: list[str] = []
    policy_status = policy.status.lower()
    coverage_limit = policy.coverage_limits.get(claim.claim_type, 0.0)

    if policy_status != "active":
        reasons.append("policy_status_not_active")

    incident_date = parse_date(claim.incident_date)
    effective_date = parse_date(policy.effective_date)
    expiration_date = parse_date(policy.expiration_date)
    if incident_date < effective_date or incident_date > expiration_date:
        reasons.append("incident_outside_policy_period")

    if coverage_limit <= 0:
        reasons.append("coverage_not_supported")
    elif claim.amount > coverage_limit:
        reasons.append("amount_exceeds_coverage_limit")

    covered_amount = max(0.0, min(claim.amount, coverage_limit) - policy.deductible)

    return PolicyCheckResult(
        passed=len(reasons) == 0,
        reasons=reasons or ["coverage_valid"],
        coverage_limit=coverage_limit,
        deductible=policy.deductible,
        covered_amount=round(covered_amount, 2),
        policy_status=policy.status,
    )

