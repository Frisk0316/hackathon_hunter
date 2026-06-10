from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

from claimclear.models import Claim, Policy, PolicyCheckResult, RiskScoreResult, parse_date

EXPECTED_DOCUMENTS = {
    "auto_physical_damage": {"photos", "repair_estimate"},
    "property_water_damage": {"photos", "contractor_estimate", "plumber_invoice"},
    "theft": {"police_report", "itemized_loss_list"},
}

VAGUE_TERMS = {
    "unclear",
    "unknown",
    "not sure",
    "changed twice",
    "not attached",
    "missing",
    "maybe",
}


def _claude_rationale_hint(
    claim: Claim,
    policy_check: PolicyCheckResult,
    flags: list[str],
) -> str | None:
    if os.getenv("CLAIMCLEAR_USE_CLAUDE", "false").lower() != "true":
        return None
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    prompt = (
        "You are auditing a synthetic insurance-claim triage decision. "
        "Write one concise sentence explaining why these risk flags matter. "
        f"Claim type: {claim.claim_type}. Amount: {claim.amount}. "
        f"Policy result: {policy_check.reasons}. Flags: {flags}."
    )
    try:
        client = Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        return None

    content = response.content[0]
    text = getattr(content, "text", None)
    return text.strip() if text else None


def score_risk(
    claim: Claim,
    policy: Policy | None,
    policy_check: PolicyCheckResult,
) -> RiskScoreResult:
    risk_score = 0.05
    flags: list[str] = []

    if not policy_check.passed:
        risk_score += 0.35
        flags.extend(reason for reason in policy_check.reasons if reason != "coverage_valid")

    if policy_check.coverage_limit > 0:
        amount_ratio = claim.amount / policy_check.coverage_limit
        if amount_ratio >= 0.75:
            risk_score += 0.16
            flags.append("amount_near_policy_limit")

    incident_date = parse_date(claim.incident_date)
    reported_date = parse_date(claim.reported_date)
    if reported_date - incident_date > timedelta(days=7):
        risk_score += 0.16
        flags.append("late_reported_claim")

    expected_docs = EXPECTED_DOCUMENTS.get(claim.claim_type, set())
    submitted_docs = {document.lower() for document in claim.documents}
    missing_docs = sorted(expected_docs - submitted_docs)
    if missing_docs:
        risk_score += min(0.18, 0.07 * len(missing_docs))
        flags.append("missing_expected_documents:" + ",".join(missing_docs))

    narrative = claim.narrative.lower()
    matched_terms = sorted(term for term in VAGUE_TERMS if term in narrative)
    if matched_terms:
        risk_score += min(0.18, 0.06 * len(matched_terms))
        flags.append("ambiguous_narrative:" + ",".join(matched_terms))

    if policy and policy.prior_claims_last_24_months >= 2:
        risk_score += 0.12
        flags.append("multiple_recent_claims")

    if policy and policy.holder_name.lower() != claim.claimant_name.lower():
        risk_score += 0.2
        flags.append("claimant_policyholder_mismatch")

    risk_score = round(min(risk_score, 0.95), 2)
    confidence = round(max(0.05, 1.0 - risk_score), 2)

    if flags:
        rationale = "Risk score reflects: " + "; ".join(flags) + "."
    else:
        rationale = (
            "Low-risk claim: coverage is valid, documents are present, "
            "and no ambiguity flags fired."
        )

    claude_hint = _claude_rationale_hint(claim, policy_check, flags)
    if claude_hint:
        rationale = f"{rationale} Claude review note: {claude_hint}"

    return RiskScoreResult(
        confidence=confidence,
        risk_score=risk_score,
        risk_flags=flags,
        rationale=rationale,
    )


def score_risk_payload(payload: dict[str, Any]) -> dict[str, Any]:
    claim = Claim.from_dict(payload["claim"])
    policy_data = payload.get("policy")
    policy = Policy.from_dict(policy_data) if policy_data else None
    policy_check = PolicyCheckResult(**payload["policy_check"])
    return score_risk(claim, policy, policy_check).to_dict()
