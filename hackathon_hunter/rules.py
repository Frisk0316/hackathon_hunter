from __future__ import annotations

from datetime import datetime

from hackathon_hunter.models import Hackathon, RulesCheckResult
from hackathon_hunter.storage import utcish_now


def _evidence_note(hackathon: Hackathon, field: str) -> str:
    evidence = hackathon.evidence_for(field)
    if not evidence:
        return "來源不足"
    best = max(evidence, key=lambda item: item.confidence)
    source = best.url or "來源不足"
    return f"{source} (confidence={best.confidence:.2f})"


def check_rules(
    hackathon: Hackathon,
    now: datetime | None = None,
    active_buffer_days: int = 7,
) -> RulesCheckResult:
    checked_at = now or utcish_now()
    blocking: list[str] = []
    warnings: list[str] = []

    if hackathon.is_expired(checked_at):
        blocking.append(f"Deadline has passed: {_evidence_note(hackathon, 'deadline')}")
    elif not hackathon.is_active_candidate(checked_at, active_buffer_days):
        blocking.append(
            f"Deadline is inside the {active_buffer_days}-day freshness buffer: "
            f"{_evidence_note(hackathon, 'deadline')}"
        )

    normalized_format = str(hackathon.format).lower()
    if normalized_format == "in_person":
        blocking.append(f"In-person only format: {_evidence_note(hackathon, 'format')}")
    elif normalized_format == "unknown":
        warnings.append(f"Participation format is unknown: {_evidence_note(hackathon, 'format')}")

    policy = str(hackathon.ai_policy).lower()
    if policy == "forbidden":
        blocking.append(
            "AI-assisted development appears forbidden: "
            f"{_evidence_note(hackathon, 'ai_policy')}"
        )
    elif policy == "unknown":
        blocking.append(
            "AI policy is unknown; build stage is blocked: "
            f"{_evidence_note(hackathon, 'ai_policy')}"
        )
    elif policy == "restricted":
        warnings.append(
            "AI policy is restricted; human review required: "
            f"{_evidence_note(hackathon, 'ai_policy')}"
        )

    if hackathon.eligibility.region_restricted:
        blocking.append(
            "Region/account eligibility must be verified before build: "
            f"{_evidence_note(hackathon, 'eligibility')}"
        )
    if hackathon.eligibility.student_only:
        blocking.append(f"Student-only eligibility: {_evidence_note(hackathon, 'eligibility')}")
    if hackathon.eligibility.team_required:
        warnings.append(f"Team may be required: {_evidence_note(hackathon, 'eligibility')}")

    for field in ["deadline", "eligibility"]:
        confidence = hackathon.evidence_confidence(field)
        if confidence is None:
            warnings.append(f"Missing evidence for {field}: 來源不足")
        elif confidence < 0.7:
            warnings.append(
                f"Low confidence evidence for {field}: {_evidence_note(hackathon, field)}"
            )

    for field in ["prize_total_usd", "cash_prize", "required_apis"]:
        if hackathon.evidence_confidence(field) is None:
            warnings.append(f"Missing evidence for {field}: 來源不足")

    human_review_required = bool(blocking or warnings) or True
    return RulesCheckResult(
        hackathon_id=hackathon.id,
        eligible=not blocking,
        blocking_issues=blocking,
        warnings=warnings,
        submission_requirements=hackathon.submission_requirements,
        human_review_required=human_review_required,
    )
