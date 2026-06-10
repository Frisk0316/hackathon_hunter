from __future__ import annotations

from typing import Any

from claimclear.models import Claim, IntakeResult

REQUIRED_FIELDS = [
    "claim_id",
    "policy_id",
    "claimant_name",
    "claimant_email",
    "claim_type",
    "amount",
    "incident_date",
    "reported_date",
    "loss_location",
    "narrative",
    "documents",
]


def _parse_key_value_form(raw_text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in raw_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if key == "amount":
            parsed[key] = float(value.replace("$", "").replace(",", ""))
        elif key == "documents":
            parsed[key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            parsed[key] = value
    return parsed


def extract_claim(payload: Claim | dict[str, Any] | str) -> IntakeResult:
    if isinstance(payload, Claim):
        raw_claim = payload.to_dict()
    elif isinstance(payload, str):
        raw_claim = _parse_key_value_form(payload)
    else:
        raw_claim = dict(payload)

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in raw_claim or raw_claim[field] in ("", None, [])
    ]
    extraction_confidence = round(max(0.2, 0.98 - (0.08 * len(missing_fields))), 2)

    claim = Claim.from_dict(raw_claim)
    if missing_fields:
        rationale = "Claim intake normalized the submitted record, but required fields need review."
    else:
        rationale = "Claim intake normalized all required fields from the submitted form."

    return IntakeResult(
        structured_claim=claim,
        extraction_confidence=extraction_confidence,
        missing_fields=missing_fields,
        rationale=rationale,
    )

