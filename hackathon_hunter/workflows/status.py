from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hackathon_hunter.rules import check_rules
from hackathon_hunter.scoring import (
    STALE_EVIDENCE_FIELDS,
    days_until_deadline,
    load_scoring_config,
    score_hackathon,
)
from hackathon_hunter.storage import load_hackathons, load_latest_processed, utcish_now


@dataclass(frozen=True)
class StatusRow:
    hackathon_id: str
    name: str
    days_until_deadline: float
    overall_score: float
    taiwan_gate: str
    ai_policy: str
    rules_status: str
    stale_fields: list[str]


@dataclass(frozen=True)
class StatusResult:
    input_path: Path
    rows: list[StatusRow]


def _taiwan_gate_label(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "review"


def run_status(
    input_path: Path | None = None,
    now: datetime | None = None,
) -> StatusResult:
    checked_at = now or utcish_now()
    if input_path:
        source_path = Path(input_path)
        hackathons = load_hackathons(source_path)
    else:
        source_path, hackathons = load_latest_processed()

    scoring_config = load_scoring_config()
    rows: list[StatusRow] = []
    for hackathon in hackathons:
        if str(hackathon.status).lower() == "closed" or hackathon.is_expired(checked_at):
            continue
        score = score_hackathon(hackathon, now=checked_at)
        rules = check_rules(hackathon, now=checked_at)
        stale_fields = hackathon.stale_evidence_fields(
            checked_at,
            max_age_days=int(scoring_config["evidence_max_age_days"]),
            fields=STALE_EVIDENCE_FIELDS,
        )
        rows.append(
            StatusRow(
                hackathon_id=hackathon.id,
                name=hackathon.name,
                days_until_deadline=days_until_deadline(hackathon, checked_at),
                overall_score=score.overall_score,
                taiwan_gate=_taiwan_gate_label(hackathon.eligibility.taiwan_eligible),
                ai_policy=str(hackathon.ai_policy),
                rules_status="eligible" if rules.eligible else "blocked",
                stale_fields=stale_fields,
            )
        )
    rows.sort(key=lambda item: item.days_until_deadline)
    return StatusResult(input_path=source_path, rows=rows)
