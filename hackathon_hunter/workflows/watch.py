from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from hackathon_hunter.models import Hackathon
from hackathon_hunter.scoring import STALE_EVIDENCE_FIELDS, load_scoring_config
from hackathon_hunter.storage import (
    load_hackathons,
    load_latest_processed,
    project_path,
    save_report,
    unique_path,
    utcish_now,
)

FAST_LANE_SECONDS_PER_DAY = 24 * 60 * 60
DEFAULT_STALE_EVIDENCE_DAYS = 14


@dataclass(frozen=True)
class WatchEvent:
    kind: str
    hackathon_id: str
    name: str
    deadline: datetime
    detail: str
    fields: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WatchResult:
    input_path: Path
    report_path: Path
    events: list[WatchEvent]


def _is_watch_candidate(hackathon: Hackathon) -> bool:
    return str(hackathon.status).lower() != "closed"


def _format_days(total_seconds: float) -> str:
    days = abs(total_seconds) / FAST_LANE_SECONDS_PER_DAY
    return f"{days:.1f} days"


def build_watch_events(
    hackathons: list[Hackathon],
    now: datetime,
    fast_lane_days: int = 3,
    stale_evidence_days: int = DEFAULT_STALE_EVIDENCE_DAYS,
) -> list[WatchEvent]:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must include timezone")
    events: list[WatchEvent] = []
    fast_lane_seconds = fast_lane_days * FAST_LANE_SECONDS_PER_DAY

    for hackathon in hackathons:
        if not _is_watch_candidate(hackathon):
            continue
        seconds_until_deadline = (hackathon.deadline - now).total_seconds()
        days_label = _format_days(seconds_until_deadline)
        if seconds_until_deadline < 0:
            events.append(
                WatchEvent(
                    kind="EXPIRED",
                    hackathon_id=hackathon.id,
                    name=hackathon.name,
                    deadline=hackathon.deadline,
                    detail=f"Deadline passed {days_label} ago; mark closed or review immediately.",
                )
            )
        elif seconds_until_deadline <= fast_lane_seconds:
            events.append(
                WatchEvent(
                    kind="ENTERING_FAST_LANE",
                    hackathon_id=hackathon.id,
                    name=hackathon.name,
                    deadline=hackathon.deadline,
                    detail=(
                        f"{days_label} until deadline; "
                        f"fast-lane threshold is {fast_lane_days} days."
                    ),
                )
            )

        stale_fields = hackathon.stale_evidence_fields(
            now,
            max_age_days=stale_evidence_days,
            fields=STALE_EVIDENCE_FIELDS,
        )
        if stale_fields:
            events.append(
                WatchEvent(
                    kind="STALE_EVIDENCE",
                    hackathon_id=hackathon.id,
                    name=hackathon.name,
                    deadline=hackathon.deadline,
                    detail=f"Evidence older than {stale_evidence_days} days.",
                    fields=stale_fields,
                )
            )
    return events


def render_watch_report(
    input_path: Path,
    events: list[WatchEvent],
    generated_at: datetime,
    fast_lane_days: int,
    stale_evidence_days: int,
) -> str:
    lines = [
        f"# Deadline Watch — {generated_at.date().isoformat()}",
        "",
        f"- Source: `{input_path}`",
        f"- Fast-lane threshold: `{fast_lane_days}` days",
        f"- Stale evidence threshold: `{stale_evidence_days}` days",
        "",
        "## Events",
        "",
    ]
    if not events:
        lines.append("No watch events.")
        return "\n".join(lines)

    lines.extend(
        [
            "| Event | Candidate | Deadline | Detail |",
            "|---|---|---|---|",
        ]
    )
    for event in events:
        field_suffix = f" Fields: {', '.join(event.fields)}." if event.fields else ""
        lines.append(
            "| "
            f"{event.kind} | `{event.hackathon_id}` — {event.name} | "
            f"{event.deadline.isoformat()} | {event.detail}{field_suffix} |"
        )
    return "\n".join(lines)


def run_watch(
    input_path: Path | None = None,
    fast_lane_days: int = 3,
    stale_evidence_days: int | None = None,
    now: datetime | None = None,
    root: Path | None = None,
) -> WatchResult:
    base = root or project_path()
    generated_at = now or utcish_now()
    scoring_config = load_scoring_config()
    effective_stale_days = int(
        stale_evidence_days
        if stale_evidence_days is not None
        else scoring_config["evidence_max_age_days"]
    )
    if input_path is None:
        resolved_input_path, hackathons = load_latest_processed(root=base)
    else:
        resolved_input_path = Path(input_path)
        hackathons = load_hackathons(resolved_input_path)

    events = build_watch_events(
        hackathons,
        now=generated_at,
        fast_lane_days=fast_lane_days,
        stale_evidence_days=effective_stale_days,
    )
    report_path = unique_path(
        base / "reports" / f"watch_{generated_at.strftime('%Y%m%d')}.md"
    )
    save_report(
        report_path,
        render_watch_report(
            resolved_input_path,
            events,
            generated_at,
            fast_lane_days,
            effective_stale_days,
        ),
    )
    return WatchResult(input_path=resolved_input_path, report_path=report_path, events=events)
