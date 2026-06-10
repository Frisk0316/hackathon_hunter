from datetime import datetime, timedelta, timezone
from pathlib import Path

from hackathon_hunter.models import Evidence, Hackathon
from hackathon_hunter.storage import write_json
from hackathon_hunter.workflows.watch import build_watch_events, run_watch

FIXED_NOW = datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc)


def _hackathon(**overrides) -> Hackathon:
    payload = {
        "id": "candidate",
        "name": "Candidate",
        "platform": "fixture",
        "url": "https://example.com",
        "deadline": FIXED_NOW + timedelta(days=10),
        "deadline_timezone": "UTC",
        "format": "online",
        "ai_policy": "allowed",
        "status": "open",
        "source_evidence": [
            Evidence(
                field="deadline",
                url="https://example.com",
                quote="fixture",
                fetched_at=FIXED_NOW,
                confidence=0.9,
            )
        ],
    }
    payload.update(overrides)
    return Hackathon(**payload)


def test_watch_events_cover_expired_fast_lane_and_stale_evidence_boundaries() -> None:
    expired = _hackathon(
        id="expired",
        name="Expired",
        deadline=FIXED_NOW - timedelta(hours=1),
    )
    fast_lane = _hackathon(
        id="fast",
        name="Fast",
        deadline=FIXED_NOW + timedelta(days=3),
    )
    outside_fast_lane = _hackathon(
        id="outside",
        name="Outside",
        deadline=FIXED_NOW + timedelta(days=3, seconds=1),
    )
    stale = _hackathon(
        id="stale",
        name="Stale",
        deadline=FIXED_NOW + timedelta(days=30),
        source_evidence=[
            Evidence(
                field="deadline",
                url="https://example.com",
                quote="fixture",
                fetched_at=FIXED_NOW - timedelta(days=15),
                confidence=0.9,
            )
        ],
    )

    events = build_watch_events(
        [expired, fast_lane, outside_fast_lane, stale],
        now=FIXED_NOW,
        fast_lane_days=3,
        stale_evidence_days=14,
    )
    by_id_and_kind = {(event.hackathon_id, event.kind) for event in events}

    assert ("expired", "EXPIRED") in by_id_and_kind
    assert ("fast", "ENTERING_FAST_LANE") in by_id_and_kind
    assert ("outside", "ENTERING_FAST_LANE") not in by_id_and_kind
    assert ("stale", "STALE_EVIDENCE") in by_id_and_kind


def test_run_watch_writes_report(tmp_path: Path) -> None:
    input_path = tmp_path / "data" / "processed" / "hackathons_20260610.json"
    write_json(
        input_path,
        {
            "generated_at": FIXED_NOW.isoformat(),
            "hackathons": [
                _hackathon(
                    id="fast",
                    deadline=FIXED_NOW + timedelta(days=1),
                )
            ],
        },
    )

    result = run_watch(input_path=input_path, root=tmp_path, now=FIXED_NOW)

    assert result.report_path.exists()
    assert result.events[0].kind == "ENTERING_FAST_LANE"
    assert "ENTERING_FAST_LANE" in result.report_path.read_text(encoding="utf-8")
