from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hackathon_hunter.storage import load_hackathons, read_json, write_json
from hackathon_hunter.workflows.import_hackathons import ImportValidationError, run_import

FIXED_NOW = datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc)


def _hackathon_payload(
    hackathon_id: str,
    *,
    name: str | None = None,
    deadline: datetime | None = None,
    prize_total_usd: float = 1000,
    fetched_at: datetime | None = None,
) -> dict:
    fetched = fetched_at or FIXED_NOW
    active_deadline = deadline or (FIXED_NOW + timedelta(days=30))
    return {
        "id": hackathon_id,
        "name": name or hackathon_id.title(),
        "platform": "fixture",
        "url": f"https://example.com/{hackathon_id}",
        "deadline": active_deadline.isoformat(),
        "deadline_timezone": "UTC",
        "format": "online",
        "prize_total_usd": prize_total_usd,
        "cash_prize": True,
        "required_apis": ["Example API"],
        "ai_policy": "allowed",
        "eligibility": {"region_restricted": False, "student_only": False},
        "status": "open",
        "source_evidence": [
            {
                "field": "deadline",
                "url": f"https://example.com/{hackathon_id}",
                "quote": "deadline fixture",
                "fetched_at": fetched.isoformat(),
                "confidence": 0.9,
            },
            {
                "field": "prize_total_usd",
                "url": f"https://example.com/{hackathon_id}",
                "quote": f"{prize_total_usd} USD",
                "fetched_at": fetched.isoformat(),
                "confidence": 0.9,
            },
        ],
    }


def test_import_rejects_bad_schema_without_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.json"
    write_json(
        input_path,
        [
            {
                "id": "bad",
                "name": "Bad",
                "platform": "fixture",
                "url": "https://example.com",
                "deadline": "2026-07-01T00:00:00",
                "deadline_timezone": "UTC",
            }
        ],
    )

    with pytest.raises(ImportValidationError):
        run_import(input_path, root=tmp_path)

    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "data" / "processed").exists()


def test_import_merge_updates_newer_fields_unions_evidence_and_marks_missing(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "data" / "processed" / "hackathons_20260601.json"
    old_keep = _hackathon_payload(
        "keep",
        prize_total_usd=1000,
        fetched_at=FIXED_NOW - timedelta(days=3),
    )
    removed = _hackathon_payload("removed", fetched_at=FIXED_NOW - timedelta(days=3))
    write_json(
        processed,
        {"generated_at": FIXED_NOW.isoformat(), "hackathons": [old_keep, removed]},
    )

    input_path = tmp_path / "incoming.json"
    updated_keep = _hackathon_payload(
        "keep",
        prize_total_usd=5000,
        fetched_at=FIXED_NOW + timedelta(hours=1),
    )
    new_item = _hackathon_payload("new", fetched_at=FIXED_NOW + timedelta(hours=1))
    write_json(input_path, {"hackathons": [updated_keep, new_item]})

    result = run_import(input_path, merge=True, root=tmp_path, now=FIXED_NOW)

    assert result.processed_path is not None
    imported = {item.id: item for item in load_hackathons(result.processed_path)}
    assert set(imported) == {"keep", "removed", "new"}
    assert imported["keep"].prize_total_usd == 5000
    assert len(imported["keep"].source_evidence) == 4

    metadata = read_json(result.processed_path)
    assert metadata["expired_for_review"] == [
        {"id": "removed", "name": "Removed", "status": "expired_for_review"}
    ]
    diff_text = result.diff_report_path.read_text(encoding="utf-8")
    assert "CHANGED `keep`" in diff_text
    assert "ADDED `new`" in diff_text
    assert "MISSING `removed`" in diff_text


def test_import_dry_run_writes_diff_but_no_processed_file(tmp_path: Path) -> None:
    existing_path = tmp_path / "data" / "processed" / "hackathons_20260601.json"
    write_json(
        existing_path,
        {"generated_at": FIXED_NOW.isoformat(), "hackathons": [_hackathon_payload("keep")]},
    )
    input_path = tmp_path / "incoming.json"
    write_json(input_path, {"hackathons": [_hackathon_payload("keep", prize_total_usd=2000)]})

    before = set((tmp_path / "data" / "processed").glob("*.json"))
    result = run_import(input_path, merge=True, dry_run=True, root=tmp_path, now=FIXED_NOW)
    after = set((tmp_path / "data" / "processed").glob("*.json"))

    assert result.processed_path is None
    assert before == after
    assert result.diff_report_path.exists()
