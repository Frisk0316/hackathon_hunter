from datetime import datetime, timedelta, timezone
from pathlib import Path

from hackathon_hunter.models import ResultRecord
from hackathon_hunter.storage import read_json, write_json
from hackathon_hunter.workflows.calibrate import run_calibrate
from hackathon_hunter.workflows.results import run_record_result

FIXED_NOW = datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc)


def _hackathon_payload(hackathon_id: str) -> dict:
    return {
        "id": hackathon_id,
        "name": "Recorded Hackathon",
        "platform": "fixture",
        "url": "https://example.com",
        "deadline": (FIXED_NOW + timedelta(days=30)).isoformat(),
        "deadline_timezone": "UTC",
        "format": "online",
        "prize_total_usd": 10000,
        "cash_prize": True,
        "required_apis": ["Example API"],
        "ai_policy": "allowed",
        "eligibility": {"region_restricted": False},
        "status": "open",
    }


def test_record_result_persists_feedback_and_score_trace(tmp_path: Path) -> None:
    input_path = tmp_path / "processed.json"
    write_json(input_path, {"hackathons": [_hackathon_payload("record-me")]})

    outputs = run_record_result(
        "record-me",
        "finalist",
        project_slug="project",
        hours_spent=12,
        what_worked=["sponsor API depth"],
        what_failed=["demo video was rushed"],
        notes="Useful round.",
        input_path=input_path,
        root=tmp_path,
    )

    record = read_json(outputs["record"])
    report = outputs["report"].read_text(encoding="utf-8")

    assert record["what_worked"] == ["sponsor API depth"]
    assert record["what_failed"] == ["demo video was rushed"]
    assert record["notes"] == "Useful round."
    assert record["score_trace"]["sponsor_api_fit"] > 0
    assert record["overall_score"] is not None
    assert "sponsor API depth" in report
    assert "demo video was rushed" in report


def test_calibrate_suggests_weight_diff_from_success_and_failure_traces(
    tmp_path: Path,
) -> None:
    rounds = tmp_path / "logs" / "rounds"
    rounds.mkdir(parents=True)
    write_json(
        rounds / "winner.json",
        ResultRecord(
            hackathon_id="winner",
            outcome="winner",
            what_worked=["deep sponsor integration"],
            score_trace={"sponsor_api_fit": 0.9, "deadline_buffer": 0.8},
            overall_score=0.8,
        ),
    )
    write_json(
        rounds / "rejected.json",
        ResultRecord(
            hackathon_id="rejected",
            outcome="rejected",
            what_failed=["weak sponsor integration"],
            score_trace={"sponsor_api_fit": 0.3, "deadline_buffer": 0.8},
            overall_score=0.4,
        ),
    )

    result = run_calibrate(root=tmp_path)
    report = result.report_path.read_text(encoding="utf-8")

    assert result.records_read == 2
    assert result.records_used == 2
    assert [suggestion.feature for suggestion in result.suggestions] == ["sponsor_api_fit"]
    assert "+ sponsor_api_fit" in report
    assert "deep sponsor integration" in report
    assert "weak sponsor integration" in report
