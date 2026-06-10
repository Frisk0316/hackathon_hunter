from datetime import datetime, timezone
from pathlib import Path

from hackathon_hunter.models import Evidence, Hackathon
from hackathon_hunter.rules import check_rules
from hackathon_hunter.storage import load_hackathons

FIXED_NOW = datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc)


def _by_id(hackathon_id: str):
    return {
        item.id: item for item in load_hackathons(Path("data/processed/hackathons_20260610.json"))
    }[hackathon_id]


def test_region_restriction_blocks_qwen_until_human_verification() -> None:
    result = check_rules(_by_id("qwen-cloud-global-ai-2026"), now=FIXED_NOW)

    assert not result.eligible
    assert any("Region/account eligibility" in item for item in result.blocking_issues)
    assert result.human_review_required


def test_unknown_ai_policy_blocks_build_stage() -> None:
    result = check_rules(_by_id("munichtech-innovation-2026"), now=FIXED_NOW)

    assert not result.eligible
    assert any("AI policy is unknown" in item for item in result.blocking_issues)


def test_rules_check_always_requires_human_review_by_design() -> None:
    hackathon = Hackathon(
        id="clean",
        name="Clean Hackathon",
        platform="fixture",
        url="https://example.com",
        deadline="2026-07-01T00:00:00+00:00",
        deadline_timezone="UTC",
        format="online",
        prize_total_usd=10000,
        cash_prize=True,
        required_apis=["Example API"],
        ai_policy="allowed",
        eligibility={"region_restricted": False, "student_only": False, "team_required": False},
        source_evidence=[
            Evidence(
                field=field,
                url="https://example.com",
                quote="fixture",
                fetched_at=FIXED_NOW,
                confidence=0.9,
            )
            for field in [
                "deadline",
                "eligibility",
                "prize_total_usd",
                "cash_prize",
                "required_apis",
            ]
        ],
    )

    result = check_rules(hackathon, now=FIXED_NOW)

    assert result.eligible
    assert result.blocking_issues == []
    assert result.warnings == []
    assert result.human_review_required


def test_rules_warns_when_critical_evidence_is_stale() -> None:
    hackathon = Hackathon(
        id="stale",
        name="Stale Hackathon",
        platform="fixture",
        url="https://example.com",
        deadline="2026-07-01T00:00:00+00:00",
        deadline_timezone="UTC",
        format="online",
        ai_policy="allowed",
        eligibility={"region_restricted": False},
        source_evidence=[
            Evidence(
                field="deadline",
                url="https://example.com",
                quote="fixture",
                fetched_at=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
                confidence=0.9,
            )
        ],
    )

    result = check_rules(hackathon, now=FIXED_NOW)

    assert any("Evidence stale for deadline" in warning for warning in result.warnings)
