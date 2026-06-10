from datetime import datetime, timezone
from pathlib import Path

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
