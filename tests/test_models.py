from pathlib import Path

import pytest
from pydantic import ValidationError

from hackathon_hunter.models import Hackathon
from hackathon_hunter.storage import load_hackathons


def test_current_radar_fixture_parses_with_timezone_deadlines() -> None:
    hackathons = load_hackathons(Path("data/processed/hackathons_20260610.json"))

    assert {item.id for item in hackathons} >= {
        "qwen-cloud-global-ai-2026",
        "elevenlabs-voice-ai-2026",
    }
    assert all(item.deadline.tzinfo is not None for item in hackathons)
    assert all(item.source_evidence for item in hackathons)


def test_deadline_requires_timezone() -> None:
    with pytest.raises(ValidationError):
        Hackathon.model_validate(
            {
                "id": "bad",
                "name": "Bad",
                "platform": "fixture",
                "url": "https://example.com",
                "deadline": "2026-07-01T12:00:00",
                "deadline_timezone": "UTC",
            }
        )
