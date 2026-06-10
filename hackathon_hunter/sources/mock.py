from __future__ import annotations

from datetime import timedelta

from hackathon_hunter.models import Evidence, Hackathon
from hackathon_hunter.sources.base import SourceAdapter, SourceResult
from hackathon_hunter.storage import utcish_now


class MockSource(SourceAdapter):
    name = "mock"

    def collect(self) -> SourceResult:
        now = utcish_now()
        deadline = now + timedelta(days=35)
        closed_deadline = now - timedelta(days=3)
        fetched_at = now
        hackathons = [
            Hackathon(
                id="mock-hackathon-001",
                name="Mock Global AI Agents Hackathon",
                platform="mock",
                url="https://example.com/mock-ai-agents",
                rules_url="https://example.com/mock-ai-agents/rules",
                deadline=deadline,
                deadline_timezone="UTC",
                format="online",
                prize_total_usd=10000,
                cash_prize=True,
                prize_breakdown="$10,000 cash prize pool",
                tracks=["AI Agents", "Developer Tools"],
                sponsors=["Example Cloud"],
                required_apis=["Example Cloud Inference API"],
                judging_criteria=[
                    "Technical execution",
                    "Sponsor API usage",
                    "Impact",
                    "Presentation",
                ],
                ai_policy="allowed",
                eligibility={
                    "region_restricted": False,
                    "student_only": False,
                    "team_required": False,
                    "notes": "Open to online participants age 18+.",
                },
                submission_requirements={
                    "github_repo": True,
                    "demo_url": True,
                    "video": True,
                    "deck": False,
                    "social_post": False,
                    "public_profile": True,
                    "notes": "Submit public repo, demo URL, and a short video.",
                },
                source_evidence=[
                    Evidence(
                        field="deadline",
                        url="https://example.com/mock-ai-agents",
                        quote="Submissions close 35 days from this fixture run.",
                        fetched_at=fetched_at,
                        confidence=0.95,
                    ),
                    Evidence(
                        field="prize_total_usd",
                        url="https://example.com/mock-ai-agents",
                        quote="$10,000 cash prize pool",
                        fetched_at=fetched_at,
                        confidence=0.95,
                    ),
                    Evidence(
                        field="cash_prize",
                        url="https://example.com/mock-ai-agents",
                        quote="Cash prize pool",
                        fetched_at=fetched_at,
                        confidence=0.95,
                    ),
                    Evidence(
                        field="required_apis",
                        url="https://example.com/mock-ai-agents/rules",
                        quote="Projects must use Example Cloud Inference API.",
                        fetched_at=fetched_at,
                        confidence=0.90,
                    ),
                    Evidence(
                        field="eligibility",
                        url="https://example.com/mock-ai-agents/rules",
                        quote="Open to online participants age 18+.",
                        fetched_at=fetched_at,
                        confidence=0.90,
                    ),
                    Evidence(
                        field="ai_policy",
                        url="https://example.com/mock-ai-agents/rules",
                        quote="AI-assisted development is allowed if disclosed.",
                        fetched_at=fetched_at,
                        confidence=0.90,
                    ),
                ],
                status="open",
                notes="Fixture candidate for end-to-end CLI smoke tests.",
            ),
            Hackathon(
                id="mock-closed-hackathon",
                name="Mock Closed Hackathon",
                platform="mock",
                url="https://example.com/mock-closed",
                rules_url="https://example.com/mock-closed/rules",
                deadline=closed_deadline,
                deadline_timezone="UTC",
                format="online",
                prize_total_usd=5000,
                cash_prize=True,
                tracks=["AI"],
                sponsors=[],
                required_apis=[],
                judging_criteria=[],
                ai_policy="allowed",
                eligibility={
                    "region_restricted": False,
                    "student_only": False,
                    "team_required": False,
                },
                submission_requirements={"github_repo": True},
                source_evidence=[
                    Evidence(
                        field="deadline",
                        url="https://example.com/mock-closed",
                        quote="This fixture is intentionally closed.",
                        fetched_at=fetched_at,
                        confidence=0.95,
                    )
                ],
                status="open",
                notes="Used to verify freshness excludes expired candidates.",
            ),
        ]
        return SourceResult(source=self.name, hackathons=hackathons)
