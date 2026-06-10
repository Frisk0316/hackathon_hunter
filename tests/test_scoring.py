from datetime import datetime, timedelta, timezone
from pathlib import Path

from hackathon_hunter.models import Eligibility, Evidence, Hackathon, SubmissionRequirements
from hackathon_hunter.scoring import rank_hackathons, score_hackathon
from hackathon_hunter.storage import load_hackathons

FIXED_NOW = datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc)


def _hackathon(**overrides) -> Hackathon:
    payload = {
        "id": "synthetic",
        "name": "Synthetic Agent Hack",
        "platform": "devpost",
        "url": "https://example.com",
        "deadline": FIXED_NOW + timedelta(days=21),
        "deadline_timezone": "UTC",
        "format": "online",
        "prize_total_usd": 50000,
        "cash_prize": True,
        "tracks": ["AI Agents"],
        "sponsors": ["Example Cloud"],
        "required_apis": ["Example API"],
        "ai_policy": "allowed",
        "eligibility": Eligibility(region_restricted=False),
        "submission_requirements": SubmissionRequirements(
            github_repo=True,
            demo_url=True,
            video=True,
        ),
        "status": "open",
    }
    payload.update(overrides)
    return Hackathon(**payload)


def test_current_fixture_ranks_candidates_and_penalizes_qwen_region_uncertainty() -> None:
    hackathons = load_hackathons(Path("data/processed/hackathons_20260610.json"))
    ranked, rejected = rank_hackathons(hackathons, now=FIXED_NOW)
    by_id = {item.id: score for item, score in ranked}

    assert ranked
    assert by_id["qwen-cloud-global-ai-2026"].trace["taiwan_eligibility_gate"] == 0.25
    assert by_id["qwen-cloud-global-ai-2026"].delivery_risk_score > 0.4
    assert any(item.id == "munichtech-innovation-2026" for item, _ in rejected)


def test_score_trace_contains_configured_dimensions() -> None:
    hackathon = load_hackathons(Path("data/processed/hackathons_20260610.json"))[0]
    score = score_hackathon(hackathon, now=FIXED_NOW)

    assert 0 <= score.overall_score <= 1
    assert {
        "prize_cash",
        "online_allowed",
        "deadline_buffer",
        "ai_policy_clear",
        "sponsor_api_fit",
        "past_winner_analyzable",
        "low_submission_estimate",
        "user_domain_fit",
        "taiwan_eligibility_gate",
        "competition_pressure_score",
        "submission_complexity_score",
        "fast_lane_mode",
    } == set(score.trace)


def test_taiwan_eligibility_gate_rejects_known_incompatible_regions() -> None:
    india_only = _hackathon(
        eligibility=Eligibility(
            region_restricted=True,
            allowed_regions=["India"],
            notes="students across India only",
        )
    )
    ranked, rejected = rank_hackathons([india_only], now=FIXED_NOW)

    assert not ranked
    assert rejected[0][1] == "taiwan_eligibility_gate blocks participation"


def test_fast_lane_mode_includes_near_deadline_candidates() -> None:
    near_deadline = _hackathon(deadline=FIXED_NOW + timedelta(days=3))

    normal_ranked, normal_rejected = rank_hackathons([near_deadline], now=FIXED_NOW)
    fast_ranked, fast_rejected = rank_hackathons(
        [near_deadline],
        now=FIXED_NOW,
        fast_lane_mode=True,
    )

    assert not normal_ranked
    assert normal_rejected
    assert fast_ranked
    assert not fast_rejected
    assert fast_ranked[0][1].trace["fast_lane_mode"] > 0.0

    profile_ranked, _ = rank_hackathons(
        [near_deadline],
        now=FIXED_NOW,
        profile={"constraints": {"fast_lane_mode": True}},
    )
    assert profile_ranked


def test_competition_pressure_and_submission_complexity_affect_trace() -> None:
    crowded = _hackathon(registrations_count=14088)
    niche = _hackathon(
        id="niche",
        registrations_count=172,
        submission_requirements=SubmissionRequirements(github_repo=True),
    )

    crowded_score = score_hackathon(crowded, now=FIXED_NOW)
    niche_score = score_hackathon(niche, now=FIXED_NOW)

    assert crowded_score.trace["competition_pressure_score"] < niche_score.trace[
        "competition_pressure_score"
    ]
    assert crowded_score.trace["submission_complexity_score"] < niche_score.trace[
        "submission_complexity_score"
    ]


def test_stale_evidence_penalizes_evidence_quality_score() -> None:
    fresh_evidence = [
        Evidence(
            field=field,
            url="https://example.com",
            quote="fixture",
            fetched_at=FIXED_NOW,
            confidence=0.9,
        )
        for field in ["deadline", "prize_total_usd", "cash_prize", "eligibility", "required_apis"]
    ]
    stale_evidence = [
        item.model_copy(update={"fetched_at": FIXED_NOW - timedelta(days=30)})
        for item in fresh_evidence
    ]

    fresh_score = score_hackathon(_hackathon(source_evidence=fresh_evidence), now=FIXED_NOW)
    stale_score = score_hackathon(_hackathon(source_evidence=stale_evidence), now=FIXED_NOW)

    assert stale_score.evidence_quality_score < fresh_score.evidence_quality_score
