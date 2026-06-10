from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from hackathon_hunter.models import Hackathon, ScoreBreakdown
from hackathon_hunter.storage import project_path, utcish_now

DEFAULT_WEIGHTS: dict[str, float] = {
    "prize_cash": 0.11,
    "online_allowed": 0.09,
    "deadline_buffer": 0.10,
    "ai_policy_clear": 0.10,
    "sponsor_api_fit": 0.11,
    "past_winner_analyzable": 0.05,
    "low_submission_estimate": 0.03,
    "user_domain_fit": 0.07,
    "taiwan_eligibility_gate": 0.14,
    "competition_pressure_score": 0.07,
    "submission_complexity_score": 0.08,
    "fast_lane_mode": 0.05,
}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_weights(path: str | Path | None = None) -> dict[str, float]:
    config_path = Path(path) if path else project_path("config", "scoring.yaml")
    if not config_path.exists():
        return DEFAULT_WEIGHTS
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    weights = payload.get("weights", {})
    merged = {**DEFAULT_WEIGHTS, **weights}
    total = sum(merged.values())
    if total <= 0:
        return DEFAULT_WEIGHTS
    return {key: value / total for key, value in merged.items()}


def load_profile(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return {}
    profile_path = Path(path)
    if not profile_path.exists():
        return {}
    return yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}


def days_until_deadline(hackathon: Hackathon, now: datetime | None = None) -> float:
    checked_at = now or utcish_now()
    return (hackathon.deadline - checked_at).total_seconds() / 86400


def _score_prize(hackathon: Hackathon) -> float:
    if hackathon.cash_prize is False:
        return 0.10
    if hackathon.cash_prize is None or hackathon.prize_total_usd is None:
        return 0.25
    amount = max(0.0, hackathon.prize_total_usd)
    return clamp(0.45 + (amount / 70000) * 0.55)


def _score_format(hackathon: Hackathon) -> float:
    normalized = str(hackathon.format).lower()
    if normalized == "online":
        return 1.0
    if normalized == "hybrid":
        return 0.7
    if normalized == "in_person":
        return 0.0
    return 0.25


def _score_deadline(hackathon: Hackathon, now: datetime | None = None) -> float:
    days = days_until_deadline(hackathon, now)
    if days >= 30:
        return 1.0
    if days >= 21:
        return 0.85
    if days >= 14:
        return 0.70
    if days >= 7:
        return 0.45
    if days > 0:
        return 0.15
    return 0.0


def _score_ai_policy(hackathon: Hackathon) -> float:
    policy = str(hackathon.ai_policy).lower()
    if policy == "allowed":
        return 1.0
    if policy == "restricted":
        return 0.45
    return 0.0


def _score_sponsor_fit(hackathon: Hackathon) -> float:
    if hackathon.required_apis:
        return 0.9
    if hackathon.sponsors:
        return 0.7
    track_text = " ".join(hackathon.tracks).lower()
    if "api" in track_text or "agent" in track_text or "ai" in track_text:
        return 0.55
    return 0.25


def _score_past_winner_analyzable(hackathon: Hackathon) -> float:
    platform = hackathon.platform.lower()
    if platform in {"devpost", "lablab", "dorahacks"}:
        return 0.70
    return 0.45


def _score_low_submission_estimate(hackathon: Hackathon) -> float:
    text = " ".join(
        [*hackathon.tracks, *(hackathon.required_apis or []), hackathon.notes or ""]
    ).lower()
    if "sponsor" in text or "required" in text or "track" in text:
        return 0.65
    if hackathon.prize_total_usd and hackathon.prize_total_usd > 50000:
        return 0.45
    return 0.50


def _normalized_regions(values: list[str] | None) -> set[str]:
    return {str(value).strip().lower() for value in values or [] if str(value).strip()}


def _profile_country(profile: dict[str, Any] | None) -> str:
    active_profile = profile or {}
    return str(
        active_profile.get("country") or active_profile.get("location") or "Taiwan"
    ).strip().lower()


def _profile_fast_lane_enabled(profile: dict[str, Any] | None) -> bool:
    active_profile = profile or {}
    constraints = active_profile.get("constraints") or {}
    return bool(active_profile.get("fast_lane_mode") or constraints.get("fast_lane_mode"))


def _text_mentions_any(text: str, phrases: set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _score_taiwan_eligibility_gate(
    hackathon: Hackathon, profile: dict[str, Any] | None
) -> float:
    country = _profile_country(profile)
    eligibility = hackathon.eligibility
    notes = (eligibility.notes or "").lower()
    allowed = _normalized_regions(eligibility.allowed_regions)
    excluded = _normalized_regions(eligibility.excluded_regions)
    country_aliases = {country}
    if country == "taiwan":
        country_aliases.update({"tw", "twn", "taiwan, province of china", "republic of china"})

    if eligibility.taiwan_eligible is True:
        return 1.0
    if eligibility.taiwan_eligible is False:
        return 0.0
    if country_aliases & excluded:
        return 0.0
    if allowed:
        if country_aliases & allowed:
            return 1.0
        if {"worldwide", "global", "all countries", "any country"} & allowed:
            return 1.0
        return 0.0
    if eligibility.student_only:
        return 0.10
    if _text_mentions_any(notes, {"india only", "students across india", "u.s. only", "us only"}):
        return 0.0
    if _text_mentions_any(notes, {"any country", "all countries", "worldwide", "global"}):
        return 1.0
    if eligibility.region_restricted is False:
        return 1.0
    if eligibility.region_restricted is True:
        if _text_mentions_any(
            notes,
            {
                "must verify",
                "depends",
                "not explicitly",
                "unsupported",
                "restricted jurisdictions",
            },
        ):
            return 0.25
        if "taiwan" in notes and _text_mentions_any(notes, {"allowed", "eligible", "included"}):
            return 0.80
        return 0.35
    return 0.55


def _extract_competition_count(hackathon: Hackathon) -> int | None:
    direct_fields = [
        "participants_count",
        "participant_count",
        "registrations_count",
        "registration_count",
        "submissions_count",
        "submission_count",
    ]
    for field in direct_fields:
        raw_value = getattr(hackathon, field, None)
        if raw_value is None:
            continue
        try:
            return max(0, int(raw_value))
        except (TypeError, ValueError):
            continue

    text = " ".join(
        [
            hackathon.notes or "",
            hackathon.prize_breakdown or "",
            " ".join(hackathon.tracks),
        ]
    )
    patterns = [
        r"([\d,]+)\s+(?:participants|registrations|registered|submissions)",
        r"(?:participants|registrations|registered|submissions)\D+([\d,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def _score_competition_pressure(hackathon: Hackathon) -> float:
    count = _extract_competition_count(hackathon)
    if count is not None:
        if count >= 10000:
            return 0.15
        if count >= 5000:
            return 0.25
        if count >= 2500:
            return 0.35
        if count >= 1000:
            return 0.50
        if count >= 500:
            return 0.65
        if count >= 100:
            return 0.80
        return 0.90
    if hackathon.prize_total_usd and hackathon.prize_total_usd >= 250000:
        return 0.25
    if hackathon.prize_total_usd and hackathon.prize_total_usd >= 70000:
        return 0.45
    if str(hackathon.platform).lower() == "devpost":
        return 0.55
    return 0.60


def _score_submission_complexity(hackathon: Hackathon) -> float:
    requirements = hackathon.submission_requirements
    penalties = {
        "github_repo": 0.03,
        "demo_url": 0.06,
        "video": 0.08,
        "deck": 0.06,
        "social_post": 0.04,
        "public_profile": 0.02,
        "deploy_proof": 0.10,
        "architecture_diagram": 0.07,
    }
    score = 0.88
    for field, penalty in penalties.items():
        if getattr(requirements, field):
            score -= penalty
    required_api_count = len(hackathon.required_apis)
    if required_api_count:
        score -= min(0.12, 0.05 + (required_api_count - 1) * 0.02)
    notes = " ".join(
        [
            requirements.notes or "",
            hackathon.notes or "",
            hackathon.prize_breakdown or "",
        ]
    ).lower()
    for keyword, penalty in {
        "revenue": 0.12,
        "expenses": 0.08,
        "hardware": 0.10,
        "live presentation": 0.08,
        "accelerator": 0.04,
        "deployment proof": 0.08,
    }.items():
        if keyword in notes:
            score -= penalty
    return clamp(score)


def _score_fast_lane_mode(
    hackathon: Hackathon,
    now: datetime | None = None,
    *,
    enabled: bool = False,
) -> float:
    if not enabled:
        return 0.50
    days = days_until_deadline(hackathon, now)
    if days <= 0:
        return 0.0
    complexity = _score_submission_complexity(hackathon)
    shippability = clamp(
        (complexity * 0.45)
        + (_score_sponsor_fit(hackathon) * 0.25)
        + (_score_format(hackathon) * 0.20)
        + (_score_ai_policy(hackathon) * 0.10)
    )
    if days <= 3:
        return clamp(shippability - 0.15)
    if days <= 7:
        return shippability
    if days <= 14:
        return clamp(0.70 + shippability * 0.20)
    return 0.55


def _score_user_domain_fit(hackathon: Hackathon, profile: dict[str, Any] | None) -> float:
    domains = [str(item).lower() for item in (profile or {}).get("domains", [])]
    text = " ".join(
        [
            hackathon.name,
            *hackathon.tracks,
            *hackathon.required_apis,
            *hackathon.sponsors,
            hackathon.notes or "",
        ]
    ).lower()
    if not domains:
        domains = ["ai", "agent", "cloud", "voice", "developer"]
    hits = sum(1 for domain in domains if any(token in text for token in domain.split()))
    return clamp(0.35 + min(hits, 4) * 0.15)


def evidence_quality(hackathon: Hackathon) -> float:
    key_fields = ["deadline", "prize_total_usd", "cash_prize", "eligibility", "required_apis"]
    confidences: list[float] = []
    for field in key_fields:
        confidence = hackathon.evidence_confidence(field)
        confidences.append(confidence if confidence is not None else 0.0)
    if hackathon.source_evidence:
        confidences.append(hackathon.average_evidence_confidence())
    return clamp(sum(confidences) / len(confidences))


def delivery_risk(
    hackathon: Hackathon,
    now: datetime | None = None,
    profile: dict[str, Any] | None = None,
) -> float:
    risk = 0.0
    days = days_until_deadline(hackathon, now)
    if days < 7:
        risk += 0.40
    elif days < 14:
        risk += 0.25
    elif days < 21:
        risk += 0.12
    if str(hackathon.ai_policy).lower() in {"unknown", "forbidden"}:
        risk += 0.35
    if hackathon.eligibility.region_restricted:
        risk += 0.25
    if _score_taiwan_eligibility_gate(hackathon, profile) <= 0.25:
        risk += 0.20
    if evidence_quality(hackathon) < 0.7:
        risk += 0.18
    if _score_competition_pressure(hackathon) <= 0.25:
        risk += 0.08
    if _score_submission_complexity(hackathon) <= 0.45:
        risk += 0.08
    requirements = hackathon.submission_requirements
    extra_requirements = [
        requirements.video,
        requirements.deploy_proof,
        requirements.architecture_diagram,
    ]
    for required in extra_requirements:
        if required:
            risk += 0.04
    return clamp(risk)


def score_hackathon(
    hackathon: Hackathon,
    weights: dict[str, float] | None = None,
    profile: dict[str, Any] | None = None,
    now: datetime | None = None,
    fast_lane_mode: bool = False,
) -> ScoreBreakdown:
    active_weights = weights or load_weights()
    trace = {
        "prize_cash": _score_prize(hackathon),
        "online_allowed": _score_format(hackathon),
        "deadline_buffer": _score_deadline(hackathon, now),
        "ai_policy_clear": _score_ai_policy(hackathon),
        "sponsor_api_fit": _score_sponsor_fit(hackathon),
        "past_winner_analyzable": _score_past_winner_analyzable(hackathon),
        "low_submission_estimate": _score_low_submission_estimate(hackathon),
        "user_domain_fit": _score_user_domain_fit(hackathon, profile),
        "taiwan_eligibility_gate": _score_taiwan_eligibility_gate(hackathon, profile),
        "competition_pressure_score": _score_competition_pressure(hackathon),
        "submission_complexity_score": _score_submission_complexity(hackathon),
        "fast_lane_mode": _score_fast_lane_mode(
            hackathon, now, enabled=fast_lane_mode or _profile_fast_lane_enabled(profile)
        ),
    }
    weighted = sum(trace[key] * active_weights.get(key, 0.0) for key in trace)
    risk = delivery_risk(hackathon, now, profile)
    overall = clamp(weighted * (1 - risk * 0.10))
    roi = clamp((trace["prize_cash"] * 0.75) + (trace["low_submission_estimate"] * 0.25))
    feasibility = clamp(
        (trace["deadline_buffer"] * 0.45)
        + (trace["online_allowed"] * 0.30)
        + (trace["submission_complexity_score"] * 0.15)
        + ((1 - risk) * 0.10)
    )
    strategic = clamp(
        (trace["sponsor_api_fit"] * 0.45)
        + (trace["user_domain_fit"] * 0.35)
        + (trace["ai_policy_clear"] * 0.20)
    )
    reason = (
        f"{hackathon.name}: overall {overall:.2f}; "
        f"cash/prize {trace['prize_cash']:.2f}, sponsor/API fit "
        f"{trace['sponsor_api_fit']:.2f}, Taiwan eligibility "
        f"{trace['taiwan_eligibility_gate']:.2f}, competition "
        f"{trace['competition_pressure_score']:.2f}, submission complexity "
        f"{trace['submission_complexity_score']:.2f}, evidence "
        f"{evidence_quality(hackathon):.2f}, delivery risk {risk:.2f}."
    )
    return ScoreBreakdown(
        hackathon_id=hackathon.id,
        roi_score=roi,
        feasibility_score=feasibility,
        strategic_fit_score=strategic,
        evidence_quality_score=evidence_quality(hackathon),
        delivery_risk_score=risk,
        overall_score=overall,
        ranking_reason=reason,
        trace=trace,
    )


def rank_hackathons(
    hackathons: list[Hackathon],
    weights: dict[str, float] | None = None,
    profile: dict[str, Any] | None = None,
    now: datetime | None = None,
    active_buffer_days: int = 7,
    fast_lane_mode: bool = False,
) -> tuple[list[tuple[Hackathon, ScoreBreakdown]], list[tuple[Hackathon, str]]]:
    checked_at = now or utcish_now()
    lane_enabled = fast_lane_mode or _profile_fast_lane_enabled(profile)
    effective_buffer_days = 1 if lane_enabled else active_buffer_days
    ranked: list[tuple[Hackathon, ScoreBreakdown]] = []
    rejected: list[tuple[Hackathon, str]] = []
    for hackathon in hackathons:
        if not hackathon.is_active_candidate(checked_at, effective_buffer_days):
            rejected.append((hackathon, "deadline closed or inside freshness buffer"))
            continue
        if str(hackathon.ai_policy).lower() in {"unknown", "forbidden"}:
            rejected.append((hackathon, f"ai_policy={hackathon.ai_policy} blocks build stage"))
            continue
        taiwan_gate = _score_taiwan_eligibility_gate(hackathon, profile)
        if taiwan_gate <= 0.05:
            rejected.append((hackathon, "taiwan_eligibility_gate blocks participation"))
            continue
        score = score_hackathon(
            hackathon,
            weights=weights,
            profile=profile,
            now=checked_at,
            fast_lane_mode=lane_enabled,
        )
        ranked.append((hackathon, score))
    ranked.sort(key=lambda item: item[1].overall_score, reverse=True)
    return ranked, rejected
