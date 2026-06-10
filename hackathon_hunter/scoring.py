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
DEFAULT_SCORING_SETTINGS: dict[str, float | int] = {
    "minimum_evidence_confidence": 0.7,
    "active_buffer_days": 7,
    "fast_lane_active_buffer_days": 1,
    "evidence_max_age_days": 14,
    "stale_evidence_penalty": 0.7,
}
DEFAULT_SCORING_CONSTANTS: dict[str, Any] = {
    "prize": {
        # Human-reviewed market anchor. Update when the radar's active prize market shifts.
        "prize_normalization_usd": 70000,
        "non_cash_score": 0.10,
        "unknown_score": 0.25,
        "base_score": 0.45,
        "amount_weight": 0.55,
    },
    "format": {
        "online": 1.0,
        "hybrid": 0.7,
        "in_person": 0.0,
        "unknown": 0.25,
    },
    "deadline": {
        "tiers": [
            {"min_days": 30, "score": 1.0},
            {"min_days": 21, "score": 0.85},
            {"min_days": 14, "score": 0.70},
            {"min_days": 7, "score": 0.45},
        ],
        "positive_score": 0.15,
        "expired_score": 0.0,
    },
    "ai_policy": {
        "allowed": 1.0,
        "restricted": 0.45,
        "default": 0.0,
    },
    "sponsor_fit": {
        "required_api_score": 0.9,
        "sponsor_score": 0.7,
        "track_keyword_score": 0.55,
        "default_score": 0.25,
    },
    "past_winner": {
        "known_platforms": ["devpost", "lablab", "dorahacks"],
        "known_platform_score": 0.70,
        "default_score": 0.45,
    },
    "low_submission_estimate": {
        "sponsor_or_track_score": 0.65,
        "high_prize_threshold_usd": 50000,
        "high_prize_score": 0.45,
        "default_score": 0.50,
    },
    "taiwan_eligibility": {
        "eligible_score": 1.0,
        "blocked_score": 0.0,
        "student_only_score": 0.10,
        "restricted_uncertain_score": 0.25,
        "restricted_taiwan_included_score": 0.80,
        "restricted_default_score": 0.35,
        "unknown_score": 0.55,
        "block_threshold": 0.05,
    },
    "competition_pressure": {
        "count_tiers": [
            {"min_count": 10000, "score": 0.15},
            {"min_count": 5000, "score": 0.25},
            {"min_count": 2500, "score": 0.35},
            {"min_count": 1000, "score": 0.50},
            {"min_count": 500, "score": 0.65},
            {"min_count": 100, "score": 0.80},
        ],
        "low_count_score": 0.90,
        "very_high_prize_threshold_usd": 250000,
        "very_high_prize_score": 0.25,
        "high_prize_threshold_usd": 70000,
        "high_prize_score": 0.45,
        "devpost_default_score": 0.55,
        "default_score": 0.60,
    },
    "submission_complexity": {
        "base_score": 0.88,
        "requirement_penalties": {
            "github_repo": 0.03,
            "demo_url": 0.06,
            "video": 0.08,
            "deck": 0.06,
            "social_post": 0.04,
            "public_profile": 0.02,
            "deploy_proof": 0.10,
            "architecture_diagram": 0.07,
        },
        "required_api_base_penalty": 0.05,
        "required_api_extra_penalty": 0.02,
        "required_api_max_penalty": 0.12,
        "keyword_penalties": {
            "revenue": 0.12,
            "expenses": 0.08,
            "hardware": 0.10,
            "live presentation": 0.08,
            "accelerator": 0.04,
            "deployment proof": 0.08,
        },
    },
    "fast_lane": {
        "disabled_score": 0.50,
        "expired_score": 0.0,
        "complexity_weight": 0.45,
        "sponsor_fit_weight": 0.25,
        "format_weight": 0.20,
        "ai_policy_weight": 0.10,
        "three_day_threshold": 3,
        "three_day_penalty": 0.15,
        "seven_day_threshold": 7,
        "fourteen_day_threshold": 14,
        "medium_buffer_base": 0.70,
        "medium_buffer_weight": 0.20,
        "default_score": 0.55,
    },
    "domain_fit": {
        "default_domains": ["ai", "agent", "cloud", "voice", "developer"],
        "base_score": 0.35,
        "hit_weight": 0.15,
        "max_hits": 4,
    },
    "evidence_quality": {
        "key_fields": ["deadline", "prize_total_usd", "cash_prize", "eligibility", "required_apis"],
    },
    "delivery_risk": {
        "deadline_tiers": [
            {"lt_days": 7, "risk": 0.40},
            {"lt_days": 14, "risk": 0.25},
            {"lt_days": 21, "risk": 0.12},
        ],
        "ai_policy_unknown_or_forbidden": 0.35,
        "region_restricted": 0.25,
        "taiwan_gate_threshold": 0.25,
        "taiwan_gate_risk": 0.20,
        "evidence_quality_threshold": 0.7,
        "low_evidence_quality": 0.18,
        "competition_pressure_threshold": 0.25,
        "competition_pressure_risk": 0.08,
        "submission_complexity_threshold": 0.45,
        "submission_complexity_risk": 0.08,
        "extra_requirement_risk": 0.04,
        "overall_risk_weight": 0.10,
    },
    "score_dimensions": {
        "roi_prize_weight": 0.75,
        "roi_submission_weight": 0.25,
        "feasibility_deadline_weight": 0.45,
        "feasibility_format_weight": 0.30,
        "feasibility_submission_weight": 0.15,
        "feasibility_risk_weight": 0.10,
        "strategic_sponsor_weight": 0.45,
        "strategic_domain_weight": 0.35,
        "strategic_ai_policy_weight": 0.20,
    },
}
STALE_EVIDENCE_FIELDS = [
    "deadline",
    "eligibility",
    "ai_policy",
    "prize_total_usd",
    "prize_breakdown",
    "cash_prize",
]


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_scoring_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else project_path("config", "scoring.yaml")
    if not config_path.exists():
        return {
            "weights": DEFAULT_WEIGHTS,
            "constants": DEFAULT_SCORING_CONSTANTS,
            **DEFAULT_SCORING_SETTINGS,
        }
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    constants = _deep_merge(
        DEFAULT_SCORING_CONSTANTS,
        payload.get("constants", {}) or {},
    )
    return {
        "weights": payload.get("weights", {}),
        "constants": constants,
        **DEFAULT_SCORING_SETTINGS,
        **{key: value for key, value in payload.items() if key != "constants"},
    }


def load_scoring_constants(path: str | Path | None = None) -> dict[str, Any]:
    return load_scoring_config(path).get("constants", DEFAULT_SCORING_CONSTANTS)


def _active_constants(constants: dict[str, Any] | None) -> dict[str, Any]:
    return constants or load_scoring_constants()


def load_weights(path: str | Path | None = None) -> dict[str, float]:
    payload = load_scoring_config(path)
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


def _score_prize(hackathon: Hackathon, constants: dict[str, Any] | None = None) -> float:
    config = _active_constants(constants)["prize"]
    if hackathon.cash_prize is False:
        return float(config["non_cash_score"])
    if hackathon.cash_prize is None or hackathon.prize_total_usd is None:
        return float(config["unknown_score"])
    amount = max(0.0, hackathon.prize_total_usd)
    return clamp(
        float(config["base_score"])
        + (amount / float(config["prize_normalization_usd"]))
        * float(config["amount_weight"])
    )


def _score_format(hackathon: Hackathon, constants: dict[str, Any] | None = None) -> float:
    config = _active_constants(constants)["format"]
    normalized = str(hackathon.format).lower()
    return float(config.get(normalized, config["unknown"]))


def _score_deadline(
    hackathon: Hackathon,
    now: datetime | None = None,
    constants: dict[str, Any] | None = None,
) -> float:
    config = _active_constants(constants)["deadline"]
    days = days_until_deadline(hackathon, now)
    for tier in config["tiers"]:
        if days >= tier["min_days"]:
            return float(tier["score"])
    if days > 0:
        return float(config["positive_score"])
    return float(config["expired_score"])


def _score_ai_policy(hackathon: Hackathon, constants: dict[str, Any] | None = None) -> float:
    config = _active_constants(constants)["ai_policy"]
    policy = str(hackathon.ai_policy).lower()
    return float(config.get(policy, config["default"]))


def _score_sponsor_fit(hackathon: Hackathon, constants: dict[str, Any] | None = None) -> float:
    config = _active_constants(constants)["sponsor_fit"]
    if hackathon.required_apis:
        return float(config["required_api_score"])
    if hackathon.sponsors:
        return float(config["sponsor_score"])
    track_text = " ".join(hackathon.tracks).lower()
    if "api" in track_text or "agent" in track_text or "ai" in track_text:
        return float(config["track_keyword_score"])
    return float(config["default_score"])


def _score_past_winner_analyzable(
    hackathon: Hackathon,
    constants: dict[str, Any] | None = None,
) -> float:
    config = _active_constants(constants)["past_winner"]
    platform = hackathon.platform.lower()
    if platform in set(config["known_platforms"]):
        return float(config["known_platform_score"])
    return float(config["default_score"])


def _score_low_submission_estimate(
    hackathon: Hackathon,
    constants: dict[str, Any] | None = None,
) -> float:
    config = _active_constants(constants)["low_submission_estimate"]
    text = " ".join(
        [*hackathon.tracks, *(hackathon.required_apis or []), hackathon.notes or ""]
    ).lower()
    if "sponsor" in text or "required" in text or "track" in text:
        return float(config["sponsor_or_track_score"])
    if hackathon.prize_total_usd and hackathon.prize_total_usd > float(
        config["high_prize_threshold_usd"]
    ):
        return float(config["high_prize_score"])
    return float(config["default_score"])


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
    hackathon: Hackathon,
    profile: dict[str, Any] | None,
    constants: dict[str, Any] | None = None,
) -> float:
    config = _active_constants(constants)["taiwan_eligibility"]
    country = _profile_country(profile)
    eligibility = hackathon.eligibility
    notes = (eligibility.notes or "").lower()
    allowed = _normalized_regions(eligibility.allowed_regions)
    excluded = _normalized_regions(eligibility.excluded_regions)
    country_aliases = {country}
    if country == "taiwan":
        country_aliases.update({"tw", "twn", "taiwan, province of china", "republic of china"})

    if eligibility.taiwan_eligible is True:
        return float(config["eligible_score"])
    if eligibility.taiwan_eligible is False:
        return float(config["blocked_score"])
    if country_aliases & excluded:
        return float(config["blocked_score"])
    if allowed:
        if country_aliases & allowed:
            return float(config["eligible_score"])
        if {"worldwide", "global", "all countries", "any country"} & allowed:
            return float(config["eligible_score"])
        return float(config["blocked_score"])
    if eligibility.student_only:
        return float(config["student_only_score"])
    if _text_mentions_any(notes, {"india only", "students across india", "u.s. only", "us only"}):
        return float(config["blocked_score"])
    if _text_mentions_any(notes, {"any country", "all countries", "worldwide", "global"}):
        return float(config["eligible_score"])
    if eligibility.region_restricted is False:
        return float(config["eligible_score"])
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
            return float(config["restricted_uncertain_score"])
        if "taiwan" in notes and _text_mentions_any(notes, {"allowed", "eligible", "included"}):
            return float(config["restricted_taiwan_included_score"])
        return float(config["restricted_default_score"])
    return float(config["unknown_score"])


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


def _score_competition_pressure(
    hackathon: Hackathon,
    constants: dict[str, Any] | None = None,
) -> float:
    config = _active_constants(constants)["competition_pressure"]
    count = _extract_competition_count(hackathon)
    if count is not None:
        for tier in config["count_tiers"]:
            if count >= tier["min_count"]:
                return float(tier["score"])
        return float(config["low_count_score"])
    if hackathon.prize_total_usd and hackathon.prize_total_usd >= float(
        config["very_high_prize_threshold_usd"]
    ):
        return float(config["very_high_prize_score"])
    if hackathon.prize_total_usd and hackathon.prize_total_usd >= float(
        config["high_prize_threshold_usd"]
    ):
        return float(config["high_prize_score"])
    if str(hackathon.platform).lower() == "devpost":
        return float(config["devpost_default_score"])
    return float(config["default_score"])


def _score_submission_complexity(
    hackathon: Hackathon,
    constants: dict[str, Any] | None = None,
) -> float:
    config = _active_constants(constants)["submission_complexity"]
    requirements = hackathon.submission_requirements
    score = float(config["base_score"])
    for field, penalty in config["requirement_penalties"].items():
        if getattr(requirements, field):
            score -= float(penalty)
    required_api_count = len(hackathon.required_apis)
    if required_api_count:
        score -= min(
            float(config["required_api_max_penalty"]),
            float(config["required_api_base_penalty"])
            + (required_api_count - 1) * float(config["required_api_extra_penalty"]),
        )
    notes = " ".join(
        [
            requirements.notes or "",
            hackathon.notes or "",
            hackathon.prize_breakdown or "",
        ]
    ).lower()
    for keyword, penalty in config["keyword_penalties"].items():
        if keyword in notes:
            score -= float(penalty)
    return clamp(score)


def _score_fast_lane_mode(
    hackathon: Hackathon,
    now: datetime | None = None,
    *,
    enabled: bool = False,
    constants: dict[str, Any] | None = None,
) -> float:
    config = _active_constants(constants)["fast_lane"]
    if not enabled:
        return float(config["disabled_score"])
    days = days_until_deadline(hackathon, now)
    if days <= 0:
        return float(config["expired_score"])
    complexity = _score_submission_complexity(hackathon, constants)
    shippability = clamp(
        (complexity * float(config["complexity_weight"]))
        + (_score_sponsor_fit(hackathon, constants) * float(config["sponsor_fit_weight"]))
        + (_score_format(hackathon, constants) * float(config["format_weight"]))
        + (_score_ai_policy(hackathon, constants) * float(config["ai_policy_weight"]))
    )
    if days <= float(config["three_day_threshold"]):
        return clamp(shippability - float(config["three_day_penalty"]))
    if days <= float(config["seven_day_threshold"]):
        return shippability
    if days <= float(config["fourteen_day_threshold"]):
        return clamp(
            float(config["medium_buffer_base"])
            + shippability * float(config["medium_buffer_weight"])
        )
    return float(config["default_score"])


def _score_user_domain_fit(
    hackathon: Hackathon,
    profile: dict[str, Any] | None,
    constants: dict[str, Any] | None = None,
) -> float:
    config = _active_constants(constants)["domain_fit"]
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
        domains = [str(item).lower() for item in config["default_domains"]]
    hits = sum(1 for domain in domains if any(token in text for token in domain.split()))
    return clamp(
        float(config["base_score"])
        + min(hits, int(config["max_hits"])) * float(config["hit_weight"])
    )


def evidence_quality(
    hackathon: Hackathon,
    now: datetime | None = None,
    *,
    max_age_days: int | None = None,
    stale_penalty: float | None = None,
    constants: dict[str, Any] | None = None,
) -> float:
    config = _active_constants(constants)["evidence_quality"]
    key_fields = config["key_fields"]
    confidences: list[float] = []
    for field in key_fields:
        confidence = hackathon.evidence_confidence(field)
        confidences.append(confidence if confidence is not None else 0.0)
    if hackathon.source_evidence:
        confidences.append(hackathon.average_evidence_confidence())
    score = clamp(sum(confidences) / len(confidences))
    if now is None:
        return score
    scoring_config = load_scoring_config()
    effective_max_age = int(
        max_age_days if max_age_days is not None else scoring_config["evidence_max_age_days"]
    )
    effective_penalty = float(
        stale_penalty if stale_penalty is not None else scoring_config["stale_evidence_penalty"]
    )
    stale_fields = hackathon.stale_evidence_fields(
        now,
        max_age_days=effective_max_age,
        fields=STALE_EVIDENCE_FIELDS,
    )
    if stale_fields:
        score *= effective_penalty
    return clamp(score)


def delivery_risk(
    hackathon: Hackathon,
    now: datetime | None = None,
    profile: dict[str, Any] | None = None,
    constants: dict[str, Any] | None = None,
) -> float:
    config = _active_constants(constants)["delivery_risk"]
    risk = 0.0
    days = days_until_deadline(hackathon, now)
    for tier in config["deadline_tiers"]:
        if days < tier["lt_days"]:
            risk += float(tier["risk"])
            break
    if str(hackathon.ai_policy).lower() in {"unknown", "forbidden"}:
        risk += float(config["ai_policy_unknown_or_forbidden"])
    if hackathon.eligibility.region_restricted:
        risk += float(config["region_restricted"])
    if _score_taiwan_eligibility_gate(hackathon, profile, constants) <= float(
        config["taiwan_gate_threshold"]
    ):
        risk += float(config["taiwan_gate_risk"])
    if evidence_quality(hackathon, now, constants=constants) < float(
        config["evidence_quality_threshold"]
    ):
        risk += float(config["low_evidence_quality"])
    if _score_competition_pressure(hackathon, constants) <= float(
        config["competition_pressure_threshold"]
    ):
        risk += float(config["competition_pressure_risk"])
    if _score_submission_complexity(hackathon, constants) <= float(
        config["submission_complexity_threshold"]
    ):
        risk += float(config["submission_complexity_risk"])
    requirements = hackathon.submission_requirements
    extra_requirements = [
        requirements.video,
        requirements.deploy_proof,
        requirements.architecture_diagram,
    ]
    for required in extra_requirements:
        if required:
            risk += float(config["extra_requirement_risk"])
    return clamp(risk)


def score_hackathon(
    hackathon: Hackathon,
    weights: dict[str, float] | None = None,
    profile: dict[str, Any] | None = None,
    now: datetime | None = None,
    fast_lane_mode: bool = False,
    constants: dict[str, Any] | None = None,
) -> ScoreBreakdown:
    active_weights = weights or load_weights()
    scoring_config = load_scoring_config()
    active_constants = constants or scoring_config["constants"]
    evidence_score = evidence_quality(
        hackathon,
        now,
        max_age_days=int(scoring_config["evidence_max_age_days"]),
        stale_penalty=float(scoring_config["stale_evidence_penalty"]),
        constants=active_constants,
    )
    trace = {
        "prize_cash": _score_prize(hackathon, active_constants),
        "online_allowed": _score_format(hackathon, active_constants),
        "deadline_buffer": _score_deadline(hackathon, now, active_constants),
        "ai_policy_clear": _score_ai_policy(hackathon, active_constants),
        "sponsor_api_fit": _score_sponsor_fit(hackathon, active_constants),
        "past_winner_analyzable": _score_past_winner_analyzable(hackathon, active_constants),
        "low_submission_estimate": _score_low_submission_estimate(hackathon, active_constants),
        "user_domain_fit": _score_user_domain_fit(hackathon, profile, active_constants),
        "taiwan_eligibility_gate": _score_taiwan_eligibility_gate(
            hackathon,
            profile,
            active_constants,
        ),
        "competition_pressure_score": _score_competition_pressure(hackathon, active_constants),
        "submission_complexity_score": _score_submission_complexity(
            hackathon,
            active_constants,
        ),
        "fast_lane_mode": _score_fast_lane_mode(
            hackathon,
            now,
            enabled=fast_lane_mode or _profile_fast_lane_enabled(profile),
            constants=active_constants,
        ),
    }
    weighted = sum(trace[key] * active_weights.get(key, 0.0) for key in trace)
    risk = delivery_risk(hackathon, now, profile, active_constants)
    dimension_config = active_constants["score_dimensions"]
    risk_config = active_constants["delivery_risk"]
    overall = clamp(weighted * (1 - risk * float(risk_config["overall_risk_weight"])))
    roi = clamp(
        (trace["prize_cash"] * float(dimension_config["roi_prize_weight"]))
        + (
            trace["low_submission_estimate"]
            * float(dimension_config["roi_submission_weight"])
        )
    )
    feasibility = clamp(
        (trace["deadline_buffer"] * float(dimension_config["feasibility_deadline_weight"]))
        + (trace["online_allowed"] * float(dimension_config["feasibility_format_weight"]))
        + (
            trace["submission_complexity_score"]
            * float(dimension_config["feasibility_submission_weight"])
        )
        + ((1 - risk) * float(dimension_config["feasibility_risk_weight"]))
    )
    strategic = clamp(
        (trace["sponsor_api_fit"] * float(dimension_config["strategic_sponsor_weight"]))
        + (trace["user_domain_fit"] * float(dimension_config["strategic_domain_weight"]))
        + (
            trace["ai_policy_clear"]
            * float(dimension_config["strategic_ai_policy_weight"])
        )
    )
    reason = (
        f"{hackathon.name}: overall {overall:.2f}; "
        f"cash/prize {trace['prize_cash']:.2f}, sponsor/API fit "
        f"{trace['sponsor_api_fit']:.2f}, Taiwan eligibility "
        f"{trace['taiwan_eligibility_gate']:.2f}, competition "
        f"{trace['competition_pressure_score']:.2f}, submission complexity "
        f"{trace['submission_complexity_score']:.2f}, evidence "
        f"{evidence_score:.2f}, delivery risk {risk:.2f}."
    )
    return ScoreBreakdown(
        hackathon_id=hackathon.id,
        roi_score=roi,
        feasibility_score=feasibility,
        strategic_fit_score=strategic,
        evidence_quality_score=evidence_score,
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
    constants: dict[str, Any] | None = None,
) -> tuple[list[tuple[Hackathon, ScoreBreakdown]], list[tuple[Hackathon, str]]]:
    checked_at = now or utcish_now()
    active_constants = constants or load_scoring_constants()
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
        taiwan_gate = _score_taiwan_eligibility_gate(hackathon, profile, active_constants)
        if taiwan_gate <= float(active_constants["taiwan_eligibility"]["block_threshold"]):
            rejected.append((hackathon, "taiwan_eligibility_gate blocks participation"))
            continue
        score = score_hackathon(
            hackathon,
            weights=weights,
            profile=profile,
            now=checked_at,
            fast_lane_mode=lane_enabled,
            constants=active_constants,
        )
        ranked.append((hackathon, score))
    ranked.sort(key=lambda item: item[1].overall_score, reverse=True)
    return ranked, rejected
