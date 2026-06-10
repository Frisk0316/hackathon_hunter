from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    field: str
    url: str | None = None
    quote: str | None = None
    fetched_at: datetime
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("fetched_at")
    @classmethod
    def fetched_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evidence fetched_at must include timezone")
        return value


class Eligibility(BaseModel):
    model_config = ConfigDict(extra="allow")

    taiwan_eligible: bool | None = None
    allowed_regions: list[str] = Field(default_factory=list)
    excluded_regions: list[str] = Field(default_factory=list)
    region_restricted: bool | None = None
    student_only: bool | None = None
    team_required: bool | None = None
    team_size: str | None = None
    notes: str | None = None


class Prize(BaseModel):
    total_usd: float | None = None
    cash_prize: bool | None = None
    breakdown: str | None = None


class SubmissionRequirements(BaseModel):
    model_config = ConfigDict(extra="allow")

    github_repo: bool | None = None
    demo_url: bool | None = None
    video: bool | None = None
    deck: bool | None = None
    social_post: bool | None = None
    public_profile: bool | None = None
    deploy_proof: bool | None = None
    architecture_diagram: bool | None = None
    notes: str | None = None


HackathonStatus = Literal["open", "upcoming", "closed", "closing", "unknown"]
HackathonFormat = Literal["online", "hybrid", "in_person", "unknown"]
AiPolicy = Literal["allowed", "restricted", "forbidden", "unknown"]


class Hackathon(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    platform: str
    url: str
    rules_url: str | None = None
    deadline: datetime
    deadline_timezone: str
    format: HackathonFormat | str = "unknown"
    prize_total_usd: float | None = None
    cash_prize: bool | None = None
    prize_breakdown: str | None = None
    tracks: list[str] = Field(default_factory=list)
    sponsors: list[str] = Field(default_factory=list)
    required_apis: list[str] = Field(default_factory=list)
    judging_criteria: list[str] = Field(default_factory=list)
    ai_policy: AiPolicy | str = "unknown"
    eligibility: Eligibility = Field(default_factory=Eligibility)
    submission_requirements: SubmissionRequirements = Field(
        default_factory=SubmissionRequirements
    )
    source_evidence: list[Evidence] = Field(default_factory=list)
    status: HackathonStatus | str = "unknown"
    notes: str | None = None

    @field_validator("deadline")
    @classmethod
    def deadline_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("deadline must include timezone")
        return value

    @property
    def prize(self) -> Prize:
        return Prize(
            total_usd=self.prize_total_usd,
            cash_prize=self.cash_prize,
            breakdown=self.prize_breakdown,
        )

    def evidence_for(self, field: str) -> list[Evidence]:
        return [item for item in self.source_evidence if item.field == field]

    def evidence_confidence(self, field: str) -> float | None:
        matches = self.evidence_for(field)
        if not matches:
            return None
        return max(item.confidence for item in matches)

    def stale_evidence_fields(
        self,
        now: datetime,
        max_age_days: int,
        fields: list[str] | None = None,
    ) -> list[str]:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must include timezone")
        cutoff = now - timedelta(days=max_age_days)
        fields_to_check = fields or sorted({item.field for item in self.source_evidence})
        stale_fields: list[str] = []
        for field in fields_to_check:
            evidence = self.evidence_for(field)
            if not evidence:
                continue
            newest = max(item.fetched_at for item in evidence)
            if newest < cutoff:
                stale_fields.append(field)
        return stale_fields

    def average_evidence_confidence(self) -> float:
        if not self.source_evidence:
            return 0.0
        return sum(item.confidence for item in self.source_evidence) / len(self.source_evidence)

    def is_expired(self, now: datetime) -> bool:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must include timezone")
        return self.deadline < now

    def is_active_candidate(self, now: datetime, buffer_days: int = 7) -> bool:
        if self.status == "closed":
            return False
        seconds = (self.deadline - now).total_seconds()
        return seconds >= buffer_days * 24 * 60 * 60


class ScoreBreakdown(BaseModel):
    hackathon_id: str
    roi_score: float = Field(ge=0.0, le=1.0)
    feasibility_score: float = Field(ge=0.0, le=1.0)
    strategic_fit_score: float = Field(ge=0.0, le=1.0)
    evidence_quality_score: float = Field(ge=0.0, le=1.0)
    delivery_risk_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    ranking_reason: str
    trace: dict[str, Any] = Field(default_factory=dict)


class RulesCheckResult(BaseModel):
    hackathon_id: str
    eligible: bool
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    submission_requirements: SubmissionRequirements = Field(
        default_factory=SubmissionRequirements
    )
    human_review_required: bool = True


class ProjectIdea(BaseModel):
    id: str
    hackathon_id: str
    name: str
    tagline: str
    problem: str
    target_user: str
    why_now: str
    sponsor_api_usage: str
    mvp_scope: list[str]
    non_goals: list[str] = Field(default_factory=list)
    demo_flow: str
    tech_stack: list[str]
    risks: list[str] = Field(default_factory=list)
    estimated_build_hours: float
    feasibility_score: float = Field(ge=0.0, le=1.0)
    judging_fit_score: float = Field(ge=0.0, le=1.0)
    differentiation_score: float = Field(ge=0.0, le=1.0)
    human_decision_required: bool = True


class RunLog(BaseModel):
    run_id: str
    workflow: str
    started_at: datetime
    completed_at: datetime | None = None
    status: Literal["started", "succeeded", "failed"] = "started"
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class ResultRecord(BaseModel):
    hackathon_id: str
    project_slug: str | None = None
    hours_spent: float | None = None
    api_cost_usd: float | None = None
    infra_cost_usd: float | None = None
    submitted: bool = False
    outcome: Literal["submitted", "finalist", "winner", "rejected", "abandoned"]
    what_worked: list[str] = Field(default_factory=list)
    what_failed: list[str] = Field(default_factory=list)
    scoring_adjustments: dict[str, Any] = Field(default_factory=dict)
