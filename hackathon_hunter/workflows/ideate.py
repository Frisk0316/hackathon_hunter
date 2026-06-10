from __future__ import annotations

from pathlib import Path

from hackathon_hunter.models import ProjectIdea
from hackathon_hunter.reports import render_ideas_report
from hackathon_hunter.storage import find_hackathon, project_path, save_report, write_json


def _base_stack(hackathon_id: str) -> list[str]:
    if "voice" in hackathon_id:
        return ["Next.js", "FastAPI", "ElevenLabs API", "SQLite", "Playwright"]
    if "qwen" in hackathon_id:
        return ["Next.js", "FastAPI", "Qwen Cloud", "PostgreSQL", "Pytest"]
    return ["Streamlit", "FastAPI", "Sponsor API", "SQLite", "Pytest"]


def generate_ideas(hackathon_id: str, n: int, input_path: Path | None = None) -> list[ProjectIdea]:
    hackathon = find_hackathon(hackathon_id, input_path)
    api = hackathon.required_apis[0] if hackathon.required_apis else "the required sponsor API"
    primary_track = hackathon.tracks[0] if hackathon.tracks else "main track"
    templates = [
        (
            "JudgeOps Copilot",
            "A review-room simulator that maps a project to the official judging rubric.",
            "Teams lose points because they cannot see how judges will read their demo.",
            "Hackathon builders preparing a final submission.",
            "It turns the sponsor API into a rubric-aware critique and rewrite loop.",
        ),
        (
            "SignalDesk",
            "A compact intelligence dashboard for one high-stakes operational decision.",
            "Domain teams drown in raw updates and miss the next best action.",
            "Small teams that need daily decisions, not generic analytics.",
            "It uses the sponsor API to extract signals and produce explainable next actions.",
        ),
        (
            "DemoPilot",
            "A guided demo generator that turns product state into judge-ready walkthroughs.",
            "Good prototypes are often undersold because the demo path is improvised.",
            "Solo hackers and small teams under deadline pressure.",
            "It uses the sponsor API to narrate, validate, and package demo flows.",
        ),
        (
            "TrustTrail",
            "An evidence ledger for AI decisions in regulated or high-risk workflows.",
            "AI demos look magical but judges still need to trust outputs and sources.",
            "Operators who need auditability before adopting AI systems.",
            "It uses the sponsor API while preserving citations, confidence, and fallbacks.",
        ),
        (
            "MicroMentor",
            "A personalized coaching loop for users who need fast, practical feedback.",
            "Most learning tools explain content but do not adapt to messy user context.",
            "Learners and builders who need targeted critique in minutes.",
            "It uses the sponsor API to run short feedback cycles with measurable progress.",
        ),
    ]
    ideas: list[ProjectIdea] = []
    for index, template in enumerate(templates[:n], start=1):
        name, tagline, problem, target_user, sponsor_usage = template
        ideas.append(
            ProjectIdea(
                id=f"idea-{index:03d}",
                hackathon_id=hackathon.id,
                name=name,
                tagline=tagline,
                problem=problem,
                target_user=target_user,
                why_now=f"{hackathon.name} rewards {primary_track} work with clear demo value.",
                sponsor_api_usage=f"{sponsor_usage} Integration anchor: {api}.",
                mvp_scope=[
                    "One complete happy-path workflow",
                    "Evidence-backed output with confidence or citations",
                    "Minimal persistence for demo history",
                    "Submission-ready README, video script, and architecture note",
                ],
                non_goals=[
                    "No production billing",
                    "No automatic final submission",
                    "No hidden real-user data collection",
                ],
                demo_flow=(
                    "User enters a concrete scenario, runs the sponsor-powered analysis, "
                    "reviews evidence, exports a judge-ready result, and sees clear failure states."
                ),
                tech_stack=_base_stack(hackathon.id),
                risks=[
                    "Sponsor API access may require manual account approval",
                    "Demo must disclose synthetic or sample data if used",
                ],
                estimated_build_hours=24 + (index * 2),
                feasibility_score=max(0.55, 0.86 - index * 0.04),
                judging_fit_score=max(0.55, 0.88 - index * 0.03),
                differentiation_score=max(0.55, 0.75 + index * 0.02),
            )
        )
    return ideas


def run_ideate(hackathon_id: str, n: int = 5, input_path: Path | None = None) -> dict[str, Path]:
    hackathon = find_hackathon(hackathon_id, input_path)
    ideas = generate_ideas(hackathon_id, n, input_path)
    strategy_path = project_path("strategy", f"{hackathon.id}_ideas.json")
    report_path = project_path("reports", "ideas", f"{hackathon.id}.md")
    write_json(
        strategy_path,
        {
            "hackathon_id": hackathon.id,
            "human_decision_required": True,
            "ideas": ideas,
        },
    )
    save_report(report_path, render_ideas_report(hackathon, ideas))
    return {"strategy": strategy_path, "report": report_path}
