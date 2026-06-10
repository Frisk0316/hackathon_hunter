from __future__ import annotations

from datetime import datetime

from hackathon_hunter.models import Hackathon, ProjectIdea, RulesCheckResult, ScoreBreakdown


def render_ranking_report(
    ranked: list[tuple[Hackathon, ScoreBreakdown]],
    rejected: list[tuple[Hackathon, str]],
    generated_at: datetime,
) -> str:
    lines = [
        f"# Hackathon Ranking — {generated_at.date().isoformat()}",
        "",
        "Scores are multi-dimensional and traceable. They are not win-rate estimates.",
        "",
        "## Top Picks",
        "",
    ]
    if not ranked:
        lines.extend(["No active candidates passed the ranking gates.", ""])
    for index, (hackathon, score) in enumerate(ranked, start=1):
        prize = hackathon.prize_total_usd
        prize_label = prize if prize is not None else "unknown"
        lines.extend(
            [
                f"### {index}. {hackathon.name} — overall {score.overall_score:.2f}",
                "",
                f"- ID: `{hackathon.id}`",
                f"- Deadline: {hackathon.deadline.isoformat()} ({hackathon.deadline_timezone})",
                f"- Format: {hackathon.format}",
                f"- Prize: {prize_label} USD",
                f"- AI policy: {hackathon.ai_policy}",
                f"- ROI: {score.roi_score:.2f}",
                f"- Feasibility: {score.feasibility_score:.2f}",
                f"- Strategic fit: {score.strategic_fit_score:.2f}",
                f"- Evidence quality: {score.evidence_quality_score:.2f}",
                f"- Delivery risk: {score.delivery_risk_score:.2f}",
                f"- Reason: {score.ranking_reason}",
                "",
                "| weight feature | score |",
                "|---|---:|",
            ]
        )
        for feature, value in score.trace.items():
            lines.append(f"| {feature} | {value:.2f} |")
        lines.append("")
    lines.extend(["## Rejected Candidates", ""])
    if not rejected:
        lines.append("No rejected candidates.")
    for hackathon, reason in rejected:
        lines.extend(
            [
                f"- `{hackathon.id}` — {hackathon.name}: {reason}",
            ]
        )
    lines.extend(
        [
            "",
            "## Human Gates",
            "",
            "- Confirm registration and official rules before build.",
            "- Confirm selected idea before project scaffold.",
            "- Confirm public repo/demo before publishing.",
            "- Confirm social posts manually.",
            "- Confirm final submission manually.",
        ]
    )
    return "\n".join(lines)


def render_rules_report(hackathon: Hackathon, result: RulesCheckResult) -> str:
    lines = [
        f"# Rules Check — {hackathon.name}",
        "",
        f"- Hackathon ID: `{hackathon.id}`",
        f"- Eligible for build: `{str(result.eligible).lower()}`",
        f"- Human review required: `{str(result.human_review_required).lower()}`",
        "",
        "## Blocking Issues",
        "",
    ]
    if result.blocking_issues:
        lines.extend(f"- {issue}" for issue in result.blocking_issues)
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if result.warnings:
        lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Submission Requirements",
            "",
            f"- GitHub repo: `{result.submission_requirements.github_repo}`",
            f"- Demo URL: `{result.submission_requirements.demo_url}`",
            f"- Video: `{result.submission_requirements.video}`",
            f"- Deck: `{result.submission_requirements.deck}`",
            f"- Public profile: `{result.submission_requirements.public_profile}`",
            f"- Notes: {result.submission_requirements.notes or 'None'}",
        ]
    )
    return "\n".join(lines)


def render_ideas_report(hackathon: Hackathon, ideas: list[ProjectIdea]) -> str:
    lines = [
        f"# Project Ideas — {hackathon.name}",
        "",
        "Human decision required: `true`",
        "",
    ]
    for idea in ideas:
        lines.extend(
            [
                f"## {idea.id}: {idea.name}",
                "",
                f"**Tagline:** {idea.tagline}",
                "",
                f"**Problem:** {idea.problem}",
                "",
                f"**Target user:** {idea.target_user}",
                "",
                f"**Why now:** {idea.why_now}",
                "",
                f"**Sponsor API usage:** {idea.sponsor_api_usage}",
                "",
                "**MVP scope:**",
                "",
                *[f"- {item}" for item in idea.mvp_scope],
                "",
                "**Non-goals:**",
                "",
                *[f"- {item}" for item in idea.non_goals],
                "",
                f"**Demo flow:** {idea.demo_flow}",
                "",
                f"**Tech stack:** {', '.join(idea.tech_stack)}",
                "",
                f"**Scores:** feasibility {idea.feasibility_score:.2f}, "
                f"judging fit {idea.judging_fit_score:.2f}, "
                f"differentiation {idea.differentiation_score:.2f}",
                "",
            ]
        )
    return "\n".join(lines)


def render_winner_template(hackathon: Hackathon) -> str:
    return "\n".join(
        [
            f"# Winner Intelligence — {hackathon.name}",
            "",
            "This template is intentionally evidence-gated. Fill it after fetching "
            "winner/finalist sources.",
            "",
            "## Hypotheses",
            "",
            "1. Sponsor API depth is likely rewarded when it is visible in the demo.",
            "   - Evidence: source needed.",
            "   - Confidence: low.",
            "2. A polished public video may matter because submission requirements "
            "include demo materials.",
            "   - Evidence: source needed.",
            "   - Confidence: low.",
            "3. Projects should map explicitly to judging criteria rather than only "
            "showcase technology.",
            "   - Evidence: judging criteria in current hackathon record.",
            "   - Confidence: medium.",
            "",
            "## Strategy",
            "",
            "- Convert each hypothesis into project requirements after evidence is added.",
            "- Do not build from this report until evidence confidence is upgraded.",
        ]
    )
