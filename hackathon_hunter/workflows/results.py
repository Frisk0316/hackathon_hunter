from __future__ import annotations

from pathlib import Path

from hackathon_hunter.models import ResultRecord
from hackathon_hunter.scoring import score_hackathon
from hackathon_hunter.storage import (
    find_hackathon,
    project_path,
    save_report,
    utcish_now,
    write_json,
)


def _format_list(items: list[str]) -> list[str]:
    if not items:
        return ["- Not recorded"]
    return [f"- {item}" for item in items]


def run_record_result(
    hackathon_id: str,
    outcome: str,
    project_slug: str | None = None,
    hours_spent: float | None = None,
    api_cost_usd: float | None = None,
    infra_cost_usd: float | None = None,
    what_worked: list[str] | None = None,
    what_failed: list[str] | None = None,
    notes: str | None = None,
    input_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, Path]:
    checked_at = utcish_now()
    score_trace: dict[str, float] = {}
    overall_score: float | None = None
    try:
        hackathon = find_hackathon(hackathon_id, input_path)
        score = score_hackathon(hackathon, now=checked_at)
        score_trace = score.trace
        overall_score = score.overall_score
    except (FileNotFoundError, LookupError):
        pass

    record = ResultRecord(
        hackathon_id=hackathon_id,
        project_slug=project_slug,
        hours_spent=hours_spent,
        api_cost_usd=api_cost_usd,
        infra_cost_usd=infra_cost_usd,
        submitted=outcome in {"submitted", "finalist", "winner", "rejected"},
        outcome=outcome,  # type: ignore[arg-type]
        what_worked=what_worked or [],
        what_failed=what_failed or [],
        notes=notes,
        score_trace=score_trace,
        overall_score=overall_score,
    )
    base = root or project_path()
    date = checked_at.strftime("%Y%m%d")
    json_path = base / "logs" / "rounds" / f"{date}_{hackathon_id}.json"
    report_path = base / "reports" / "retrospectives" / f"{hackathon_id}.md"
    write_json(json_path, record)
    save_report(
        report_path,
        "\n".join(
            [
                f"# Retrospective — {hackathon_id}",
                "",
                f"- Outcome: {outcome}",
                f"- Project: {project_slug or 'not recorded'}",
                f"- Hours spent: {hours_spent if hours_spent is not None else 'not recorded'}",
                f"- API cost USD: {api_cost_usd if api_cost_usd is not None else 'not recorded'}",
                "- Infra cost USD: "
                f"{infra_cost_usd if infra_cost_usd is not None else 'not recorded'}",
                "- Overall score at recording: "
                f"{overall_score if overall_score is not None else 'not recorded'}",
                "",
                "## What Worked",
                "",
                *_format_list(record.what_worked),
                "",
                "## What Failed",
                "",
                *_format_list(record.what_failed),
                "",
                "## Notes",
                "",
                notes or "Not recorded",
                "",
                "## Follow-up",
                "",
                "- Run `hackathon-hunter calibrate` after multiple completed rounds.",
                "- Adjust scoring weights only after human review.",
            ]
        ),
    )
    return {"record": json_path, "report": report_path}
