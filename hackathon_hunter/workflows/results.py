from __future__ import annotations

from pathlib import Path

from hackathon_hunter.models import ResultRecord
from hackathon_hunter.storage import project_path, save_report, utcish_now, write_json


def run_record_result(
    hackathon_id: str,
    outcome: str,
    project_slug: str | None = None,
    hours_spent: float | None = None,
    api_cost_usd: float | None = None,
    infra_cost_usd: float | None = None,
) -> dict[str, Path]:
    record = ResultRecord(
        hackathon_id=hackathon_id,
        project_slug=project_slug,
        hours_spent=hours_spent,
        api_cost_usd=api_cost_usd,
        infra_cost_usd=infra_cost_usd,
        submitted=outcome in {"submitted", "finalist", "winner", "rejected"},
        outcome=outcome,  # type: ignore[arg-type]
    )
    date = utcish_now().strftime("%Y%m%d")
    json_path = project_path("logs", "rounds", f"{date}_{hackathon_id}.json")
    report_path = project_path("reports", "retrospectives", f"{hackathon_id}.md")
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
                "",
                "## Follow-up",
                "",
                "- Add what worked.",
                "- Add what failed.",
                "- Adjust scoring weights only after human review.",
            ]
        ),
    )
    return {"record": json_path, "report": report_path}
