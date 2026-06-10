from __future__ import annotations

from pathlib import Path

from hackathon_hunter.reports import render_rules_report
from hackathon_hunter.rules import check_rules
from hackathon_hunter.storage import (
    load_hackathons,
    load_latest_processed,
    project_path,
    save_report,
)


def run_check_rules(input_path: Path | None = None) -> list[Path]:
    if input_path:
        hackathons = load_hackathons(input_path)
    else:
        _, hackathons = load_latest_processed()
    output_paths: list[Path] = []
    for hackathon in hackathons:
        result = check_rules(hackathon)
        path = project_path("reports", f"rules_check_{hackathon.id}.md")
        output_paths.append(save_report(path, render_rules_report(hackathon, result)))
    return output_paths
