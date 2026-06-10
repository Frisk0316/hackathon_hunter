from __future__ import annotations

from pathlib import Path

from hackathon_hunter.reports import render_winner_template
from hackathon_hunter.storage import find_hackathon, project_path, save_report


def run_analyze_winners(hackathon_id: str, input_path: Path | None = None) -> Path:
    hackathon = find_hackathon(hackathon_id, input_path)
    path = project_path("reports", "winners", f"{hackathon.id}.md")
    return save_report(path, render_winner_template(hackathon))
