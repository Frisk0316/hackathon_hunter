from __future__ import annotations

from pathlib import Path

from hackathon_hunter.reports import render_ranking_report
from hackathon_hunter.scoring import load_profile, load_weights, rank_hackathons
from hackathon_hunter.storage import (
    load_hackathons,
    load_latest_processed,
    project_path,
    save_report,
    unique_path,
    utcish_now,
)


def run_rank(
    input_path: Path | None = None,
    profile_path: Path | None = None,
    weights_path: Path | None = None,
    fast_lane_mode: bool = False,
) -> Path:
    source_path = input_path
    if input_path:
        hackathons = load_hackathons(input_path)
    else:
        source_path, hackathons = load_latest_processed()
    ranked, rejected = rank_hackathons(
        hackathons,
        weights=load_weights(weights_path),
        profile=load_profile(profile_path),
        fast_lane_mode=fast_lane_mode,
    )
    stem = "radar_mock" if source_path and source_path.name.startswith("mock_") else "radar_ranked"
    report_path = unique_path(
        project_path("reports", f"{stem}_{utcish_now().strftime('%Y%m%d')}.md")
    )
    return save_report(report_path, render_ranking_report(ranked, rejected, utcish_now()))
