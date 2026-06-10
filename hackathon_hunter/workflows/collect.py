from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from hackathon_hunter.reports import render_ranking_report
from hackathon_hunter.scoring import rank_hackathons
from hackathon_hunter.sources.mock import MockSource
from hackathon_hunter.sources.web_search import WebSearchSource
from hackathon_hunter.storage import (
    apply_freshness,
    deduplicate_hackathons,
    project_path,
    save_processed_hackathons,
    save_raw_snapshot,
    save_report,
    unique_path,
    utcish_now,
)


def _passes_filters(
    item: Any,
    days_ahead: int,
    min_prize_usd: float,
    online_only: bool,
) -> bool:
    now = utcish_now()
    if online_only and str(item.format).lower() not in {"online", "hybrid"}:
        return False
    if item.prize_total_usd is not None and item.prize_total_usd < min_prize_usd:
        return False
    if item.prize_total_usd is None and min_prize_usd > 0:
        return False
    if item.deadline > now + timedelta(days=days_ahead):
        return False
    return True


def run_collect(
    days_ahead: int = 90,
    min_prize_usd: float = 1000,
    online_only: bool = False,
    mock: bool = False,
) -> dict[str, Path]:
    source = MockSource() if mock else WebSearchSource()
    result = source.collect()
    raw_path = save_raw_snapshot(
        result.source,
        {
            "source": result.source,
            "errors": result.errors,
            "hackathons": result.hackathons,
        },
    )
    filtered = [
        item
        for item in result.hackathons
        if _passes_filters(item, days_ahead, min_prize_usd, online_only)
    ]
    processed_items = apply_freshness(deduplicate_hackathons(filtered))
    processed_name = "mock_hackathons.json" if mock else None
    processed_path = save_processed_hackathons(
        processed_items,
        name=processed_name,
        metadata={
            "source": result.source,
            "errors": result.errors,
            "filters": {
                "days_ahead": days_ahead,
                "min_prize_usd": min_prize_usd,
                "online_only": online_only,
            },
        },
    )
    ranked, rejected = rank_hackathons(processed_items)
    report_stem = "radar_mock" if mock else "radar"
    report_path = unique_path(
        project_path("reports", f"{report_stem}_{utcish_now().strftime('%Y%m%d')}.md")
    )
    save_report(report_path, render_ranking_report(ranked, rejected, utcish_now()))
    return {"raw": raw_path, "processed": processed_path, "report": report_path}
