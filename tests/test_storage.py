import os
from pathlib import Path

import pytest

from hackathon_hunter.storage import (
    load_latest_processed,
    save_processed_hackathons,
    write_json,
)
from hackathon_hunter.workflows.collect import CollectError, run_collect


def test_collect_error_does_not_overwrite_existing_processed_file(tmp_path: Path) -> None:
    protected = tmp_path / "data" / "processed" / "hackathons_20260610.json"
    protected.parent.mkdir(parents=True)
    protected.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(CollectError) as exc_info:
        run_collect(source_name="web_search", root=tmp_path)

    assert protected.read_text(encoding="utf-8") == "sentinel\n"
    assert exc_info.value.raw_path is not None
    assert exc_info.value.raw_path.exists()
    assert not list((tmp_path / "reports").glob("radar*.md"))


def test_source_mock_uses_mock_output_contract(tmp_path: Path) -> None:
    outputs = run_collect(source_name="mock", root=tmp_path)

    assert outputs["processed"].name == "mock_hackathons.json"
    assert outputs["report"].name.startswith("radar_mock_")


def test_save_processed_uses_unique_path_unless_overwrite_is_explicit(tmp_path: Path) -> None:
    filename = "hackathons_20260610.json"
    first = save_processed_hackathons([], name=filename, root=tmp_path)
    second = save_processed_hackathons([], name=filename, root=tmp_path)
    third = save_processed_hackathons([], name=filename, root=tmp_path, overwrite=True)

    assert first.name == filename
    assert second.name != filename
    assert second.exists()
    assert third == first


def test_load_latest_processed_prefers_filename_date_over_mtime_and_mock(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    old = processed / "hackathons_20260601.json"
    latest = processed / "hackathons_20260610.json"
    mock = processed / "mock_hackathons.json"
    write_json(old, {"generated_at": "2026-06-01T12:00:00+00:00", "hackathons": []})
    write_json(latest, {"generated_at": "2026-06-10T12:00:00+00:00", "hackathons": []})
    write_json(mock, {"generated_at": "2026-06-12T12:00:00+00:00", "hackathons": []})

    newer_mtime = latest.stat().st_mtime + 1000
    os.utime(old, (newer_mtime, newer_mtime))
    os.utime(mock, (newer_mtime + 1000, newer_mtime + 1000))

    selected, hackathons = load_latest_processed(root=tmp_path)

    assert selected == latest
    assert hackathons == []


def test_load_latest_processed_breaks_same_day_ties_with_generated_at(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    early = processed / "hackathons_20260610.json"
    late = processed / "hackathons_20260610_120000.json"
    write_json(early, {"generated_at": "2026-06-10T03:00:00+00:00", "hackathons": []})
    write_json(late, {"generated_at": "2026-06-10T12:00:00+00:00", "hackathons": []})

    selected, _ = load_latest_processed(root=tmp_path)

    assert selected == late
