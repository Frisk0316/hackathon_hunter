from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from hackathon_hunter.models import Hackathon

ROOT = Path(__file__).resolve().parents[1]
DATED_PROCESSED_RE = re.compile(r"^hackathons_(\d{8})(?:_.+)?\.json$")


def project_path(*parts: str | Path) -> Path:
    return ROOT.joinpath(*parts)


def utcish_now() -> datetime:
    return datetime.now().astimezone()


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def _parse_processed_date(path: Path) -> date | None:
    match = DATED_PROCESSED_RE.match(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _generated_at_timestamp(path: Path) -> float:
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return float("-inf")
    if not isinstance(payload, dict) or "generated_at" not in payload:
        return float("-inf")
    raw_value = str(payload["generated_at"]).replace("Z", "+00:00")
    try:
        generated_at = datetime.fromisoformat(raw_value)
    except ValueError:
        return float("-inf")
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    return generated_at.timestamp()


def save_raw_snapshot(source: str, payload: Any, root: Path | None = None) -> Path:
    base = root or ROOT
    timestamp = utcish_now().strftime("%Y%m%d_%H%M%S")
    return write_json(base / "data" / "raw" / source / f"{timestamp}.json", payload)


def load_hackathons(path: str | Path) -> list[Hackathon]:
    payload = read_json(path)
    if isinstance(payload, dict):
        items = payload.get("hackathons", [])
    else:
        items = payload
    return [Hackathon.model_validate(item) for item in items]


def load_latest_processed(root: Path | None = None) -> tuple[Path, list[Hackathon]]:
    base = root or ROOT
    processed_dir = base / "data" / "processed"
    dated_candidates = [
        (parsed_date, path)
        for path in processed_dir.glob("hackathons_*.json")
        if (parsed_date := _parse_processed_date(path)) is not None
    ]
    if dated_candidates:
        _, latest = max(
            dated_candidates,
            key=lambda item: (
                item[0],
                _generated_at_timestamp(item[1]),
                item[1].stat().st_mtime,
            ),
        )
        return latest, load_hackathons(latest)

    mock_candidates = list(processed_dir.glob("mock_hackathons*.json"))
    if not mock_candidates:
        raise FileNotFoundError("No processed hackathon JSON files found in data/processed")
    latest = max(mock_candidates, key=lambda item: item.stat().st_mtime)
    return latest, load_hackathons(latest)


def save_processed_hackathons(
    hackathons: Iterable[Hackathon],
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    base = root or ROOT
    filename = name or f"hackathons_{utcish_now().strftime('%Y%m%d')}.json"
    destination = base / "data" / "processed" / filename
    if destination.exists() and not overwrite:
        destination = unique_path(destination)
    payload: dict[str, Any] = {
        "generated_at": utcish_now().isoformat(),
        "hackathons": list(hackathons),
    }
    if metadata:
        payload.update(metadata)
    return write_json(destination, payload)


def save_report(path: str | Path, markdown: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return destination


def unique_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        return candidate
    timestamp = utcish_now().strftime("%H%M%S")
    stamped = candidate.with_name(f"{candidate.stem}_{timestamp}{candidate.suffix}")
    if not stamped.exists():
        return stamped
    counter = 2
    while True:
        numbered = candidate.with_name(f"{candidate.stem}_{timestamp}_{counter}{candidate.suffix}")
        if not numbered.exists():
            return numbered
        counter += 1


def deduplicate_hackathons(items: Iterable[Hackathon]) -> list[Hackathon]:
    seen: set[str] = set()
    deduped: list[Hackathon] = []
    for item in items:
        key = item.id
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def apply_freshness(items: Iterable[Hackathon], now: datetime | None = None) -> list[Hackathon]:
    checked_at = now or utcish_now()
    refreshed: list[Hackathon] = []
    for item in items:
        if item.is_expired(checked_at):
            refreshed.append(item.model_copy(update={"status": "closed"}))
        else:
            refreshed.append(item)
    return refreshed


def find_hackathon(hackathon_id: str, path: str | Path | None = None) -> Hackathon:
    if path is None:
        _, hackathons = load_latest_processed()
    else:
        hackathons = load_hackathons(path)
    for hackathon in hackathons:
        if hackathon.id == hackathon_id:
            return hackathon
    raise LookupError(f"Hackathon not found: {hackathon_id}")
