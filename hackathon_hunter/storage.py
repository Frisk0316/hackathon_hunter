from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from hackathon_hunter.models import Hackathon

ROOT = Path(__file__).resolve().parents[1]


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
    candidates = [
        path
        for path in processed_dir.glob("*.json")
        if path.name.startswith(("hackathons_", "mock_hackathons"))
    ]
    if not candidates:
        raise FileNotFoundError("No processed hackathon JSON files found in data/processed")
    latest = max(candidates, key=lambda item: item.stat().st_mtime)
    return latest, load_hackathons(latest)


def save_processed_hackathons(
    hackathons: Iterable[Hackathon],
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    base = root or ROOT
    filename = name or f"hackathons_{utcish_now().strftime('%Y%m%d')}.json"
    payload: dict[str, Any] = {
        "generated_at": utcish_now().isoformat(),
        "hackathons": list(hackathons),
    }
    if metadata:
        payload.update(metadata)
    return write_json(base / "data" / "processed" / filename, payload)


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
    return candidate.with_name(f"{candidate.stem}_{timestamp}{candidate.suffix}")


def deduplicate_hackathons(items: Iterable[Hackathon]) -> list[Hackathon]:
    seen: set[str] = set()
    deduped: list[Hackathon] = []
    for item in items:
        key = item.id or item.url
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
