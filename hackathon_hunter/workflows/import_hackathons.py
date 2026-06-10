from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hackathon_hunter.models import Evidence, Hackathon
from hackathon_hunter.storage import (
    load_latest_processed,
    project_path,
    read_json,
    save_processed_hackathons,
    save_report,
    unique_path,
    utcish_now,
)


class ImportValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("\n".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class ImportChange:
    kind: str
    hackathon_id: str
    name: str
    field: str | None = None
    old: str | None = None
    new: str | None = None


@dataclass(frozen=True)
class ImportResult:
    input_path: Path
    diff_report_path: Path
    processed_path: Path | None
    changes: list[ImportChange]
    dry_run: bool


def _payload_items(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        items = payload.get("hackathons")
    else:
        items = payload
    if not isinstance(items, list):
        raise ImportValidationError(["input must be a list or an object with a hackathons list"])
    return items


def _validate_hackathons(path: Path) -> list[Hackathon]:
    payload = read_json(path)
    items = _payload_items(payload)
    hackathons: list[Hackathon] = []
    errors: list[str] = []
    for index, item in enumerate(items):
        try:
            hackathons.append(Hackathon.model_validate(item))
        except ValidationError as error:
            for detail in error.errors():
                location = ".".join(str(part) for part in detail["loc"])
                errors.append(f"hackathons[{index}].{location}: {detail['msg']}")
    if errors:
        raise ImportValidationError(errors)
    return hackathons


def _value_label(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, datetime):
        return value.isoformat()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _evidence_key(evidence: Evidence) -> str:
    return json.dumps(evidence.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)


def _merge_evidence(existing: list[Evidence], incoming: list[Evidence]) -> list[Evidence]:
    by_key = {_evidence_key(item): item for item in existing}
    for item in incoming:
        by_key.setdefault(_evidence_key(item), item)
    return sorted(by_key.values(), key=lambda item: (item.field, item.fetched_at, item.url or ""))


def _latest_evidence_at(hackathon: Hackathon, field: str) -> datetime | None:
    evidence = hackathon.evidence_for(field)
    if not evidence:
        return None
    return max(item.fetched_at for item in evidence)


def _incoming_field_wins(existing: Hackathon, incoming: Hackathon, field: str) -> bool:
    existing_at = _latest_evidence_at(existing, field)
    incoming_at = _latest_evidence_at(incoming, field)
    if incoming_at and existing_at:
        return incoming_at >= existing_at
    if incoming_at and not existing_at:
        return True
    if not incoming_at and not existing_at:
        return True
    return False


def _merge_one(existing: Hackathon, incoming: Hackathon) -> tuple[Hackathon, list[ImportChange]]:
    payload = existing.model_dump(mode="python")
    changes: list[ImportChange] = []
    for field in Hackathon.model_fields:
        if field in {"id", "source_evidence"}:
            continue
        old_value = getattr(existing, field)
        new_value = getattr(incoming, field)
        if old_value == new_value or not _incoming_field_wins(existing, incoming, field):
            continue
        payload[field] = new_value
        changes.append(
            ImportChange(
                kind="changed",
                hackathon_id=existing.id,
                name=incoming.name,
                field=field,
                old=_value_label(old_value),
                new=_value_label(new_value),
            )
        )

    payload["source_evidence"] = _merge_evidence(existing.source_evidence, incoming.source_evidence)
    return Hackathon.model_validate(payload), changes


def _merge_hackathons(
    existing: list[Hackathon],
    incoming: list[Hackathon],
) -> tuple[list[Hackathon], list[ImportChange], list[dict[str, str]]]:
    existing_by_id = {item.id: item for item in existing}
    incoming_by_id = {item.id: item for item in incoming}
    merged_by_id = dict(existing_by_id)
    changes: list[ImportChange] = []

    for item in incoming:
        if item.id not in existing_by_id:
            merged_by_id[item.id] = item
            changes.append(ImportChange(kind="added", hackathon_id=item.id, name=item.name))
            continue
        merged, item_changes = _merge_one(existing_by_id[item.id], item)
        merged_by_id[item.id] = merged
        changes.extend(item_changes)

    missing = [
        {"id": item.id, "name": item.name, "status": "expired_for_review"}
        for item in existing
        if item.id not in incoming_by_id
    ]
    for item in missing:
        changes.append(ImportChange(kind="missing", hackathon_id=item["id"], name=item["name"]))

    ordered: list[Hackathon] = []
    for item in existing:
        ordered.append(merged_by_id[item.id])
    for item in incoming:
        if item.id not in existing_by_id:
            ordered.append(merged_by_id[item.id])
    return ordered, changes, missing


def _render_import_diff(
    input_path: Path,
    changes: list[ImportChange],
    generated_at: datetime,
    *,
    merge: bool,
    dry_run: bool,
    previous_path: Path | None,
) -> str:
    lines = [
        f"# Import Diff — {generated_at.date().isoformat()}",
        "",
        f"- Input: `{input_path}`",
        f"- Merge: `{str(merge).lower()}`",
        f"- Dry run: `{str(dry_run).lower()}`",
        (
            f"- Previous processed: `{previous_path}`"
            if previous_path
            else "- Previous processed: `none`"
        ),
        "",
        "## Changes",
        "",
    ]
    if not changes:
        lines.append("No changes.")
        return "\n".join(lines)

    for change in changes:
        if change.kind == "changed":
            lines.extend(
                [
                    f"### CHANGED `{change.hackathon_id}` — {change.name}",
                    "",
                    f"- Field: `{change.field}`",
                    f"- Old: {change.old}",
                    f"- New: {change.new}",
                    "",
                ]
            )
        elif change.kind == "added":
            lines.extend([f"- ADDED `{change.hackathon_id}` — {change.name}"])
        elif change.kind == "missing":
            lines.extend(
                [
                    f"- MISSING `{change.hackathon_id}` — {change.name} "
                    "(retained in processed, flagged expired_for_review in metadata)"
                ]
            )
    return "\n".join(lines)


def run_import(
    input_path: Path,
    *,
    merge: bool = False,
    dry_run: bool = False,
    root: Path | None = None,
    now: datetime | None = None,
) -> ImportResult:
    base = root or project_path()
    generated_at = now or utcish_now()
    resolved_input = Path(input_path)
    incoming = _validate_hackathons(resolved_input)
    previous_path: Path | None = None
    expired_for_review: list[dict[str, str]] = []

    if merge:
        previous_path, existing = load_latest_processed(root=base)
        final_items, changes, expired_for_review = _merge_hackathons(existing, incoming)
    else:
        final_items = incoming
        changes = [
            ImportChange(kind="added", hackathon_id=item.id, name=item.name)
            for item in incoming
        ]

    diff_path = unique_path(base / "reports" / f"import_diff_{generated_at.strftime('%Y%m%d')}.md")
    save_report(
        diff_path,
        _render_import_diff(
            resolved_input,
            changes,
            generated_at,
            merge=merge,
            dry_run=dry_run,
            previous_path=previous_path,
        ),
    )

    processed_path = None
    if not dry_run:
        processed_path = save_processed_hackathons(
            final_items,
            metadata={
                "source": "import",
                "input": str(resolved_input),
                "merge": merge,
                "previous_processed": str(previous_path) if previous_path else None,
                "expired_for_review": expired_for_review,
            },
            root=base,
        )

    return ImportResult(
        input_path=resolved_input,
        diff_report_path=diff_path,
        processed_path=processed_path,
        changes=changes,
        dry_run=dry_run,
    )
