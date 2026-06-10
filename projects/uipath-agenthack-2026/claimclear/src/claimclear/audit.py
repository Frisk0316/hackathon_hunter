from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claimclear.models import AuditEntry, utc_now_iso


class AuditLogger:
    def __init__(self, audit_path: str | Path | None = None) -> None:
        self.audit_path = Path(audit_path) if audit_path else None
        self.entries: list[AuditEntry] = []
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            self.audit_path.write_text("", encoding="utf-8")

    def record(
        self,
        *,
        case_id: str,
        claim_id: str,
        actor: str,
        action: str,
        details: dict[str, Any],
    ) -> AuditEntry:
        entry = AuditEntry(
            timestamp=utc_now_iso(),
            case_id=case_id,
            claim_id=claim_id,
            actor=actor,
            action=action,
            details=details,
        )
        self.entries.append(entry)
        if self.audit_path:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
        return entry

    def entries_for_case(self, case_id: str) -> list[AuditEntry]:
        return [entry for entry in self.entries if entry.case_id == case_id]

