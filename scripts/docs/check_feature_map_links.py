#!/usr/bin/env python3
"""Check repo-relative paths referenced in docs/FEATURE_MAP.md."""

from __future__ import annotations

import re
from pathlib import Path


def looks_like_path(value: str) -> bool:
    if value.startswith(("http://", "https://", "mailto:")):
        return False
    if " " in value:
        return False
    if "*" in value or "..." in value or "<" in value or ">" in value:
        return False
    return "/" in value or "." in Path(value).name


def main() -> int:
    feature_map = Path("docs/FEATURE_MAP.md")
    if not feature_map.exists():
        print("WARNING: docs/FEATURE_MAP.md not found.")
        return 0

    text = feature_map.read_text(encoding="utf-8", errors="replace")
    refs = sorted(
        {
            match.group(1).strip()
            for match in re.finditer(r"`([^`]+)`", text)
            if looks_like_path(match.group(1).strip())
        }
    )
    missing = [ref for ref in refs if "[" not in ref and "]" not in ref and not Path(ref).exists()]

    print(f"Checked referenced paths: {len(refs)}")
    if missing:
        print("WARNING: referenced paths not found:")
        for ref in missing:
            print("  -", ref)
    else:
        print("PASS: all referenced paths exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
