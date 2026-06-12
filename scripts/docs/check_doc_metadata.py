#!/usr/bin/env python3
"""Check markdown files for YAML front matter."""

from __future__ import annotations

import argparse
from pathlib import Path


def has_front_matter(text: str) -> bool:
    return text.startswith("---\n") and "\n---\n" in text[4:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--paths", nargs="*", default=["docs"])
    args = parser.parse_args()

    files: list[Path] = []
    for root in args.paths:
        path = Path(root)
        if path.exists():
            files.extend(sorted(path.rglob("*.md")))

    if not files:
        print("No markdown files found.")
        return 0

    missing = [
        str(path)
        for path in files
        if not has_front_matter(path.read_text(encoding="utf-8", errors="replace"))
    ]
    print(f"Checked markdown files: {len(files)}")

    if missing:
        level = "ERROR" if args.strict else "WARNING"
        print(f"{level}: files missing YAML front matter:")
        for path in missing:
            print("  -", path)
        return 1 if args.strict else 0

    print("PASS: all checked files have YAML front matter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
