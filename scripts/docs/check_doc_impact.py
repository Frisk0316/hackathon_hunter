#!/usr/bin/env python3
"""Infer documentation updates from changed files."""

from __future__ import annotations

import argparse
import subprocess

Rule = dict[str, list[str] | str]

DEFAULT_RULES: list[Rule] = [
    {
        "name": "cli",
        "prefixes": ["hackathon_hunter/cli.py", "hackathon_hunter/__main__.py"],
        "docs": [
            "docs/API_MAP.md",
            "docs/FEATURE_MAP.md",
            "docs/RUNBOOK.md",
            "README.md",
        ],
    },
    {
        "name": "workflow",
        "prefixes": ["hackathon_hunter/workflows/"],
        "docs": [
            "docs/ARCHITECTURE.md",
            "docs/FEATURE_MAP.md",
            "docs/DATA_FLOW.md",
            "docs/RUNBOOK.md",
        ],
    },
    {
        "name": "models_schema",
        "prefixes": ["hackathon_hunter/models.py"],
        "docs": [
            "docs/API_MAP.md",
            "docs/DATA_FLOW.md",
            "docs/DOMAIN_RULES.md",
            "docs/CONTEXT_PACKS/data_pipeline.md",
        ],
    },
    {
        "name": "scoring",
        "prefixes": ["hackathon_hunter/scoring.py", "config/scoring.yaml"],
        "docs": [
            "docs/DOMAIN_RULES.md",
            "docs/DATA_FLOW.md",
            "docs/MENTAL_MODELS.md",
            "docs/INVARIANTS.md",
            "docs/ADR/",
        ],
    },
    {
        "name": "rules",
        "prefixes": ["hackathon_hunter/rules.py"],
        "docs": [
            "docs/DOMAIN_RULES.md",
            "docs/INVARIANTS.md",
            "docs/FAILURE_MODES.md",
            "README.md",
        ],
    },
    {
        "name": "sources",
        "prefixes": ["hackathon_hunter/sources/"],
        "docs": [
            "docs/ARCHITECTURE.md",
            "docs/FEATURE_MAP.md",
            "docs/DATA_FLOW.md",
            "docs/RUNBOOK.md",
        ],
    },
    {
        "name": "storage_reports",
        "prefixes": ["hackathon_hunter/storage.py", "hackathon_hunter/reports.py"],
        "docs": [
            "docs/DATA_FLOW.md",
            "docs/RUNBOOK.md",
            "docs/FEATURE_MAP.md",
        ],
    },
    {
        "name": "generated_projects",
        "prefixes": [
            "projects/",
            "hackathon_hunter/workflows/build_spec.py",
            "hackathon_hunter/workflows/ideate.py",
            "hackathon_hunter/workflows/qa.py",
        ],
        "docs": [
            "docs/UI_MAP.md",
            "docs/DOMAIN_RULES.md",
            "docs/RUNBOOK.md",
            "docs/CONTEXT_PACKS/generated_projects.md",
        ],
    },
    {
        "name": "harness",
        "prefixes": [
            "AI_CONTEXT.md",
            "AGENTS.md",
            "CLAUDE.md",
            "Makefile",
            "docs/",
            "tasks/",
            "scripts/docs/",
        ],
        "docs": [
            "docs/DOC_IMPACT_MATRIX.md",
            "docs/CURRENT_STATE.md",
            "docs/CONTEXT_INDEX.md",
            "docs/CHANGELOG_AI.md",
        ],
    },
]


def git(args: list[str]) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as error:  # noqa: BLE001 - diagnostic script should not crash.
        print(f"WARNING: git command failed: git {' '.join(args)} ({error})")
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def changed_files(base: str | None) -> list[str]:
    if base:
        return git(["diff", "--name-only", f"{base}...HEAD"])
    return sorted(
        set(
            git(["diff", "--name-only"])
            + git(["diff", "--cached", "--name-only"])
            + git(["ls-files", "--others", "--exclude-standard"])
        )
    )


def matches(path: str, prefixes: list[str]) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in prefixes)


def doc_was_changed(doc: str, changed: set[str], docs_changed: set[str]) -> bool:
    if doc.endswith("/"):
        return any(path.startswith(doc) for path in docs_changed)
    return doc in changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    changed = changed_files(args.base)
    if not changed:
        print("No changed files detected.")
        return 0

    expected: dict[str, set[str]] = {}
    for path in changed:
        for rule in DEFAULT_RULES:
            prefixes = rule["prefixes"]
            docs = rule["docs"]
            if isinstance(prefixes, list) and isinstance(docs, list) and matches(path, prefixes):
                expected.setdefault(str(rule["name"]), set()).update(docs)

    changed_set = set(changed)
    docs_changed = {
        path
        for path in changed
        if path.endswith(".md")
        or path.startswith("docs/")
        or path in {"AI_CONTEXT.md", "AGENTS.md"}
    }

    missing: list[tuple[str, str]] = []
    print("Changed files:")
    for path in changed:
        print("  -", path)

    print("\nInferred areas:")
    if not expected:
        print("  - none")
    for area, docs in expected.items():
        print("  -", area)
        for doc in sorted(docs):
            ok = doc_was_changed(doc, changed_set, docs_changed)
            print(f"      {'changed' if ok else 'missing'}: {doc}")
            if not ok:
                missing.append((area, doc))

    if missing:
        level = "ERROR" if args.strict else "WARNING"
        print(f"\n{level}: possible missing documentation updates:")
        for area, doc in missing:
            print(f"  - area={area} expected={doc}")
        return 1 if args.strict else 0

    print("\nPASS: documentation impact appears covered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
