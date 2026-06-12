---
title: Runbook
owner: human
status: active
last_updated: 2026-06-12
---

# Runbook

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Common Commands

```bash
make test
make lint
make docs-check
make docs-impact
make verify
```

## Mock Workflow

```bash
make collect-mock
make rank-mock
make rules-mock
make status-mock
make watch-mock
```

`watch-mock` may exit with code 2 when watch events exist. That is workflow behavior, not automatically a test failure.

## Evidence Import

```bash
python3 -m hackathon_hunter import --input examples/hackathons.sample.json --dry-run
python3 -m hackathon_hunter import --input path/to/radar.json --merge
```

Use dry-run before merge when importing human or LLM-produced radar JSON.

## Generated Project QA

```bash
python3 -m hackathon_hunter qa --project projects/<hackathon-id>/<idea-id>
```

QA is advisory. Final submission still requires human review.

## Troubleshooting

| Symptom | Check |
|---|---|
| No processed radar found | Run `make collect-mock` or import a validated radar |
| Candidate rejected unexpectedly | Read rejected reasons in ranking report and inspect `rules.py` / `scoring.py` |
| Import schema errors | Validate timezone-aware `deadline` and `source_evidence.fetched_at` fields |
| Watch exits 2 | Inspect watch report; events may be deadline or stale-evidence alerts |
| QA fails on placeholders | Replace placeholders or remove unready submission claims |
| Docs impact warning | Update docs named by `make docs-impact` or explain why not impacted |

## Rollback

For code changes, use git diff to identify touched files and revert only the files from the task. Do not reset unrelated user changes.

For generated artifacts, delete only the task-specific output directory or report after confirming it is not needed.

For scoring/config changes, restore both `hackathon_hunter/scoring.py` and `config/scoring.yaml` together if semantics changed.
