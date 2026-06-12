---
title: Architecture
owner: human
status: active
last_updated: 2026-06-12
---

# Architecture

Hackathon Hunter is a Python package plus Typer CLI. It separates the hunter system from generated hackathon projects and historical artifacts.

## System Shape

```text
CLI commands
  -> workflow modules
    -> domain modules
      -> Pydantic models
      -> storage/report helpers
        -> data, reports, strategy, projects, logs
```

## Main Layers

| Layer | Files | Responsibility |
|---|---|---|
| CLI | `hackathon_hunter/cli.py`, `hackathon_hunter/__main__.py` | Typer commands, console output, run-log wrapping, process exit codes |
| Workflows | `hackathon_hunter/workflows/` | Orchestrate one user-facing action per module |
| Models | `hackathon_hunter/models.py` | Pydantic contracts for hackathons, evidence, scores, ideas, run logs, results |
| Scoring | `hackathon_hunter/scoring.py`, `config/scoring.yaml` | Weighted ranking, delivery risk, evidence quality, fast-lane behavior |
| Rules | `hackathon_hunter/rules.py` | Eligibility, AI-policy, deadline, region, stale-evidence, and human-review gates |
| Sources | `hackathon_hunter/sources/` | Adapter interface and collectors for mock, web search, Devpost, Lablab, DoraHacks |
| Storage | `hackathon_hunter/storage.py` | JSON IO, raw snapshots, processed radar files, reports, latest-file selection |
| Reports | `hackathon_hunter/reports.py` | Markdown rendering for ranking and rules outputs |
| Run logs | `hackathon_hunter/runlog.py`, `logs/` | Structured workflow audit trail |
| Tests | `tests/` | Model, workflow, scoring, rule, storage, QA, import, watch, calibration coverage |

## Artifact Boundaries

| Path | Meaning |
|---|---|
| `data/raw/` | Source snapshots from collectors |
| `data/processed/` | Normalized hackathon radar JSON |
| `reports/` | Human-readable radar, rules, import diff, winner analysis, calibration reports |
| `strategy/` | Generated project idea sets |
| `projects/` | Active generated hackathon project packages |
| `archive/generated-projects/` | Historical examples, not active hunter code |
| `logs/` | Run logs and workflow audit artifacts |

## Key Flows

Collection creates raw snapshots, filters candidates, writes processed JSON, ranks active candidates, and renders radar reports.

Import validates an evidence-backed radar JSON, optionally merges it with the latest processed file using evidence recency, writes an import diff, and writes a new processed file unless dry-run.

Ranking rejects expired, blocked, or ineligible candidates before scoring. Scoring combines ROI, feasibility, strategic fit, evidence quality, and delivery risk.

Build-spec is a generator. It creates a project package, but it must not publish or submit anything.

QA checks generated project materials for missing README sections, placeholders, secrets, synthetic-data disclosure, and smoke-test presence. It is advisory; final submission is human-only.

## Extension Points

- Add new collectors by implementing `SourceAdapter` in `hackathon_hunter/sources/base.py` and registering the factory in `hackathon_hunter/workflows/collect.py`.
- Add or tune scoring factors in `hackathon_hunter/scoring.py` and `config/scoring.yaml`, then update `docs/DOMAIN_RULES.md`, `docs/DATA_FLOW.md`, tests, and an ADR if semantics change.
- Add CLI commands in `hackathon_hunter/cli.py` by wrapping workflow calls with `run_logged`.
- Add generated-project checks in `hackathon_hunter/workflows/qa.py` and update `docs/DOMAIN_RULES.md`.

## Non-Goals

- No automatic final submission.
- No live account registration.
- No secret storage.
- No claim that a generated project is ready without local checks and human review.
