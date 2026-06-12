---
title: UI Map
owner: human
status: active
last_updated: 2026-06-12
---

# UI Map

Hackathon Hunter has no first-party web UI. Its user surfaces are CLI output, Markdown reports, JSON artifacts, and generated project packages.

## Surfaces

| Surface | Files | User-facing behavior |
|---|---|---|
| CLI help and commands | `hackathon_hunter/cli.py` | Typer command names, options, help text, exit codes |
| Rich status table | `hackathon_hunter/cli.py`, `hackathon_hunter/workflows/status.py` | Candidate ID, days, score, Taiwan gate, AI policy, rules status, stale fields |
| Radar reports | `hackathon_hunter/reports.py`, `hackathon_hunter/workflows/rank.py`, `hackathon_hunter/workflows/collect.py` | Ranked/rejected candidate Markdown |
| Rules reports | `hackathon_hunter/reports.py`, `hackathon_hunter/workflows/rules_check.py` | Blocking issues, warnings, human-review flag |
| Watch reports | `hackathon_hunter/workflows/watch.py` | Deadline and stale-evidence events |
| Import diff reports | `hackathon_hunter/workflows/import_hackathons.py` | Added, changed, and missing radar records |
| Generated project docs | `hackathon_hunter/workflows/build_spec.py` | SPEC, TASKS, README, SUBMISSION_DRAFT, AGENT_BRIEF |
| QA reports | `hackathon_hunter/workflows/qa.py` | Failures, warnings, and human gate reminder |

## UI Change Rule

Changes to command names, option names, report headings, table columns, generated project text, or QA wording are user-facing. Update `docs/API_MAP.md`, `docs/FEATURE_MAP.md`, `docs/RUNBOOK.md`, and tests where relevant.
