---
title: Doc Impact Matrix
owner: human
status: active
last_updated: 2026-06-12
---

# Doc Impact Matrix

Use this whenever code, configuration, or workflow behavior changes.

| Changed area | Examples | Required docs to check | ADR trigger |
|---|---|---|---|
| CLI contract | `hackathon_hunter/cli.py`, command names, options, exit codes | `API_MAP.md`, `FEATURE_MAP.md`, `RUNBOOK.md`, `README.md` | New workflow or breaking option change |
| Models / schema | `hackathon_hunter/models.py`, JSON artifact fields | `API_MAP.md`, `DATA_FLOW.md`, `DOMAIN_RULES.md`, `CONTEXT_PACKS/data_pipeline.md` | Field semantics, validation, compatibility decision |
| Scoring | `hackathon_hunter/scoring.py`, `config/scoring.yaml` | `DOMAIN_RULES.md`, `DATA_FLOW.md`, `MENTAL_MODELS.md`, `INVARIANTS.md`, tests | New scoring dimension or changed gate semantics |
| Rules / gates | `hackathon_hunter/rules.py`, human-review behavior | `DOMAIN_RULES.md`, `INVARIANTS.md`, `FAILURE_MODES.md`, `README.md` | Any relaxed or added gate |
| Source adapters | `hackathon_hunter/sources/`, `collect.py` | `ARCHITECTURE.md`, `FEATURE_MAP.md`, `DATA_FLOW.md`, `RUNBOOK.md` | New external source or trust model |
| Storage / reports | `hackathon_hunter/storage.py`, `reports.py` | `DATA_FLOW.md`, `RUNBOOK.md`, `FEATURE_MAP.md` | Artifact naming or retention change |
| Generated projects | `hackathon_hunter/workflows/build_spec.py`, `qa.py`, `projects/` | `UI_MAP.md`, `DOMAIN_RULES.md`, `RUNBOOK.md`, `CONTEXT_PACKS/generated_projects.md` | Submission material policy change |
| Run logs | `hackathon_hunter/runlog.py`, log model | `ARCHITECTURE.md`, `DATA_FLOW.md`, `CURRENT_STATE.md` | Audit-trail semantics change |
| Tests | `tests/` | `RUNBOOK.md`, affected feature docs | Usually no, unless behavior is redefined |
| Harness / docs | `AI_CONTEXT.md`, `AGENTS.md`, `docs/`, `tasks/`, `scripts/docs/` | `CONTEXT_INDEX.md`, `CURRENT_STATE.md`, `COMPRESSION_RULES.md` | Harness policy or source-of-truth change |

## Required Final Handoff Field

```text
Doc Sync Status: updated / not impacted / deferred with reason
```
