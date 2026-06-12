---
title: Data Pipeline Context Pack
owner: human
status: active
last_updated: 2026-06-12
---

# Data Pipeline Context Pack

Load this for collection, import, ranking, rules, storage, reports, and scoring work.

## Core Files

- `hackathon_hunter/models.py`
- `hackathon_hunter/storage.py`
- `hackathon_hunter/workflows/collect.py`
- `hackathon_hunter/workflows/import_hackathons.py`
- `hackathon_hunter/workflows/rank.py`
- `hackathon_hunter/workflows/status.py`
- `hackathon_hunter/workflows/watch.py`
- `hackathon_hunter/scoring.py`
- `hackathon_hunter/rules.py`
- `config/scoring.yaml`

## Tests

- `tests/test_models.py`
- `tests/test_storage.py`
- `tests/test_import_workflow.py`
- `tests/test_scoring.py`
- `tests/test_rules.py`
- `tests/test_watch.py`
- `tests/test_workflows.py`

## Safety Notes

- Preserve timezone-aware validation.
- Preserve evidence recency merge semantics.
- Preserve AI-policy and eligibility gates.
- Preserve stale-evidence warnings.
- Do not overwrite non-mock processed outputs without `unique_path`.
