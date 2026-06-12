---
title: Feature Map
owner: human
status: active
last_updated: 2026-06-12
---

# Feature Map

Use this before editing. Locate the feature, read the files, then run the smallest relevant tests.

| Feature | User command / entry | Core files | Tests / checks | Outputs |
|---|---|---|---|---|
| CLI wiring | `python3 -m hackathon_hunter --help` | `hackathon_hunter/cli.py`, `hackathon_hunter/__main__.py` | `tests/test_workflows.py`, `python3 -m pytest` | console output |
| Collect radar | `python3 -m hackathon_hunter collect --mock` | `hackathon_hunter/workflows/collect.py`, `hackathon_hunter/sources/base.py`, `hackathon_hunter/sources/mock.py`, `hackathon_hunter/storage.py` | `tests/test_workflows.py`, `tests/test_storage.py` | `data/processed/`, `reports/` |
| Import evidence-backed radar | `python3 -m hackathon_hunter import --input examples/hackathons.sample.json --dry-run` | `hackathon_hunter/workflows/import_hackathons.py`, `hackathon_hunter/models.py`, `hackathon_hunter/storage.py` | `tests/test_import_workflow.py`, `tests/test_models.py` | `reports/import_diff_*.md`, `data/processed/` |
| Status dashboard | `python3 -m hackathon_hunter status --input data/processed/mock_hackathons.json` | `hackathon_hunter/workflows/status.py`, `hackathon_hunter/cli.py`, `hackathon_hunter/scoring.py` | `tests/test_workflows.py` | Rich table |
| Deadline watch | `python3 -m hackathon_hunter watch --input data/processed/mock_hackathons.json` | `hackathon_hunter/workflows/watch.py`, `hackathon_hunter/scoring.py`, `hackathon_hunter/models.py` | `tests/test_watch.py` | `reports/watch_*.md`, exit code 2 when events exist |
| Rank opportunities | `python3 -m hackathon_hunter rank --input data/processed/mock_hackathons.json` | `hackathon_hunter/workflows/rank.py`, `hackathon_hunter/scoring.py`, `config/scoring.yaml`, `config/user_profile.example.yaml` | `tests/test_scoring.py`, `tests/test_workflows.py` | `reports/radar_*.md` |
| Check rules | `python3 -m hackathon_hunter check-rules --input data/processed/mock_hackathons.json` | `hackathon_hunter/workflows/rules_check.py`, `hackathon_hunter/rules.py`, `hackathon_hunter/models.py` | `tests/test_rules.py`, `tests/test_workflows.py` | rules reports |
| Analyze winners | `python3 -m hackathon_hunter analyze-winners --hackathon-id ...` | `hackathon_hunter/workflows/analyze_winners.py`, `hackathon_hunter/storage.py` | `tests/test_workflows.py` | `reports/winners/` |
| Generate ideas | `python3 -m hackathon_hunter ideate --hackathon-id ...` | `hackathon_hunter/workflows/ideate.py`, `hackathon_hunter/models.py`, `hackathon_hunter/storage.py` | `tests/test_workflows.py` | `strategy/` |
| Build project spec | `python3 -m hackathon_hunter build-spec --hackathon-id ... --idea-id ...` | `hackathon_hunter/workflows/build_spec.py`, `hackathon_hunter/workflows/ideate.py`, `hackathon_hunter/models.py` | `tests/test_workflows.py` | `projects/` |
| QA generated project | `python3 -m hackathon_hunter qa --project projects/...` | `hackathon_hunter/workflows/qa.py` | `tests/test_qa.py` | QA report and submission package in the generated project |
| Record results | `python3 -m hackathon_hunter record-result ...` | `hackathon_hunter/workflows/results.py`, `hackathon_hunter/models.py` | `tests/test_results_calibration.py` | result records |
| Calibrate scoring | `python3 -m hackathon_hunter calibrate` | `hackathon_hunter/workflows/calibrate.py`, `hackathon_hunter/scoring.py`, `config/scoring.yaml` | `tests/test_results_calibration.py`, `tests/test_scoring.py` | calibration report |
| Run logs | all CLI commands | `hackathon_hunter/runlog.py`, `hackathon_hunter/models.py`, `hackathon_hunter/cli.py` | `tests/test_workflows.py`, `tests/test_import_workflow.py` | `logs/` |
| Harness docs | `make docs-check`, `make docs-impact` | `AI_CONTEXT.md`, `AGENTS.md`, `docs/`, `scripts/docs/`, `tasks/` | `make docs-check`, `make docs-impact` | warnings or pass output |

## Locate Before Edit Checklist

1. Identify whether the change is CLI, workflow, scoring/rules, model/schema, storage/reporting, source adapter, generated project, or harness docs.
2. Read the core files and matching tests above.
3. Check `docs/DOC_IMPACT_MATRIX.md` for required docs.
4. Keep generated artifacts separate from package code.
5. Preserve human gates.
