---
title: API Map
owner: human
status: active
last_updated: 2026-06-12
---

# API Map

This repo has no HTTP API. The public contract is the CLI plus Pydantic data models and JSON artifact shapes.

## CLI Contract

| Command | Purpose | Important options | Notes |
|---|---|---|---|
| `collect` | Collect candidates and write processed radar/report | `--days-ahead`, `--min-prize-usd`, `--online-only`, `--mock`, `--source` | Writes raw snapshot first; mock overwrites mock processed file |
| `rank` | Rank processed candidates | `--input`, `--profile`, `--weights`, `--fast-lane` | Rejects blocked candidates before scoring |
| `import` | Validate and import evidence-backed radar JSON | `--input`, `--merge`, `--dry-run` | Produces diff report; merge uses evidence recency |
| `status` | Show active candidate dashboard | `--input` | Read-only summary |
| `watch` | Emit deadline/stale-evidence events | `--input`, `--fast-lane-days` | Exits 2 when events exist |
| `check-rules` | Check eligibility and submission gates | `--input` | Always requires human review |
| `analyze-winners` | Create winner intelligence template | `--hackathon-id`, `--input` | Evidence-gated research artifact |
| `ideate` | Generate project idea candidates | `--hackathon-id`, `--n`, `--input` | Ideas require human selection |
| `build-spec` | Generate project package scaffold | `--hackathon-id`, `--idea-id`, `--input` | Does not publish or submit |
| `qa` | Check generated submission package | `--project` | Advisory only |
| `record-result` | Record outcome learning | command options in `cli.py` | Feeds calibration |
| `calibrate` | Suggest scoring adjustments | `--results-dir`, `--weights` | Suggestions require human review |

## Core Data Contracts

| Model | File | Meaning |
|---|---|---|
| `Evidence` | `hackathon_hunter/models.py` | Field-level source quote, URL, fetched time, confidence |
| `Hackathon` | `hackathon_hunter/models.py` | Normalized candidate with timezone-aware deadline and evidence |
| `ScoreBreakdown` | `hackathon_hunter/models.py` | ROI, feasibility, strategic fit, evidence quality, delivery risk, trace |
| `RulesCheckResult` | `hackathon_hunter/models.py` | Blocking issues, warnings, submission requirements, human-review flag |
| `ProjectIdea` | `hackathon_hunter/models.py` | Human-selectable project idea |
| `RunLog` | `hackathon_hunter/models.py` | Workflow audit record |
| `ResultRecord` | `hackathon_hunter/models.py` | Outcome and calibration input |

## Contract Change Rule

Any change to CLI commands/options, JSON model fields, model validation, exit codes, report outputs, or generated project file names requires tests and doc updates.
