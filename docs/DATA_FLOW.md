---
title: Data Flow
owner: human
status: active
last_updated: 2026-06-12
---

# Data Flow

## Collection Flow

```text
source adapter
  -> SourceResult
  -> raw snapshot in data/raw
  -> filter by deadline/prize/format
  -> deduplicate
  -> apply freshness
  -> processed JSON in data/processed
  -> rank active candidates
  -> radar report in reports
  -> run log in logs
```

## Import Flow

```text
external radar JSON
  -> Pydantic validation
  -> optional merge with latest processed file
  -> evidence recency decides field replacement
  -> import diff report
  -> processed JSON unless dry-run
  -> run log
```

## Ranking Flow

```text
processed Hackathon records
  -> reject expired / inside buffer / AI-policy-blocked / Taiwan-gate-blocked
  -> score remaining candidates
  -> compute evidence quality and delivery risk
  -> sort by overall score
  -> render report
```

## Project Generation Flow

```text
selected hackathon + selected idea
  -> build-spec workflow
  -> projects/<hackathon-id>/<idea-id>
  -> SPEC / TASKS / README / SUBMISSION_DRAFT / AGENT_BRIEF
  -> human builds or delegates project
  -> QA workflow
  -> QA_REPORT / SUBMISSION_PACKAGE
```

## Outcome Learning Flow

```text
submitted / finalist / winner / rejected / abandoned outcome
  -> result record
  -> calibrate workflow
  -> scoring suggestions report
  -> human review before config changes
```

## Important Data Semantics

- `deadline` and `Evidence.fetched_at` must be timezone-aware.
- `source_evidence` is field-level provenance, not generic notes.
- Missing or stale evidence should reduce confidence or produce warnings.
- `ai_policy=unknown` blocks build-stage ranking.
- `human_review_required=True` remains true by design for rules checks.
