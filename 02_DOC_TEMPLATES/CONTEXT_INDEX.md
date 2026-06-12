---
created_at: 2026-06-12
updated_at: 2026-06-12
status: current
owner: ai-maintained
purpose: tell AI what to read based on task type
when_to_read: TBD
when_to_update: TBD
---

# CONTEXT_INDEX


## Always Read First

- `AI_CONTEXT.md`
- `AGENTS.md`
- `docs/CURRENT_STATE.md`
- `docs/FEATURE_MAP.md`
- task spec

## If Task Involves Frontend

Read:

- `docs/UI_MAP.md`
- `docs/API_MAP.md`
- `docs/CONTEXT_PACKS/frontend.md`
- relevant frontend files

## If Task Involves API

Read:

- `docs/API_MAP.md`
- `docs/DATA_FLOW.md`
- relevant API route files
- relevant frontend API callers

## If Task Involves Core / Business Logic

Read:

- `docs/DOMAIN_RULES.md`
- `docs/INVARIANTS.md`
- relevant ADRs
- relevant tests

## If Task Involves Data Pipeline

Read:

- `docs/DATA_FLOW.md`
- `docs/CONTEXT_PACKS/data_pipeline.md`
- data quality tests
