---
created_at: 2026-06-12
updated_at: 2026-06-12
status: current
owner: ai-maintained
purpose: compact backend/API context for AI sessions
when_to_read: before API route, service, schema, or server changes
when_to_update: when backend structure or API ownership changes
---

# Backend Context Pack

## Purpose

[Describe backend/API purpose]

## Key Files

- `[path]`
- `[path]`

## Safe Changes

- validation messages
- non-breaking response additions
- logging
- internal refactor with tests

## Dangerous Changes

- changing public API contract
- changing business logic inside route handlers
- bypassing service/core layer
- altering persistence semantics

## Read More

- `docs/API_MAP.md`
- `docs/DATA_FLOW.md`
- `docs/ARCHITECTURE.md`
