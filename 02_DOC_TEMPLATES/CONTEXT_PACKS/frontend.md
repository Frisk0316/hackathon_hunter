---
created_at: 2026-06-12
updated_at: 2026-06-12
status: current
owner: ai-maintained
purpose: compact frontend context for AI sessions
when_to_read: before frontend UI, chart, API caller, or state changes
when_to_update: when frontend structure or ownership changes
---

# Frontend Context Pack

## Purpose

[Describe frontend purpose]

## Key Files

- `[path]`
- `[path]`

## Safe Changes

- layout
- labels
- display formatting
- loading states
- non-semantic UI improvements

## Dangerous Changes

- computing business logic in frontend
- changing API response semantics
- mutating persisted result semantics
- silently transforming data interpretation

## Common Debug Flow

```text
visible label -> page/component -> state/props -> api client -> backend route -> CSS/chart helper
```

## Read More

- `docs/UI_MAP.md`
- `docs/API_MAP.md`
- `docs/FEATURE_MAP.md`
