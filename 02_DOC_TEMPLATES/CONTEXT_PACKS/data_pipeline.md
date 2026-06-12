---
created_at: 2026-06-12
updated_at: 2026-06-12
status: current
owner: ai-maintained
purpose: compact data pipeline context for AI sessions
when_to_read: before ingestion, transformation, data quality, schema, or storage changes
when_to_update: when data flow or storage semantics change
---

# Data Pipeline Context Pack

## Purpose

[Describe data pipeline purpose]

## Main Flow

```text
source -> ingestion -> validation -> storage -> consumer -> output
```

## Key Files

- `[path]`
- `[path]`

## Safe Changes

- logging
- retry behavior with same semantics
- additional diagnostics
- non-breaking validation output

## Dangerous Changes

- changing canonical data semantics
- changing missing data treatment
- changing schema
- changing source priority without ADR

## Read More

- `docs/DATA_FLOW.md`
- `docs/DOMAIN_RULES.md`
- `docs/DOC_IMPACT_MATRIX.md`
