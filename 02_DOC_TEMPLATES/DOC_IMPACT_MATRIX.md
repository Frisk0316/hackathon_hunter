---
created_at: 2026-06-12
updated_at: 2026-06-12
status: current
owner: ai-maintained
purpose: map code changes to documentation and test impact
when_to_read: TBD
when_to_update: TBD
---

# DOC_IMPACT_MATRIX


## How to Use

When files change, find the matching section and check required docs/tests.

## Frontend UI

If changing:

```text
frontend/
```

Must check:

```text
docs/UI_MAP.md
docs/FEATURE_MAP.md
docs/API_MAP.md
docs/CHANGELOG_AI.md
tests/frontend/
```

## API Contract

If changing:

```text
src/api/
backend/api/
frontend/api.js
```

Must check:

```text
docs/API_MAP.md
docs/FEATURE_MAP.md
docs/DATA_FLOW.md
tests/integration/
```

## Core / Business Logic

If changing:

```text
src/core/
src/domain/
services/
```

Must check:

```text
docs/DOMAIN_RULES.md
docs/INVARIANTS.md
docs/CHANGELOG_AI.md
docs/ADR/
tests/unit/
tests/integration/
```

## Data Pipeline / DB

If changing:

```text
src/data/
src/db/
db/
migrations/
ingest/
pipeline/
```

Must check:

```text
docs/DATA_FLOW.md
docs/DOMAIN_RULES.md
docs/RUNBOOK.md
docs/ADR/
```
