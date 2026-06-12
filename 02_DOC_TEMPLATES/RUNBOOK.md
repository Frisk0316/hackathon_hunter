---
created_at: 2026-06-12
updated_at: 2026-06-12
status: current
owner: ai-maintained
purpose: provide reproducible commands for setup, dev, tests, verification, rollback
when_to_read: TBD
when_to_update: TBD
---

# RUNBOOK


## Setup

```bash
make setup
```

## Start Development

```bash
make dev
```

## Run Tests

```bash
make test-unit
make test-integration
make test-frontend
```

## Documentation Checks

```bash
make docs-check
make docs-impact
```

## Full Verification

```bash
make verify
make verify-full
```

## Rollback

```bash
git status --short
git diff
git restore <file>
git revert <commit>
```

## Common Problems

### Problem: [Symptom]

Check:

1. [Step 1]
2. [Step 2]
3. [Step 3]
