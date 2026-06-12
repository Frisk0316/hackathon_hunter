---
title: Compression Rules
owner: human
status: active
last_updated: 2026-06-12
---

# Compression Rules

When summarizing or handing off a long AI session, do not omit:

1. Current branch and dirty files.
2. Active task and why it matters.
3. Files changed.
4. Files intentionally not touched.
5. Business logic, scoring, evidence, or human-gate changes.
6. Failed tests and exact failure names.
7. Skipped tests with reasons.
8. Docs impacted and doc sync status.
9. ADRs needed.
10. Human decisions still required.
11. Rollback plan.
12. Uncertainty.

Never compress "unknown" into "confirmed".

## Handoff Shape

```text
Goal:
What changed:
Current files:
Checks run:
Checks failed/skipped:
Docs updated:
Risks:
Human decisions:
Rollback:
Next step:
```
