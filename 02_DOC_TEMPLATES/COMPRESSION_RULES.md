---
created_at: 2026-06-12
updated_at: 2026-06-12
status: current
owner: ai-maintained
purpose: define what cannot be lost during context summarization
when_to_read: TBD
when_to_update: TBD
---

# COMPRESSION_RULES


When summarizing or compacting context, never omit:

1. Current branch and commit
2. Active task
3. Do-not-touch files
4. Business logic changes
5. Open risks
6. Failed tests
7. Tests skipped with reason
8. Docs that need updates
9. ADRs required
10. Human decisions not yet made
11. Rollback plan
12. Uncertainty

Never compress away uncertainty.

If information is unknown, write `unknown`, not an invented answer.
