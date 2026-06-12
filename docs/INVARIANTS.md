---
title: Invariants
owner: human
status: active
last_updated: 2026-06-12
---

# Invariants

- Final submission is never automatic.
- The five human gates remain explicit.
- `deadline` and `Evidence.fetched_at` are timezone-aware.
- `ai_policy=unknown` or `ai_policy=forbidden` blocks build-stage ranking.
- Rules checks always require human review.
- Expired hackathons are not active candidates.
- Stale evidence creates warnings; it is not silently ignored.
- Scoring trace must remain explainable.
- Generated projects must not contain real secrets.
- Synthetic data and fallback behavior must be disclosed.
- Run logs should record workflow inputs, outputs, errors, status, and timestamps.
- Archived generated projects are historical examples unless explicitly targeted.
