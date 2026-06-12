---
title: Domain Rules
owner: human
status: active
last_updated: 2026-06-12
---

# Domain Rules

## Non-Negotiable Human Gates

The system must never claim these actions happened unless the human performed or explicitly confirmed them:

1. Hackathon registration.
2. Project idea selection.
3. Public repo or demo publication.
4. Social post publication.
5. Final submission.

## Evidence Rules

- Deadline, eligibility, AI policy, prize, cash prize, and required API claims need source evidence.
- Evidence must include a field name, optional URL, optional quote, timezone-aware fetched time, and confidence.
- Low-confidence evidence should warn.
- Stale evidence should warn before committing to a build.
- LLM-generated guesses are not evidence.

## Deadline Rules

- Expired hackathons are not active candidates.
- Normal ranking requires the configured active buffer.
- Fast-lane mode may use a smaller buffer, but must preserve explicit risk.
- Deadline comparisons require timezone-aware datetimes.

## AI Policy Rules

- `allowed` can proceed.
- `restricted` requires human review.
- `unknown` blocks build stage.
- `forbidden` blocks build stage.

## Eligibility Rules

- Region restriction blocks until human verification.
- Student-only requirements block by default for this user's workflow.
- Team-required status warns unless it otherwise blocks participation.
- Taiwan/account eligibility must remain explicit.

## Scoring Rules

- Ranking is transparent: every score should be explainable through the trace.
- Scoring constants are configuration and domain semantics, not cosmetic values.
- Calibration reports suggest changes only; humans approve scoring changes.
- Delivery risk reduces overall confidence and must not be removed silently.

## Generated Project Rules

- Generated projects must include `.env.example` and must not include real secrets.
- Synthetic data or fallback behavior must be disclosed in README and submission materials.
- Submission drafts are drafts only.
- QA is advisory, not approval to submit.
