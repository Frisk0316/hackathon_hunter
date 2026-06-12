---
title: Generated Projects Context Pack
owner: human
status: active
last_updated: 2026-06-12
---

# Generated Projects Context Pack

Load this for `ideate`, `build-spec`, generated project files, QA, and submission package work.

## Core Files

- `hackathon_hunter/workflows/ideate.py`
- `hackathon_hunter/workflows/build_spec.py`
- `hackathon_hunter/workflows/qa.py`
- `hackathon_hunter/models.py`
- `tests/test_qa.py`
- `tests/test_workflows.py`

## Generated Output

Generated project packages live under `projects/`. Historical examples live under `archive/generated-projects/`.

## Safety Notes

- Generated projects are drafts until human review.
- Keep repo/demo/video/social/final submission as pending human actions.
- Require `.env.example`; reject real `.env` files in generated packages.
- Disclose synthetic data and fallbacks in README and submission drafts.
