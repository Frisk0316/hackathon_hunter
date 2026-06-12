---
title: Task Template
owner: human
status: template
last_updated: 2026-06-12
---

# Task Spec

## Task Title

[任務名稱]

## Goal

[這次要完成什麼]

## Layer

- [ ] Docs
- [ ] Frontend
- [ ] API
- [ ] Data
- [ ] Core logic
- [ ] Infra
- [ ] Tests

## Current Problem

[目前遇到什麼問題]

## Expected Behavior

[修好後應該長怎樣]

## Scope

Allowed to change:

- `[path]`

Not allowed to change:

- `[path]`

## Constraints

- Prefer minimal changes.
- Do not refactor unrelated code.
- Do not add dependencies unless necessary.
- Update impacted docs after changes.
- Provide rollback plan.

## Test Plan

```bash
make verify
```

## Docs to Update

- [ ] `docs/FEATURE_MAP.md`
- [ ] `docs/RUNBOOK.md`
- [ ] `docs/CHANGELOG_AI.md`

## Definition of Done

- [ ] Code changed
- [ ] Tests run
- [ ] Docs impact checked
- [ ] Docs updated if needed
- [ ] Handoff summary written
- [ ] Rollback plan provided
