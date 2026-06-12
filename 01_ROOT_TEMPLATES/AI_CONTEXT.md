# AI_CONTEXT.md

## Project Name

[專案名稱]

## One-Sentence Summary

[用一句話說明這個專案在做什麼]

## Project Summary

[說明此 repo 的核心用途、主要模組、主要使用者、目前狀態]

## Primary Human Goal

The human user wants fast AI-assisted development while keeping the project human-maintainable, auditable, and safe from unreviewed complexity.

## Current Project Stage

- [ ] Prototype
- [ ] Research
- [ ] Internal tool
- [ ] Production candidate
- [ ] Production
- [ ] Archived

Current description:

[說明目前階段]

## Source of Truth

| Topic | Source |
|---|---|
| Project context | `AI_CONTEXT.md` |
| Agent rules | `AGENTS.md`, `CLAUDE.md` |
| Architecture | `docs/ARCHITECTURE.md` |
| Feature navigation | `docs/FEATURE_MAP.md` |
| Runtime / test / rollback | `docs/RUNBOOK.md` |
| Domain rules | `docs/DOMAIN_RULES.md` |
| API contract | `docs/API_MAP.md` |
| Data flow | `docs/DATA_FLOW.md` |
| Current state | `docs/CURRENT_STATE.md` |

## Human-Maintainability Rule

Every meaningful change must be traceable as:

```text
Feature / bug -> files touched -> data flow -> test command -> artifact/result -> rollback plan
```

If this cannot be explained, the change is not ready.

## Do Not Do Without Explicit Approval

- Do not rewrite unrelated files.
- Do not introduce new dependencies without explanation.
- Do not change public APIs without updating docs and tests.
- Do not change database schema casually.
- Do not change business logic silently.
- Do not make broad refactors during small bug fixes.
- Do not claim production/live readiness without evidence.
- Do not rely on chat history as source of truth.

## Preferred AI Workflow

1. Read context.
2. Run `git status --short`.
3. Locate before edit.
4. Propose minimal plan.
5. Implement scoped change.
6. Run relevant checks.
7. Update impacted docs.
8. Provide handoff with rollback plan.
9. Add learning notes if non-trivial.

## Main Risk Areas

- [列出高風險區，例如 auth、payments、trading、DB migration、P&L、external API 等]

## Definition of Done

- [ ] Code changed as intended
- [ ] Relevant tests/checks run or skipped with reason
- [ ] Relevant docs updated or explicitly marked not impacted
- [ ] Change summary provided
- [ ] Risks stated
- [ ] Rollback plan provided
- [ ] Human learning notes provided for non-trivial changes
