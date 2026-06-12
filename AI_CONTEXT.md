---
title: Hackathon Hunter AI Context
owner: human
status: active
last_updated: 2026-06-12
---

# AI Context

## Project Name

Hackathon Hunter

## One-Sentence Summary

Hackathon Hunter is a Python CLI pipeline for finding evidence-backed online hackathons, ranking opportunities, generating project specs, and preparing human-reviewed submission packages.

## Project Summary

This repo is an internal research and planning tool. It discovers hackathon candidates from source adapters, normalizes them into Pydantic models, preserves evidence, ranks opportunities with transparent scoring, checks eligibility and submission gates, creates project specs, and records outcomes for future calibration.

The package code lives in `hackathon_hunter/`. Operational artifacts live in `data/`, `reports/`, `strategy/`, `logs/`, and `projects/`. Historical generated submissions live under `archive/generated-projects/` and are not the hunter system itself.

## Primary Human Goal

The human wants fast AI-assisted hackathon research and build planning while keeping the repo auditable, evidence-based, and safe from automatic submission or silent rule changes.

## Current Project Stage

- [ ] Prototype
- [x] Research
- [x] Internal tool
- [ ] Production candidate
- [ ] Production
- [ ] Archived

Current description: working Python CLI with tests, scoring/rules workflows, import/status/run-log support, and a full AI-native harness added for long-session continuity.

## Source Of Truth

| Topic | Source |
|---|---|
| Project context | `AI_CONTEXT.md` |
| Agent rules | `AGENTS.md`, `CLAUDE.md` |
| Architecture | `docs/ARCHITECTURE.md` |
| Feature navigation | `docs/FEATURE_MAP.md` |
| Runtime / test / rollback | `docs/RUNBOOK.md` |
| Domain rules | `docs/DOMAIN_RULES.md` |
| API / CLI contract | `docs/API_MAP.md` |
| Data flow | `docs/DATA_FLOW.md` |
| Current state | `docs/CURRENT_STATE.md` |
| Context loading | `docs/CONTEXT_INDEX.md` |
| Harness templates | `00_PLAYBOOK/`, `01_ROOT_TEMPLATES/`, `02_DOC_TEMPLATES/`, `03_TASK_TEMPLATES/`, `04_PROMPTS/`, `05_SCRIPTS/` |

## Human-Maintainability Rule

Every meaningful change must be traceable as:

```text
Feature / bug -> files touched -> data flow -> test command -> artifact/result -> rollback plan
```

If this cannot be explained, the change is not ready.

## Do Not Do Without Explicit Approval

- Do not automatically register for a hackathon.
- Do not automatically accept official rules.
- Do not automatically publish a repo, demo, video, or social post.
- Do not automatically submit a final entry.
- Do not commit secrets, cookies, real API keys, or private keys.
- Do not treat LLM guesses as facts.
- Do not relax evidence, eligibility, deadline, or AI-policy gates silently.
- Do not modify archived generated projects unless the task explicitly targets them.
- Do not rewrite unrelated files or do broad refactors during small fixes.
- Do not rely on chat history as the source of truth.

## Preferred AI Workflow

1. Read `AI_CONTEXT.md`.
2. Read `docs/CURRENT_STATE.md` and `docs/CONTEXT_INDEX.md`.
3. Read the relevant context pack from `docs/CONTEXT_PACKS/`.
4. Run `git status --short`.
5. Locate before edit using `docs/FEATURE_MAP.md`.
6. Implement the smallest safe change.
7. Run relevant checks.
8. Check doc impact with `make docs-impact`.
9. Update impacted docs.
10. Provide handoff with risks, rollback plan, and human learning notes for non-trivial work.

## Main Risk Areas

- Evidence-backed fields: `deadline`, `eligibility`, `ai_policy`, prizes, and required APIs.
- Scoring weights and constants in `hackathon_hunter/scoring.py` and `config/scoring.yaml`.
- Human-gate behavior in `hackathon_hunter/rules.py`, `build_spec.py`, and `qa.py`.
- Generated project artifacts under `projects/`.
- Run logs under `logs/runs/`.
- Source adapters that may rely on network research.

## Definition Of Done

- [ ] Change is scoped to the requested feature, bug, docs, or harness work.
- [ ] Relevant tests/checks ran, or skipped checks are named with reasons.
- [ ] Impacted docs are updated, or doc impact is explicitly marked not applicable.
- [ ] Evidence and human-gate rules remain intact.
- [ ] Risks and rollback plan are stated.
- [ ] Handoff includes current artifacts and next recommended task when useful.
