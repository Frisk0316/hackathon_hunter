---
title: Context Index
owner: human
status: active
last_updated: 2026-06-12
---

# Context Index

Use this file to load only the context needed for a task.

## Always Load

1. `AI_CONTEXT.md`
2. `AGENTS.md`
3. `docs/CURRENT_STATE.md`
4. `docs/FEATURE_MAP.md`
5. `docs/RUNBOOK.md`

## Load By Task

| Task type | Additional context |
|---|---|
| CLI command change | `docs/API_MAP.md`, `docs/UI_MAP.md` |
| Workflow change | `docs/ARCHITECTURE.md`, `docs/DATA_FLOW.md`, matching tests |
| Scoring or ranking | `docs/DOMAIN_RULES.md`, `docs/MENTAL_MODELS.md`, `docs/INVARIANTS.md`, `docs/CONTEXT_PACKS/data_pipeline.md`, `config/scoring.yaml` |
| Rules or safety gate | `docs/DOMAIN_RULES.md`, `docs/INVARIANTS.md`, `docs/FAILURE_MODES.md` |
| Source adapter | `docs/DATA_FLOW.md`, `docs/CONTEXT_PACKS/data_pipeline.md` |
| Generated project | `docs/CONTEXT_PACKS/generated_projects.md`, `docs/DOMAIN_RULES.md`, `docs/UI_MAP.md` |
| Harness or docs | `docs/DOC_IMPACT_MATRIX.md`, `docs/COMPRESSION_RULES.md`, `00_PLAYBOOK/` |

## Artifact Map

| Artifact | Meaning |
|---|---|
| `data/processed/*.json` | Normalized radar state |
| `reports/*.md` | Human-readable analysis and diffs |
| `strategy/*.json` | Generated ideas |
| `projects/` | Active generated project packages |
| `logs/` | Workflow audit logs |
| `CODEX_HANDOFF.md` | Older planning handoff |

## Context Budget Rule

If context is tight, load source-of-truth docs first, then only the code files named by `FEATURE_MAP.md`. Do not load archive projects unless the task targets them.
