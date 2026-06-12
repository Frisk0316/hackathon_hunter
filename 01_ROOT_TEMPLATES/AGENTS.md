# AGENTS.md

Rules for GPT / Codex / generic coding agents working in this repo.

## Core Principle

This project must remain human-maintainable. Optimize for small, reviewable changes; clear file ownership; reproducible commands; updated documentation; easy rollback; and human learning.

## Mandatory Session Start

Before editing code, read:

1. `AI_CONTEXT.md`
2. `docs/CURRENT_STATE.md` if present
3. `docs/CONTEXT_INDEX.md` if present
4. `docs/ARCHITECTURE.md`
5. `docs/FEATURE_MAP.md`
6. `docs/RUNBOOK.md`
7. Relevant task spec / issue / review summary

Then run:

```bash
git status --short
```

Do not overwrite unrelated human, Claude, Codex, or GPT changes.

## Locate Before Edit

Before modifying files, identify:

1. Layer: frontend / API / service / data / DB / core / docs / infra
2. Likely files involved
3. Files that must not be touched
4. Smallest safe change
5. Tests/checks to run
6. Docs that may need updates

## Modification Rules

Unless explicitly requested:

- Do not rewrite unrelated files.
- Do not introduce new dependencies.
- Do not change public APIs.
- Do not change database schema.
- Do not reformat entire files.
- Do not perform large refactors.
- Do not silently change behavior.

## Doc Sync Required

For every change, classify whether it affects business logic, API contract, data flow, UI behavior, database schema, test behavior, or runtime behavior.

If affected:

1. Create or update Change Manifest if non-trivial.
2. Consult `docs/DOC_IMPACT_MATRIX.md` if present.
3. Update impacted docs.
4. Add entry to `docs/CHANGELOG_AI.md` if present.
5. Add ADR for major business rules, formulas, data semantics, schema, or architecture decisions.
6. Include doc sync status in final handoff.

## Intelligence Harness Required for Non-Trivial Tasks

Before non-trivial design or implementation, perform design-space expansion:

1. What problem are we actually solving?
2. What assumptions are being made?
3. What are at least 3 possible approaches?
4. What are tradeoffs?
5. What is the smallest validation?
6. What would go wrong if this assumption is false?

After non-trivial tasks, include Human Learning Notes.

## Context Resilience Rule

Do not rely on chat history as source of truth. Never compress away uncertainty.

## Required Completion Report

```text
Goal:
Layer:
Files added:
Files changed:
Diff scope:
Tests/checks run:
Tests/checks failed or skipped:
Docs updated:
Doc Sync Status:
Risks:
Rollback plan:
Human Learning Notes:
Questions for human review:
Next recommended task:
```
