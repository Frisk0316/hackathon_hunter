<claude-mem-context>
# Memory Context

# [hackathon_hunter] recent context, 2026-06-12 10:57pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 48 obs (21,718t read) | 701,307t work | 97% savings

### Jun 10, 2026
146 1:42p 🔵 ClaimClear Project — UiPath AgentHack 2026 Initial State Surveyed
147 1:43p 🔵 ClaimClear — Full Spec, Tasks, and Submission Draft Surveyed
150 1:44p ✅ ClaimClear — Directory Scaffold Created
152 1:46p 🟣 ClaimClear — Core Scaffold, Data Models, and Synthetic Dataset Written
153 1:48p 🟣 ClaimClear — Full Agent Pipeline Implemented: 8 Modules + Tests
161 2:07p 🔵 Hackathon Hunter — Round 2 Improvement Plan for Codex Reviewed
163 2:08p 🔵 Hackathon Hunter — P0 Bug Confirmation via Code Inspection
164 2:10p 🔴 Hackathon Hunter — storage.py P0 Bugs Fixed: Overwrite Protection + mtime-Based File Selection
168 2:11p 🔴 Hackathon Hunter — collect.py, cli.py, rules.py P0 Fixes Applied
172 " 🟣 Hackathon Hunter — Hackathon.stale_evidence_fields() Method Added to models.py
175 2:12p 🟣 Hackathon Hunter — watch.py Workflow Created: Deadline Watch with EXPIRED/ENTERING_FAST_LANE/STALE_EVIDENCE Events
176 2:14p 🔴 Hackathon Hunter — Pytest Run: 16/17 Pass; test_mock_cli_end_to_end Fails + Two Ruff Errors Fixed
181 2:15p 🔴 Hackathon Hunter — test_mock_cli_end_to_end Regression Fixed + README Updated + Ruff Import Sorted
184 " ✅ Hackathon Hunter — P0 Round 2 Complete: 17/17 Tests Pass, Ruff Clean
185 " ✅ Hackathon Hunter — P0 Round 2 Fully Verified: 18/18 Tests Pass, CLI Smoke Tests Confirmed
187 2:28p ⚖️ Hackathon Hunter — P1 Implementation Plan Initiated (5-Step Execution)
188 2:29p 🔵 Hackathon Hunter — P1 Pre-Implementation Audit: Workflow Patterns + Uncommitted State
191 " 🔵 Hackathon Hunter — scoring.py Full Pipeline: evidence_quality, delivery_risk, rank_hackathons
192 2:31p 🟣 Hackathon Hunter — import_hackathons.py: New Import Command with Validation, Merge, Diff, Dry-run
193 " 🟣 Hackathon Hunter — Evidence Staleness Wired into scoring.py + config/scoring.yaml
198 2:33p 🔄 Hackathon Hunter — watch.py + rules.py: STALE_EVIDENCE_FIELDS Centralized from scoring.py
199 " 🟣 Hackathon Hunter — runlog.py: WorkflowRunLog Context Manager for Structured Run Logging
200 " 🟣 Hackathon Hunter — status.py + cli.py: New status Command and import/RunLog Integration
207 2:34p 🟣 Hackathon Hunter — CLI: import + status Commands Added; rank + watch + collect Wrapped in RunLog
209 2:36p 🟣 Hackathon Hunter — CLI: Full RunLog Coverage for All Commands + Error Propagation to Logs
210 " 🟣 Hackathon Hunter — Tests: import_workflow, rules staleness, status + RunLog Integration Tests Added
212 2:38p 🔴 Hackathon Hunter — Ruff + NameError Fixes: watch.py, runlog.py, import_hackathons.py, status.py
213 " ✅ Hackathon Hunter — P1 QA Gate Passes: 23 Tests + Ruff Clean + Live Smoke Tests Verified
219 2:39p 🔵 Hackathon Hunter — examples/hackathons.sample.json Migrated to Evidence-Backed Schema; RunLog JSONs Gitignored
### Jun 12, 2026
220 10:46p ✅ AI-Native Harness Full Scaffold Requested for Active Project
222 " 🔵 AI-Native Harness Pack Confirmed: 50 Files in ai_native_harness_pack.zip
224 " 🔵 Harness Pack Template Contents Confirmed: AGENTS.md, AI_CONTEXT.md, and 3 Python Scripts
226 10:47p ✅ AI-Native Harness Pack Scaffold Deployed to hackathon_hunter Project Root
228 " ✅ Task Templates and Doc Scripts Deployed to Project-Standard Paths
231 10:48p ✅ AI_CONTEXT.md, AGENTS.md, CLAUDE.md, and Makefile Written with Project-Specific Harness Content
234 10:50p 🟣 Full Documentation Source-of-Truth Suite Written: 18 New docs/ Files
235 10:51p 🔵 Hackathon Hunter — AI-Native Harness Scripts and FEATURE_MAP Confirmed Present
237 10:52p ✅ Hackathon Hunter — Legacy Docs and Task Templates Patched with YAML Front Matter
239 " 🔴 Hackathon Hunter — check_doc_impact.py Fixed: Untracked Files + E501 Line Length
240 " 🟣 Hackathon Hunter — make verify Gate Passes: 31 Tests + Ruff Clean + Docs-Check + Docs-Impact
243 10:53p ✅ Hackathon Hunter — CURRENT_STATE.md Updated with Verified make verify Result
244 " 🔵 Hackathon Hunter — Full AI-Native Harness Pack File Tree Confirmed In Repo
247 10:54p 🔵 Hackathon Hunter — Git Index Readonly Error Requires Escalated Sandbox Permission for Staging
248 " ✅ Hackathon Hunter — AI-Native Harness Pack Staged: 88 Files, 5015 Insertions
252 " 🟣 Hackathon Hunter — AI-Native Harness Pack Committed and Pushed to GitHub
249 " ✅ Hackathon Hunter — AI-Native Harness Pack zip deleted, commit attempted
250 " 🔴 check_doc_impact.py — Ruff E501 line-length fixed with multi-line boolean expression
251 " ✅ docs/CURRENT_STATE.md — Updated with verified make verify results from 2026-06-12

Access 701k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>

# AGENTS.md

Rules for GPT / Codex / generic coding agents working in this repo.

## Core Principle

Hackathon Hunter must remain human-maintainable, evidence-backed, and safe. Optimize for small reviewable changes, clear file ownership, reproducible commands, updated documentation, easy rollback, and human learning.

## Mandatory Session Start

Before editing code, read:

1. `AI_CONTEXT.md`
2. `docs/CURRENT_STATE.md`
3. `docs/CONTEXT_INDEX.md`
4. `docs/ARCHITECTURE.md`
5. `docs/FEATURE_MAP.md`
6. `docs/RUNBOOK.md`
7. Relevant `docs/CONTEXT_PACKS/*.md`
8. Relevant task spec / issue / review summary

Then run:

```bash
git status --short
```

Do not overwrite unrelated human, Claude, Codex, or GPT changes.

## Locate Before Edit

Before modifying files, identify:

1. Layer: CLI / workflow / model / scoring / rule / source adapter / storage / report / generated project / docs / infra.
2. Likely files involved.
3. Files that must not be touched.
4. Smallest safe change.
5. Tests/checks to run.
6. Docs that may need updates.

## Modification Rules

Unless explicitly requested:

- Do not rewrite unrelated files.
- Do not introduce new dependencies.
- Do not change CLI behavior without updating `docs/API_MAP.md`, `docs/FEATURE_MAP.md`, tests, and README if user-facing.
- Do not change scoring, eligibility, evidence, stale-evidence, or submission-gate logic silently.
- Do not reformat entire files.
- Do not perform large refactors during a narrow task.
- Do not modify `archive/generated-projects/` unless the task is about historical examples.

## Hackathon Safety Rules

- Final submission always requires human approval.
- Registration, idea choice, public repo/demo, social post, and final submission are human gates.
- Unknown AI policy blocks build stage.
- Expired or inside-buffer deadlines must not be ranked as active unless fast-lane rules explicitly allow it.
- Synthetic data and fallbacks must be disclosed in project materials.
- Evidence is required for deadlines, rules, prizes, eligibility, and AI policy.

## Doc Sync Required

For every change, classify whether it affects business logic, CLI contract, data flow, scoring/rules behavior, generated-project behavior, test behavior, runtime behavior, or documentation.

If affected:

1. Consult `docs/DOC_IMPACT_MATRIX.md`.
2. Update impacted docs.
3. Add an ADR for major business rules, formulas, data semantics, or architecture decisions.
4. Include Doc Sync Status in final handoff.

Use:

```bash
make docs-check
make docs-impact
```

## Intelligence Harness Required For Non-Trivial Tasks

Before non-trivial design or implementation, expand the design space:

1. What problem are we actually solving?
2. What assumptions are being made?
3. What are at least 3 possible approaches?
4. What are tradeoffs?
5. What is the smallest validation?
6. What would go wrong if the assumption is false?

After non-trivial tasks, include Human Learning Notes.

## Context Resilience Rule

Do not rely on chat history as source of truth. Repo docs, git diff, tests, run logs, and generated artifacts are the source of truth. Never compress away uncertainty, failed tests, skipped checks, pending human decisions, or rollback plans.

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
