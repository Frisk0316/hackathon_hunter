# TASKS — ClaimClear

Coding worker = Codex. Tasks are ordered so each milestone is demoable. Check off as completed.
Human-only steps are tagged **[HUMAN]** — Codex should stop and request them, not fake them.

## Milestone 0 — Account & platform (mostly human)
- [ ] **[HUMAN]** Create UiPath Automation Cloud (Community) account.
- [ ] **[HUMAN]** Register the team on the UiPath AgentHack Devpost; read & accept rules.
- [x] Codex: scaffold repo (`src/`, `data/`, `docs/`, `tests/`, `.env.example`, `.gitignore`, `Makefile`/`scripts/`).
- [x] Codex: add `AGENT_BUILD_LOG.md` and start logging the Claude→Codex workflow (bonus points).

## Milestone 1 — Skeleton + synthetic data
- [x] Codex: synthetic claims generator → realistic claim records (id, type, amount, policy_id, narrative, docs).
- [x] Codex: mock policy store (coverage, limits, status) keyed by policy_id.
- [x] Codex: `.env.example` with `ANTHROPIC_API_KEY=` (empty value only, never a real key).
- [x] Codex: basic CI (lint + tests).

## Milestone 2 — Core agent pipeline (local, runnable without UiPath cloud first)
- [x] Codex: **Intake agent** — parse a claim form/PDF → structured fields (deterministic + LLM fallback).
- [x] Codex: **Policy-check** — validate claim vs mock policy store; return pass/fail + reasons.
- [x] Codex: **Risk-scoring agent** — Claude-based; returns `confidence` (0–1) + `rationale` + `risk_flags`.
- [x] Codex: **Decision router** — rules: clear + high-confidence → auto-approve; else → escalate.
- [x] Codex: **Audit log** — append every step (input, agent output, decision) to a viewable log/JSON.
- [x] Codex: deliberate **exception path** — at least one synthetic claim that forces low confidence → escalation.

## Milestone 3 — UiPath Maestro Case integration
- [x] Codex: package the agents as **UiPath Coded Agents** (Python) consumable by Maestro.
- [x] Codex: write `docs/UIPATH_SETUP.md` — exact steps to build the Maestro Case process + wire agents + human approval task.
- [ ] **[HUMAN + Codex]** Build the Maestro Case process in UiPath cloud per the doc; connect bot + agents + human gate.
- [x] Codex: a small UI or CLI to submit a claim and view status/audit (so the demo is visual).

## Milestone 4 — Submission quality
- [x] Codex: README complete — Quick Start, Tech Stack, Architecture diagram, **synthetic-data disclosure**, no scaffold markers.
- [x] Codex: `make demo` (or documented script) runs the clean + ambiguous scenarios end-to-end.
- [x] Codex: smoke test passes; tests green; lint clean.
- [x] Codex: finalize `AGENT_BUILD_LOG.md`.
- [x] Codex: produce `HANDOFF_REPORT.md` (what's done, what's manual, how to run).
- [x] Codex: produce `VIDEO_SCRIPT.md` and `SUBMISSION_CHECKLIST.md` for human recording/submission.
- [ ] **[HUMAN]** Record ≤5-min demo video following demo script in SPEC.
- [ ] **[HUMAN]** Confirm public repo, then submit on Devpost.

## Done criteria (Codex's portion)
- Local `make demo` shows clean→auto-approve and ambiguous→escalate→resolve with audit entries.
- Coded agents packaged for UiPath; `UIPATH_SETUP.md` lets a human finish the cloud wiring.
- README + AGENT_BUILD_LOG + HANDOFF_REPORT complete; QA placeholder/secret checks pass.
