# AGENT_BRIEF — ClaimClear (for Codex)

You are the **coding worker** for ClaimClear, our UiPath AgentHack 2026 entry. Build the code; do not make strategy or submission decisions.

## Project goal
Agentic insurance-claims triage on **UiPath Maestro Case**: auto-clear straightforward claims, escalate ambiguous ones to a human, with a full audit trail. Full design in [SPEC.md](SPEC.md); ordered work in [TASKS.md](TASKS.md).

## Non-goals (do NOT build)
- Real payment rails, real carrier/EHR integration, multi-insurer support, mobile app.
- Any use of real claims data / PII. **Synthetic data only.**
- Auto-submitting the hackathon or posting socials.

## Required platform
- **UiPath Automation Cloud is the required orchestration layer** — agents are nodes inside a Maestro Case process, not a standalone web app.
- Build the agent logic + demo so it runs **locally first** (without cloud) for fast iteration, then package as UiPath **Coded Agents** and document the cloud wiring in `docs/UIPATH_SETUP.md`.

## Build order (see TASKS.md)
Skeleton+synthetic data → local agent pipeline (intake / policy-check / risk-scoring / router / audit + exception path) → UiPath Maestro packaging → submission-quality docs.

## Demo path (must work)
1. Clean claim → auto-cleared in seconds, audit shown.
2. Ambiguous claim → low confidence → human approval queue → approve → resolved.
3. Audit log: every agent decision + rationale traceable.

## Acceptance tests
- `make demo` (or documented script) runs both scenarios end-to-end on synthetic data.
- Clean claim auto-approves; ambiguous escalates; both write audit entries.
- `pytest` green; lint clean; smoke test passes.
- README: Quick Start, Tech Stack, Architecture, **synthetic-data disclosure**, zero placeholder tokens (`TODO`, `TBD`, `VIDEO_URL`, `DEMO_URL`, `YOUR_REPO`, `PLACEHOLDER`).
- `.env.example` present; **no real keys/secrets committed.**

## Bonus (do not skip — free points)
Maintain `AGENT_BUILD_LOG.md` documenting that this was built with **Claude Code + Codex**. The judging page explicitly awards bonus points for coding-agent usage. Be specific (what each agent did, key decisions).

## Time budget
~30–32 hours of build within the window to 2026-06-29. Prioritize a flawless core demo (clean + ambiguous + audit) over breadth.

## Files / boundaries
- Work only inside `projects/uipath-agenthack-2026/claimclear/`.
- Do **not** modify the hunter pipeline (`hackathon_hunter/`), other `projects/*`, `data/`, `reports/`, or `strategy/`.
- Secrets go in `.env` (gitignored); only `.env.example` is committed.

## Stop-and-ask (human gates)
Pause and request the human for: UiPath account creation, building the Maestro Case process in the cloud, recording the demo video, publishing the repo, and final Devpost submission. Never fake these.

## Done = produce `HANDOFF_REPORT.md`
When your portion is complete, write `HANDOFF_REPORT.md`: what's done, what remains manual (cloud wiring, video, submit), and exact run instructions.
