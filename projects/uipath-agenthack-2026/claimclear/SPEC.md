# SPEC — ClaimClear

> UiPath AgentHack 2026 · Track: **Maestro Case** · deadline 2026-06-29 23:45 EDT
> Source strategy: [reports/winners/uipath-agenthack-2026.md](../../../reports/winners/uipath-agenthack-2026.md), [strategy/uipath-agenthack-2026_ideas.json](../../../strategy/uipath-agenthack-2026_ideas.json) (idea-001).

## One-liner
Agentic insurance-claims triage that auto-clears the straightforward majority and escalates ambiguous claims to a human, with a full audit trail.

## Problem (lead with this number)
Routine P&C claims commonly take **7–15 days** to triage because simple claims queue behind complex ones. Manual triage is slow, inconsistent, and hard to audit.

## Solution
A **Maestro Case**-orchestrated flow with three agents + one bot + one human gate:
1. **Intake agent** — extracts structured claim data from a submitted form/PDF.
2. **Policy-check bot** (RPA) — validates the claim against a policy record (coverage, limits, status).
3. **Risk-scoring agent** — scores fraud/ambiguity; emits a confidence score + rationale.
4. **Decision router** — auto-approves clear, low-risk claims; routes ambiguous/low-confidence claims to a **human approval task**.
5. **Audit log** — every step, input, agent rationale, and decision recorded and viewable.

## Architecture
```
Claim (form/PDF)
  -> [Maestro Case process]
       -> Intake Agent (extract)            (Agent Builder / coded agent)
       -> Policy-Check Bot (validate)       (UiPath RPA, mocked policy store)
       -> Risk-Scoring Agent (score+why)    (coded agent, Claude reasoning)
       -> Decision Router
            -> auto-approve  (clear + low risk)
            -> Human Approval Gate (ambiguous)  -> resolve
       -> Audit Log (every decision)
```
- **Orchestration layer = UiPath Maestro Case (required).** Agents are nodes inside the orchestrated case, not a standalone app.
- Coded agents are Python (UiPath Coded Agents). LLM reasoning via Claude.

## Scope
**MVP (must build):**
- Ingest a claim (form or PDF) → structured fields.
- Validate against a (mock) policy record.
- Risk-score with an agent → confidence + rationale.
- Auto-approve clear cases.
- Escalate ambiguous case to a human approval task.
- Audit log of every decision.

**Explicitly NOT doing:** real payment rails; multi-insurer support; mobile app; real carrier/EHR integration.

## Data
- **Synthetic claims dataset only** (generated). No real PII/claims data.
- Must be disclosed as synthetic in README + submission (QA fails undisclosed synthetic data).

## Demo script (the win is here — H3)
1. Submit a **clean** claim → auto-cleared in seconds, audit trail shown.
2. Submit an **ambiguous** claim → risk agent emits low confidence → routed to **human approval queue** → human approves → resolved.
3. Show the audit log: every agent decision + rationale is traceable.
*(The deliberate exception path + human governance is the differentiator.)*

## Judging alignment (map every feature to a criterion)
| feature | criterion it scores |
|---|---|
| Maestro Case orchestration of bot+agents+human | Platform Usage |
| 7–15 day cycle-time ROI framing | Business Impact & Adoption |
| deliberate ambiguous-claim exception path | Technical Execution (edge cases) |
| human approval gate + audit log | governance (required across tracks) |
| working prototype + README + 5-min video | Completeness |
| `AGENT_BUILD_LOG.md` (Claude Code + Codex) | **Bonus points** |

## Required platform / accounts (HUMAN steps — not Codex)
- UiPath Automation Cloud account (Community is fine) — **human must create.**
- Maestro Case process + Agent Builder agents configured in UiPath cloud — **human-assisted** (Codex provides the coded-agent logic + configuration notes, but cloud-side modeling is manual/low-code).
- `ANTHROPIC_API_KEY` for Claude reasoning — `.env`, never committed.

## Acceptance criteria
- One-command local run of the coded-agent + synthetic data demo (`make demo` or documented script).
- Clean claim auto-approves; ambiguous claim escalates; both produce audit entries.
- README has Quick Start, Tech Stack, Architecture, **synthetic-data disclosure**, and no unresolved scaffold markers.
- `AGENT_BUILD_LOG.md` documents the Claude→Codex build.
- `.env.example` present; no secrets committed.

## Human gates (do not auto-pass)
1. Confirm registration on UiPath AgentHack + accept rules.
2. Confirm this idea/scope (DONE — user picked idea-001).
3. Confirm public repo/demo before publishing.
4. Confirm any social post manually.
5. Confirm final Devpost submission manually.
