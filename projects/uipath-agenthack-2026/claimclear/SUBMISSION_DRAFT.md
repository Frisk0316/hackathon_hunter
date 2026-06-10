# SUBMISSION DRAFT — ClaimClear (Devpost)

> Draft only. **Final submission is a human gate** — do not auto-submit. Links marked "(provided at publish time)" are filled by the human after the repo/video are public.

**Title:** ClaimClear — agentic insurance-claims triage that clears the easy 80% and escalates the rest, with a full audit trail

**Track:** Maestro Case

---

**Inspiration**
Routine property & casualty claims commonly take 7–15 days to triage because simple claims queue behind complex ones. The cost isn't just speed — manual triage is inconsistent and hard to audit. We wanted to show that agentic orchestration can clear the straightforward majority instantly while keeping a human in control of the ambiguous minority.

**What it does**
ClaimClear runs a UiPath Maestro Case process that orchestrates three agents, one RPA bot, and one human approval gate. An intake agent extracts claim data; a policy-check bot validates coverage; a risk-scoring agent emits a confidence score with a written rationale; a decision router auto-approves clear, low-risk claims and escalates ambiguous ones to a human queue. Every step is recorded in an audit log, so each decision is fully traceable.

**How we built it**
UiPath Maestro Case is the orchestration spine; the agents are nodes inside the case, not a standalone app. Agent reasoning uses Claude-compatible coded-agent hooks; the deterministic local demo keeps scoring repeatable. We built and tested the agent pipeline locally on a synthetic claims dataset for fast iteration, then documented the Maestro wiring and human approval task setup. The whole project was built using coding agents — Claude Code for planning and Codex for implementation — documented in our AGENT_BUILD_LOG.

**Challenges we ran into**
Designing the escalation boundary — when an agent should defer to a human rather than auto-decide — took the most iteration. We made the confidence threshold and risk flags explicit and visible, so the governance decision is auditable rather than a black box.

**Accomplishments we're proud of**
A working end-to-end flow that handles a real exception path: an ambiguous claim is detected, escalated, approved by a human, and resolved — all captured in the audit trail. Governance is a visible feature, not an afterthought.

**What we learned**
Orchestration plus visible human governance beats a clever single agent. Handling the unhappy path is where agentic automation earns trust.

**What's next for ClaimClear**
Real carrier-system connectors, multi-line support beyond P&C, and a Marketplace-ready Maestro Case template.

**Built with**
UiPath Maestro Case, UiPath Agent Builder, UiPath Coded Agents, Claude, Python, Claude Code, Codex.

---

## Data disclosure (required)
ClaimClear runs entirely on a **synthetically generated claims dataset**. No real claims, policies, or personal data are used. This is stated in the README and demo.

## Submission assets checklist (human-completed)
- GitHub repo (public): provided at publish time (human gate 3).
- Demo video (≤5 min): recorded by human following the SPEC demo script.
- UiPath project / Maestro Case: built per `docs/UIPATH_SETUP.md`.
- Presentation deck: UiPath-provided template.
