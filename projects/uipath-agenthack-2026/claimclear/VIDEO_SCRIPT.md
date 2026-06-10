# VIDEO SCRIPT — ClaimClear

Target length: 4:30 to 5:00. Keep the recording focused on the working flow: clean claim clears automatically, ambiguous claim escalates to a human, and every step is auditable.

## Pre-flight

- Run from `projects/uipath-agenthack-2026/claimclear`.
- Use a terminal large enough for the output to be readable.
- Have `README.md`, `docs/UIPATH_SETUP.md`, and `artifacts/demo_summary.json` ready in tabs.
- If the cloud Maestro Case is not fully wired yet, say this is the local coded-agent simulation of the documented Maestro flow.
- Open with the synthetic-data disclosure: no real claims, policies, or personal data.

## Recording Timeline

### 0:00-0:25 — Title and Problem

Narration:

> This is ClaimClear, a UiPath Maestro Case entry for insurance-claims triage. Routine claims often wait behind complex cases, so ClaimClear clears straightforward claims in seconds while routing ambiguous ones to a governed human approval path.

Show:

- `README.md` title and one-line description.
- Synthetic data disclosure section.

### 0:25-0:55 — Architecture

Narration:

> Maestro Case is the orchestration spine. The flow runs an intake agent, a policy-check bot, a risk-scoring agent, a decision router, and then either straight-through approval or a human approval task. Every step writes to an audit log.

Show:

- `README.md` architecture diagram.
- `docs/UIPATH_SETUP.md` section 4 process flow.

### 0:55-1:25 — Start the Demo

Command:

```bash
make demo
```

Narration:

> The local demo mirrors the Maestro flow with deterministic synthetic fixtures, so the judging walkthrough is repeatable without relying on live secrets or real carrier systems.

Show:

- Terminal header: `ClaimClear local Maestro Case simulation`.
- Audit path and synthetic-data line.

### 1:25-2:05 — Clean Claim Auto-Approval

Narration:

> The first claim is clean: low amount, valid active policy, complete supporting documents, and no ambiguity flags. The decision router sends it through straight-through processing.

Show terminal lines:

- `Scenario 1: clean claim auto-clears`
- `Claim: CLM-1001`
- `Policy check: pass`
- `Risk: score=0.05, confidence=0.95`
- `Initial route: auto_approved / resolved`
- `Final: auto_approved / resolved`

### 2:05-3:10 — Ambiguous Claim Human Gate

Narration:

> The second claim is covered, but it should not be auto-approved. The amount is near the policy limit, the report is late, expected documents are missing, and the narrative is ambiguous. ClaimClear creates a human review task instead of pretending certainty.

Show terminal lines:

- `Scenario 2: ambiguous claim escalates, then human approves`
- `Claim: CLM-1002`
- `Risk: score=0.81, confidence=0.19`
- `Flags: amount_near_policy_limit, late_reported_claim, ...`
- `Initial route: escalated / pending_human`
- `Human task: TASK-CLM-1002-REVIEW`
- `Final: human_approved / resolved`

### 3:10-4:05 — Audit Trail

Command:

```bash
python3 -m json.tool artifacts/demo_summary.json
```

Narration:

> The important part is not just the decision, but the trace. The summary and JSONL audit log preserve the intake result, policy validation, risk rationale, router decision, human task, and final resolution.

Show:

- `audit_entries` for `CASE-CLM-1002`.
- `human_approval_gate` task creation.
- `human_reviewer` task completion.

### 4:05-4:40 — UiPath Handoff

Narration:

> The coded-agent contracts are packaged in `uipath_coded_agents.py`, and the cloud wiring steps are documented for UiPath Maestro Case. In the cloud version, the same branch creates the reviewer task inside Maestro instead of simulating approval locally.

Show:

- `src/claimclear/uipath_coded_agents.py`.
- `docs/UIPATH_SETUP.md` sections 3-5.

### 4:40-5:00 — Close

Narration:

> ClaimClear demonstrates governed agentic automation: clear claims move immediately, ambiguous claims stay human-controlled, and every step is auditable. The demo uses synthetic data only, and final submission remains a human gate.

Show:

- `SUBMISSION_DRAFT.md` data disclosure.
- `AGENT_BUILD_LOG.md` bonus-point build log.

## Backup Short Version

If time is tight, show only:

1. README architecture and synthetic-data disclosure.
2. `make demo`.
3. `CLM-1001` auto-approved.
4. `CLM-1002` escalated to `TASK-CLM-1002-REVIEW` and resolved.
5. `artifacts/demo_summary.json` audit entries.
6. `docs/UIPATH_SETUP.md` for Maestro Case wiring.
