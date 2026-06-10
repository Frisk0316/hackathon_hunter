# ClaimClear

ClaimClear is a UiPath AgentHack 2026 Maestro Case entry for agentic insurance-claims triage. It auto-clears straightforward synthetic claims, escalates ambiguous claims to a human approval gate, and records every decision in an audit log.

## Quick Start

```bash
cd projects/uipath-agenthack-2026/claimclear
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make demo
make test
```

The local demo does not require UiPath cloud access or an Anthropic key. It uses deterministic scoring by default so the hackathon walkthrough is repeatable.

## Demo

```bash
make demo
```

The demo runs two synthetic cases:

1. `CLM-1001` is a clean auto-physical-damage claim. It passes policy validation, scores as low risk, and is auto-approved.
2. `CLM-1002` is an ambiguous property-water-damage claim. It passes coverage checks but has late reporting, missing documents, a high amount-to-limit ratio, and narrative ambiguity. It escalates to `TASK-CLM-1002-REVIEW`, then the local demo simulates human approval.

Generated demo files:

- `artifacts/audit_log.jsonl` records every case, agent, bot, router, and human-gate event.
- `artifacts/demo_summary.json` captures the structured output for both scenarios.

## Tech Stack

- Python 3.10+
- UiPath Maestro Case as the target orchestration layer
- UiPath Coded Agents compatible Python callables in `src/claimclear/uipath_coded_agents.py`
- Synthetic JSON fixtures in `data/`
- Optional Claude rationale enrichment through Anthropic when `CLAIMCLEAR_USE_CLAUDE=true`
- Pytest and Ruff for local quality checks

## Architecture

```text
Synthetic claim form
  -> Maestro Case process
       -> Intake Agent
       -> Policy-Check Bot
       -> Risk-Scoring Agent
       -> Decision Router
            -> straight-through auto approval
            -> human approval gate
       -> Audit Log
```

Local execution mirrors the Maestro flow in `ClaimClearPipeline` so development and judging demos can run without waiting on cloud setup. The cloud handoff steps are documented in `docs/UIPATH_SETUP.md`; the deeper architecture is in `docs/ARCHITECTURE.md`.

## Data Disclosure

ClaimClear uses only synthetically generated claims, policies, names, emails, narratives, and audit records. No real claims, policies, personal data, carrier data, or payment rails are used.

## Commands

```bash
make demo          # run both required demo scenarios
make run-clean     # run only CLM-1001
make run-ambiguous # run only CLM-1002 with simulated human approval
make test          # run pytest
make lint          # run ruff
make qa            # run test, lint, and smoke demo
```

## UiPath Integration

`src/claimclear/uipath_coded_agents.py` exposes the callable contracts for:

- `intake_agent`
- `policy_check_bot`
- `risk_scoring_agent`
- `decision_router`

Human cloud setup is still a gate: create the UiPath Automation Cloud account, build the Maestro Case process, configure the human approval task, and record the final video manually.

