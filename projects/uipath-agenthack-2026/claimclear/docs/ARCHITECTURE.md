# ClaimClear Architecture

## Objective

ClaimClear demonstrates a Maestro Case pattern for insurance triage: automate the clear claims, escalate the ambiguous ones, and make every decision auditable.

The project is intentionally narrow:

- Property and casualty claim triage only.
- Synthetic data only.
- Local demo first, then UiPath Maestro Case wiring.
- Human governance is part of the main path, not an afterthought.

## Runtime Views

### Local Demo Runtime

```text
data/claims.json
  -> claimclear.cli
  -> ClaimClearPipeline
       -> intake_agent.extract_claim
       -> policy_check.check_policy
       -> risk_scoring_agent.score_risk
       -> decision_router.route_decision
       -> human approval simulator when escalated
       -> audit.AuditLogger
  -> artifacts/audit_log.jsonl
  -> artifacts/demo_summary.json
```

This runtime is used by `make demo`, `make run-clean`, `make run-ambiguous`, and tests.

### UiPath Maestro Runtime

```text
Claim submission
  -> Maestro Case: ClaimClear Triage
       -> Coded Agent: Intake Agent
       -> RPA / Coded Bot: Policy Check
       -> Coded Agent: Risk Scoring Agent
       -> Coded Agent: Decision Router
       -> Human Task: Claims Reviewer Approval
       -> Data Service / Orchestrator storage: Audit Log
```

The UiPath setup is manual because the cloud account, Maestro Case canvas, and final submission are human gates. The agent contracts are provided in `src/claimclear/uipath_coded_agents.py`.

## Data Model

### Claim

Core fields:

- `claim_id`
- `policy_id`
- `claimant_name`
- `claimant_email`
- `claim_type`
- `amount`
- `incident_date`
- `reported_date`
- `loss_location`
- `narrative`
- `documents`
- `metadata`

### Policy

Core fields:

- `policy_id`
- `holder_name`
- `status`
- `effective_date`
- `expiration_date`
- `coverage_limits`
- `deductible`
- `prior_claims_last_24_months`
- `notes`

### Agent Outputs

- `IntakeResult`: structured claim, extraction confidence, missing fields, rationale.
- `PolicyCheckResult`: pass/fail, reasons, coverage limit, deductible, covered amount, policy status.
- `RiskScoreResult`: confidence, risk score, risk flags, rationale.
- `DecisionResult`: outcome, status, route, payout estimate, rationale, next owner, human task id.
- `AuditEntry`: timestamp, case id, claim id, actor, action, details.

## Agent Responsibilities

### Intake Agent

Normalizes a submitted claim form into the Claim schema. The local version supports dictionaries and simple key-value text forms. In UiPath, this maps to form submission or Document Understanding output.

### Policy-Check Bot

Validates the claim against the mock policy store:

- Policy exists.
- Policy status is active.
- Incident date is inside the policy period.
- Claim type is covered.
- Amount is inside the coverage limit.

This is modeled as a bot because the same pattern would query a carrier policy system in a production deployment.

### Risk-Scoring Agent

Scores ambiguity and fraud risk using deterministic rules for demo repeatability. It can optionally call Claude for rationale enrichment when `CLAIMCLEAR_USE_CLAUDE=true` and `ANTHROPIC_API_KEY` is configured.

Risk signals include:

- Failed policy checks.
- Amount near policy limit.
- Late reporting.
- Missing expected documents.
- Ambiguous narrative terms.
- Multiple recent claims.
- Claimant and policyholder mismatch.

### Decision Router

Routes the claim:

- Auto-approve when coverage passes, confidence is at least `0.72`, and risk score is at most `0.28`.
- Escalate to human review for anything outside that auto-clear boundary.

The demo intentionally includes an ambiguous case to prove the human-governance path.

### Human Approval Gate

The local demo simulates approval for `CLM-1002`. In Maestro, this becomes a human task assigned to a claims reviewer. The reviewer sees the policy result, risk flags, rationale, payout estimate, and audit history.

## Audit Strategy

Every stage appends a JSONL audit entry:

- `maestro_case.case_started`
- `intake_agent.completed`
- `policy_check_bot.completed`
- `risk_scoring_agent.completed`
- `decision_router.completed`
- `human_approval_gate.task_created`
- `human_reviewer.task_completed`
- `maestro_case.case_resolved`

Local path: `artifacts/audit_log.jsonl`.

UiPath path: store equivalent records in Data Service, a queue item payload, or Orchestrator logs depending on the final tenant setup.

## Demo Scenarios

### Clean Claim

`CLM-1001`:

- Active policy.
- Low amount-to-limit ratio.
- Required documents present.
- No ambiguity flags.
- Final outcome: `auto_approved`.

### Ambiguous Claim

`CLM-1002`:

- Active policy.
- Amount near limit.
- Reported late.
- Missing plumber invoice and contractor estimate.
- Narrative contains ambiguity terms.
- Multiple recent claims.
- Initial outcome: `escalated`.
- Final outcome in local demo: `human_approved`.

## Quality Gates

Local gates:

- `make demo`
- `make test`
- `make lint`
- `make qa`

Human gates:

- UiPath Automation Cloud account creation.
- Maestro Case process creation.
- Cloud agent wiring.
- Demo video recording.
- Public repo confirmation.
- Final Devpost submission.

## Security And Privacy

- Synthetic data only.
- No real PII.
- No committed secrets.
- `.env` is gitignored.
- `.env.example` contains empty configuration keys.
- Optional Claude usage is disabled by default.

