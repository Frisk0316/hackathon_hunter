# UiPath Maestro Case Setup

This guide lets a human build the cloud version after the local demo is working. Codex cannot create accounts, click through the tenant UI, publish the final repo, record the video, or submit the hackathon entry.

## 1. Prerequisites

- UiPath Automation Cloud account with Maestro Case access.
- A workspace or tenant where you can create processes, agents, queues, and human tasks.
- Local ClaimClear demo passing with `make demo` and `make test`.
- Optional Anthropic API key stored as a secure asset if Claude rationale enrichment is enabled.

## 2. Create Project Assets

Create these UiPath assets or equivalent Data Service tables:

- `ClaimClearPolicies`: policy records from `data/policies.json`.
- `ClaimClearClaims`: incoming claim payloads from `data/claims.json`.
- `ClaimClearAuditLog`: append-only audit records with timestamp, case id, claim id, actor, action, and details.
- `ClaimClearHumanReviewQueue`: queue or human-task source for escalated claims.

Keep the sample data marked as synthetic in descriptions and demo notes.

## 3. Create Coded Agents

Use `src/claimclear/uipath_coded_agents.py` as the source contract.

Create these Coded Agents or bot steps:

1. `ClaimClear Intake Agent`
   - Function: `intake_agent(payload)`
   - Input: raw claim form payload.
   - Output: `IntakeResult` JSON.

2. `ClaimClear Policy Check Bot`
   - Function: `policy_check_bot(payload)`
   - Input: claim JSON plus policy JSON.
   - Output: `PolicyCheckResult` JSON.

3. `ClaimClear Risk Scoring Agent`
   - Function: `risk_scoring_agent(payload)`
   - Input: claim JSON, policy JSON, policy-check JSON.
   - Output: `RiskScoreResult` JSON.

4. `ClaimClear Decision Router`
   - Function: `decision_router(payload)`
   - Input: claim JSON, policy-check JSON, risk-score JSON.
   - Output: `DecisionResult` JSON.

Package the `src/claimclear` module with each coded-agent deployment or place it in a shared library package, depending on the tenant workflow.

## 4. Build The Maestro Case

Create a Maestro Case process named `ClaimClear Triage`.

Recommended case variables:

- `case_id`
- `claim`
- `policy`
- `intake_result`
- `policy_check_result`
- `risk_score_result`
- `decision_result`
- `human_review_result`
- `audit_entries`

Recommended process flow:

1. Start from claim form submission.
2. Generate `case_id` as `CASE-<claim_id>`.
3. Run `ClaimClear Intake Agent`.
4. Append intake output to `ClaimClearAuditLog`.
5. Fetch matching policy by `policy_id`.
6. Run `ClaimClear Policy Check Bot`.
7. Append policy-check output to `ClaimClearAuditLog`.
8. Run `ClaimClear Risk Scoring Agent`.
9. Append risk output to `ClaimClearAuditLog`.
10. Run `ClaimClear Decision Router`.
11. Append router output to `ClaimClearAuditLog`.
12. Branch on `decision_result.outcome`.
13. If `auto_approved`, mark case resolved.
14. If `escalated`, create a human approval task.
15. On human approval, append the human decision to `ClaimClearAuditLog`.
16. Mark case resolved.

## 5. Configure Human Task

Human task name: `ClaimClear Review`.

Show the reviewer:

- Claim id.
- Claimant name.
- Claim type.
- Amount.
- Policy-check reasons.
- Risk score and confidence.
- Risk flags.
- Agent rationale.
- Payout estimate.
- Full audit history for the case.

Actions:

- Approve claim.
- Reject claim.

For the recorded demo, use the synthetic ambiguous claim `CLM-1002` and approve it after showing the risk flags and audit entries.

## 6. Audit Mapping

Map each local audit entry to cloud storage:

```json
{
  "timestamp": "2026-06-10T00:00:00+00:00",
  "case_id": "CASE-CLM-1002",
  "claim_id": "CLM-1002",
  "actor": "risk_scoring_agent",
  "action": "completed",
  "details": {}
}
```

The timestamp shown above is an example format. Cloud records should use the real runtime timestamp.

## 7. Demo Recording Path

Record a video no longer than five minutes:

1. Show the project title and synthetic-data disclosure.
2. Submit `CLM-1001`.
3. Show auto approval and audit entries.
4. Submit `CLM-1002`.
5. Show escalation to the human task.
6. Approve the task as a reviewer.
7. Show final resolved status and audit entries.
8. Close with the business impact: routine claims can move from days of queue time to seconds while ambiguous claims remain governed.

## 8. Troubleshooting

- If a claim routes unexpectedly, inspect `risk_flags` and `policy_check_result.reasons`.
- If a coded agent cannot import `claimclear`, package the full `src/claimclear` folder with the agent or convert the imports to the tenant's shared-library format.
- If Claude enrichment is unavailable, leave `CLAIMCLEAR_USE_CLAUDE=false`; the deterministic risk scorer remains valid for the demo.

