# SUBMISSION CHECKLIST — ClaimClear

Final submission is a human gate. Do not submit until every item here is reviewed against the current UiPath AgentHack Devpost form.

## Before Recording

- [ ] Run `make test`.
- [ ] Run `make lint`.
- [ ] Run `make demo`.
- [ ] Confirm `artifacts/demo_summary.json` shows `CLM-1001` auto-approved.
- [ ] Confirm `artifacts/demo_summary.json` shows `CLM-1002` escalated to `TASK-CLM-1002-REVIEW` and then resolved.
- [ ] Confirm `README.md` synthetic-data disclosure is visible.
- [ ] Confirm `SUBMISSION_DRAFT.md` synthetic-data disclosure is visible.
- [ ] Confirm no `.env` file is present.

## Video Assets

- [ ] Record video using `VIDEO_SCRIPT.md`.
- [ ] Keep video at or under 5 minutes.
- [ ] Show synthetic-data disclosure near the beginning.
- [ ] Show clean claim auto-approval.
- [ ] Show ambiguous claim human approval gate.
- [ ] Show audit trail.
- [ ] Show UiPath Maestro Case setup documentation or the actual cloud process if wired.
- [ ] Upload video to the platform accepted by Devpost.

## Repository

- [ ] Confirm repo is public.
- [ ] Confirm `projects/uipath-agenthack-2026/claimclear/README.md` renders correctly.
- [ ] Confirm `.env.example` has empty or placeholder values only.
- [ ] Confirm generated local virtual environments are not committed.
- [ ] Confirm no real claims, policies, personal data, or API keys are committed.

## UiPath / Devpost Form

- [ ] Confirm team registration and official rules acceptance.
- [ ] Select track: Maestro Case.
- [ ] Use title: `ClaimClear`.
- [ ] Paste the human-reviewed content from `SUBMISSION_DRAFT.md`.
- [ ] Add public repo URL.
- [ ] Add demo video URL.
- [ ] Add UiPath project or Maestro Case notes, if requested.
- [ ] Attach or link presentation deck if required by the current form.

## Final Human Review

- [ ] Business impact is clear: routine claim triage moves from days to seconds.
- [ ] Platform usage is clear: Maestro Case orchestrates agents, bot, and human task.
- [ ] Governance is clear: ambiguous claims never auto-submit; they route to a human.
- [ ] Auditability is clear: every agent/bot/human step is logged.
- [ ] Synthetic-data disclosure appears in README, video, and submission text.
- [ ] Final submit button is clicked by a human only.
