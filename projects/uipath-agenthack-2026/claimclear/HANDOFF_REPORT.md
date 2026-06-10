# ClaimClear Handoff Report

## Status

Codex portion is implemented for a local, end-to-end demo. The project can run the required clean and ambiguous claim scenarios without UiPath cloud access.

## What Is Done

- Project skeleton under `projects/uipath-agenthack-2026/claimclear/`.
- Synthetic claims and policies in `data/`.
- Local agent pipeline in `src/claimclear/`.
- CLI demo through `make demo`.
- JSONL audit trail in `artifacts/audit_log.jsonl` after demo runs.
- UiPath Coded Agent wrapper functions in `src/claimclear/uipath_coded_agents.py`.
- Tests in `tests/test_pipeline.py`.
- Architecture and UiPath setup docs in `docs/`.
- Build log in `AGENT_BUILD_LOG.md`.

## Run Instructions

```bash
cd projects/uipath-agenthack-2026/claimclear
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make demo
make test
```

Useful direct commands:

```bash
make run-clean
make run-ambiguous
make qa
```

## Verified Locally

- `make demo` completed successfully.
- `make test` completed successfully with 3 passing tests.
- `make qa` completed successfully: tests, Ruff lint, and smoke demo.

## Manual Work Remaining

- Human creates or confirms UiPath Automation Cloud access.
- Human builds the Maestro Case process using `docs/UIPATH_SETUP.md`.
- Human wires the coded agents and human approval task in UiPath cloud.
- Human records the demo video.
- Human confirms public repository visibility.
- Human submits the final Devpost entry.

## Demo Script

1. Run `make demo` locally or submit `CLM-1001` in Maestro.
2. Show `CLM-1001` auto-approved with audit entries.
3. Submit `CLM-1002`.
4. Show low confidence, risk flags, and human task creation.
5. Approve the task.
6. Show final resolved status and audit entries.
