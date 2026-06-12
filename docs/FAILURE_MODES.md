---
title: Failure Modes
owner: human
status: active
last_updated: 2026-06-12
---

# Failure Modes

| Failure | Early signal | Prevention / response |
|---|---|---|
| Stale evidence drives a build | `watch` reports stale fields | Re-verify source fields before build-spec |
| Unknown AI policy slips through | `rules.py` or ranking no longer blocks unknown | Preserve tests and domain rule |
| Deadline window is too tight | Watch events or rejected ranking reason | Use fast-lane only with explicit risk |
| LLM hallucinated source data | Evidence quote/URL missing or low confidence | Treat as missing evidence |
| Secret committed to generated project | QA secret scan or `.env` present | Keep only `.env.example`; rotate if leaked |
| Duplicate or overwritten processed radar | Unexpected file replacement | Preserve `unique_path` and overwrite only mock fixture intentionally |
| Human gate implied as complete | Submission text says published/submitted without proof | Keep "pending human approval" language |
| Docs drift after scoring/rules change | `make docs-impact` warnings | Update required docs and tests |
| Context loss after long AI session | Agent cannot name active task or risks | Read `CURRENT_STATE.md`, `CONTEXT_INDEX.md`, and provide handoff |
