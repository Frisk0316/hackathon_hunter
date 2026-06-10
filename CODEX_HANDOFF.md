# Codex Handoff — 2026-06-10

Division of labor: **Claude = planning + search (intelligence)**, **Codex = pipeline implementation (code)**.
This note is the contract between the two. Source of truth for the refactor is `hackathon_hunter_codex_improvement_plan.md`.

---

## What Claude did this pass (done)

1. **Freshness check / cleanup.** All round-1 candidates are past-deadline → moved stale data to:
   - `examples/hackathons.sample.json` (was `hackathons.json`)
   - `examples/strategy.sample.json` (was `strategy.json`)
   These are **samples only**, not active candidates.
2. **Fresh Radar sweep** (Devpost, lablab, DoraHacks, web search) → produced, in the spec's schema/locations:
   - `data/processed/hackathons_20260610.json` — evidence-backed, timezone-aware deadlines, per-field `source_evidence` + confidence, `status`, `expired_for_review`.
   - `reports/radar_20260610.md` — multi-dimensional scoring + ranking + human gates.
3. Created `data/raw/.gitkeep`, `data/processed/.gitkeep`, `reports/`, `examples/`.

## Current intelligence verdict (drives idea/spec stages)

| rank | target | overall | status | gate |
|---|---|---|---|---|
| 1 | Qwen Cloud Global AI Hackathon ($70K, Jul 9) | 0.82 | open | ⚠ verify Taiwan eligibility on Qwen Cloud |
| 2 | ElevenLabs Voice AI ($5K, Jul 28–31) | 0.71 | open | short 3-day build window |
| 3 | DevNetwork API+Cloud+AI (Sep 3) | 0.58 | open | non-cash; sponsor prizes "Coming Soon" |
| 4 | MunichTech (Sep 20) | N/A | unknown | ai_policy unknown → blocked until verified |

**Blocking human decision before any build:** confirm a Qwen Cloud account can be created from Taiwan. Result decides primary target (Qwen) vs. fallback (ElevenLabs).

---

## What Codex should build next (per improvement plan §3–§8)

Recommended order — these consume the artifacts above so they stay testable:

1. **Phase 1** — `pyproject.toml`, `hackathon_hunter/` package, `models.py` (Pydantic: `Hackathon`, `Evidence`, `Eligibility`, `Prize`, `SubmissionRequirements`, `ScoreBreakdown`, `ProjectIdea`, `RunLog`), `cli.py` (typer). Make `models.Hackathon` parse `data/processed/hackathons_20260610.json` as the first fixture — **the schema there is the target shape**.
2. **Phase 2** — `sources/base.py` adapter interface + `web_search`/`mock` adapter; `storage.py` (`save_raw_snapshot`, `load_latest_processed`, `deduplicate_hackathons`); freshness check (deadline < now → `status=closed`, exclude from active).
3. **Phase 3** — `rules.py` (eligibility checker, §2.2 schema) + `scoring.py` (weights in `config/scoring.yaml`, mirror the weights I applied in the radar report) + ranking report generator.
4. **Phase 4–5** — `ideate` / `build_spec` workflows.
5. **Phase 6** — QA placeholder/secret scan + pytest + GitHub Actions CI.

### Contracts to honor
- Deadlines must stay timezone-aware. My JSON already encodes offsets + IANA tz.
- Every key field needs `source_evidence`; don't let the model invent values (see MunichTech: left `null`/`unknown` on purpose — do not backfill with guesses).
- **No auto-submit, no secret commits.** Keep all 5 human gates from spec §6.
- Don't delete `hackathon-project/` or `hackathon-project-algofest/` — move to `archive/generated-projects/` (spec §3) or mark as historical examples in README.

### Open items for Claude (next search pass, ~1 week)
- Re-fetch MunichTech detail page (verify prize/rules/AI policy).
- Re-check DevNetwork once sponsor-challenge prizes announced.
- Winner-intelligence (§2.4) + ideation (§2.5) on the confirmed primary target once the eligibility gate clears.

---

## What Codex did after this handoff

Implemented the first reusable pipeline slice:

1. Added Python package + CLI:
   - `pyproject.toml`, `setup.cfg`, `setup.py`
   - `hackathon_hunter/cli.py`
   - `python -m hackathon_hunter` and `hackathon-hunter`
2. Added core models:
   - timezone-aware `Hackathon`
   - `Evidence`, `Eligibility`, `Prize`, `SubmissionRequirements`
   - `ScoreBreakdown`, `ProjectIdea`, `RunLog`
3. Added storage/source/rules/scoring/report layers:
   - offline `--mock` source adapter
   - raw snapshots and processed JSON
   - freshness check + dedup
   - AI-policy and eligibility gates
   - traceable weighted scoring from `config/scoring.yaml`
4. Added workflows:
   - `collect`
   - `rank`
   - `check-rules`
   - `analyze-winners`
   - `ideate`
   - `build-spec`
   - `qa`
   - `record-result`
5. Moved historical generated projects to:
   - `archive/generated-projects/hackathon-project`
   - `archive/generated-projects/hackathon-project-algofest`
6. Added docs, root README, config examples, tests, and GitHub Actions CI.

Verification run:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .[dev]
.venv/bin/hackathon-hunter collect --mock --online-only --min-prize-usd 1000
.venv/bin/hackathon-hunter rank --input data/processed/mock_hackathons.json
.venv/bin/hackathon-hunter check-rules --input data/processed/mock_hackathons.json
.venv/bin/hackathon-hunter ideate --hackathon-id mock-hackathon-001 --n 5
.venv/bin/hackathon-hunter build-spec --hackathon-id mock-hackathon-001 --idea-id idea-001
.venv/bin/hackathon-hunter qa --project projects/mock-hackathon-001/idea-001
.venv/bin/pytest
.venv/bin/ruff check .
```

Results:

- `pytest`: 7 passed.
- `ruff check .`: passed.
- Mock QA report: passed.

Note: mock smoke-test artifacts are intentionally ignored by `.gitignore`; they can be regenerated with the commands above.

---

## Scoring rewrite after extended radar

Codex added the ranking changes requested after the 2026-06-10 extended search:

1. Added trace features in `hackathon_hunter/scoring.py` and `config/scoring.yaml`:
   - `taiwan_eligibility_gate`
   - `competition_pressure_score`
   - `submission_complexity_score`
   - `fast_lane_mode`
2. Added optional eligibility fields in `hackathon_hunter/models.py`:
   - `taiwan_eligible`
   - `allowed_regions`
   - `excluded_regions`
3. Added `hackathon-hunter rank --fast-lane` so near-deadline sprint candidates can be ranked with a 1-day freshness gate instead of the default 7-day gate.
4. Updated scoring behavior:
   - known Taiwan-incompatible events are rejected at rank time
   - uncertain region/account eligibility is still rankable, but heavily penalized
   - large registration/participant counts lower `competition_pressure_score`
   - video, repo, deploy proof, architecture diagram, revenue evidence, hardware, and live presentation requirements lower `submission_complexity_score`

Verification:

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/hackathon-hunter rank --input data/processed/hackathons_20260610.json --fast-lane
```

Results:

- `pytest`: 10 passed.
- `ruff check .`: passed.
- Fast-lane ranking report: `reports/radar_ranked_20260610.md`.

---

## Build kickoff — ClaimClear (2026-06-10)

**Decisions locked (human-approved):**
- Target hackathon: **UiPath AgentHack** (`uipath-agenthack-2026`) — $50K cash, online, Taiwan eligible, deadline 2026-06-29.
- Project: **idea-001 ClaimClear** (Maestro Case track).

**Why this target/idea (Claude intelligence):** UiPath is the only high-cash candidate with confirmed Taiwan eligibility, AND the judging page explicitly awards bonus points for coding-agent use (Claude Code + Codex named) — our exact workflow. ClaimClear = best judging-fit-to-feasibility. Full reasoning: `reports/winners/uipath-agenthack-2026.md`, `reports/ideas/uipath-agenthack-2026.md`.

**Build package for Codex (start here):**
- `projects/uipath-agenthack-2026/claimclear/SPEC.md` — full design + acceptance criteria.
- `projects/uipath-agenthack-2026/claimclear/TASKS.md` — ordered milestones ([HUMAN] steps tagged).
- `projects/uipath-agenthack-2026/claimclear/AGENT_BRIEF.md` — Codex's working brief (<200 lines, done criteria, boundaries).
- `projects/uipath-agenthack-2026/claimclear/SUBMISSION_DRAFT.md` — Devpost draft (do not auto-submit).

**Hard constraints for Codex:** synthetic data only (disclose it); UiPath Automation Cloud is the required orchestration layer (build agents locally first, then package as Coded Agents + write `docs/UIPATH_SETUP.md`); maintain `AGENT_BUILD_LOG.md` for the bonus; no secrets committed; stop-and-ask on all human gates (UiPath account, cloud wiring, video, publish, submit).
