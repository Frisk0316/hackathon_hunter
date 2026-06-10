# Project Ideas — UiPath AgentHack (2026)

> Target `uipath-agenthack-2026` · $50K cash · deadline 2026-06-29 · **human decision required: pick ONE before scaffold.**
> Grounded in [reports/winners/uipath-agenthack-2026.md](winners/uipath-agenthack-2026.md). Structured data: [strategy/uipath-agenthack-2026_ideas.json](../strategy/uipath-agenthack-2026_ideas.json).

**Every idea assumes:** Maestro orchestration as the spine · a visible human approval gate · a demo that deliberately triggers an exception · a quantified ROI claim · a documented Claude Code + Codex build log (bonus points).

| id | name | track | feasibility | judging fit | differentiation | build hrs |
|---|---|---|---:|---:|---:|---:|
| idea-001 | **ClaimClear** | Maestro Case | 0.78 | 0.90 | 0.60 | 32 |
| idea-002 | DisputeDesk | Maestro Case | 0.72 | 0.88 | 0.68 | 34 |
| idea-003 | OnboardPilot | Maestro BPMN | 0.82 | 0.78 | 0.50 | 28 |
| idea-004 | **TestPilot** | Test Cloud | 0.75 | 0.74 | 0.82 | 30 |
| idea-005 | CareCoord | Maestro Case | 0.70 | 0.90 | 0.75 | 36 |

---

### idea-001 · ClaimClear — *recommended (safe)*
Agentic insurance-claims triage: auto-clear the easy ~80%, escalate the rest with an audit trail. Maestro Case orchestrates doc-intake agent → policy check (bot) → risk-scoring agent → auto-pay clear cases / human gate for ambiguous ones.
**Why it wins:** named UiPath use case + strongest ROI narrative (cycle-time cut) + clean exception+governance demo. Best fit-to-effort for ~19 days.
**Watch:** Automation Cloud + document-understanding setup time.

### idea-002 · DisputeDesk
Agentic loan/payment dispute resolution with a compliance trail. Evidence-gathering agent + DMN rules + human adjudication → compliance-ready decision letter.
**Why:** rides UiPath's own Lake Michigan CU "10 days out of lending" proof point; slightly more novel than claims.
**Watch:** compliance realism + DMN modeling time.

### idea-003 · OnboardPilot
Agentic employee onboarding across IT/HR/access provisioning, modeled in Maestro **BPMN**, with agents resolving exceptions and a human gate for privileged access.
**Why:** highest feasibility (predictable sequence), least build time.
**Watch:** mocked systems can look thin; lowest differentiation.

### idea-004 · TestPilot — *high-upside contrarian*
Agentic test generation in **Test Cloud**: requirements → risk-ranked test suites, flag fragile tests, human review before commit.
**Why:** Test Cloud was the least-crowded track in 2025 (Maestro/Agents dominated) → **best odds + highest differentiation**.
**Watch:** narrower "wow", less ROI-familiar to judges, needs a believable sample suite.

### idea-005 · CareCoord — *highest ceiling*
Agentic patient-care coordination (referrals / prior-auth / follow-ups) in Maestro Case with clinician oversight.
**Why:** healthcare = high judge resonance + our domain-depth edge; strong governance story.
**Watch:** most build hours; needs careful synthetic-data disclosure (no real PHI).

---

## Recommendation
- **Safe pick → idea-001 ClaimClear**: best judging-fit-to-feasibility for the window.
- **Contrarian → idea-004 TestPilot**: lowest competition, highest differentiation.
- **Ceiling → idea-005 CareCoord**: best narrative if we accept more build risk.

Once you pick, I'll write the Codex build handoff (SPEC / TASKS / AGENT_BRIEF / SUBMISSION_DRAFT scaffolding inputs) so Codex can start coding.
