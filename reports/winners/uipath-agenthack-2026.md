# Winner Intelligence — UiPath AgentHack (2026)

> Target: `uipath-agenthack-2026` · $50K cash · deadline 2026-06-29 23:45 EDT · online · Taiwan eligible
> Purpose (spec §2.4): turn past-winner + judging signals into a concrete project strategy, not just a summary.

## Field context (what we're up against)

- **AgentHack 2025: 400+ submissions from 50+ countries.** This is a *competitive* field, not a thin one — winning comes from fit + polish + robustness, not from merely submitting. *(evidence: UiPath Community Blog, "AgentHack 2025", confidence 0.85)*
- **"Maestro and Agents ruled the roost."** Winning 2025 projects leaned on **Maestro orchestration**, not standalone single agents. *(evidence: same, confidence 0.8)*
- Winning projects spanned regulated verticals: **healthcare, public sector, manufacturing, banking, customer service, finance, insurance.** Domain realism mattered. *(evidence: same, confidence 0.8)*
- Many winners went on to become **tutorials / Marketplace components / enterprise solutions** → judges reward reusability + production viability, not throwaway demos. *(evidence: same, confidence 0.75)*

## Judging signal (verbatim from the official page)

| criterion | what it really rewards |
|---|---|
| Business Impact & Adoption Potential | "real-world relevance and production viability" — pick a real enterprise pain with quantified ROI |
| Platform Usage | depth of UiPath: Agent Builder + **Maestro** + external frameworks |
| Technical Execution | "handling of exceptions, failures, and edge cases" — **robustness is explicitly scored** |
| Completeness | working prototype + GitHub README + 5-min demo video |
| Creativity & Innovation | "novel design decisions, unexpected orchestration patterns" |
| Presentation | "logical flow from problem to solution to impact" |
| **Bonus Points** | **"Solutions using coding agents (Claude Code, Codex, Cursor, Gemini CLI) receive additional scoring"** + blending coding agents with low-code / UiPath-native + external agents |

*(evidence: https://uipath-agenthack.devpost.com/ , confidence 0.9)*

"Strong submissions" per the host: *"a working prototype, an end-to-end flow, handle real-world complexity, and be documented clearly enough that another developer could understand and build on your work."* **Human involvement and governance at decision points are essential across all tracks.** *(confidence 0.9)*

---

## Winning hypotheses (strategy, not summary)

### H1 — Build *on Maestro orchestration*, never a lone chatbot. (confidence 0.85)
2025 winners were Maestro-centric. The required platform layer is UiPath; the differentiator is an **orchestrated multi-actor flow** (RPA bot + AI agent + human approval) modeled in Maestro/BPMN. A single LLM wrapper will score low on Platform Usage and Creativity.
**Apply:** make Maestro the spine; agents are nodes inside an orchestrated process with explicit human decision gates.

### H2 — Pick one regulated vertical with a *quantified* ROI story. (confidence 0.8)
Judges weight "production viability" and adoption. The host's own proof point: *Lake Michigan Credit Union "pulled ten days out of their consumer lending cycle and now runs 15% more loan volume."* That's the register to speak in — concrete cycle-time / volume / cost numbers, not "AI for X."
**Apply:** insurance claims, loan/dispute resolution, or patient-care coordination — all named by UiPath as Maestro Case use cases.

### H3 — Robustness + governance is the cheap differentiator most entrants skip. (confidence 0.85)
"Technical Execution = handling of exceptions, failures, edge cases" is a named criterion, and "governance at decision points is essential." In a 400-entry field, most demos show the happy path only. Visibly handling a failure case + a human-in-the-loop approval gate is a fast way to stand out.
**Apply:** script the demo to deliberately trigger an exception → show the agent escalating to a human → resolution. Make the governance step a feature, not an afterthought.

### H4 — The coding-agent bonus is a free multiplier we are *uniquely* positioned to claim. (confidence 0.9)
The page **literally names Claude Code and Codex** as bonus-scoring tools. Our entire build process is Claude (planning) + Codex (coding). Almost no other team can authentically document this.
**Apply:** keep an `AGENT_BUILD_LOG.md` showing the Claude→Codex workflow, and call it out in the README + demo to bank the bonus. Bonus is *additive* — claim it regardless of track.

### H5 — Track choice is a competition-vs-appeal trade-off. (confidence 0.6)
- **Maestro Case** — highest judge familiarity & richest ROI narratives (claims/loans/onboarding) but most crowded.
- **Maestro BPMN** — structured order-to-cash / procure-to-pay; clean to demo, mid competition.
- **Test Cloud** (agentic testing: requirement→test, fragile-test detection, risk-based orchestration) — likely **fewer, less-polished entries** since 2025 was dominated by Maestro/Agents → better odds, but narrower "wow" and needs a testing angle.
**Apply:** default to **Maestro Case** for ROI narrative + judge familiarity; consider Test Cloud only if we want a lower-competition lane and can make agentic testing visually compelling.

---

## Archetypes

**Recommended:** a Maestro-orchestrated, **exception-heavy case-management agent** for one regulated vertical (claims / loan-dispute / patient coordination), with (a) multi-actor orchestration, (b) a visible human approval gate, (c) deliberate exception handling in the demo, (d) a quantified cycle-time/cost ROI claim, (e) documented Claude Code + Codex build process for the bonus.

**Avoid:** generic chatbot / RAG Q&A; a single agent with no orchestration; anything where UiPath isn't the orchestration layer (it's *required*); happy-path-only demos with no error handling or governance.

## Sources
- [UiPath Community Blog — AgentHack 2025](https://www.uipath.com/community-blog/community-news/uipath-community-annual-global-hackathon-2025)
- [UiPath AgentHack (Devpost) — judging & guidance](https://uipath-agenthack.devpost.com/)
- [UiPath Maestro — Agentic Orchestration](https://www.uipath.com/platform/agentic-automation/agentic-orchestration)
- [Introducing Maestro Case](https://www.uipath.com/blog/product-and-updates/introducing-maestro-case-new-uipath-capability)
