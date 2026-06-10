# Extended Hackathon Radar - 2026-06-10

This report extends `reports/radar_20260610.md` with additional opportunities found after
checking Devpost, lablab.ai, MLH, ETHGlobal, USAGov prize challenges, and selected news/official
event pages.

## New High-Priority Candidates

| rank | candidate | deadline | prize | status | gate |
|---:|---|---|---:|---|---|
| 1 | H0: Vercel v0 + AWS Databases | 2026-06-29 17:00 PDT | $80,000 cash + AWS credits | open | verify excluded countries |
| 2 | UiPath AgentHack | 2026-06-29 23:45 EDT | $50,000 cash | open | learn UiPath fast |
| 3 | Slack Agent Builder Challenge | 2026-07-13 17:00 PDT | $42,000 cash | open | verify Taiwan eligibility |
| 4 | Build with Gemini XPRIZE | 2026-08-17 13:00 PDT | $2,000,000 cash | open | requires real users/revenue |
| 5 | Arm Create: AI Optimization Challenge | 2026-08-14 | $8,000 cash | open | page needs direct verification |
| 6 | ETHOnline 2026 | 2026-09-04 to 2026-09-16 | unknown | open/upcoming | web3-specific |

## Fast-Lane Candidates

These are too close for the default 7-day freshness buffer but may still be worth a sprint if a
small scoped project already exists.

| candidate | deadline | prize | note |
|---|---|---:|---|
| Google Cloud Rapid Agent Hackathon | 2026-06-11 14:00 PDT | $60,000 cash | extremely suitable, but almost closed |
| Splunk Agentic Ops Hackathon | 2026-06-15 09:00 PDT | $20,000 cash | good MCP/AI ops fit, under buffer |
| FIND EVIL! | 2026-06-15 23:45 EDT | $22,000 cash | AI cybersecurity fit, under buffer |

## Blocked Or Low-ROI Finds

| candidate | reason |
|---|---|
| ET AI Hackathon 2.0 | India student-only eligibility; registration closes 2026-06-11 |
| NHAI Hackathon 7.0 | application window ended 2026-06-05 |
| IIT Madras AI Road Safety Hackathon | submissions ended 2026-05-31 |
| MLH Global Hack Week: Agents | useful practice/scouting, but prize ROI unclear |
| ACL Caregiver AI Prize Challenge | huge prize, but long federal prize challenge rather than hackathon |
| Connecting Talent to Opportunity Challenge | huge prize, but likely state/workforce-system fit rather than solo software sprint |

## Scoring Notes

Current scoring is good at preferring online, cash, AI/API, and long-enough deadlines. It is weak at:

- direct competitor pressure, because `registrations_count` is not used;
- explicit evidence-quality weighting, because evidence quality only affects delivery risk;
- fast-lane opportunities, because the 7-day freshness buffer rejects them outright;
- regional eligibility gates, because ranking only gates unknown/forbidden AI policy and deadline;
- submission complexity, because revenue evidence, marketplace submission, and platform onboarding are not scored.

Recommended scoring additions:

- `claude_codex_leverage_score`: agent/MCP/API/v0/no-code/coding-agent fit.
- `competition_pressure_score`: registrations divided by track count and prize buckets.
- `submission_complexity_score`: video, deck, marketplace, deploy proof, revenue evidence.
- `api_onboarding_risk_score`: sponsor account, credits, cloud deployment, platform restrictions.
- `fast_lane_mode`: separate mode for high-fit opportunities inside the normal freshness buffer.

## Sources Checked

- https://rapid-agent.devpost.com/
- https://xprize.devpost.com/
- https://h01.devpost.com/
- https://uipath-agenthack.devpost.com/
- https://slackhack.devpost.com/
- https://splunk.devpost.com/
- https://findevil.devpost.com/
- https://ethglobal.com/events
- https://www.mlh.com/seasons/2026/events
- https://events.mlh.com/events/14312-global-hack-week-agents
- https://www.usa.gov/find-active-challenge
- https://www.usa.gov/challenges/acl-caregiver-ai-prize
- https://www.usa.gov/challenges/cto-challenge
- https://economictimes.indiatimes.com/et-ai-hackathon/2nd-edition
