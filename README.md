# Hackathon Hunter

Hackathon Hunter is a research and planning pipeline for discovering online prize hackathons, checking eligibility, ranking opportunities, analyzing winners, generating project specs, and preparing submission packages.

It does not automatically submit hackathon entries. Final submission always requires human approval.

## Workflows

1. Collect hackathons
2. Check rules and eligibility
3. Rank candidates
4. Analyze winners
5. Generate ideas
6. Build project specs
7. Hand off to coding agents
8. QA submission package
9. Record results

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
hackathon-hunter collect --mock
hackathon-hunter rank --input data/processed/mock_hackathons.json
hackathon-hunter check-rules --input data/processed/mock_hackathons.json
hackathon-hunter ideate --hackathon-id mock-hackathon-001 --input data/processed/mock_hackathons.json
hackathon-hunter build-spec --hackathon-id mock-hackathon-001 --idea-id idea-001 --input data/processed/mock_hackathons.json
hackathon-hunter watch --input data/processed/mock_hackathons.json
hackathon-hunter status --input data/processed/mock_hackathons.json
hackathon-hunter qa --project projects/mock-hackathon-001/idea-001
```

Import the next evidence-backed radar from Claude with the schema gate:

```bash
hackathon-hunter import --input examples/hackathons.sample.json --dry-run
hackathon-hunter import --input path/to/claude_radar.json --merge
```

## Current Intelligence

The latest handoff from the search/planning pass is in `CODEX_HANDOFF.md`.
The current processed radar fixture is `data/processed/hackathons_20260610.json`, and the human-readable radar report is `reports/radar_20260610.md`.

The old generated hackathon submissions are historical examples, not the hunter system itself. They live under `archive/generated-projects/`.

## Safety

- No auto-submit.
- No secret commits.
- Evidence is required for deadlines, rules, prize, and AI policy.
- Unknown AI policy blocks the build stage.
- Synthetic data must be disclosed in README and submission drafts.
- The five human gates are preserved: registration, idea choice, public repo/demo, social post, and final submission.
