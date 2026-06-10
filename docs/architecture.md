# Architecture

Hackathon Hunter separates research artifacts from generated project code.

The package exposes a Typer CLI. Workflows load normalized `Hackathon` models, preserve source evidence, write processed JSON, and produce Markdown reports. Generated hackathon submissions belong under `projects/` or `archive/generated-projects/`, not in the package root.

Core modules:

- `models.py`: Pydantic contracts for hackathons, evidence, scoring, ideas, and run logs.
- `storage.py`: JSON loading, raw snapshots, processed outputs, deduplication, and reports.
- `rules.py`: eligibility and human-gate checks.
- `scoring.py`: transparent weighted ranking.
- `reports.py`: Markdown report rendering.
- `sources/`: adapter interface and offline mock source.
- `workflows/`: CLI orchestration for collect, rank, rules, ideas, specs, QA, and results.
