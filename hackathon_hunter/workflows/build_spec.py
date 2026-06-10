from __future__ import annotations

from pathlib import Path

from hackathon_hunter.models import ProjectIdea
from hackathon_hunter.storage import find_hackathon, project_path, read_json
from hackathon_hunter.workflows.ideate import generate_ideas


def _load_idea(hackathon_id: str, idea_id: str, input_path: Path | None = None) -> ProjectIdea:
    strategy_path = project_path("strategy", f"{hackathon_id}_ideas.json")
    if strategy_path.exists():
        payload = read_json(strategy_path)
        ideas = [ProjectIdea.model_validate(item) for item in payload.get("ideas", [])]
    else:
        ideas = generate_ideas(hackathon_id, 5, input_path)
    for idea in ideas:
        if idea.id == idea_id:
            return idea
    raise LookupError(f"Idea not found: {idea_id}")


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path


def run_build_spec(
    hackathon_id: str,
    idea_id: str,
    input_path: Path | None = None,
) -> dict[str, Path]:
    hackathon = find_hackathon(hackathon_id, input_path)
    idea = _load_idea(hackathon_id, idea_id, input_path)
    project_dir = project_path("projects", hackathon.id, idea.id)
    required_api = hackathon.required_apis[0] if hackathon.required_apis else "sponsor API"
    spec = f"""# SPEC

## Goal

Build {idea.name}, {idea.tagline}

## Hackathon

- Name: {hackathon.name}
- ID: {hackathon.id}
- URL: {hackathon.url}
- Deadline: {hackathon.deadline.isoformat()}
- Required API: {required_api}

## Problem

{idea.problem}

## Target User

{idea.target_user}

## MVP Scope

{chr(10).join(f"- {item}" for item in idea.mvp_scope)}

## Non-goals

{chr(10).join(f"- {item}" for item in idea.non_goals)}

## Demo Path

{idea.demo_flow}

## Human Approval Points

- Confirm official registration and account eligibility.
- Confirm this idea before coding starts.
- Confirm public repo and demo URL before publishing.
- Confirm any social post manually.
- Confirm final submission manually.
"""
    tasks = """# TASKS

## Milestone 1 - Skeleton

- [ ] Create app scaffold
- [ ] Add `.env.example`
- [ ] Add basic CI or smoke test

## Milestone 2 - Core Demo Path

- [ ] Implement the primary user flow
- [ ] Integrate the required sponsor API
- [ ] Add fallback error handling with visible disclosure

## Milestone 3 - Submission Quality

- [ ] Complete README
- [ ] Complete demo script
- [ ] Generate screenshots
- [ ] Run smoke test
- [ ] Run Hackathon Hunter QA
"""
    readme = f"""# {idea.name}

{idea.tagline}

## Demo

Demo URL is pending human approval before publishing.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

## Tech Stack

{", ".join(idea.tech_stack)}

## Architecture

The MVP has a small web interface, a sponsor API integration layer, local
persistence for demo history, and an export path for submission materials.

## Data Disclosure

The initial scaffold uses demo data only when live API access is unavailable.
Any synthetic fallback must be disclosed in the final submission.
"""
    submission = f"""# SUBMISSION DRAFT

Hackathon: {hackathon.name}

Project: {idea.name}

Tagline: {idea.tagline}

## Description

{idea.name} helps {idea.target_user.lower()} by solving this problem: {idea.problem}

## Sponsor API Usage

{idea.sponsor_api_usage}

## Demo Materials

- Repository: pending human publication
- Demo: pending human publication
- Video: pending human recording

## Required Human Gate

Do not submit this draft automatically. Final submission requires manual review
and acceptance of official rules.
"""
    brief = f"""# AGENT BRIEF

Project: {idea.name}

Goal: Build the MVP described in `SPEC.md`.

Non-goals:
{chr(10).join(f"- {item}" for item in idea.non_goals)}

Required API: {required_api}

Demo path: {idea.demo_flow}

Acceptance tests:
- App starts locally.
- Primary demo path works with clear fallback errors.
- README has Demo, Quick Start, Tech Stack, and Architecture.
- `.env.example` exists and contains no real secrets.
- `python3 -m hackathon_hunter qa --project {project_dir.relative_to(project_path())}`
  produces a report.

Files not to touch:
- `data/processed/*`
- `reports/radar_*.md`
- `CODEX_HANDOFF.md`

Done criteria: produce `HANDOFF_REPORT.md` with implemented features, test
results, known gaps, and submission risks.
"""
    outputs = {
        "spec": _write(project_dir / "SPEC.md", spec),
        "tasks": _write(project_dir / "TASKS.md", tasks),
        "readme_draft": _write(project_dir / "README_DRAFT.md", readme),
        "readme": _write(project_dir / "README.md", readme),
        "submission": _write(project_dir / "SUBMISSION_DRAFT.md", submission),
        "brief": _write(project_dir / "AGENT_BRIEF.md", brief),
        "env": _write(project_dir / ".env.example", "SPONSOR_API_KEY=\n"),
        "requirements": _write(
            project_dir / "requirements.txt",
            "# Add runtime dependencies here.\n",
        ),
        "app": _write(
            project_dir / "app.py",
            'print("Scaffold ready. Replace this with the hackathon MVP app.")\n',
        ),
    }
    return outputs
