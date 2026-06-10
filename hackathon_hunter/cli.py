from pathlib import Path

import typer
from rich.console import Console

from hackathon_hunter.workflows.analyze_winners import run_analyze_winners
from hackathon_hunter.workflows.build_spec import run_build_spec
from hackathon_hunter.workflows.collect import run_collect
from hackathon_hunter.workflows.ideate import run_ideate
from hackathon_hunter.workflows.qa import run_qa
from hackathon_hunter.workflows.rank import run_rank
from hackathon_hunter.workflows.results import run_record_result
from hackathon_hunter.workflows.rules_check import run_check_rules

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def collect(
    days_ahead: int = typer.Option(90, "--days-ahead"),
    min_prize_usd: float = typer.Option(1000, "--min-prize-usd"),
    online_only: bool = typer.Option(False, "--online-only"),
    mock: bool = typer.Option(False, "--mock"),
) -> None:
    """Collect hackathon candidates and write processed JSON plus radar report."""
    outputs = run_collect(
        days_ahead=days_ahead,
        min_prize_usd=min_prize_usd,
        online_only=online_only,
        mock=mock,
    )
    for label, path in outputs.items():
        console.print(f"{label}: {path}")


@app.command()
def rank(
    input: Path | None = typer.Option(None, "--input", exists=True, file_okay=True),
    profile: Path | None = typer.Option(None, "--profile", exists=True, file_okay=True),
    weights: Path | None = typer.Option(None, "--weights", exists=True, file_okay=True),
    fast_lane: bool = typer.Option(False, "--fast-lane"),
) -> None:
    """Rank processed hackathons and write reports/radar_YYYYMMDD.md."""
    path = run_rank(
        input_path=input,
        profile_path=profile,
        weights_path=weights,
        fast_lane_mode=fast_lane,
    )
    console.print(f"report: {path}")


@app.command("check-rules")
def check_rules(
    input: Path | None = typer.Option(None, "--input", exists=True, file_okay=True),
) -> None:
    """Check eligibility and submission rule gates."""
    paths = run_check_rules(input_path=input)
    for path in paths:
        console.print(f"report: {path}")


@app.command("analyze-winners")
def analyze_winners(
    hackathon_id: str = typer.Option(..., "--hackathon-id"),
    input: Path | None = typer.Option(None, "--input", exists=True, file_okay=True),
) -> None:
    """Generate an evidence-gated winner intelligence template."""
    path = run_analyze_winners(hackathon_id, input_path=input)
    console.print(f"report: {path}")


@app.command()
def ideate(
    hackathon_id: str = typer.Option(..., "--hackathon-id"),
    n: int = typer.Option(5, "--n", min=1, max=10),
    input: Path | None = typer.Option(None, "--input", exists=True, file_okay=True),
) -> None:
    """Generate multiple project ideas and pause for human selection."""
    outputs = run_ideate(hackathon_id, n=n, input_path=input)
    for label, path in outputs.items():
        console.print(f"{label}: {path}")


@app.command("build-spec")
def build_spec(
    hackathon_id: str = typer.Option(..., "--hackathon-id"),
    idea_id: str = typer.Option(..., "--idea-id"),
    input: Path | None = typer.Option(None, "--input", exists=True, file_okay=True),
) -> None:
    """Build project spec, tasks, agent brief, and submission draft."""
    outputs = run_build_spec(hackathon_id, idea_id, input_path=input)
    for label, path in outputs.items():
        console.print(f"{label}: {path}")


@app.command()
def qa(project: Path = typer.Option(..., "--project", exists=True, file_okay=False)) -> None:
    """Run local submission-package QA checks."""
    outputs = run_qa(project)
    console.print(f"passed: {outputs['passed']}")
    console.print(f"qa_report: {outputs['qa_report']}")
    console.print(f"submission_package: {outputs['submission_package']}")


@app.command("record-result")
def record_result(
    hackathon_id: str = typer.Option(..., "--hackathon-id"),
    outcome: str = typer.Option(..., "--outcome"),
    project_slug: str | None = typer.Option(None, "--project-slug"),
    hours_spent: float | None = typer.Option(None, "--hours-spent"),
    api_cost_usd: float | None = typer.Option(None, "--api-cost-usd"),
    infra_cost_usd: float | None = typer.Option(None, "--infra-cost-usd"),
) -> None:
    """Record an outcome and produce a retrospective stub."""
    outputs = run_record_result(
        hackathon_id,
        outcome,
        project_slug=project_slug,
        hours_spent=hours_spent,
        api_cost_usd=api_cost_usd,
        infra_cost_usd=infra_cost_usd,
    )
    for label, path in outputs.items():
        console.print(f"{label}: {path}")
