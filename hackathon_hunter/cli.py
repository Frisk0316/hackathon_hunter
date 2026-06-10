from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from hackathon_hunter.runlog import run_logged
from hackathon_hunter.workflows.analyze_winners import run_analyze_winners
from hackathon_hunter.workflows.build_spec import run_build_spec
from hackathon_hunter.workflows.collect import CollectError, run_collect
from hackathon_hunter.workflows.ideate import run_ideate
from hackathon_hunter.workflows.import_hackathons import ImportValidationError, run_import
from hackathon_hunter.workflows.qa import run_qa
from hackathon_hunter.workflows.rank import run_rank
from hackathon_hunter.workflows.results import run_record_result
from hackathon_hunter.workflows.rules_check import run_check_rules
from hackathon_hunter.workflows.status import run_status
from hackathon_hunter.workflows.watch import run_watch

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def collect(
    days_ahead: int = typer.Option(90, "--days-ahead"),
    min_prize_usd: float = typer.Option(1000, "--min-prize-usd"),
    online_only: bool = typer.Option(False, "--online-only"),
    mock: bool = typer.Option(False, "--mock"),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Source adapter to use: mock, web_search, devpost, lablab, dorahacks.",
    ),
) -> None:
    """Collect hackathon candidates and write processed JSON plus radar report."""
    with run_logged(
        "collect",
        {
            "days_ahead": days_ahead,
            "min_prize_usd": min_prize_usd,
            "online_only": online_only,
            "mock": mock,
            "source": source,
        },
    ) as run_log:
        try:
            outputs = run_collect(
                days_ahead=days_ahead,
                min_prize_usd=min_prize_usd,
                online_only=online_only,
                mock=mock,
                source_name=source,
            )
        except CollectError as error:
            if error.raw_path:
                console.print(f"raw: {error.raw_path}")
            for message in error.errors:
                console.print(f"[red]error:[/red] {message}")
            run_log.errors.extend(error.errors)
            raise typer.Exit(code=1) from error
        run_log.set_outputs(**outputs)
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
    with run_logged(
        "rank",
        {"input": input, "profile": profile, "weights": weights, "fast_lane": fast_lane},
    ) as run_log:
        path = run_rank(
            input_path=input,
            profile_path=profile,
            weights_path=weights,
            fast_lane_mode=fast_lane,
        )
        run_log.set_outputs(report=path)
    console.print(f"report: {path}")


@app.command("import")
def import_hackathons(
    input: Path = typer.Option(..., "--input", exists=True, file_okay=True),
    merge: bool = typer.Option(False, "--merge"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Validate and import Claude evidence-backed radar JSON."""
    with run_logged(
        "import",
        {"input": input, "merge": merge, "dry_run": dry_run},
    ) as run_log:
        try:
            result = run_import(input_path=input, merge=merge, dry_run=dry_run)
        except ImportValidationError as error:
            for message in error.errors:
                console.print(f"[red]schema error:[/red] {message}")
            run_log.errors.extend(error.errors)
            raise typer.Exit(code=1) from error
        run_log.set_outputs(
            diff_report=result.diff_report_path,
            processed=result.processed_path,
            changes=len(result.changes),
            dry_run=result.dry_run,
        )
    console.print(f"diff_report: {result.diff_report_path}")
    if result.processed_path:
        console.print(f"processed: {result.processed_path}")
    else:
        console.print("processed: dry-run")


@app.command()
def status(input: Path | None = typer.Option(None, "--input", exists=True, file_okay=True)) -> None:
    """Show active candidate scores, gates, and stale evidence at a glance."""
    with run_logged("status", {"input": input}) as run_log:
        result = run_status(input_path=input)
        run_log.set_outputs(input=result.input_path, rows=len(result.rows))
    table = Table(title=f"Hackathon Hunter Status — {result.input_path}")
    table.add_column("ID")
    table.add_column("Days", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("TW")
    table.add_column("AI")
    table.add_column("Rules")
    table.add_column("Stale")
    for row in result.rows:
        table.add_row(
            row.hackathon_id,
            f"{row.days_until_deadline:.1f}",
            f"{row.overall_score:.2f}",
            row.taiwan_gate,
            row.ai_policy,
            row.rules_status,
            ", ".join(row.stale_fields) if row.stale_fields else "-",
        )
    console.print(table)


@app.command()
def watch(
    input: Path | None = typer.Option(None, "--input", exists=True, file_okay=True),
    fast_lane_days: int = typer.Option(3, "--fast-lane-days", min=0),
) -> None:
    """Check processed candidates for deadline and evidence watch events."""
    with run_logged(
        "watch",
        {"input": input, "fast_lane_days": fast_lane_days},
    ) as run_log:
        result = run_watch(input_path=input, fast_lane_days=fast_lane_days)
        run_log.set_outputs(report=result.report_path, events=len(result.events))
    console.print(f"report: {result.report_path}")
    if result.events:
        for event in result.events:
            console.print(f"[yellow]{event.kind}[/yellow] {event.hackathon_id}: {event.detail}")
        raise typer.Exit(code=2)
    console.print("No watch events.")


@app.command("check-rules")
def check_rules(
    input: Path | None = typer.Option(None, "--input", exists=True, file_okay=True),
) -> None:
    """Check eligibility and submission rule gates."""
    with run_logged("check-rules", {"input": input}) as run_log:
        paths = run_check_rules(input_path=input)
        run_log.set_outputs(reports=paths)
    for path in paths:
        console.print(f"report: {path}")


@app.command("analyze-winners")
def analyze_winners(
    hackathon_id: str = typer.Option(..., "--hackathon-id"),
    input: Path | None = typer.Option(None, "--input", exists=True, file_okay=True),
) -> None:
    """Generate an evidence-gated winner intelligence template."""
    with run_logged(
        "analyze-winners",
        {"hackathon_id": hackathon_id, "input": input},
    ) as run_log:
        path = run_analyze_winners(hackathon_id, input_path=input)
        run_log.set_outputs(report=path)
    console.print(f"report: {path}")


@app.command()
def ideate(
    hackathon_id: str = typer.Option(..., "--hackathon-id"),
    n: int = typer.Option(5, "--n", min=1, max=10),
    input: Path | None = typer.Option(None, "--input", exists=True, file_okay=True),
) -> None:
    """Generate multiple project ideas and pause for human selection."""
    with run_logged(
        "ideate",
        {"hackathon_id": hackathon_id, "n": n, "input": input},
    ) as run_log:
        outputs = run_ideate(hackathon_id, n=n, input_path=input)
        run_log.set_outputs(**outputs)
    for label, path in outputs.items():
        console.print(f"{label}: {path}")


@app.command("build-spec")
def build_spec(
    hackathon_id: str = typer.Option(..., "--hackathon-id"),
    idea_id: str = typer.Option(..., "--idea-id"),
    input: Path | None = typer.Option(None, "--input", exists=True, file_okay=True),
) -> None:
    """Build project spec, tasks, agent brief, and submission draft."""
    with run_logged(
        "build-spec",
        {"hackathon_id": hackathon_id, "idea_id": idea_id, "input": input},
    ) as run_log:
        outputs = run_build_spec(hackathon_id, idea_id, input_path=input)
        run_log.set_outputs(**outputs)
    for label, path in outputs.items():
        console.print(f"{label}: {path}")


@app.command()
def qa(project: Path = typer.Option(..., "--project", exists=True, file_okay=False)) -> None:
    """Run local submission-package QA checks."""
    with run_logged("qa", {"project": project}) as run_log:
        outputs = run_qa(project)
        run_log.set_outputs(**outputs)
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
    with run_logged(
        "record-result",
        {
            "hackathon_id": hackathon_id,
            "outcome": outcome,
            "project_slug": project_slug,
            "hours_spent": hours_spent,
            "api_cost_usd": api_cost_usd,
            "infra_cost_usd": infra_cost_usd,
        },
    ) as run_log:
        outputs = run_record_result(
            hackathon_id,
            outcome,
            project_slug=project_slug,
            hours_spent=hours_spent,
            api_cost_usd=api_cost_usd,
            infra_cost_usd=infra_cost_usd,
        )
        run_log.set_outputs(**outputs)
    for label, path in outputs.items():
        console.print(f"{label}: {path}")
