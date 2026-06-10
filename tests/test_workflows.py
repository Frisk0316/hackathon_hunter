from pathlib import Path

from typer.testing import CliRunner

from hackathon_hunter.cli import app

runner = CliRunner()


def test_mock_cli_end_to_end() -> None:
    result = runner.invoke(
        app,
        ["collect", "--mock", "--online-only", "--min-prize-usd", "1000"],
    )
    assert result.exit_code == 0, result.output
    assert Path("data/processed/mock_hackathons.json").exists()

    result = runner.invoke(app, ["rank", "--input", "data/processed/mock_hackathons.json"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["check-rules", "--input", "data/processed/mock_hackathons.json"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["ideate", "--hackathon-id", "mock-hackathon-001", "--n", "2"])
    assert result.exit_code == 0, result.output
    assert Path("strategy/mock-hackathon-001_ideas.json").exists()

    result = runner.invoke(
        app,
        ["build-spec", "--hackathon-id", "mock-hackathon-001", "--idea-id", "idea-001"],
    )
    assert result.exit_code == 0, result.output
    assert Path("projects/mock-hackathon-001/idea-001/SPEC.md").exists()

    result = runner.invoke(
        app,
        ["qa", "--project", "projects/mock-hackathon-001/idea-001"],
    )
    assert result.exit_code == 0, result.output
    assert "passed: True" in result.output
