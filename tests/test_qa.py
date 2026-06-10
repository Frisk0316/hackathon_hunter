from pathlib import Path

from hackathon_hunter.workflows.qa import run_qa


def _write_project(project: Path, *, readme_extra: str = "", submission: str = "") -> None:
    project.mkdir(parents=True)
    (project / "README.md").write_text(
        "\n".join(
            [
                "# Demo Project",
                "",
                "## Demo",
                "Demo link will be added after deployment.",
                "",
                "## Quick Start",
                "Run the app locally.",
                "",
                "## Tech Stack",
                "Python and tests.",
                "",
                "## Architecture",
                "Small local pipeline.",
                readme_extra,
            ]
        ),
        encoding="utf-8",
    )
    (project / "SUBMISSION_DRAFT.md").write_text(
        submission or "# Submission\n\nReady for review.\n",
        encoding="utf-8",
    )
    (project / ".env.example").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=your_openai_key_changeme",
                "GITHUB_TOKEN=xxx",
                "AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX",
            ]
        ),
        encoding="utf-8",
    )
    (project / "tests").mkdir()


def _report_text(result: dict[str, Path | bool]) -> str:
    return Path(result["qa_report"]).read_text(encoding="utf-8")


def test_qa_ignores_dependency_and_cache_directories(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_project(project)
    nested = project / "node_modules" / "package"
    nested.mkdir(parents=True)
    (nested / "index.js").write_text("const token = 'sk-abcdefghijklmnopqrstuvwx'\n")

    result = run_qa(project)

    assert result["passed"] is True


def test_qa_detects_broad_secret_patterns_but_allows_env_example_placeholders(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write_project(project)
    (project / "app.py").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY = 'sk-abcdefghijklmnopqrstuvwxyz'",
                "GITHUB_TOKEN = 'ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ'",
                "AWS_ACCESS_KEY_ID = 'AKIA1234567890ABCDEF'",
                "SLACK_BOT_TOKEN = 'xoxb-1234567890-abcdef'",
            ]
        ),
        encoding="utf-8",
    )

    result = run_qa(project)
    report = _report_text(result)

    assert result["passed"] is False
    assert report.count("Possible secret detected") == 4
    assert ".env.example" not in report


def test_qa_checks_submission_draft_placeholders(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_project(project, submission="# Submission\n\nDemo: DEMO_URL\nVideo: VIDEO_URL\n")

    result = run_qa(project)
    report = _report_text(result)

    assert result["passed"] is False
    assert "SUBMISSION_DRAFT.md contains placeholder: VIDEO_URL" in report
    assert "SUBMISSION_DRAFT.md contains placeholder: DEMO_URL" in report


def test_qa_reports_synthetic_disclosure_once_per_missing_document(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_project(project)
    data_dir = project / "data"
    data_dir.mkdir()
    (data_dir / "one.txt").write_text("synthetic claim fixture\n", encoding="utf-8")
    (data_dir / "two.txt").write_text("synthetic policy fixture\n", encoding="utf-8")

    result = run_qa(project)
    report = _report_text(result)

    assert result["passed"] is False
    assert report.count("Synthetic data is referenced but not disclosed in README") == 1
    assert (
        report.count("Synthetic data is referenced but not disclosed in SUBMISSION_DRAFT.md")
        == 1
    )
