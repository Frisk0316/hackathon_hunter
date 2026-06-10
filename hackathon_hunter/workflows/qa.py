from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from hackathon_hunter.storage import save_report

PLACEHOLDERS = ["TODO", "TBD", "VIDEO_URL", "DEMO_URL", "YOUR_REPO", "PLACEHOLDER"]
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|token|secret)\s*=\s*['\"]?[A-Za-z0-9_\-]{24,}"),
]


def _readme_path(project: Path) -> Path | None:
    for name in ["README.md", "README_DRAFT.md"]:
        candidate = project / name
        if candidate.exists():
            return candidate
    return None


def _scan_text_files(project: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path in project.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".zip", ".pyc"}:
            continue
        try:
            files.append((path, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    return files


def run_qa(project_path: Path) -> dict[str, Path | bool]:
    project = project_path.resolve()
    failures: list[str] = []
    warnings: list[str] = []

    readme = _readme_path(project)
    readme_text = ""
    if readme is None:
        failures.append("Missing README.md or README_DRAFT.md")
    else:
        readme_text = readme.read_text(encoding="utf-8")
        for section in ["Demo", "Quick Start", "Tech Stack", "Architecture"]:
            if f"## {section}" not in readme_text:
                failures.append(f"README missing section: {section}")
        for placeholder in PLACEHOLDERS:
            if placeholder in readme_text:
                failures.append(f"README contains placeholder: {placeholder}")
        headings = re.findall(r"^##\s+(.+)$", readme_text, flags=re.MULTILINE)
        duplicates = [heading for heading, count in Counter(headings).items() if count > 1]
        for heading in duplicates:
            warnings.append(f"README has duplicate section: {heading}")

    if not (project / ".env.example").exists():
        failures.append("Missing .env.example")
    if (project / ".env").exists():
        failures.append("A real .env file is present in the project directory")

    all_text = _scan_text_files(project)
    for path, text in all_text:
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                failures.append(f"Possible secret detected in {path.relative_to(project)}")
        if "synthetic" in text.lower() and "synthetic" not in readme_text.lower():
            failures.append("Synthetic data is referenced but not disclosed in README")

    if not any((project / name).exists() for name in ["tests", "scripts", "smoke_test.py"]):
        warnings.append("No tests or smoke-test script found")

    passed = not failures
    report = "\n".join(
        [
            "# QA REPORT",
            "",
            f"Status: {'PASS' if passed else 'FAIL'}",
            "",
            "## Failures",
            "",
            *(f"- {item}" for item in failures or ["None"]),
            "",
            "## Warnings",
            "",
            *(f"- {item}" for item in warnings or ["None"]),
            "",
            "## Human Gate",
            "",
            "This QA report is advisory. Final submission still requires manual review.",
        ]
    )
    package = "\n".join(
        [
            "# SUBMISSION PACKAGE",
            "",
            f"QA status: {'PASS' if passed else 'FAIL'}",
            "",
            "- Review README",
            "- Review submission draft",
            "- Confirm demo/video links manually",
            "- Submit only after accepting official rules",
        ]
    )
    qa_path = save_report(project / "QA_REPORT.md", report)
    package_path = save_report(project / "SUBMISSION_PACKAGE.md", package)
    return {"passed": passed, "qa_report": qa_path, "submission_package": package_path}
