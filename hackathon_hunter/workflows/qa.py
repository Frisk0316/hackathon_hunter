from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from hackathon_hunter.storage import save_report

PLACEHOLDERS = ["TODO", "TBD", "VIDEO_URL", "DEMO_URL", "YOUR_REPO", "PLACEHOLDER"]
SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
BINARY_SUFFIXES = {
    ".7z",
    ".gif",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pyc",
    ".tar",
    ".webp",
    ".zip",
}
PLACEHOLDER_SECRET_TOKENS = {
    "changeme",
    "dummy",
    "example",
    "not_a_real",
    "placeholder",
    "xxx",
    "your_",
}
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|token|secret)\s*=\s*['\"]?[A-Za-z0-9_\-]{24,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[bp]-[A-Za-z0-9-]{10,}"),
]


def _readme_path(project: Path) -> Path | None:
    for name in ["README.md", "README_DRAFT.md"]:
        candidate = project / name
        if candidate.exists():
            return candidate
    return None


def _scan_text_files(project: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path in _iter_candidate_files(project):
        if not path.is_file():
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            files.append((path, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    return files


def _iter_candidate_files(project: Path) -> list[Path]:
    files: list[Path] = []
    for path in project.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(project).parts):
            continue
        files.append(path)
    return files


def _document_text(project: Path, name: str) -> str:
    path = project / name
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _contains_placeholder_secret(path: Path, line: str) -> bool:
    if path.name != ".env.example" and not path.name.endswith(".example"):
        return False
    normalized = line.lower()
    return any(token in normalized for token in PLACEHOLDER_SECRET_TOKENS)


def _secret_findings(project: Path, path: Path, text: str) -> list[str]:
    findings: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _contains_placeholder_secret(path, line):
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                relative = path.relative_to(project)
                findings.append(f"Possible secret detected in {relative}:{line_number}")
                break
    return findings


def _check_placeholders(label: str, text: str, failures: list[str]) -> None:
    for placeholder in PLACEHOLDERS:
        if placeholder in text:
            failures.append(f"{label} contains placeholder: {placeholder}")


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
        _check_placeholders("README", readme_text, failures)
        headings = re.findall(r"^##\s+(.+)$", readme_text, flags=re.MULTILINE)
        duplicates = [heading for heading, count in Counter(headings).items() if count > 1]
        for heading in duplicates:
            warnings.append(f"README has duplicate section: {heading}")

    submission_text = _document_text(project, "SUBMISSION_DRAFT.md")
    if submission_text:
        _check_placeholders("SUBMISSION_DRAFT.md", submission_text, failures)

    if not (project / ".env.example").exists():
        failures.append("Missing .env.example")
    if (project / ".env").exists():
        failures.append("A real .env file is present in the project directory")

    all_text = _scan_text_files(project)
    synthetic_referenced = False
    for path, text in all_text:
        failures.extend(_secret_findings(project, path, text))
        if "synthetic" in text.lower():
            synthetic_referenced = True

    if synthetic_referenced:
        if "synthetic" not in readme_text.lower():
            failures.append("Synthetic data is referenced but not disclosed in README")
        if submission_text and "synthetic" not in submission_text.lower():
            failures.append(
                "Synthetic data is referenced but not disclosed in SUBMISSION_DRAFT.md"
            )

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
