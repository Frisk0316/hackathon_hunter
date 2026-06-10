from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from pydantic import ValidationError

from hackathon_hunter.models import ResultRecord
from hackathon_hunter.scoring import DEFAULT_WEIGHTS, load_scoring_config
from hackathon_hunter.storage import project_path, read_json, save_report, unique_path, utcish_now

SUCCESS_OUTCOMES = {"winner", "finalist"}
FAILURE_OUTCOMES = {"rejected", "abandoned"}
MIN_TRACE_DELTA = 0.15
WEIGHT_STEP = 0.01


@dataclass(frozen=True)
class CalibrationSuggestion:
    feature: str
    direction: str
    current_weight: float
    suggested_weight: float
    success_average: float
    failure_average: float
    reason: str


@dataclass(frozen=True)
class CalibrationResult:
    report_path: Path
    records_read: int
    records_used: int
    suggestions: list[CalibrationSuggestion]


def _load_result_records(results_dir: Path) -> list[ResultRecord]:
    records: list[ResultRecord] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = read_json(path)
            records.append(ResultRecord.model_validate(payload))
        except (OSError, ValidationError, ValueError):
            continue
    return records


def _trace_averages(records: list[ResultRecord]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for record in records:
        for feature, value in record.score_trace.items():
            values.setdefault(feature, []).append(float(value))
    return {feature: mean(items) for feature, items in values.items()}


def _suggest_weight_changes(
    records: list[ResultRecord],
    weights: dict[str, float],
) -> list[CalibrationSuggestion]:
    successes = [
        record
        for record in records
        if record.outcome in SUCCESS_OUTCOMES and record.score_trace
    ]
    failures = [
        record
        for record in records
        if record.outcome in FAILURE_OUTCOMES and record.score_trace
    ]
    if not successes or not failures:
        return []

    success_averages = _trace_averages(successes)
    failure_averages = _trace_averages(failures)
    suggestions: list[CalibrationSuggestion] = []
    for feature in sorted(set(success_averages) & set(failure_averages)):
        delta = success_averages[feature] - failure_averages[feature]
        current = float(weights.get(feature, DEFAULT_WEIGHTS.get(feature, 0.0)))
        if delta >= MIN_TRACE_DELTA:
            suggestions.append(
                CalibrationSuggestion(
                    feature=feature,
                    direction="increase",
                    current_weight=current,
                    suggested_weight=current + WEIGHT_STEP,
                    success_average=success_averages[feature],
                    failure_average=failure_averages[feature],
                    reason="Successful rounds scored materially higher on this feature.",
                )
            )
        elif delta <= -MIN_TRACE_DELTA:
            suggestions.append(
                CalibrationSuggestion(
                    feature=feature,
                    direction="decrease",
                    current_weight=current,
                    suggested_weight=max(0.0, current - WEIGHT_STEP),
                    success_average=success_averages[feature],
                    failure_average=failure_averages[feature],
                    reason="Failed rounds scored materially higher on this feature.",
                )
            )
    return suggestions


def _top_lessons(records: list[ResultRecord], field: str) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(getattr(record, field))
    return counter.most_common(8)


def _render_lesson_section(title: str, lessons: list[tuple[str, int]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not lessons:
        lines.append("- None recorded")
        return lines
    lines.extend(f"- {lesson} ({count}x)" for lesson, count in lessons)
    return lines


def _render_report(
    records: list[ResultRecord],
    suggestions: list[CalibrationSuggestion],
    weights: dict[str, float],
) -> str:
    outcome_counts = Counter(record.outcome for record in records)
    lines = [
        f"# Calibration — {utcish_now().date().isoformat()}",
        "",
        "This report suggests scoring changes only. Do not apply them without human review.",
        "",
        "## Records",
        "",
        f"- Records read: {len(records)}",
        *[f"- {outcome}: {count}" for outcome, count in sorted(outcome_counts.items())],
        "",
        "## Suggested Weight Diff",
        "",
    ]
    if suggestions:
        lines.append("```diff")
        for suggestion in suggestions:
            lines.extend(
                [
                    f"- {suggestion.feature}: {suggestion.current_weight:.2f}",
                    f"+ {suggestion.feature}: {suggestion.suggested_weight:.2f}",
                ]
            )
        lines.append("```")
        lines.append("")
        lines.append("## Rationale")
        lines.append("")
        for suggestion in suggestions:
            lines.extend(
                [
                    f"### `{suggestion.feature}`",
                    "",
                    f"- Direction: {suggestion.direction}",
                    f"- Success average: {suggestion.success_average:.2f}",
                    f"- Failure average: {suggestion.failure_average:.2f}",
                    f"- Reason: {suggestion.reason}",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "No numeric weight changes suggested.",
                "",
                "Need at least one `winner`/`finalist` and one `rejected`/`abandoned` "
                "record with `score_trace` before calibration can compare features.",
                "",
            ]
        )

    lines.extend(_render_lesson_section("What Worked", _top_lessons(records, "what_worked")))
    lines.append("")
    lines.extend(_render_lesson_section("What Failed", _top_lessons(records, "what_failed")))
    lines.extend(
        [
            "",
            "## Current Weights",
            "",
            "| Feature | Weight |",
            "|---|---:|",
        ]
    )
    for feature, weight in sorted(weights.items()):
        lines.append(f"| {feature} | {float(weight):.2f} |")
    return "\n".join(lines)


def run_calibrate(
    results_dir: Path | None = None,
    weights_path: Path | None = None,
    root: Path | None = None,
) -> CalibrationResult:
    base = root or project_path()
    active_results_dir = results_dir or base / "logs" / "rounds"
    records = _load_result_records(active_results_dir)
    weights = load_scoring_config(weights_path).get("weights", {})
    suggestions = _suggest_weight_changes(records, weights)
    report_path = unique_path(
        base / "reports" / f"calibration_{utcish_now().strftime('%Y%m%d')}.md"
    )
    save_report(report_path, _render_report(records, suggestions, weights))
    records_used = sum(1 for record in records if record.score_trace)
    return CalibrationResult(
        report_path=report_path,
        records_read=len(records),
        records_used=records_used,
        suggestions=suggestions,
    )
