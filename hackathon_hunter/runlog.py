from __future__ import annotations

from pathlib import Path
from typing import Any

from hackathon_hunter.models import RunLog
from hackathon_hunter.storage import project_path, unique_path, utcish_now, write_json


def _sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    return value


class WorkflowRunLog:
    def __init__(self, workflow: str, inputs: dict[str, Any]) -> None:
        self.workflow = workflow
        self.inputs = _sanitize(inputs)
        self.outputs: dict[str, Any] = {}
        self.errors: list[str] = []
        self.path: Path | None = None
        self.record: RunLog | None = None

    def __enter__(self) -> WorkflowRunLog:
        started_at = utcish_now()
        stem = f"{started_at.strftime('%Y%m%d_%H%M%S')}_{self.workflow}"
        self.path = unique_path(project_path("logs", "runs", f"{stem}.json"))
        self.record = RunLog(
            run_id=self.path.stem,
            workflow=self.workflow,
            started_at=started_at,
            inputs=self.inputs,
        )
        write_json(self.path, self.record)
        return self

    def set_outputs(self, **outputs: Any) -> None:
        self.outputs.update(_sanitize(outputs))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _tb,
    ) -> bool:
        if self.record is None or self.path is None:
            return False
        status = "failed" if exc else "succeeded"
        errors = self.errors.copy()
        if exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        self.record = self.record.model_copy(
            update={
                "completed_at": utcish_now(),
                "status": status,
                "outputs": _sanitize(self.outputs),
                "errors": errors,
            }
        )
        write_json(self.path, self.record)
        return False


def run_logged(workflow: str, inputs: dict[str, Any]) -> WorkflowRunLog:
    return WorkflowRunLog(workflow, inputs)
