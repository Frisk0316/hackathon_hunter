from __future__ import annotations

import json

from claimclear.pipeline import ClaimClearPipeline
from claimclear.synthetic_data import load_claims


def test_clean_claim_auto_approves(tmp_path):
    pipeline = ClaimClearPipeline.from_fixtures(tmp_path / "audit.jsonl")
    result = pipeline.run(claim_payload=load_claims()["CLM-1001"])

    assert result.initial_decision.outcome == "auto_approved"
    assert result.final_decision.status == "resolved"
    assert result.risk_score.confidence >= 0.72
    assert result.policy_check.passed is True
    assert any(entry.actor == "decision_router" for entry in result.audit_entries)


def test_ambiguous_claim_escalates_then_human_approves(tmp_path):
    pipeline = ClaimClearPipeline.from_fixtures(tmp_path / "audit.jsonl")
    result = pipeline.run(claim_payload=load_claims()["CLM-1002"], resolve_human="approve")

    assert result.initial_decision.outcome == "escalated"
    assert result.initial_decision.human_task_id == "TASK-CLM-1002-REVIEW"
    assert result.final_decision.outcome == "human_approved"
    assert result.final_decision.status == "resolved"
    assert result.risk_score.confidence < 0.72
    assert any(entry.actor == "human_reviewer" for entry in result.audit_entries)


def test_audit_log_is_jsonl(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    pipeline = ClaimClearPipeline.from_fixtures(audit_path)
    pipeline.run(claim_payload=load_claims()["CLM-1001"])

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 5
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["actor"] == "maestro_case"
    assert parsed[-1]["action"] == "case_resolved"
