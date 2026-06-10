from __future__ import annotations

import argparse
import json
from pathlib import Path

from claimclear.pipeline import ClaimClearPipeline
from claimclear.synthetic_data import load_claims

DEFAULT_AUDIT_PATH = Path("artifacts/audit_log.jsonl")


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _print_result(label: str, result) -> None:
    print(f"\n{label}")
    print(f"  Case: {result.case_id}")
    claim_line = (
        f"  Claim: {result.claim.claim_id} | "
        f"{result.claim.claim_type} | {_money(result.claim.amount)}"
    )
    print(claim_line)
    print(
        "  Policy check: "
        f"{'pass' if result.policy_check.passed else 'review'} "
        f"({', '.join(result.policy_check.reasons)})"
    )
    print(
        "  Risk: "
        f"score={result.risk_score.risk_score:.2f}, "
        f"confidence={result.risk_score.confidence:.2f}"
    )
    if result.risk_score.risk_flags:
        print(f"  Flags: {', '.join(result.risk_score.risk_flags)}")
    print(
        "  Initial route: "
        f"{result.initial_decision.outcome} / {result.initial_decision.status}"
    )
    if result.initial_decision.human_task_id:
        print(f"  Human task: {result.initial_decision.human_task_id}")
    print(f"  Final: {result.final_decision.outcome} / {result.final_decision.status}")
    print(f"  Payout estimate: {_money(result.final_decision.payout_estimate)}")
    print(f"  Audit entries: {len(result.audit_entries)}")


def _write_summary(path: Path, results: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.to_dict() for result in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def command_demo(args: argparse.Namespace) -> int:
    audit_path = Path(args.audit_path)
    claims = load_claims()
    pipeline = ClaimClearPipeline.from_fixtures(audit_path)

    clean_result = pipeline.run(claim_payload=claims["CLM-1001"])
    ambiguous_result = pipeline.run(
        claim_payload=claims["CLM-1002"],
        resolve_human="approve",
    )

    print("ClaimClear local Maestro Case simulation")
    print("Synthetic data only. No real claims, policies, or personal data are used.")
    print(f"Audit log: {audit_path}")
    _print_result("Scenario 1: clean claim auto-clears", clean_result)
    _print_result("Scenario 2: ambiguous claim escalates, then human approves", ambiguous_result)

    summary_path = Path(args.summary_path)
    _write_summary(summary_path, [clean_result, ambiguous_result])
    print(f"\nDemo summary: {summary_path}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    claims = load_claims()
    if args.claim_id not in claims:
        known_ids = ", ".join(sorted(claims))
        raise SystemExit(f"Unknown claim id '{args.claim_id}'. Known claims: {known_ids}")

    pipeline = ClaimClearPipeline.from_fixtures(args.audit_path)
    result = pipeline.run(
        claim_payload=claims[args.claim_id],
        resolve_human=args.resolve_human,
    )
    print("ClaimClear single-claim run")
    print(f"Audit log: {args.audit_path}")
    _print_result(f"Claim {args.claim_id}", result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ClaimClear local agent demos.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run clean and ambiguous demo scenarios.")
    demo.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    demo.add_argument("--summary-path", default="artifacts/demo_summary.json")
    demo.set_defaults(func=command_demo)

    run = subparsers.add_parser("run", help="Run one synthetic claim.")
    run.add_argument("--claim-id", required=True)
    run.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    run.add_argument("--resolve-human", choices=["approve", "reject"])
    run.set_defaults(func=command_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
