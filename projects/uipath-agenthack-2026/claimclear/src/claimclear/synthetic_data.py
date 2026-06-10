from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claimclear.models import Claim, Policy


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_POLICIES: list[dict[str, Any]] = [
    {
        "policy_id": "POL-2001",
        "holder_name": "Ava Lin",
        "status": "active",
        "effective_date": "2026-01-01",
        "expiration_date": "2026-12-31",
        "coverage_limits": {
            "auto_physical_damage": 15000,
            "property_water_damage": 12000,
            "theft": 5000,
        },
        "deductible": 500,
        "prior_claims_last_24_months": 0,
        "notes": ["Synthetic policy for clean demo path."],
    },
    {
        "policy_id": "POL-2002",
        "holder_name": "Mateo Chen",
        "status": "active",
        "effective_date": "2026-01-01",
        "expiration_date": "2026-12-31",
        "coverage_limits": {
            "auto_physical_damage": 10000,
            "property_water_damage": 12000,
            "theft": 4000,
        },
        "deductible": 750,
        "prior_claims_last_24_months": 2,
        "notes": ["Synthetic policy for human-review demo path."],
    },
    {
        "policy_id": "POL-2003",
        "holder_name": "Nora Patel",
        "status": "lapsed",
        "effective_date": "2025-01-01",
        "expiration_date": "2025-12-31",
        "coverage_limits": {
            "auto_physical_damage": 8000,
            "property_water_damage": 8000,
        },
        "deductible": 1000,
        "prior_claims_last_24_months": 1,
        "notes": ["Synthetic policy for policy-exception testing."],
    },
]

SYNTHETIC_CLAIMS: list[dict[str, Any]] = [
    {
        "claim_id": "CLM-1001",
        "policy_id": "POL-2001",
        "claimant_name": "Ava Lin",
        "claimant_email": "ava.lin@example.invalid",
        "claim_type": "auto_physical_damage",
        "amount": 2400,
        "incident_date": "2026-06-04",
        "reported_date": "2026-06-05",
        "loss_location": "Seattle, WA",
        "narrative": (
            "Minor rear bumper damage after a low-speed parking lot collision. "
            "Photos, repair estimate, and police report are attached."
        ),
        "documents": ["photos", "repair_estimate", "police_report"],
        "metadata": {
            "scenario": "clean_auto_clear",
            "synthetic": True,
        },
    },
    {
        "claim_id": "CLM-1002",
        "policy_id": "POL-2002",
        "claimant_name": "Mateo Chen",
        "claimant_email": "mateo.chen@example.invalid",
        "claim_type": "property_water_damage",
        "amount": 9850,
        "incident_date": "2026-05-15",
        "reported_date": "2026-06-06",
        "loss_location": "Austin, TX",
        "narrative": (
            "Water damage noticed after returning from travel. The exact source is unclear, "
            "the plumber invoice is not attached, and the first estimate changed twice."
        ),
        "documents": ["photos"],
        "metadata": {
            "scenario": "ambiguous_human_review",
            "synthetic": True,
        },
    },
    {
        "claim_id": "CLM-1003",
        "policy_id": "POL-2003",
        "claimant_name": "Nora Patel",
        "claimant_email": "nora.patel@example.invalid",
        "claim_type": "auto_physical_damage",
        "amount": 6200,
        "incident_date": "2026-05-20",
        "reported_date": "2026-05-21",
        "loss_location": "Denver, CO",
        "narrative": "Front-end collision with photos and a repair estimate attached.",
        "documents": ["photos", "repair_estimate"],
        "metadata": {
            "scenario": "policy_exception",
            "synthetic": True,
        },
    },
]


def _load_json_or_default(filename: str, default: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "data" / filename
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_claims() -> dict[str, Claim]:
    records = _load_json_or_default("claims.json", SYNTHETIC_CLAIMS)
    return {record["claim_id"]: Claim.from_dict(record) for record in records}


def load_policies() -> dict[str, Policy]:
    records = _load_json_or_default("policies.json", SYNTHETIC_POLICIES)
    return {record["policy_id"]: Policy.from_dict(record) for record in records}


def write_fixtures(output_dir: Path | None = None) -> None:
    target_dir = output_dir or PROJECT_ROOT / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in {
        "claims.json": SYNTHETIC_CLAIMS,
        "policies.json": SYNTHETIC_POLICIES,
    }.items():
        path = target_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
            handle.write("\n")

