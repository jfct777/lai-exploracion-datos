#!/usr/bin/env python3
"""Build the fixed M34 128-mosaic replication plan after the locus audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import m34_adaptive_sweep as sweep


ROOTS = ("R0", "R1", "R2")
FINALISTS = (("bilstm", "bilstm_r1"), ("unet_1d", "unet_r1"))
ARMS = ("RD", "RE")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(contract_path: Path, audit_path: Path) -> dict[str, Any]:
    contract = sweep.validate_contract(sweep.strict_json(contract_path))
    audit = sweep.strict_json(audit_path)
    if audit.get("stage") != "M34_RARE_LOCUS_DISTRIBUTION_AUDIT":
        raise ValueError("unexpected rare-locus audit stage")
    if audit.get("status") != "PASS_DESCRIPTIVE_AUDIT_NO_MODEL_SELECTION":
        raise ValueError("rare-locus audit did not pass")
    selection = audit.get("selection", {})
    if selection.get("selected_loci") != 660:
        raise ValueError("rare-locus audit does not reproduce 660 loci")
    if selection.get("minor_alt_loci") != 373 or selection.get("minor_ref_loci") != 287:
        raise ValueError("rare-locus orientation counts differ")
    for ancestry in ("AFR", "EUR", "NAM"):
        ancestry_audit = audit.get("ancestry", {}).get(ancestry, {})
        if ancestry_audit.get("minimum_callability") != 1.0:
            raise ValueError(f"{ancestry} callability differs from the audited panel")

    stage = contract["stages"]["replication_128"]
    seed = int(stage["seeds"][0])
    if list(stage["rotations"]) != list(ROOTS):
        raise ValueError("finalist roots differ from R0/R1/R2")
    if stage["maximum_families"] != 2 or stage["one_config_per_family"] is not True:
        raise ValueError("finalist capacity differs from the two-family design")
    tasks: list[dict[str, Any]] = []
    for root in ROOTS:
        for family, config_id in FINALISTS:
            for arm in ARMS:
                tasks.append(sweep._task(
                    contract, family, config_id, seed, root, arm,
                    int(stage["maximum_updates"]), float(stage["radius_cM"]),
                    "replication_128",
                ))
    return {
        "schema_version": "1.0.0",
        "stage": "M34_EXPLORATORY_128_REPLICATION_PLAN",
        "status": "PLAN_ONLY_NO_EXECUTION_TEST_CLOSED",
        "claim_level": "exploratory",
        "target_size": {"people": 128, "fit": 96, "valid": 32},
        "roots": list(ROOTS),
        "training_seeds": [seed],
        "maximum_updates": int(stage["maximum_updates"]),
        "warmup_updates": int(stage["training_overrides"]["warmup_updates"]),
        "validation_every_updates": int(
            stage["training_overrides"]["validation_every_updates"]
        ),
        "budget_rationale": (
            "3200 updates preserve the 800-update small-pilot exposure per FIT "
            "person when FIT grows from 24 to 96 people"
        ),
        "radius_cM": 0.2,
        "models": [config_id for _, config_id in FINALISTS],
        "comparisons": ["RE_minus_RD", "RE_minus_F0"],
        "test_opened": False,
        "task_count": len(tasks),
        "tasks": tasks,
        "audit_observations": {
            "selected_loci": selection["selected_loci"],
            "nam_enriched_loci": audit["nam_enrichment"][
                "loci_nam_af_ge_0_05_and_afr_eur_below_0_01"
            ],
            "nam_enriched_in_ge_2_units": audit["nam_enrichment"][
                "of_these_in_ge_2_nam_units"
            ],
            "nam_enriched_in_ge_3_units": audit["nam_enrichment"][
                "of_these_in_ge_3_nam_units"
            ],
        },
        "inputs": {
            "adaptive_contract_sha256": sha256_file(contract_path),
            "rare_locus_audit_sha256": sha256_file(audit_path),
        },
        "stop_rule": {
            "rare_signal": (
                "For each root compare RE-RD at boundary F1 0.2 cM; require positive "
                "differences in all three roots, at least 0.005 in two roots and a "
                "root median of at least 0.005."
            ),
            "sensitivity": (
                "RE-RD at 0.1 and 0.5 cM must be nonnegative in at least two roots, "
                "and all preregistered guardrails must remain within tolerance."
            ),
            "lai_improvement": (
                "The root-median RE-F0 must be positive and RE-F0 must be "
                "nonnegative in at least two roots."
            ),
            "scope_if_failed": "close these locus-ordered diploid finalists without opening TEST",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = build_plan(args.contract, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": plan["stage"], "task_count": plan["task_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
