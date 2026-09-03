#!/usr/bin/env python3
"""Run historical TRACE materialization with an authenticated strict-SHAM adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from m37_trace_core import require
from m37_trace_materialize import materialize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--factors-receipt", type=Path, required=True)
    parser.add_argument("--reference-receipt", type=Path)
    parser.add_argument("--f0", type=Path, required=True)
    parser.add_argument("--marker-axis", type=Path, required=True)
    parser.add_argument("--marker-axis-receipt", type=Path, required=True)
    parser.add_argument("--arm", choices=("RE", "RD", "SHAM"), required=True)
    parser.add_argument("--beta-prior-strength", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    factors = json.loads(args.factors_receipt.read_text(encoding="utf-8"))
    require(
        factors.get("stage") == "M38B_APPLY_FROZEN_LOO_PRIMARY_MASK"
        and factors.get("decision") == "PASS_PRIMARY_FACTORS_FROZEN_FOR_MODEL"
        and factors.get("counts", {}).get("primary_loci") == 123
        and factors.get("counts", {}).get("target_people") == 96,
        "M38B primary-factor receipt differs",
    )
    for artifact in (args.selected, args.target):
        require(
            factors.get("outputs", {}).get(artifact.name, {}).get("sha256") == sha256(artifact),
            f"M38B primary-factor hash differs: {artifact.name}",
        )
    marker_document = json.loads(args.marker_axis_receipt.read_text(encoding="utf-8"))
    require(
        marker_document.get("stage") == "M37_BIND_MARKER_AXIS"
        and marker_document.get("source_receipt_sha256") is not None
        and marker_document.get("f0_source_sha256") == sha256(args.f0),
        "M38B F-minus/marker-axis binding differs",
    )
    adapter: Path | None = None
    if args.arm == "SHAM":
        require(args.reference_receipt is not None, "SHAM needs strict receipt")
        strict = json.loads(args.reference_receipt.read_text(encoding="utf-8"))
        require(
            strict.get("stage") == "M38B_STRICT_SHAM_REFERENCE"
            and strict.get("status") == "PASS_SINGLE_FROZEN_STRICT_DERANGEMENT"
            and strict.get("fixed_ancestry_labels") == 0
            and strict.get("per_locus_count_multiset_preserved") is True
            and strict.get("output_sha256") == sha256(args.reference),
            "M38B strict SHAM receipt differs",
        )
        require(
            strict.get("source_sha256") == factors.get("outputs", {}).get(
                "m38b_primary_reference_rare_summary.npz", {}
            ).get("sha256"),
            "M38B strict SHAM source is not the authenticated primary reference",
        )
        adapter = Path("m38b.strict_sham.m37.adapter.receipt.json")
        adapter.write_text(json.dumps({
            "schema_version": "1.0.0", "stage": "M37_TRACE_SHAM_REFERENCE",
            "status": "PASS_M38B_STRICT_SHAM_ADAPTER",
            "output_sha256": sha256(args.reference),
            "m38b_strict_sham_receipt_sha256": sha256(args.reference_receipt),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        require(args.reference_receipt is None,
                "reference receipt is forbidden outside SHAM")
        require(
            factors.get("outputs", {}).get(args.reference.name, {}).get("sha256") == sha256(args.reference),
            "M38B primary reference hash differs",
        )
    payload = materialize(
        args.selected, args.target, args.reference, args.f0, args.marker_axis,
        args.marker_axis_receipt, args.arm, args.beta_prior_strength, adapter,
    )
    # Preserve the historical deterministic format and receipt schema used by
    # downstream binders, but make the strict SHAM source visible in its receipt.
    from m33_safe_bridge_core import write_deterministic_npz
    write_deterministic_npz(args.output, payload)
    document = {
        "schema_version": "1.0.0", "stage": "M37_TRACE_MATERIALIZE",
        "arm": args.arm, "target_ref_disjoint": True,
        "target_fold_assignment": "forbidden", "output_sha256": sha256(args.output),
        "strict_sham_receipt_sha256": (
            sha256(args.reference_receipt) if args.reference_receipt else None
        ),
        "primary_factors_receipt_sha256": sha256(args.factors_receipt),
        "marker_axis_receipt_sha256": sha256(args.marker_axis_receipt),
    }
    args.output.with_suffix(".receipt.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS_M38B_MATERIALIZE", "arm": args.arm}, sort_keys=True))


if __name__ == "__main__":
    main()
