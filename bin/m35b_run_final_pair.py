#!/usr/bin/env python3
"""Run the single preassigned balanced FLARE/FLARE2 pair after the 9/9 gate."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m34_run_flare as flare_common
import m35_flare2_paired as m35
from m35b_cluster_screen import load_contract, require


def run_final(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    require(not args.outdir.exists(), "refusing to overwrite M35B final output")
    args.outdir.mkdir(parents=True)
    contract = load_contract(args.contract)
    primary = contract["primary_final_pair"]
    token = json.loads(args.go_token.read_text(encoding="utf-8"))
    gate = json.loads(args.gate_receipt.read_text(encoding="utf-8"))
    require(gate.get("status") == "PASS_M35B_PRIMARY_9_OF_9_GO_PREASSIGNED_FINAL" and
            gate.get("truth_opened") is False,
            "M35B final is blocked by the cluster gate")
    require(token.get("status") == "GO_PREASSIGNED_FINAL_ONLY" and
            token.get("gate_sha256") == m35.sha256_file(args.gate_receipt),
            "M35B final token differs from aggregate gate")
    require({key: token[key] for key in ("selection_seed", "gmm_seed", "granularity")} ==
            {key: primary[key] for key in ("selection_seed", "gmm_seed", "granularity")},
            "M35B final pair is not the prospectively assigned pair")
    evidence_path = args.screen_dir / "m35b.cluster_evidence.json"
    model_path = args.screen_dir / "m35b.cluster.model"
    panel_path = args.screen_dir / "m35b.ref-panel.tsv"
    map_path = args.screen_dir / "m35b.map"
    for path in (evidence_path, model_path, panel_path, map_path, args.reference_vcf,
                 args.target_vcf, args.flare_jar):
        require(path.is_file() and not path.is_symlink(), f"invalid M35B final input: {path}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    require(evidence.get("status") == "PASS_M35B_CLUSTER_SEPARATION" and
            evidence["selection_seed"] == primary["selection_seed"] and
            evidence["gmm_seed"] == primary["gmm_seed"] and
            evidence["granularity"] == "coarse",
            "M35B final screen evidence differs from the preassigned pair")
    require(evidence["model_sha256"] == m35.sha256_file(model_path),
            "M35B final model hash differs from screen evidence")

    reference = flare_common.scan_vcf(args.reference_vcf, "22")
    target = flare_common.scan_vcf(args.target_vcf, "22")
    require(len(reference["samples"]) == 75 and reference["loci"] == target["loci"] and
            len(target["loci"]) == contract["scope"]["marker_count"],
            "M35B final geometry differs")
    require(set(reference["samples"]).isdisjoint(target["samples"]),
            "M35B final reference overlaps target")

    direct_prefix = args.outdir / "m35b.direct"
    flare2_prefix = args.outdir / "m35b.flare2.raw"
    direct_command = m35.build_command(
        args.flare_jar, args.reference_vcf, args.target_vcf, panel_path,
        map_path, direct_prefix, contract["flare_parameters"]["direct"],
    )
    flare2_command = m35.build_command(
        args.flare_jar, args.reference_vcf, args.target_vcf, panel_path,
        map_path, flare2_prefix, contract["flare_parameters"]["final"], model_path,
    )
    subprocess.run(direct_command, check=True)
    subprocess.run(flare2_command, check=True)
    canonical_path = args.outdir / "m35b.flare2.anc.vcf.gz"
    m35.relabel_flare2_vcf(
        Path(f"{flare2_prefix}.anc.vcf.gz"), canonical_path, evidence, ["AFR", "EUR", "NAM"],
    )
    direct_path = Path(f"{direct_prefix}.anc.vcf.gz")
    audits = {
        "FLARE_0_6_BALANCED": flare_common.audit_ancestry_vcf(
            direct_path, target, ["AFR", "EUR", "NAM"]),
        "FLARE2_BALANCED": flare_common.audit_ancestry_vcf(
            canonical_path, target, ["AFR", "EUR", "NAM"]),
    }
    require(audits["FLARE_0_6_BALANCED"]["marker_count"] ==
            audits["FLARE2_BALANCED"]["marker_count"] == contract["scope"]["marker_count"],
            "M35B final output marker axes differ")
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M35B_PREASSIGNED_FINAL_TRUTH_BLIND_INFERENCE",
        "status": "PASS_M35B_PAIRED_INFERENCE_READY_FOR_SEPARATE_SCORE",
        "selection_seed": primary["selection_seed"],
        "gmm_seed": primary["gmm_seed"],
        "granularity": primary["granularity"],
        "reference_sample_count": len(reference["samples"]),
        "target_sample_count": len(target["samples"]),
        "marker_count": len(target["loci"]),
        "output_audits": audits,
        "output_sha256": {
            "FLARE_0_6_BALANCED": m35.sha256_file(direct_path),
            "FLARE2_BALANCED": m35.sha256_file(canonical_path),
        },
        "contract_sha256": m35.sha256_file(args.contract),
        "gate_sha256": m35.sha256_file(args.gate_receipt),
        "cluster_evidence_sha256": m35.sha256_file(evidence_path),
        "wall_seconds": time.monotonic() - started,
        "children_max_rss_kib": usage.ru_maxrss,
        "truth_input_present": False,
        "scoring_performed": False,
    }
    (args.outdir / "m35b.final_inference_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--go-token", type=Path, required=True)
    parser.add_argument("--gate-receipt", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--reference-vcf", type=Path, required=True)
    parser.add_argument("--target-vcf", type=Path, required=True)
    parser.add_argument("--flare-jar", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run_final(parse_args())
    print(json.dumps({"status": result["status"]}, sort_keys=True))
