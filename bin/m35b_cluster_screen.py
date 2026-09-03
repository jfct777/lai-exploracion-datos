#!/usr/bin/env python3
"""Run one truth-blind M35B FLARE2 cluster-separation screen."""

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


class M35BScreenError(ValueError):
    """Raised when a frozen M35B screen invariant differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M35BScreenError(message)


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema_version") == "1.0.0", "M35B contract schema differs")
    require(value.get("experiment_id") == "M35B_FLARE2_BALANCED_SENSITIVITY_CHR22",
            "M35B contract identity differs")
    require(value.get("status") == "PREREGISTERED_EXPLORATORY_SCREEN",
            "M35B contract status differs")
    scope = value.get("scope", {})
    require(scope.get("chromosome") == "22" and scope.get("marker_count") == 42986,
            "M35B marker scope differs")
    balance = value.get("reference_balance", {})
    require(balance.get("selected_counts_per_seed") == {"AFR": 25, "EUR": 25, "NAM": 25},
            "M35B reference balance differs")
    screen = value.get("cluster_screen", {})
    require(screen.get("primary_granularity") == "coarse" and
            screen.get("sensitivity_granularity") == "fine" and
            screen.get("primary_gate") == "all_9_coarse_selection_by_gmm_combinations_must_pass" and
            screen.get("no_post_hoc_seed_selection") is True,
            "M35B screen policy differs")
    require(screen.get("nam_support_minimum") == 0.5 and
            screen.get("assignment_log_margin_minimum") == 0.25,
            "M35B cluster gate differs")
    return value


def verify_hash(path: Path, expected: str, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"invalid M35B {label}")
    require(m35.sha256_file(path) == expected, f"M35B {label} hash differs")


def screen(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    contract = load_contract(args.contract)
    require(args.selection_seed in contract["reference_balance"]["selection_seeds"],
            "M35B selection seed is not preregistered")
    require(args.gmm_seed in contract["cluster_screen"]["gmm_seeds"],
            "M35B GMM seed is not preregistered")
    require(args.granularity in {"coarse", "fine"}, "M35B panel granularity differs")
    require(not args.outdir.exists(), "refusing to overwrite M35B screen output")
    args.outdir.mkdir(parents=True)

    verify_hash(args.target_vcf, contract["inputs"]["target_vcf"]["sha256"], "target VCF")
    verify_hash(args.target_tbi, contract["inputs"]["target_tbi"]["sha256"], "target index")
    verify_hash(args.genetic_map, contract["inputs"]["genetic_map"]["sha256"], "genetic map")
    for path, label in ((args.reference_vcf, "balanced reference VCF"),
                        (args.reference_tbi, "balanced reference index"),
                        (args.sample_map, "balanced sample map"),
                        (args.panel_macro_map, "panel macro map"),
                        (args.prepare_receipt, "preparation receipt"),
                        (args.flare_jar, "FLARE jar"),
                        (args.model_wrapper, "GMM wrapper"),
                        (args.upstream_builder, "upstream GMM builder")):
        require(path.is_file() and not path.is_symlink(), f"invalid M35B {label}")

    preparation = json.loads(args.prepare_receipt.read_text(encoding="utf-8"))
    require(preparation.get("status") == "PASS_BALANCED_25_25_25" and
            preparation["selection"]["selection_seed"] == args.selection_seed and
            preparation["selection"]["selected_counts"] == {"AFR": 25, "EUR": 25, "NAM": 25},
            "M35B preparation receipt differs")
    require(preparation.get("valid_or_test_role_used_as_reference") is False,
            "M35B preparation used a reserved role as reference")
    require(preparation["source"]["target_vcf_sha256"] == contract["inputs"]["target_vcf"]["sha256"] and
            preparation["source"]["roles_sha256"] == contract["inputs"]["roles"]["sha256"],
            "M35B preparation sources differ")
    panel_receipt = preparation["panels"][args.granularity]
    require(panel_receipt["sample_map_sha256"] == m35.sha256_file(args.sample_map) and
            panel_receipt["panel_macro_map_sha256"] == m35.sha256_file(args.panel_macro_map),
            "M35B selected panel maps differ from the preparation receipt")

    reference = flare_common.scan_vcf(args.reference_vcf, "22")
    target = flare_common.scan_vcf(args.target_vcf, "22")
    require(len(reference["samples"]) == 75, "M35B balanced reference must contain 75 samples")
    require(reference["loci"] == target["loci"] and len(target["loci"]) == 42986,
            "M35B balanced reference and R0 target marker axes differ")
    require(m35.marker_axis_sha256(target["loci"]) == contract["scope"]["marker_axis_sha256"],
            "M35B marker axis hash differs")
    require(set(reference["samples"]).isdisjoint(target["samples"]),
            "M35B balanced reference overlaps R0 target")
    require(m35._axis_digest(reference["samples"]) ==
            preparation["reference"]["sample_axis_sha256"],
            "M35B balanced reference sample axis differs from its preparation receipt")

    panel_path = args.outdir / "m35b.ref-panel.tsv"
    map_path = args.outdir / "m35b.map"
    panel_audit = m35.normalize_panel_maps(
        args.sample_map, args.panel_macro_map, panel_path,
        reference["samples"], ["AFR", "EUR", "NAM"],
    )
    expected_mode = ("COARSE_MACROANCESTRY_EXPLICIT" if args.granularity == "coarse"
                     else "FINE_POPULATION_OR_SUBPOPULATION")
    require(panel_audit["mode"] == expected_mode, "M35B panel granularity label differs")
    require(panel_audit["macro_counts"] == {"AFR": 25, "EUR": 25, "NAM": 25},
            "M35B normalized panel is not balanced")
    map_audit = flare_common.normalize_genetic_map(
        args.genetic_map, map_path, "22", target["vcf_chromosome"],
        target["first_bp"], target["last_bp"],
    )

    parameters = contract["flare_parameters"]["panel_probability"]
    panel_prefix = args.outdir / "m35b.panel_probs"
    panel_command = m35.build_command(
        args.flare_jar, args.reference_vcf, args.target_vcf, panel_path,
        map_path, panel_prefix, parameters,
    )
    subprocess.run(panel_command, check=True)
    panels_path = Path(f"{panel_prefix}.panels")
    model_prefix = args.outdir / "m35b.cluster"
    model_command = [
        "python3", str(args.model_wrapper), "--seed", str(args.gmm_seed),
        "--builder", str(args.upstream_builder), "3", str(panels_path), str(model_prefix),
    ]
    subprocess.run(model_command, check=True)
    model_path = Path(f"{model_prefix}.model")

    raw = m35.cluster_assignment_evidence_from_model(
        model_path, ["AFR", "EUR", "NAM"], panel_audit["panel_to_ancestry"],
        0.0, contract["cluster_screen"]["assignment_log_margin_minimum"],
    )
    selected_by_ancestry = {
        ancestry: raw["selected_panel_probability"][cluster]
        for cluster, ancestry in raw["cluster_to_ancestry"].items()
    }
    reasons: list[str] = []
    if selected_by_ancestry["NAM"] < contract["cluster_screen"]["nam_support_minimum"]:
        reasons.append("NAM_support_below_0.5")
    if raw["log_margin"] < contract["cluster_screen"]["assignment_log_margin_minimum"]:
        reasons.append("assignment_log_margin_below_0.25")
    status = "PASS_M35B_CLUSTER_SEPARATION" if not reasons else "NO_GO_M35B_CLUSTER_SEPARATION"
    evidence = {
        **raw,
        "experiment_id": contract["experiment_id"],
        "status": status,
        "failure_reasons": reasons,
        "selection_seed": args.selection_seed,
        "gmm_seed": args.gmm_seed,
        "granularity": args.granularity,
        "selected_panel_probability_by_ancestry": selected_by_ancestry,
        "NAM_support": selected_by_ancestry["NAM"],
        "NAM_support_minimum": contract["cluster_screen"]["nam_support_minimum"],
        "assignment_log_margin_minimum": contract["cluster_screen"]["assignment_log_margin_minimum"],
        "balanced_macro_counts": panel_audit["macro_counts"],
        "reference_sample_axis_sha256": preparation["reference"]["sample_axis_sha256"],
        "marker_axis_sha256": contract["scope"]["marker_axis_sha256"],
        "target_truth_opened": False,
    }
    evidence_path = args.outdir / "m35b.cluster_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M35B_TRUTH_BLIND_CLUSTER_SCREEN",
        "status": status,
        "selection_seed": args.selection_seed,
        "gmm_seed": args.gmm_seed,
        "granularity": args.granularity,
        "wall_seconds": time.monotonic() - started,
        "children_max_rss_kib": usage.ru_maxrss,
        "contract_sha256": m35.sha256_file(args.contract),
        "prepare_receipt_sha256": m35.sha256_file(args.prepare_receipt),
        "evidence_sha256": m35.sha256_file(evidence_path),
        "model_sha256": m35.sha256_file(model_path),
        "panels_sha256": m35.sha256_file(panels_path),
        "normalized_map_sha256": map_audit["sha256"],
        "balanced_reference_vcf_sha256": m35.sha256_file(args.reference_vcf),
        "balanced_reference_tbi_sha256": m35.sha256_file(args.reference_tbi),
        "sample_map_sha256": m35.sha256_file(args.sample_map),
        "panel_macro_map_sha256": m35.sha256_file(args.panel_macro_map),
        "truth_input_present": False,
        "final_inference_performed": False,
    }
    receipt_path = args.outdir / "m35b.screen_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--gmm-seed", type=int, required=True)
    parser.add_argument("--granularity", choices=("coarse", "fine"), required=True)
    parser.add_argument("--reference-vcf", type=Path, required=True)
    parser.add_argument("--reference-tbi", type=Path, required=True)
    parser.add_argument("--target-vcf", type=Path, required=True)
    parser.add_argument("--target-tbi", type=Path, required=True)
    parser.add_argument("--sample-map", type=Path, required=True)
    parser.add_argument("--panel-macro-map", type=Path, required=True)
    parser.add_argument("--prepare-receipt", type=Path, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--flare-jar", type=Path, required=True)
    parser.add_argument("--model-wrapper", type=Path, required=True)
    parser.add_argument("--upstream-builder", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = screen(parse_args())
    print(json.dumps({"status": result["status"]}, sort_keys=True))
