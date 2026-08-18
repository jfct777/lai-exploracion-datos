#!/usr/bin/env python3
"""Bind persistent root-specific B0 artifacts into the frozen Gnomix contract."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from m28c_gnomix_training_smoke import audit_breakpoint_probability_map  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: {observed} != {expected}")
    return observed


def load_frozen(path: Path, stage: str, status: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("stage") != stage or value.get("status") != status:
        raise ValueError(f"unexpected or unfrozen contract: {path}")
    return value


def marker_positions(path: Path) -> list[int]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    positions = [int(row["position"]) for row in rows]
    if len(positions) != 79791 or positions != sorted(set(positions)):
        raise ValueError("B0 table is not 79,791 ordered unique positions")
    return positions


def manifest_authenticates(
    path: Path,
    expected_stage: str,
    outputs: dict[str, Path],
    inputs: dict[str, Path] | None = None,
) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != expected_stage:
        raise ValueError(f"manifest has unexpected stage: {path}")
    output_hashes = manifest.get("sha256", {})
    for label, file_path in outputs.items():
        if output_hashes.get(file_path.name) != sha256(file_path):
            raise ValueError(f"manifest does not authenticate {label}")
    input_hashes = manifest.get("inputs", {})
    for label, file_path in (inputs or {}).items():
        if input_hashes.get(file_path.name) != sha256(file_path):
            raise ValueError(f"manifest does not authenticate upstream {label}")
    return manifest


def run(args: argparse.Namespace) -> dict:
    pre = load_frozen(args.preregistration, "M29_ROOT_GNOMIX_B0", "PRE_FROZEN_BEFORE_ROOT_TRAINING")
    producer = load_frozen(args.production_contract, "M29_ROOT_B0_PRODUCTION", "PRE_FROZEN_BEFORE_B0_SELECTION")
    if sha256(args.production_contract) != pre["authenticated_templates"]["production_contract_sha256"]:
        raise ValueError("producer contract differs from M29 Gnomix PRE")
    if args.root_label not in pre["roots"] or int(pre["roots"][args.root_label]) != args.root_seed:
        raise ValueError("root label/seed differs from M29 Gnomix PRE")
    producer_roots = {row.get("label"): int(row["root_seed"]) for row in producer.get("roots", [])}
    if producer_roots.get(args.root_label) != args.root_seed:
        raise ValueError("root label/seed differs from producer contract")
    if pre.get("resources", {}).get("memory_per_root") != "8 GB" or float(pre["resources"].get("fail_closed_peak_rss_gib", -1)) != 6.4 or int(pre["resources"].get("max_parallel_roots", -1)) != 2:
        raise ValueError("M29 Gnomix resource contract differs from the frozen 8 GiB / 6.4 GiB / two-root design")
    template = json.loads(args.template_contract.read_text(encoding="utf-8"))
    require_hash(args.template_contract, pre["authenticated_templates"]["full_b0_contract_sha256"], "full-B0 template")
    require_hash(args.gnomix_config, pre["authenticated_templates"]["gnomix_config_sha256"], "Gnomix config")
    require_hash(args.runner, pre["authenticated_templates"]["runner_sha256"], "Gnomix runner")

    selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    ingest = json.loads(args.ingest_report.read_text(encoding="utf-8"))
    if selection.get("root_seed") != args.root_seed or selection.get("decision") != "GO_ROOT_B0_MATERIALIZATION":
        raise ValueError("selection report belongs to another root or failed")
    if materialization.get("root_seed") != args.root_seed or materialization.get("m29_stage") != "M29_ROOT_B0_MATERIALIZATION" or materialization.get("decision") != "GO_EXTERNAL_GNOMIX_INGEST_VALIDATION":
        raise ValueError("materialization report belongs to another root or failed")
    if ingest.get("root_seed") != args.root_seed or ingest.get("m29_stage") != "M29_ROOT_B0_GNOMIX_INGEST" or ingest.get("decision") != "GO_M29_ROOT_B0_READY_FOR_TRAINING":
        raise ValueError("ingest report belongs to another root or failed")
    producer_hash = sha256(args.production_contract)
    if selection.get("production_contract_sha256") != producer_hash or materialization.get("m29_production_contract_sha256") != producer_hash or ingest.get("m29_production_contract_sha256") != producer_hash:
        raise ValueError("producer report chain cites another contract")

    manifest_authenticates(args.selection_manifest, "M29_ROOT_B0_SELECTION", {"B0": args.b0_markers, "selection report": args.selection_report})
    manifest_authenticates(
        args.materialization_manifest,
        "M29_ROOT_B0_MATERIALIZATION",
        {"sample map": args.sample_map, "materialization report": args.materialization_report},
        {"selection report": args.selection_report},
    )
    manifest_authenticates(
        args.ingest_manifest,
        "M29_ROOT_B0_GNOMIX_INGEST",
        {"reference VCF": args.reference_vcf, "reference TBI": args.reference_tbi, "target VCF": args.target_vcf, "target TBI": args.target_tbi, "ingest report": args.ingest_report},
        {"materialization report": args.materialization_report},
    )
    if materialization.get("m29_selection_report_sha256") != sha256(args.selection_report):
        raise ValueError("materialization report cites another selection report")
    if selection.get("output_sha256", {}).get(args.b0_markers.name) != sha256(args.b0_markers):
        raise ValueError("selection report does not authenticate B0")
    ingest_hashes = ingest.get("output_sha256", {})
    for file_path in (args.reference_vcf, args.reference_tbi, args.target_vcf, args.target_tbi):
        if ingest_hashes.get(file_path.name) != sha256(file_path):
            raise ValueError("ingest report does not authenticate an input")

    positions = marker_positions(args.b0_markers)
    sys.path.insert(0, str(args.gnomix_root))
    guard = audit_breakpoint_probability_map(args.genetic_map, positions, "22", float(template["numerical_guard"]["probability_tolerance"]), float(template["numerical_guard"]["negative_mass_tolerance"]))
    contract = copy.deepcopy(template)
    contract["scope"] = "M29_root_specific_full_B0_training_and_inference_no_truth_no_effect_estimation"
    contract["authenticated_inputs"] = {
        "reference_vcf_sha256": sha256(args.reference_vcf),
        "reference_tbi_sha256": sha256(args.reference_tbi),
        "target_vcf_sha256": sha256(args.target_vcf),
        "target_tbi_sha256": sha256(args.target_tbi),
        "sample_map_sha256": sha256(args.sample_map),
        "b0_marker_table_sha256": sha256(args.b0_markers),
        "genetic_map_sha256": sha256(args.genetic_map),
        "gnomix_config_sha256": sha256(args.gnomix_config),
    }
    contract["source_panel"]["first_position"] = positions[0]
    contract["source_panel"]["last_position"] = positions[-1]
    contract["execution"]["canonical_replicate"] = args.root_label
    contract["execution"]["audit_replicate"] = None
    contract["execution"]["replicates"] = [args.root_label]
    contract["execution"]["launch_order"] = "Root17 and root18 are independent DEV roots; run at most two root tasks concurrently."
    contract["numerical_guard"]["expected_negative_probability_count"] = guard["negative_probability_count"]
    contract["resources"].update({
        "cpus_per_training": 4,
        "memory_per_training": "8 GB",
        "peak_rss_stop_gib": 6.4,
        "parallel_training_replicates": 2,
        "resource_basis": "Equivalent M28C full-B0 peak RSS was approximately 3.46 GiB; 8 GiB allocation and 6.4 GiB stop threshold were frozen before M29 training.",
        "cloud_batch_jobs": 0,
        "new_vms": 0,
        "review_threshold_memory": "6.4 GiB",
    })
    contract["m29_binding"] = {
        "stage": "M29_ROOT_GNOMIX_B0",
        "root_label": args.root_label,
        "root_seed": args.root_seed,
        "preregistration_sha256": sha256(args.preregistration),
        "production_contract_sha256": producer_hash,
        "selection_report_sha256": sha256(args.selection_report),
        "materialization_report_sha256": sha256(args.materialization_report),
        "ingest_report_sha256": sha256(args.ingest_report),
        "selection_manifest_sha256": sha256(args.selection_manifest),
        "materialization_manifest_sha256": sha256(args.materialization_manifest),
        "ingest_manifest_sha256": sha256(args.ingest_manifest),
        "breakpoint_guard": guard,
        "truth_accessed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-label", required=True, choices=["root17", "root18"])
    parser.add_argument("--root-seed", required=True, type=int)
    for name in ("preregistration", "production-contract", "template-contract", "selection-report", "selection-manifest", "materialization-report", "materialization-manifest", "ingest-report", "ingest-manifest", "reference-vcf", "reference-tbi", "target-vcf", "target-tbi", "sample-map", "b0-markers", "genetic-map", "gnomix-config", "runner", "gnomix-root", "out"):
        parser.add_argument(f"--{name}", dest=name.replace("-", "_"), required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    value = run(parse_args())
    print(json.dumps({"root": value["m29_binding"]["root_label"], "root_seed": value["m29_binding"]["root_seed"]}, sort_keys=True))
