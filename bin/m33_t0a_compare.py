#!/usr/bin/env python3
"""Compare independent M33 T0a processes without opening scientific outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOTS = ("root17", "root18")
FAMILIES = ("local_linear", "small_residual_cnn_1d")
RADII_CM = [0.05, 0.1, 0.2, 0.5]
STRESS_PEOPLE = [30, 256, 1024, 2619]
FALSE_FIELDS = ("truth_read", "training", "gradients", "optimizer",
                "predictions_persisted", "consumable")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="append", required=True, type=Path)
    parser.add_argument("--stress-receipt", action="append", required=True, type=Path)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--oci-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_auth = json.loads(args.source_auth.read_text(encoding="utf-8"))
    require(source_auth.get("stage") == "M33_T0A_SOURCE_AUTH" and
            source_auth.get("status") == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES" and
            source_auth.get("git_commit") == args.implementation_commit and
            source_auth.get("source_sha256", {}).get("bin/m33_t0a_compare.py") ==
            sha256_file(args.source_root / "bin/m33_t0a_compare.py"),
            "T0a comparison source authentication differs")
    source_auth_sha = sha256_file(args.source_auth)
    require(args.oci_image.startswith(
        "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:"),
        "T0a comparison image is not fixed by project digest")
    require(len(args.receipt) == 8, "T0a requires exactly eight fresh-process receipts")
    grouped = {}
    child_records = []
    for path in args.receipt:
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload.get("stage") == "M33_T0A_FORWARD_TECHNICAL_ROOT" and
                payload.get("status") == "PASS_T0A_FORWARD_ONLY_NON_CONSUMABLE",
                "T0a child did not pass")
        key = (payload["root_label"], payload["model_family"])
        require(key[0] in ROOTS and key[1] in FAMILIES and
                payload["repetition"] in (0, 1), "T0a child identity differs")
        require(payload["root_seed"] == (20260817 if key[0] == "root17" else 20260818),
                "T0a child root seed differs")
        require(payload.get("implementation_commit") == args.implementation_commit and
                payload.get("source_auth_sha256") == source_auth_sha and
                payload.get("oci_image") == args.oci_image,
                "T0a child execution identity differs from command")
        require(payload.get("marker_count") == 512 and
                payload.get("radii_cM") == RADII_CM,
                "T0a child canonical geometry differs")
        require(all(payload.get(field) is False for field in FALSE_FIELDS) and
                payload.get("model_or_radius_selected") is False and
                payload.get("device") == "cpu" and payload.get("vram_applicable") is False,
                "T0a child firewall differs")
        grouped.setdefault(key, {})[payload["repetition"]] = payload
        child_records.append({"name": path.name, "sha256": sha256_file(path),
                              "stage": payload["stage"]})
    require(set(grouped) == {(root, family) for root in ROOTS for family in FAMILIES},
            "T0a root/model coverage differs")
    exact_fields = (
        "marker_count", "target_count", "radii_cM", "channel_count", "rare_locus_count",
        "valid_tokens", "padded_tokens", "row_count", "shard_count",
        "maximum_valid_tokens_per_shard", "output_semantic_sha256",
        "feature_semantic_sha256", "marker_index_semantic_sha256",
        "technical_locus_key_axis_semantic_sha256", "parameter_count",
        "parameter_shape_sha256", "parameter_value_sha256", "zero_residual_F0_max_abs",
        "simplex_max_abs", "invariance_checks", "nonzero_probe_invariance_checks",
        "nonzero_probe_delta_max_abs", "torch_version", "oci_image",
        "nonzero_probe_output_sha256", "nonzero_probe_delta_sha256",
        "nonzero_probe_feature_sha256",
        "implementation_commit", "source_auth_sha256", "bridge_receipt_sha256",
    )
    comparisons = []
    for key in sorted(grouped):
        require(set(grouped[key]) == {0, 1}, "T0a fresh-process repetitions differ")
        first, second = grouped[key][0], grouped[key][1]
        mismatches = [field for field in exact_fields if first[field] != second[field]]
        require(not mismatches, f"T0a cross-process determinism differs: {key} {mismatches}")
        require(max(first["peak_rss_fraction"], second["peak_rss_fraction"]) < 0.7,
                f"T0a memory warning reached: {key}")
        comparisons.append({
            "root_label": key[0], "model_family": key[1],
            "cross_process_exact": True,
            "output_semantic_sha256": first["output_semantic_sha256"],
            "feature_semantic_sha256": first["feature_semantic_sha256"],
            "zero_residual_F0_max_abs": first["zero_residual_F0_max_abs"],
            "simplex_max_abs": first["simplex_max_abs"],
            "invariance_checks": first["invariance_checks"],
            "nonzero_probe_invariance_checks": first["nonzero_probe_invariance_checks"],
            "nonzero_probe_delta_max_abs": first["nonzero_probe_delta_max_abs"],
            "nonzero_probe_output_sha256": first["nonzero_probe_output_sha256"],
            "nonzero_probe_delta_sha256": first["nonzero_probe_delta_sha256"],
            "nonzero_probe_feature_sha256": first["nonzero_probe_feature_sha256"],
            "maximum_peak_rss_fraction": max(first["peak_rss_fraction"],
                                               second["peak_rss_fraction"]),
            "elapsed_seconds_by_repetition": [first["elapsed_seconds"],
                                                second["elapsed_seconds"]],
        })
    require(len(args.stress_receipt) == 4, "T0a requires four stress receipts")
    stress = {}
    for path in args.stress_receipt:
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload.get("stage") == "M33_T0A_SYNTHETIC_STRESS" and
                payload.get("status") == "PASS_T0A_SYNTHETIC_STRESS_NON_CONSUMABLE" and
                payload.get("model_family") in FAMILIES and payload.get("repetition") in (0, 1),
                "T0a stress identity differs")
        require(payload.get("implementation_commit") == args.implementation_commit and
                payload.get("source_auth_sha256") == source_auth_sha and
                payload.get("oci_image") == args.oci_image,
                "T0a stress execution identity differs from command")
        require(payload.get("stress_people") == STRESS_PEOPLE and
                payload.get("stress_loci") == 512 and
                payload.get("synthetic_context_length") == 64,
                "T0a stress canonical geometry differs")
        require(all(payload.get(field) is False for field in FALSE_FIELDS) and
                payload.get("model_or_radius_selected") is False and
                payload.get("device") == "cpu" and payload.get("vram_applicable") is False,
                "T0a stress firewall differs")
        stress.setdefault(payload["model_family"], {})[payload["repetition"]] = payload
        child_records.append({"name": path.name, "sha256": sha256_file(path),
                              "stage": payload["stage"]})
    stress_comparisons = []
    stress_exact_fields = (
        "stress_people", "stress_loci", "synthetic_context_length",
        "synthetic_people_are_not_individuals", "parameter_count",
        "stress_loci_interpretation", "stress_is_streaming_not_full_DNABR_matrix",
        "ragged_geometry_probe",
        "parameter_shape_sha256", "parameter_value_sha256", "torch_version", "oci_image",
        "implementation_commit", "source_auth_sha256",
    )
    for family in FAMILIES:
        require(set(stress.get(family, {})) == {0, 1}, "T0a stress repetitions differ")
        first, second = stress[family][0], stress[family][1]
        require(all(first[field] == second[field] for field in stress_exact_fields),
                "T0a stress metadata differs across processes")
        first_rows = [{key: value for key, value in row.items()
                       if key not in {"elapsed_seconds", "valid_tokens_per_second",
                                      "peak_rss_fraction"}} for row in first["rows"]]
        second_rows = [{key: value for key, value in row.items()
                        if key not in {"elapsed_seconds", "valid_tokens_per_second",
                                       "peak_rss_fraction"}} for row in second["rows"]]
        require(first_rows == second_rows, "T0a stress output differs across processes")
        require(max(first["peak_rss_fraction"], second["peak_rss_fraction"]) < 0.7,
                "T0a stress memory warning reached")
        stress_comparisons.append({
            "model_family": family, "cross_process_exact": True,
            "rows": first_rows,
            "maximum_peak_rss_fraction": max(first["peak_rss_fraction"],
                                               second["peak_rss_fraction"]),
        })
    all_children = [payload for repetitions in grouped.values() for payload in repetitions.values()]
    all_children += [payload for repetitions in stress.values() for payload in repetitions.values()]
    for field in ("implementation_commit", "source_auth_sha256", "torch_version", "oci_image"):
        require(len({payload[field] for payload in all_children}) == 1,
                f"T0a global execution identity differs: {field}")
    require(len(child_records) == 12 and len({row["name"] for row in child_records}) == 12,
            "T0a child receipt inventory differs")
    result = {
        "stage": "M33_T0A_CROSS_PROCESS_COMPARISON",
        "status": "PASS_T0A_CROSS_PROCESS_TECHNICAL_ONLY",
        "comparisons": comparisons, "stress_comparisons": stress_comparisons,
        "child_receipts": sorted(child_records, key=lambda row: row["name"]),
        "root_seeds": {"root17": 20260817, "root18": 20260818},
        "model_families": list(FAMILIES), "marker_count": 512,
        "radii_cM": [0.05, 0.1, 0.2, 0.5],
        "scientific_evidence": False, "truth_read": False, "training": False,
        "gradients": False, "optimizer": False, "predictions_persisted": False,
        "model_or_radius_selected": False, "development_open": False,
        "t0b_open": False, "t1_open": False, "consumable": False,
        "implementation_commit": args.implementation_commit,
        "source_auth_sha256": source_auth_sha,
        "oci_image": args.oci_image,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
