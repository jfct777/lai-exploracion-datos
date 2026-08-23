#!/usr/bin/env python3
"""Fail-closed receipt comparison for the five M33 T0b full-chr22 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


EXPECTED_CASES = {
    ("root17", "local_linear", 0),
    ("root18", "local_linear", 0),
    ("root17", "small_residual_cnn_1d", 0),
    ("root17", "small_residual_cnn_1d", 1),
    ("root18", "small_residual_cnn_1d", 0),
}
FALSE_FIELDS = (
    "truth_read", "training", "gradients", "optimizer", "predictions_persisted",
    "model_or_radius_selected", "scientific_evidence", "consumable",
)
EXACT_REPEAT_FIELDS = (
    "marker_count", "target_count", "radii_cM", "channel_count", "rare_locus_count",
    "valid_tokens", "padded_tokens", "row_count", "shard_count",
    "maximum_valid_tokens_per_shard", "maximum_padded_tokens_per_batch",
    "output_semantic_sha256", "feature_semantic_sha256",
    "marker_index_semantic_sha256", "technical_locus_key_axis_semantic_sha256",
    "parameter_count", "parameter_shape_sha256", "parameter_value_sha256",
    "zero_residual_F0_max_abs", "simplex_max_abs", "invariance_checks",
    "sentinel_replay", "sentinel_replay_exact", "sentinel_passes",
    "memory_warning_fraction", "memory_stop_fraction", "device", "vram_applicable",
    "torch_version", "oci_image", "implementation_commit", "source_auth_sha256",
    "contract_sha256", "preflight_receipt_sha256", "bridge_receipt_sha256",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON object differs: {path}")
    return payload


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def bounded_zero(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1e-6


def compare_receipts(
    receipt_paths: list[Path], source_auth_path: Path, source_root: Path,
    implementation_commit: str, oci_image: str, contract_path: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    source_auth = load_json(source_auth_path)
    source_auth_sha = sha256_file(source_auth_path)
    require(source_auth.get("stage") == "M33_T0B_SOURCE_AUTH" and
            source_auth.get("status") == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES" and
            source_auth.get("git_commit") == implementation_commit and
            source_auth.get("source_sha256", {}).get("bin/m33_t0b_compare.py") ==
            sha256_file(source_root / "bin/m33_t0b_compare.py"),
            "T0b comparison source authentication differs")
    contract = load_json(contract_path)
    contract_sha = sha256_file(contract_path)
    require(contract.get("stage") == "M33_T0B_FULL_CHR22_CONTRACT" and
            contract.get("status") == "FROZEN_BEFORE_EXECUTION",
            "T0b comparison contract differs")
    expected_inputs = contract.get("expected_inputs", {})
    require(set(expected_inputs) == {"root17", "root18"},
            "T0b comparison expected roots differ")
    expected_identity = {
        root: {**identity, "marker_count": 79_791}
        for root, identity in expected_inputs.items()
    }
    preflight = load_json(preflight_path)
    preflight_sha = sha256_file(preflight_path)
    require(preflight.get("stage") == "M33_T0B_PREFLIGHT" and
            preflight.get("status") == "PASS_T0B_PREFLIGHT_THREE_WAY_ONLY" and
            preflight.get("contract_sha256") == contract_sha and
            preflight.get("implementation_commit") == implementation_commit and
            preflight.get("source_auth_sha256") == source_auth_sha and
            preflight.get("input_identity_by_root") == expected_identity,
            "T0b comparison preflight differs")
    require(len(receipt_paths) == 5, "T0b requires exactly five child receipts")
    observed: dict[tuple[str, str, int], dict[str, Any]] = {}
    children = []
    for path in receipt_paths:
        payload = load_json(path)
        require(payload.get("stage") == "M33_T0B_FULL_CHR22_FORWARD" and
                payload.get("status") == "PASS_T0B_FULL_CHR22_FORWARD_ONLY_NON_CONSUMABLE",
                "T0b child did not pass")
        key = (payload.get("root_label"), payload.get("model_family"),
               payload.get("repetition"))
        require(key in EXPECTED_CASES and key not in observed,
                f"T0b child inventory differs: {key}")
        require(payload.get("root_seed") == (
            20260817 if key[0] == "root17" else 20260818),
            "T0b child root seed differs")
        require(payload.get("marker_count") == 79_791 and
                payload.get("radii_cM") == [0.05, 0.1, 0.2, 0.5] and
                payload.get("maximum_padded_tokens_per_batch") == 262_144,
                "T0b child full geometry differs")
        require(payload.get("target_count", 0) > 0 and
                payload.get("row_count") == payload["target_count"] * 79_791 * 4 and
                0 < payload.get("maximum_valid_tokens_per_shard", 0) <= 262_144 and
                payload.get("padded_tokens", -1) >= payload.get("valid_tokens", 0) >= 0 and
                payload.get("shard_count", 0) > 0,
                "T0b child counters differ")
        expected_root = expected_inputs[key[0]]
        require(payload.get("target_count") == expected_root.get("target_count") and
                payload.get("rare_locus_count") == expected_root.get("rare_locus_count") and
                payload.get("bridge_receipt_sha256") ==
                expected_root.get("bridge_receipt_sha256"),
                "T0b child root provenance differs")
        expected_parameters = 90 if key[1] == "local_linear" else 1651
        require(payload.get("parameter_count") == expected_parameters,
                "T0b child parameter count differs")
        hash_fields = (
            "output_semantic_sha256", "feature_semantic_sha256",
            "marker_index_semantic_sha256", "technical_locus_key_axis_semantic_sha256",
            "parameter_shape_sha256", "parameter_value_sha256", "source_auth_sha256",
            "contract_sha256", "preflight_receipt_sha256", "bridge_receipt_sha256",
        )
        require(all(is_sha256(payload.get(field)) for field in hash_fields),
                "T0b child semantic hash differs")
        invariances = payload.get("invariance_checks")
        require(bounded_zero(payload.get("zero_residual_F0_max_abs")) and
                bounded_zero(payload.get("simplex_max_abs")) and
                isinstance(invariances, dict) and invariances and
                all(bounded_zero(value) for value in invariances.values()),
                "T0b child zero-head or invariance gate differs")
        require(payload.get("sentinel_replay_exact") is True and
                payload.get("sentinel_passes") == 2 and
                len(payload.get("sentinel_replay", [])) == 12,
                "T0b child sentinel replay differs")
        sentinel_keys = {(row.get("radius_cM"), row.get("marker_index"))
                         for row in payload["sentinel_replay"]}
        require(sentinel_keys == {(radius, marker) for radius in (0.05, 0.1, 0.2, 0.5)
                                  for marker in (0, 39_895, 79_790)},
                "T0b child sentinel inventory differs")
        require(all(
            is_sha256(row.get("output_semantic_sha256")) and
            is_sha256(row.get("feature_semantic_sha256")) and
            row.get("row_count") == payload["target_count"] and
            row.get("shard_count") == math.ceil(payload["target_count"] / 8) and
            row.get("padded_tokens", -1) >= row.get("valid_tokens", 0) >= 0
            for row in payload["sentinel_replay"]),
            "T0b child sentinel hashes or counters differ")
        require(payload.get("implementation_commit") == implementation_commit and
                payload.get("source_auth_sha256") == source_auth_sha and
                payload.get("oci_image") == oci_image and
                payload.get("contract_sha256") == contract_sha and
                payload.get("preflight_receipt_sha256") == preflight_sha,
                "T0b child execution identity differs")
        require(all(payload.get(field) is False for field in FALSE_FIELDS) and
                payload.get("device") == "cpu" and payload.get("vram_applicable") is False,
                "T0b child firewall differs")
        require(payload.get("memory_stop_fraction") == 0.80 and
                payload.get("memory_warning_fraction") == 0.70 and
                type(payload.get("peak_rss_fraction")) in (int, float) and
                math.isfinite(payload["peak_rss_fraction"]) and
                0 <= payload["peak_rss_fraction"] < 0.80 and
                payload.get("memory_warning") is (payload["peak_rss_fraction"] >= 0.70),
                "T0b child memory stop rule differs")
        observed[key] = payload
        children.append({"name": path.name, "sha256": sha256_file(path)})
    require(set(observed) == EXPECTED_CASES, "T0b five-case inventory differs")
    first = observed[("root17", "small_residual_cnn_1d", 0)]
    second = observed[("root17", "small_residual_cnn_1d", 1)]
    missing = [field for field in EXACT_REPEAT_FIELDS if field not in first or field not in second]
    require(not missing, f"T0b repeated CNN field missing: {missing}")
    mismatches = [field for field in EXACT_REPEAT_FIELDS if first[field] != second[field]]
    require(not mismatches, f"T0b root17 CNN full determinism differs: {mismatches}")
    require(len(children) == len({row["name"] for row in children}) == 5,
            "T0b child receipt filenames differ")
    return {
        "stage": "M33_T0B_FULL_CHR22_COMPARISON",
        "status": "PASS_T0B_FULL_CHR22_TECHNICAL_ONLY",
        "case_inventory": [list(case) for case in sorted(EXPECTED_CASES)],
        "child_receipts": sorted(children, key=lambda row: row["name"]),
        "root17_cnn_repetitions_exact": True,
        "root17_cnn_exact_fields": list(EXACT_REPEAT_FIELDS),
        "all_sentinel_replays_exact": True,
        "marker_count": 79_791, "radii_cM": [0.05, 0.1, 0.2, 0.5],
        "model_ranking_performed": False, "radius_ranking_performed": False,
        "scientific_evidence": False, "truth_read": False, "training": False,
        "gradients": False, "optimizer": False, "predictions_persisted": False,
        "consumable": False, "t1_open": False, "development_open": False,
        "implementation_commit": implementation_commit,
        "source_auth_sha256": source_auth_sha, "oci_image": oci_image,
        "contract_sha256": contract_sha, "preflight_receipt_sha256": preflight_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="append", type=Path, required=True)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--oci-image", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_receipts(
        args.receipt, args.source_auth, args.source_root, args.implementation_commit,
        args.oci_image, args.contract, args.preflight_receipt)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
