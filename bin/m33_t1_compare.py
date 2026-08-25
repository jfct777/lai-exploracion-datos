#!/usr/bin/env python3
"""Compare the four M33 T1 receipts and close the technical gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import m33_t1_preflight as preflight


EXPECTED_CASES = {
    ("local_linear", 0), ("local_linear", 1),
    ("small_residual_cnn_1d", 0), ("small_residual_cnn_1d", 1),
}
FALSE_FIELDS = (
    "truth_read", "real_data_read", "training", "optimizer_created",
    "optimizer_step", "checkpoint_written", "predictions_persisted",
    "tensors_persisted", "scientific_evidence", "consumable", "development_open",
)
EXACT_REPEAT_FIELDS = (
    "people", "central_markers", "rows", "channels", "maximum_context",
    "padded_tokens", "valid_tokens", "context_lengths", "synthetic_target_class_counts",
    "transition_count", "transition_count_per_person_haplotype", "boundary_weight_sum",
    "first_marker_weight_max", "boundary_weight_beta", "memory_limit_gib",
    "device", "vram_applicable",
    "torch_version", "oci_image", "implementation_commit", "source_auth_sha256",
    "contract_sha256", "preflight_receipt_sha256", "synthetic_only",
) + FALSE_FIELDS
EXACT_SUBCASE_FIELDS = (
    "name", "loss", "probability_sha256", "delta_sha256", "gradient_sha256",
    "stage_gradient_norms", "global_gradient_norm", "valid_input_gradient_norm",
    "padding_input_gradient_max_abs", "parameter_value_sha256_before",
    "parameter_value_sha256_after", "parameter_count", "parameter_shape_sha256",
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


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_source(path: Path, source_root: Path, commit: str) -> str:
    preflight.validate_source_auth(path, commit, source_root)
    payload = preflight.load_json(path)
    hashes = payload.get("source_sha256", {})
    require(payload.get("stage") == "M33_T1_SOURCE_AUTH" and
            payload.get("status") == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES" and
            payload.get("git_commit") == commit and
            hashes.get("bin/m33_t1_compare.py") ==
            sha256_file(source_root / "bin/m33_t1_compare.py"),
            "T1 comparison source authentication differs")
    return sha256_file(path)


def compare_receipts(receipt_paths: list[Path], source_auth_path: Path,
                     source_root: Path, implementation_commit: str, oci_image: str,
                     contract_path: Path, preflight_path: Path) -> dict[str, Any]:
    contract_sha, contract = preflight.validate_contract(contract_path)
    require(oci_image == contract["execution"]["oci_image"], "T1 comparison OCI differs")
    source_auth_sha = validate_source(source_auth_path, source_root, implementation_commit)
    preflight_receipt = preflight.load_json(preflight_path)
    require(preflight_receipt.get("status") == "PASS_T1_PREFLIGHT_SYNTHETIC_ONLY" and
            preflight_receipt.get("contract_sha256") == contract_sha and
            preflight_receipt.get("source_auth_sha256") == source_auth_sha and
            preflight_receipt.get("implementation_commit") == implementation_commit and
            preflight_receipt.get("oci_image") == oci_image,
            "T1 comparison preflight differs")
    require(len(receipt_paths) == 4, "T1 requires exactly four child receipts")
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    children = []
    for path in receipt_paths:
        receipt = preflight.load_json(path)
        key = (receipt.get("model_family"), receipt.get("repetition"))
        require(receipt.get("stage") == "M33_T1_BACKWARD_DRY_RUN" and
                receipt.get("status") == "PASS_T1_BACKWARD_CASE_TECHNICAL_ONLY_NON_CONSUMABLE" and
                key in EXPECTED_CASES and key not in observed,
                f"T1 child inventory differs: {key}")
        require(receipt.get("people") == 8 and receipt.get("central_markers") == 256 and
                receipt.get("rows") == 2048 and receipt.get("channels") == 13 and
                receipt.get("maximum_context") == 128 and
                receipt.get("padded_tokens") == 262_144 and
                receipt.get("valid_tokens") == 98_816 and
                receipt.get("context_lengths") == [0, 1, 64, 128] and
                receipt.get("synthetic_target_class_counts") == [1366, 1365, 1365] and
                receipt.get("transition_count") == 112 and
                receipt.get("transition_count_per_person_haplotype") == 7 and
                receipt.get("boundary_weight_sum") == 4208.0 and
                receipt.get("first_marker_weight_max") == 1.0 and
                receipt.get("boundary_weight_beta") == 1.0 and
                receipt.get("synthetic_only") is True,
                "T1 child stress geometry differs")
        require(receipt.get("implementation_commit") == implementation_commit and
                receipt.get("oci_image") == oci_image and
                receipt.get("source_auth_sha256") == source_auth_sha and
                receipt.get("contract_sha256") == contract_sha and
                receipt.get("preflight_receipt_sha256") == sha256_file(preflight_path),
                "T1 child execution identity differs")
        require(all(receipt.get(field) is False for field in FALSE_FIELDS),
                "T1 child firewall differs")
        require(receipt.get("memory_stop_fraction") == 0.80 and
                receipt.get("memory_warning_fraction") == 0.70 and
                type(receipt.get("memory_limit_gib")) in (int, float) and
                math.isclose(receipt["memory_limit_gib"], 8.0,
                             rel_tol=0.0, abs_tol=0.05) and
                receipt.get("device") == "cpu" and receipt.get("vram_applicable") is False and
                type(receipt.get("peak_rss_fraction")) in (int, float) and
                math.isfinite(receipt["peak_rss_fraction"]) and
                0 <= receipt["peak_rss_fraction"] < 0.80 and
                receipt.get("memory_warning") is (receipt["peak_rss_fraction"] >= 0.70),
                "T1 child memory gate differs")
        subcases = receipt.get("subcases", [])
        require([row.get("name") for row in subcases] ==
                ["production_zero_head", "private_nonzero_probe_head"],
                "T1 child subcase inventory differs")
        for row in subcases:
            before = row.get("parameter_value_sha256_before")
            after = row.get("parameter_value_sha256_after")
            expected_parameter_count = 90 if key[0] == "local_linear" else 1651
            require(type(row.get("loss")) in (int, float) and math.isfinite(row["loss"]) and
                    row["loss"] > 0 and is_sha256(row.get("probability_sha256")) and
                    is_sha256(row.get("delta_sha256")) and
                    is_sha256(row.get("gradient_sha256")) and
                    is_sha256(before) and is_sha256(after) and before == after and
                    row.get("parameter_count") == expected_parameter_count and
                    is_sha256(row.get("parameter_shape_sha256")) and
                    row.get("global_gradient_norm", 0) > 0 and
                    row.get("padding_input_gradient_max_abs") == 0.0,
                    "T1 child gradient or parameter gate differs")
            require(type(row.get("peak_rss_fraction_after")) in (int, float) and
                    0 <= row["peak_rss_fraction_after"] < 0.80,
                    "T1 child subcase memory gate differs")
            if row["name"] == "private_nonzero_probe_head":
                require(row.get("valid_input_gradient_norm", 0) > 0 and
                        all(value > 0 for value in row.get("stage_gradient_norms", {}).values()),
                        "T1 private probe gradient flow differs")
        production, private = subcases
        expected_stages = ({"head"} if key[0] == "local_linear" else
                           {"stem", "block1", "block2", "head1", "head2"})
        require(set(production.get("stage_gradient_norms", {})) == expected_stages and
                set(private.get("stage_gradient_norms", {})) == expected_stages,
                "T1 child gradient stage inventory differs")
        if key[0] == "local_linear":
            require(production["stage_gradient_norms"]["head"] > 0,
                    "T1 production linear head gradient differs")
        else:
            require(production["stage_gradient_norms"]["head2"] > 0 and
                    all(production["stage_gradient_norms"][name] == 0.0
                        for name in ("stem", "block1", "block2", "head1")),
                    "T1 production CNN zero-head gradient pattern differs")
        observed[key] = receipt
        children.append({"name": path.name, "sha256": sha256_file(path)})
    require(set(observed) == EXPECTED_CASES, "T1 four-case inventory differs")
    for family in ("local_linear", "small_residual_cnn_1d"):
        first, second = observed[(family, 0)], observed[(family, 1)]
        missing = [field for field in EXACT_REPEAT_FIELDS if field not in first or field not in second]
        require(not missing, f"T1 repeat field missing: {missing}")
        mismatches = [field for field in EXACT_REPEAT_FIELDS if first[field] != second[field]]
        require(not mismatches, f"T1 {family} determinism differs: {mismatches}")
        for index, (first_subcase, second_subcase) in enumerate(
                zip(first["subcases"], second["subcases"])):
            subcase_missing = [field for field in EXACT_SUBCASE_FIELDS
                               if field not in first_subcase or field not in second_subcase]
            require(not subcase_missing,
                    f"T1 repeat subcase field missing: {subcase_missing}")
            subcase_mismatches = [field for field in EXACT_SUBCASE_FIELDS
                                  if first_subcase[field] != second_subcase[field]]
            require(not subcase_mismatches,
                    f"T1 {family} subcase {index} determinism differs: {subcase_mismatches}")
    return {
        "stage": "M33_T1_BACKWARD_DRY_RUN_COMPARISON",
        "status": "PASS_T1_BACKWARD_DRY_RUN_TECHNICAL_ONLY",
        "case_inventory": [list(row) for row in sorted(EXPECTED_CASES)],
        "child_receipts": sorted(children, key=lambda row: row["name"]),
        "exact_repetition_by_model_family": True,
        "synthetic_only": True,
        "scientific_evidence": False,
        "truth_read": False,
        "real_data_read": False,
        "training": False,
        "optimizer_created": False,
        "optimizer_step": False,
        "checkpoint_written": False,
        "predictions_persisted": False,
        "tensors_persisted": False,
        "consumable": False,
        "development_open": False,
        "implementation_commit": implementation_commit,
        "oci_image": oci_image,
        "source_auth_sha256": source_auth_sha,
        "contract_sha256": contract_sha,
        "preflight_receipt_sha256": sha256_file(preflight_path),
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
