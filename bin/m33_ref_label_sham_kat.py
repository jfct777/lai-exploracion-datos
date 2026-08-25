#!/usr/bin/env python3
"""Run the synthetic, receipt-only M33 REF-label sham known-answer test."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import m33_safe_bridge_core as core
from m33_ref_label_sham_source_auth import sha256_file, validate_source_auth


STATUS = "PASS_REF_LABEL_SHAM_KAT_SYNTHETIC_ONLY_NON_CONSUMABLE"
OCI = ("us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/"
       "m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99")
PRE4_SHA256 = "4308bbf33ae28f554f701da33efdc185264f9f407d62661e7048e0345687eb8b"
M0_SHA256 = "fb74cd610a36b22fe54b8681238a13b48a0243642c9a24cd036597d161361614"
CONTRACT_SHA256 = "81d3854a5c6aa58490e84c5cc9c6c2059f939c23214562e0d56936910f781ac6"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_contract(path: Path) -> dict[str, Any]:
    require(sha256_file(path) == CONTRACT_SHA256, "REF-label sham contract bytes differ")
    contract = json.loads(path.read_text(encoding="utf-8"))
    require(contract.get("stage") == "M33_REF_LABEL_SHAM_KAT_CONTRACT" and
            contract.get("status") == "FROZEN_SYNTHETIC_ONLY_BEFORE_EXECUTION" and
            contract.get("preregistered_seeds") == list(core.REF_LABEL_SHAM_SEEDS) and
            contract.get("scope", {}).get("synthetic_only") is True and
            contract.get("scope", {}).get("persisted_outputs") == "receipt_only" and
            contract.get("execution", {}).get("oci_image") == OCI and
            contract.get("effective_reference_summary_schema", {}).get("dtypes", {}).get(
                "ancestry") == "|S4" and
            contract.get("upstream_contracts") == {
                "conf/m33_pre4_preregistration.json": PRE4_SHA256,
                "conf/m33_m0_materializer_contract.json": M0_SHA256,
            },
            "REF-label sham frozen contract differs")
    return contract


def synthetic_fixture() -> dict[str, Any]:
    node_ids = list(range(12))
    people = [f"P{index}" for index in range(6) for _ in range(2)]
    person_labels = ["AFR", "AFR", "EUR", "EUR", "ASIA", "ASIA"]
    ancestries = [label for label in person_labels for _ in range(2)]
    raw_states = np.asarray([
        [0, 0, -1, 1], [1, 0, -1, 1],
        [0, 1, -1, 0], [0, 1, -1, 0],
        [1, 1, 0, -1], [1, 0, 0, -1],
        [1, 1, 1, 0], [1, -1, 1, 0],
        [0, 0, 0, 1], [0, 0, 1, 1],
        [0, 1, 1, 1], [-1, 1, 1, 0],
    ], dtype="|i1")
    minor_codes = np.asarray([0, 1, 0, 1], dtype="|i1")
    expected = [
        {"node_id": node, "person_id": person, "ancestry": ancestry}
        for node, person, ancestry in zip(node_ids, people, ancestries)
    ]
    target_sentinel = np.asarray([[0, 1, 2, 0], [2, 0, 1, 1]], dtype="|i1")
    return {
        "node_ids": node_ids, "people": people, "ancestries": ancestries,
        "raw_states": raw_states, "minor_codes": minor_codes,
        "expected": expected, "target_sentinel": target_sentinel,
        "locus_id": np.asarray([101, 102, 103, 104], dtype="<u8"),
    }


def independent_summary(raw_states: np.ndarray, minor_codes: np.ndarray,
                        node_labels: Sequence[str]) -> dict[str, np.ndarray]:
    """Independent loop oracle; deliberately does not call summarize_reference()."""
    ac = np.zeros((3, raw_states.shape[1]), dtype="<u2")
    an = np.zeros_like(ac)
    for ancestry_index, ancestry in enumerate(core.ANCESTRIES):
        for node_index, label in enumerate(node_labels):
            if label != ancestry:
                continue
            for locus_index, code in enumerate(minor_codes):
                state = int(raw_states[node_index, locus_index])
                if state >= 0:
                    an[ancestry_index, locus_index] += 1
                    ac[ancestry_index, locus_index] += int(state == int(code))
    af = np.divide(ac, an, out=np.zeros_like(ac, dtype="<f8"), where=an > 0)
    return {
        "minor_ac": ac, "callable_an": an, "minor_af": af,
        "observed_mask": (an > 0).astype("|u1"),
        "no_support": ((an > 0) & (ac == 0)).astype("|u1"),
    }


def summary_artifact(summary: Mapping[str, np.ndarray], locus_id: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ancestry": np.asarray(core.ANCESTRIES, dtype="|S4"),
        "locus_id": np.ascontiguousarray(locus_id),
        **{name: np.ascontiguousarray(value) for name, value in summary.items()},
    }


def run(contract_path: Path, pre4_path: Path, m0_path: Path,
        source_auth: Path, source_root: Path,
        implementation_commit: str, oci_image: str) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", implementation_commit or "") is not None,
            "REF-label sham implementation commit differs")
    contract = load_contract(contract_path)
    require(sha256_file(pre4_path) == PRE4_SHA256 and sha256_file(m0_path) == M0_SHA256,
            "upstream PRE4 or M0 contract differs")
    require(oci_image == contract["execution"]["oci_image"], "REF-label sham OCI differs")
    source_auth_sha = validate_source_auth(source_auth, implementation_commit, source_root)
    fixture = synthetic_fixture()
    raw_before = core.semantic_arrays_sha256(
        "m33_ref_label_sham_fixture_inputs_v1",
        {"raw_states": fixture["raw_states"], "minor_codes": fixture["minor_codes"],
         "target_sentinel": fixture["target_sentinel"], "locus_id": fixture["locus_id"]},
    )
    original = core.summarize_reference(
        fixture["raw_states"], fixture["minor_codes"], fixture["node_ids"],
        fixture["people"], fixture["ancestries"], fixture["expected"],
    )
    first, diagnostics = core.summarize_reference_label_shams(
        fixture["raw_states"], fixture["minor_codes"], fixture["node_ids"],
        fixture["people"], fixture["ancestries"], fixture["expected"],
    )
    second, repeated_diagnostics = core.summarize_reference_label_shams(
        fixture["raw_states"], fixture["minor_codes"], fixture["node_ids"],
        fixture["people"], fixture["ancestries"], fixture["expected"],
    )
    require(diagnostics == repeated_diagnostics, "REF-label sham diagnostics are not deterministic")

    original_hash = core.semantic_arrays_sha256(
        "m33_ref_label_summary_effective_s4_v1",
        summary_artifact(original, fixture["locus_id"]),
    )
    summary_hashes: dict[str, str] = {}
    for seed in core.REF_LABEL_SHAM_SEEDS:
        labels, _diagnostic = core.permute_diploid_reference_labels(
            fixture["node_ids"], fixture["people"], fixture["ancestries"], seed,
        )
        oracle = independent_summary(fixture["raw_states"], fixture["minor_codes"], labels)
        require(all(np.array_equal(first[seed][name], oracle[name]) for name in oracle),
                "REF-label sham differs from independent AC/AN/AF oracle")
        artifact = summary_artifact(first[seed], fixture["locus_id"])
        repeated = summary_artifact(second[seed], fixture["locus_id"])
        observed_hash = core.semantic_arrays_sha256(
            "m33_ref_label_summary_effective_s4_v1", artifact,
        )
        require(observed_hash == core.semantic_arrays_sha256(
                    "m33_ref_label_summary_effective_s4_v1", repeated) and
                observed_hash != original_hash,
                "REF-label sham summary is nondeterministic or identical to the real fixture")
        summary_hashes[str(seed)] = observed_hash
    require(len(set(summary_hashes.values())) == 3,
            "REF-label sham summary artifacts are duplicated")
    raw_after = core.semantic_arrays_sha256(
        "m33_ref_label_sham_fixture_inputs_v1",
        {"raw_states": fixture["raw_states"], "minor_codes": fixture["minor_codes"],
         "target_sentinel": fixture["target_sentinel"], "locus_id": fixture["locus_id"]},
    )
    require(raw_after == raw_before, "REF-label sham changed fixture inputs or TARGET")
    require(Counter(diagnostic["seed"] for diagnostic in diagnostics) ==
            Counter(core.REF_LABEL_SHAM_SEEDS), "REF-label sham seed inventory differs")
    return {
        "stage": "M33_REF_LABEL_SHAM_KAT",
        "status": STATUS,
        "implementation_commit": implementation_commit,
        "oci_image": oci_image,
        "contract_sha256": sha256_file(contract_path),
        "pre4_preregistration_sha256": PRE4_SHA256,
        "m0_materializer_contract_sha256": M0_SHA256,
        "source_auth_sha256": source_auth_sha,
        "seeds": list(core.REF_LABEL_SHAM_SEEDS),
        "person_count": 6,
        "node_count": 12,
        "locus_count": 4,
        "minor_codes": [0, 1, 0, 1],
        "assignment_diagnostics": diagnostics,
        "original_summary_semantic_sha256": original_hash,
        "sham_summary_semantic_sha256": summary_hashes,
        "input_and_TARGET_semantic_sha256_before": raw_before,
        "input_and_TARGET_semantic_sha256_after": raw_after,
        "effective_ancestry_dtype": "|S4",
        "synthetic_only": True,
        "scientific_evidence": False,
        "consumable": False,
        "truth_read": False,
        "real_asset_read": False,
        "training": False,
        "individual_reference_exported": False,
        "summary_arrays_persisted": False,
        "permutation_p_value": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--pre4-preregistration", type=Path, required=True)
    parser.add_argument("--m0-materializer-contract", type=Path, required=True)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--oci-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.contract, args.pre4_preregistration, args.m0_materializer_contract,
                  args.source_auth, args.source_root,
                  args.implementation_commit, args.oci_image)
    core.write_exclusive_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
