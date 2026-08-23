#!/usr/bin/env python3
"""Independently verify one non-consumable M33 technical-root SAFE_BRIDGE KAT.

This verifier intentionally shares no implementation with the bridge, A0, M31,
or the synthetic KAT.  It accepts only the four sanitized ``technical_kat_*``
NPZ files and their JSON receipts; it never opens source genotypes or truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


STATUS = "PASS_SAFE_BRIDGE_TECHNICAL_ROOT_KAT_ONLY_NON_CONSUMABLE"
STAGE = "M33_SAFE_BRIDGE_TECHNICAL_ROOT_KAT"
RECEIPT_SCHEMA = "tests_m33_safe_bridge_technical_kat_receipt_v1"
ANCESTRY_ORDER = ["AFR", "EUR", "ASIA"]
ARTIFACTS = {
    "technical_kat_selected_loci_incremental.npz": (
        "tests_m33_safe_bridge_technical_kat_selected_loci_incremental_v1",
        {"locus_key_sha256", "chrom", "pos", "ref", "alt", "cM", "minor_code"},
    ),
    "technical_kat_target_rare_diploid_incremental.npz": (
        "tests_m33_safe_bridge_technical_kat_target_rare_diploid_incremental_v1",
        {"sample_key_sha256", "locus_key_sha256", "minor_dosage", "observed_mask"},
    ),
    "technical_kat_reference_rare_summary_incremental.npz": (
        "tests_m33_safe_bridge_technical_kat_reference_rare_summary_incremental_v1",
        {"ancestry", "locus_key_sha256", "minor_ac", "callable_an", "minor_af",
         "observed_mask", "no_support"},
    ),
    "technical_kat_flare_f0_sanitized.npz": (
        "tests_m33_safe_bridge_technical_kat_flare_f0_sanitized_v1",
        {"sample_key_sha256", "chrom", "pos", "ref", "alt", "probabilities"},
    ),
}
FORBIDDEN_TOKENS = ("sample_id", "person_id", "individual_id", "node_id", "phase",
                    "hap_presence", "genotype", "gt", "an1", "an2", "truth",
                    "mosaic", "materialize", "ready")
EXPECTED_ROOTS = {
    "root17": {
        "root_seed": 20260817,
        "a0_receipt_sha256": "4fe79f0dc648caa2c64b9d81c752e144af6d0c9db43e3fa3ae4274887ecfff93",
        "i0_receipt_sha256": "53425116ccf0e3d9d8df7b4eaab1e248efe425d0fd0d22041bbc330cbcbb3924",
        "a0_registry_sha256": "44311fe8ef9238c81f630343857439ac16e52b7569c1348a52fb65d744ad93cd",
        "flare_sha256": "85dfd76df2c14cb8fe0a753910f25c49c88d38edc5708ec6d641053d95cc74e8",
        "flare_generation": "1787175566795248",
        "tbi_sha256": "936a5d8256a2f1610898e2a3fe00523001c203cb014a7509c9c76911fe626e52",
        "selected_count": 94029,
        "overlap_count": 0,
        "minor_code_zero_count": 526,
        "target_count": 30,
        "target_missing_count": 0,
        "target_legacy_sha256": "ea142d0a87ae4e74a6817b15f4b9dc196467ea6f0c875611aa0813b417e2eff4",
        "ref_count": 90,
        "ref_callable_an": 60,
        "ref_no_support_count": 43892,
        "ref_legacy_sha256": "6f0d91443fd5f187377111e38133e8ab823399ebb1fe095c3b78e4da06723bdd",
        "f0_locus_count": 79791,
        "f0_vector_count": 4787460,
    },
    "root18": {
        "root_seed": 20260818,
        "a0_receipt_sha256": "de1ea4ae2b67179acb9a6f56024bfe9bbb92b00663ef3684b280d077f1726b4b",
        "i0_receipt_sha256": "c9a0a5bdc74ade07cb98cfadbfadf0ed9e274adca0c18cd4250de6fffce66984",
        "a0_registry_sha256": "649993d6e098b3cf92260a95d5bfcf8a89a529f8438f753dce368598958773de",
        "flare_sha256": "edc4bcdc62f5ce0ffe04bd27e9d6d6ee892e03282a1474639fc3082fbc3832c9",
        "flare_generation": "1787175916753131",
        "tbi_sha256": "9f26af28898adbdcc8eec8ebddc10b321464923c8b244719e7c3b7a6d0dfacaf",
        "selected_count": 94703,
        "overlap_count": 0,
        "minor_code_zero_count": 541,
        "target_count": 30,
        "target_missing_count": 0,
        "target_legacy_sha256": "cc068025663bd88e3b2903685fdf76cf164a04c9ac52362b80305c0adc52ee46",
        "ref_count": 90,
        "ref_callable_an": 60,
        "ref_no_support_count": 43938,
        "ref_legacy_sha256": "f2fcf538258114df5410674fe52ef7b6abaf64a35ec4bc251d6a6c25633ccee5",
        "f0_locus_count": 79791,
        "f0_vector_count": 4787460,
    },
}
VERIFIER_SOURCE_FILES = {
    "bin/m33_safe_bridge_technical_verify.py",
    "tests/test_m33_safe_bridge_technical_verify.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"JSON is not a regular file: {path}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=reject)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_verifier_source_auth(path: Path) -> tuple[str, str, str]:
    payload = load_json(path)
    require(payload.get("stage") == "M33_SAFE_BRIDGE_TECHNICAL_KAT_SOURCE_AUTH" and
            payload.get("status") == "AUTHORIZED_EXACT_TECHNICAL_KAT_SOURCES",
            "source-auth identity drifted")
    files = payload.get("independent_verifier_files", {})
    require(set(files) == VERIFIER_SOURCE_FILES, "verifier source-auth inventory drifted")
    repo_root = Path(__file__).resolve().parents[1]
    observed = {relative: sha256_file(repo_root / relative) for relative in sorted(files)}
    require(observed == files, "verifier bytes differ from source-auth")
    implementation_commit = payload.get("implementation_commit")
    require(isinstance(implementation_commit, str) and
            re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is not None,
            "source-auth implementation commit is invalid")
    return sha256_file(path), observed["bin/m33_safe_bridge_technical_verify.py"], implementation_commit


def write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    require(path.parent.is_dir() and not path.parent.is_symlink() and not path.exists(),
            "verification receipt output must be new under a regular directory")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        os.chmod(path, 0o400)
    finally:
        if temporary.exists():
            temporary.unlink()


def array_sha256(value: np.ndarray) -> str:
    """Reproduce A0's legacy dtype/shape/byte semantic digest independently."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def artifact_semantic_sha256(schema: str, arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(schema.encode("utf-8") + b"\0")
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        require(array.dtype.kind != "O", f"object array is forbidden: {name}")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _forbidden_name(name: str) -> bool:
    lowered = name.lower()
    return any(re.search(rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)", lowered)
               for token in FORBIDDEN_TOKENS)


def open_npz(path: Path, expected_name: str, members: set[str]) -> dict[str, np.ndarray]:
    require(path.name == expected_name, f"technical KAT filename drifted: {path.name}")
    require(path.is_file() and not path.is_symlink(), f"artifact is not a regular file: {path}")
    require(not _forbidden_name(path.name), f"forbidden token in artifact filename: {path.name}")
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), f"duplicate ZIP member in {path.name}")
        require(set(names) == {f"{member}.npy" for member in members},
                f"NPZ member inventory drifted: {path.name}")
        require(all("/" not in name and "\\" not in name and not _forbidden_name(name[:-4])
                    for name in names), f"forbidden or nested NPZ member: {path.name}")
    with np.load(path, allow_pickle=False) as loaded:
        require(set(loaded.files) == members, f"NPZ key inventory drifted: {path.name}")
        arrays = {name: np.array(loaded[name], copy=True) for name in sorted(loaded.files)}
    require(all(value.dtype.kind not in "OUV" for value in arrays.values()),
            f"object, Unicode, or void array forbidden: {path.name}")
    return arrays


def _bytes_are_sha256(values: np.ndarray) -> bool:
    if values.dtype != np.dtype("|S64") or values.ndim != 1:
        return False
    return all(re.fullmatch(rb"[0-9a-f]{64}", bytes(value)) is not None for value in values)


def validate_a0(a0: dict[str, Any], expected: dict[str, Any], root: str) -> None:
    require(a0.get("stage") == "M33_A0_REAL_ADAPTER" and
            a0.get("status") == "PASS_TECHNICAL_COMPATIBILITY_ONLY",
            "A0 receipt did not pass technical compatibility")
    require(a0.get("root_label") == root and a0.get("root_seed") == expected["root_seed"],
            "A0 root identity drifted")
    require(a0.get("asset_registry_sha256") == expected["a0_registry_sha256"],
            "A0 registry anchor drifted")
    require(a0.get("scientific_evidence") is False and a0.get("ready_emitted") is False and
            a0.get("checks", {}).get("truth_not_read") is True,
            "A0 scope firewall drifted")
    counts = a0.get("counts", {})
    checks = {
        "selected_rare_sites": expected["selected_count"],
        "incremental_rare_sites": expected["selected_count"],
        "rare_overlap_flare_sites": expected["overlap_count"],
        "minor_code_zero_sites": expected["minor_code_zero_count"],
        "target_people": expected["target_count"],
        "target_missing_diploid_cells": expected["target_missing_count"],
        "target_diploid_dosage_sha256": expected["target_legacy_sha256"],
        "ref_people": expected["ref_count"],
        "ref_callable_AN_per_ancestry": expected["ref_callable_an"],
        "ref_no_support_sites": expected["ref_no_support_count"],
        "ref_ac_an_sha256": expected["ref_legacy_sha256"],
        "flare_loci": expected["f0_locus_count"],
        "flare_probability_vectors": expected["f0_vector_count"],
    }
    require(all(counts.get(key) == value for key, value in checks.items()),
            "A0 known-answer counts or hashes drifted")
    require(counts.get("phase_exported_to_M0") is False, "A0 exported forbidden phase")
    require(a0.get("input_sha256", {}).get("flare_anc") == expected["flare_sha256"],
            "A0 FLARE source hash drifted")


def validate_i0(i0: dict[str, Any], expected: dict[str, Any], root: str) -> None:
    require(i0.get("stage") == "M33_I0_REAL_INDEX" and
            i0.get("status") == "PASS_DOUBLE_INDEX_AND_QUERY_PARITY_TECHNICAL_ONLY",
            "I0 receipt did not pass technical indexing")
    require(i0.get("root_label") == root and i0.get("root_seed") == expected["root_seed"],
            "I0 root identity drifted")
    require(i0.get("source_flare_sha256") == expected["flare_sha256"] and
            i0.get("source_generation") == expected["flare_generation"] and
            i0.get("output_tbi_sha256") == expected["tbi_sha256"] and
            i0.get("independent_tbi_sha256") == expected["tbi_sha256"],
            "I0 source generation or index hash drifted")
    require(i0.get("indexed_record_count") == expected["f0_locus_count"] and
            i0.get("sequential_record_count") == expected["f0_locus_count"],
            "I0 record counts drifted")
    require(i0.get("scientific_evidence") is False and i0.get("safe_bridge") is False and
            i0.get("materialize") is False and i0.get("global_ready") is False and
            i0.get("training") is False and i0.get("truth") is False,
            "I0 scope firewall drifted")


def validate_receipt(receipt: dict[str, Any], expected: dict[str, Any], root: str,
                     paths: dict[str, Path], a0_hash: str, i0_hash: str) -> None:
    require(receipt.get("stage") == STAGE and receipt.get("status") == STATUS and
            receipt.get("schema_id") == RECEIPT_SCHEMA, "bridge receipt identity drifted")
    require(receipt.get("root_label") == root and receipt.get("root_seed") == expected["root_seed"],
            "bridge root identity drifted")
    for field in ("scientific_evidence", "consumable", "truth_read", "materialize_authorized",
                  "ready_emitted", "training_authorized", "gcs_write"):
        require(receipt.get(field) is False, f"bridge scope firewall drifted: {field}")
    require(receipt.get("append_only") is True and receipt.get("reopen_verified") is True and
            receipt.get("write_chmod_rename_probes_failed") is True,
            "bridge append-only, reopening, or read-only proof absent")
    require(receipt.get("phase_swap_invariant") is True and
            receipt.get("network_disabled") is True and
            receipt.get("credential_environment_absent") is True and
            receipt.get("f0_anp_only_projection") is True and
            receipt.get("f0_gt_an1_an2_ignored") is True and
            receipt.get("raw_identifiers_exported") is False,
            "bridge phase or isolation proof absent")
    require(type(receipt.get("runner_uid")) is int and type(receipt.get("runner_euid")) is int and
            receipt["runner_uid"] == 65534 and receipt["runner_euid"] == 65534,
            "bridge execution identity is invalid")
    require(isinstance(receipt.get("input_sha256_pre"), dict) and
            receipt.get("input_sha256_pre") == receipt.get("input_sha256_post") and
            receipt["input_sha256_pre"], "bridge inputs changed during execution")
    require(a0_hash == expected["a0_receipt_sha256"] and
            i0_hash == expected["i0_receipt_sha256"],
            "independent A0/I0 receipt anchor drifted")
    require(receipt.get("artifact_schema") ==
            {name: schema for name, (schema, _members) in ARTIFACTS.items()},
            "artifact tests_* schemas drifted")
    require(set(receipt.get("artifact_raw_sha256", {})) == set(ARTIFACTS) and
            set(receipt.get("artifact_semantic_sha256", {})) == set(ARTIFACTS),
            "bridge artifact hash inventory drifted")
    for name, path in paths.items():
        require(receipt["artifact_raw_sha256"][name] == sha256_file(path),
                f"bridge artifact raw hash mismatch: {name}")
    count_checks = {
        "selected_all_count": expected["selected_count"],
        "selected_incremental_count": expected["selected_count"],
        "selected_overlap_count": expected["overlap_count"],
        "incremental_minor_code_0_locus_count": expected["minor_code_zero_count"],
        "target_count": expected["target_count"],
        "target_missing_cells": expected["target_missing_count"],
        "ref_people": expected["ref_count"],
        "reference_no_support_loci": expected["ref_no_support_count"],
        "flare_marker_count": expected["f0_locus_count"],
        "f0_probability_vectors": expected["f0_vector_count"],
    }
    require(all(receipt.get(key) == value for key, value in count_checks.items()),
            "bridge receipt known-answer counts drifted")
    require(receipt.get("ref_people_by_ancestry") == {name: 30 for name in ANCESTRY_ORDER},
            "bridge REF ancestry counts drifted")
    require(receipt.get("target_diploid_dosage_legacy_sha256") == expected["target_legacy_sha256"] and
            receipt.get("reference_ac_an_legacy_sha256") == expected["ref_legacy_sha256"],
            "bridge legacy semantic hashes drifted")


def verify(root: str, selected_path: Path, target_path: Path, reference_path: Path,
           f0_path: Path, receipt_path: Path, a0_path: Path, i0_path: Path,
           source_auth_path: Path) -> dict[str, Any]:
    require(root in EXPECTED_ROOTS, "root must be root17 or root18")
    expected = EXPECTED_ROOTS[root]
    a0_hash, i0_hash = sha256_file(a0_path), sha256_file(i0_path)
    require(a0_hash == expected["a0_receipt_sha256"], "A0 receipt raw hash drifted")
    require(i0_hash == expected["i0_receipt_sha256"], "I0 receipt raw hash drifted")
    a0, i0, receipt = load_json(a0_path), load_json(i0_path), load_json(receipt_path)
    source_auth_sha256, verifier_source_sha256, implementation_commit = (
        validate_verifier_source_auth(source_auth_path)
    )
    require(receipt.get("source_auth_sha256") == source_auth_sha256 and
            receipt.get("git_commit") == implementation_commit,
            "bridge receipt is not bound to the verifier source-auth")
    validate_a0(a0, expected, root)
    validate_i0(i0, expected, root)
    paths = {
        selected_path.name: selected_path, target_path.name: target_path,
        reference_path.name: reference_path, f0_path.name: f0_path,
    }
    require(set(paths) == set(ARTIFACTS), "technical KAT artifact filenames drifted")
    validate_receipt(receipt, expected, root, paths, a0_hash, i0_hash)

    opened: dict[str, dict[str, np.ndarray]] = {}
    for name, (_schema, members) in ARTIFACTS.items():
        opened[name] = open_npz(paths[name], name, members)
    selected = opened["technical_kat_selected_loci_incremental.npz"]
    target = opened["technical_kat_target_rare_diploid_incremental.npz"]
    reference = opened["technical_kat_reference_rare_summary_incremental.npz"]
    f0 = opened["technical_kat_flare_f0_sanitized.npz"]

    loci = expected["selected_count"]
    require(_bytes_are_sha256(selected["locus_key_sha256"]) and
            selected["locus_key_sha256"].shape == (loci,),
            "selected pseudonymous locus-key axis drifted")
    require(selected["chrom"].dtype == np.dtype("|u1") and np.all(selected["chrom"] == 22),
            "selected chromosome drifted")
    require(selected["pos"].dtype == np.dtype("<i8") and selected["pos"].shape == (loci,) and
            np.all(selected["pos"] > 0) and len(np.unique(selected["pos"])) == loci,
            "selected position axis drifted")
    require(selected["ref"].dtype == np.dtype("|S1") and
            selected["alt"].dtype == np.dtype("|S1") and
            np.all(selected["ref"] == b"A") and np.all(selected["alt"] == b"C"),
            "selected REF/ALT code mapping drifted")
    require(selected["cM"].dtype == np.dtype("<f8") and selected["cM"].shape == (loci,) and
            np.all(np.isfinite(selected["cM"])) and np.all(np.diff(selected["cM"]) >= 0),
            "selected genetic-coordinate axis drifted")
    require(len(np.unique(selected["locus_key_sha256"])) == loci,
            "selected locus key is duplicated")
    require(selected["minor_code"].dtype == np.dtype("|i1") and
            selected["minor_code"].shape == (loci,) and
            np.all(np.isin(selected["minor_code"], [0, 1])) and
            int((selected["minor_code"] == 0).sum()) == expected["minor_code_zero_count"],
            "selected minor-code known answer drifted")

    samples = expected["target_count"]
    require(_bytes_are_sha256(target["sample_key_sha256"]) and
            target["sample_key_sha256"].shape == (samples,) and
            len(np.unique(target["sample_key_sha256"])) == samples,
            "TARGET pseudonymous sample axis drifted")
    require(_bytes_are_sha256(target["locus_key_sha256"]) and
            np.array_equal(target["locus_key_sha256"], selected["locus_key_sha256"]),
            "TARGET locus axis drifted")
    require(target["minor_dosage"].dtype == np.dtype("|i1") and
            target["minor_dosage"].shape == (samples, loci) and
            target["observed_mask"].dtype == np.dtype("|u1") and
            target["observed_mask"].shape == (samples, loci), "TARGET arrays drifted")
    require(np.all(np.isin(target["minor_dosage"], [0, 1, 2])) and
            np.all(np.isin(target["observed_mask"], [0, 1])), "TARGET state domain drifted")
    require(int((target["observed_mask"] == 0).sum()) == expected["target_missing_count"],
            "TARGET missing-cell count drifted")
    target_legacy = target["minor_dosage"].T.astype(np.int8, copy=False)
    require(array_sha256(target_legacy) == expected["target_legacy_sha256"],
            "TARGET A0 legacy array semantic hash mismatch")

    require(reference["ancestry"].dtype == np.dtype("|S4") and
            np.array_equal(reference["ancestry"], np.asarray([b"AFR", b"EUR", b"ASIA"], dtype="|S4")),
            "REF ancestry axis drifted")
    require(_bytes_are_sha256(reference["locus_key_sha256"]) and
            np.array_equal(reference["locus_key_sha256"], selected["locus_key_sha256"]),
            "REF locus axis drifted")
    shape = (3, loci)
    require(reference["minor_ac"].dtype == np.dtype("<u2") and
            reference["callable_an"].dtype == np.dtype("<u2") and
            reference["minor_af"].dtype == np.dtype("<f8") and
            reference["observed_mask"].dtype == np.dtype("|u1") and
            reference["no_support"].dtype == np.dtype("|u1") and
            all(reference[name].shape == shape for name in
                ("minor_ac", "callable_an", "minor_af", "observed_mask", "no_support")),
            "REF summary dtype or dimensions drifted")
    require(np.all(reference["minor_ac"] <= reference["callable_an"]) and
            np.all(reference["callable_an"] == expected["ref_callable_an"]),
            "REF AC/AN state drifted")
    require(np.allclose(reference["minor_af"],
                        reference["minor_ac"] / reference["callable_an"], atol=0, rtol=0),
            "REF minor AF does not equal AC/AN")
    require(np.array_equal(reference["observed_mask"], (reference["callable_an"] > 0).astype(np.uint8)) and
            np.array_equal(reference["no_support"],
                           ((reference["callable_an"] > 0) & (reference["minor_ac"] == 0)).astype(np.uint8)),
            "REF explicit masks drifted")
    require(int(np.all(reference["minor_ac"] == 0, axis=0).sum()) ==
            expected["ref_no_support_count"],
            "REF no-support count drifted")
    legacy_ref = np.column_stack((reference["minor_ac"].T.astype(np.int16),
                                  reference["callable_an"].T.astype(np.int16)))
    require(array_sha256(legacy_ref) == expected["ref_legacy_sha256"],
            "REF A0 legacy AC/AN semantic hash mismatch")

    markers = expected["f0_locus_count"]
    require(_bytes_are_sha256(f0["sample_key_sha256"]) and
            np.array_equal(f0["sample_key_sha256"], target["sample_key_sha256"]),
            "F0 pseudonymous sample axis drifted")
    require(f0["chrom"].dtype == np.dtype("|u1") and f0["chrom"].shape == (markers,) and
            np.all(f0["chrom"] == 22) and f0["pos"].dtype == np.dtype("<i8") and
            f0["pos"].shape == (markers,) and np.all(np.diff(f0["pos"]) > 0),
            "F0 marker identity axis drifted")
    require(f0["ref"].dtype == np.dtype("|S1") and f0["alt"].dtype == np.dtype("|S1") and
            np.all(f0["ref"] == b"A") and np.all(f0["alt"] == b"C"),
            "F0 REF/ALT code mapping drifted")
    require(f0["probabilities"].dtype == np.dtype("<f4") and
            f0["probabilities"].shape == (samples, 2, markers, 3),
            "F0 probability dimensions drifted")
    require(f0["probabilities"].size // 3 == expected["f0_vector_count"] and
            np.all(np.isfinite(f0["probabilities"])) and np.all(f0["probabilities"] >= 0) and
            np.allclose(f0["probabilities"].sum(axis=3), 1.0, atol=5e-6, rtol=0),
            "F0 probabilities or vector count drifted")

    for name, arrays in opened.items():
        schema = ARTIFACTS[name][0]
        require(receipt["artifact_semantic_sha256"][name] ==
                artifact_semantic_sha256(schema, arrays), f"artifact semantic hash mismatch: {name}")

    return {
        "stage": "M33_SAFE_BRIDGE_TECHNICAL_ROOT_KAT_INDEPENDENT_VERIFY",
        "schema_id": "tests_m33_safe_bridge_technical_kat_independent_verify_receipt_v1",
        "status": STATUS,
        "root_label": root,
        "root_seed": expected["root_seed"],
        "scientific_evidence": False,
        "consumable": False,
        "truth_read": False,
        "materialize": False,
        "ready_emitted": False,
        "training": False,
        "artifacts_reopened": 4,
        "raw_hashes_verified": 4,
        "legacy_known_answers_verified": 2,
        "a0_receipt_sha256": a0_hash,
        "i0_receipt_sha256": i0_hash,
        "bridge_receipt_sha256": sha256_file(receipt_path),
        "source_auth_sha256": source_auth_sha256,
        "verifier_source_sha256": verifier_source_sha256,
        "implementation_commit": implementation_commit,
        "artifact_raw_sha256": {name: sha256_file(path) for name, path in sorted(paths.items())},
        "artifact_semantic_sha256": {
            name: artifact_semantic_sha256(ARTIFACTS[name][0], opened[name])
            for name in sorted(opened)
        },
        "append_only": True,
        "reopen_verified": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, choices=sorted(EXPECTED_ROOTS))
    parser.add_argument("--selected-loci", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--f0", required=True, type=Path)
    parser.add_argument("--bridge-receipt", required=True, type=Path)
    parser.add_argument("--a0-receipt", required=True, type=Path)
    parser.add_argument("--i0-receipt", required=True, type=Path)
    parser.add_argument("--source-auth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify(args.root, args.selected_loci, args.target, args.reference, args.f0,
                    args.bridge_receipt, args.a0_receipt, args.i0_receipt, args.source_auth)
    write_exclusive_json(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
