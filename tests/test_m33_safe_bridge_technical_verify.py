#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m33_safe_bridge_technical_verify", ROOT / "bin/m33_safe_bridge_technical_verify.py")
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


def key(value: str) -> bytes:
    return hashlib.sha256(value.encode()).hexdigest().encode()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class TechnicalVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = copy.deepcopy(VERIFY.EXPECTED_ROOTS)

    def tearDown(self) -> None:
        VERIFY.EXPECTED_ROOTS.clear()
        VERIFY.EXPECTED_ROOTS.update(self.previous)

    def test_verifier_has_no_bridge_a0_or_m31_import(self) -> None:
        source = (ROOT / "bin/m33_safe_bridge_technical_verify.py").read_text(encoding="utf-8")
        for forbidden in ("import m33_safe_bridge_core", "import m33_safe_bridge_kat",
                          "import m33_a0_real_adapter", "import m31_"):
            self.assertNotIn(forbidden, source)

    def fixture(self, root: Path) -> dict[str, Path]:
        root.mkdir(parents=True, exist_ok=False)
        locus_keys = np.asarray([key("locus-a"), key("locus-b"), key("locus-c")], dtype="|S64")
        sample_keys = np.asarray([key("sample-a"), key("sample-b")], dtype="|S64")
        selected = {
            "locus_key_sha256": locus_keys,
            "chrom": np.asarray([22, 22, 22], dtype="|u1"),
            "pos": np.asarray([100, 200, 300], dtype="<i8"),
            "ref": np.asarray([b"A", b"A", b"A"], dtype="|S1"),
            "alt": np.asarray([b"C", b"C", b"C"], dtype="|S1"),
            "cM": np.asarray([0.1, 0.2, 0.3], dtype="<f8"),
            "minor_code": np.asarray([0, 1, 0], dtype="|i1"),
        }
        target = {
            "sample_key_sha256": sample_keys,
            "locus_key_sha256": locus_keys,
            "minor_dosage": np.asarray([[2, 1, 0], [1, 0, 2]], dtype="|i1"),
            "observed_mask": np.ones((2, 3), dtype="|u1"),
        }
        minor_ac = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype="<u2")
        callable_an = np.full((3, 3), 4, dtype="<u2")
        reference = {
            "ancestry": np.asarray([b"AFR", b"EUR", b"ASIA"], dtype="|S4"),
            "locus_key_sha256": locus_keys,
            "minor_ac": minor_ac,
            "callable_an": callable_an,
            "minor_af": minor_ac.astype("<f8") / callable_an,
            "observed_mask": np.ones((3, 3), dtype="|u1"),
            "no_support": (minor_ac == 0).astype("|u1"),
        }
        f0 = {
            "sample_key_sha256": sample_keys,
            "chrom": np.asarray([22, 22], dtype="|u1"),
            "pos": np.asarray([110, 210], dtype="<i8"),
            "ref": np.asarray([b"A", b"A"], dtype="|S1"),
            "alt": np.asarray([b"C", b"C"], dtype="|S1"),
            "probabilities": np.full((2, 2, 2, 3), 1 / 3, dtype="<f4"),
        }
        payloads = {
            "technical_kat_selected_loci_incremental.npz": selected,
            "technical_kat_target_rare_diploid_incremental.npz": target,
            "technical_kat_reference_rare_summary_incremental.npz": reference,
            "technical_kat_flare_f0_sanitized.npz": f0,
        }
        paths: dict[str, Path] = {}
        for name, arrays in payloads.items():
            path = root / name
            np.savez(path, **arrays)
            paths[name] = path

        expected = {
            "root_seed": 17,
            "a0_registry_sha256": "a" * 64,
            "flare_sha256": "b" * 64,
            "flare_generation": "123",
            "tbi_sha256": "c" * 64,
            "selected_count": 3,
            "overlap_count": 0,
            "minor_code_zero_count": 2,
            "target_count": 2,
            "target_missing_count": 0,
            "target_legacy_sha256": VERIFY.array_sha256(target["minor_dosage"].T),
            "ref_count": 6,
            "ref_callable_an": 4,
            "ref_no_support_count": 1,
            "ref_legacy_sha256": VERIFY.array_sha256(np.column_stack(
                (minor_ac.T.astype(np.int16), callable_an.T.astype(np.int16)))),
            "f0_locus_count": 2,
            "f0_vector_count": 8,
        }
        a0 = {
            "stage": "M33_A0_REAL_ADAPTER",
            "status": "PASS_TECHNICAL_COMPATIBILITY_ONLY",
            "root_label": "root17",
            "root_seed": 17,
            "asset_registry_sha256": "a" * 64,
            "scientific_evidence": False,
            "ready_emitted": False,
            "checks": {"truth_not_read": True},
            "input_sha256": {"flare_anc": "b" * 64},
            "counts": {
                "selected_rare_sites": 3,
                "incremental_rare_sites": 3,
                "rare_overlap_flare_sites": 0,
                "minor_code_zero_sites": 2,
                "target_people": 2,
                "target_missing_diploid_cells": 0,
                "target_diploid_dosage_sha256": expected["target_legacy_sha256"],
                "ref_people": 6,
                "ref_callable_AN_per_ancestry": 4,
                "ref_no_support_sites": 1,
                "ref_ac_an_sha256": expected["ref_legacy_sha256"],
                "flare_loci": 2,
                "flare_probability_vectors": 8,
                "phase_exported_to_M0": False,
            },
        }
        i0 = {
            "stage": "M33_I0_REAL_INDEX",
            "status": "PASS_DOUBLE_INDEX_AND_QUERY_PARITY_TECHNICAL_ONLY",
            "root_label": "root17",
            "root_seed": 17,
            "source_flare_sha256": "b" * 64,
            "source_generation": "123",
            "output_tbi_sha256": "c" * 64,
            "independent_tbi_sha256": "c" * 64,
            "indexed_record_count": 2,
            "sequential_record_count": 2,
            "scientific_evidence": False,
            "safe_bridge": False,
            "materialize": False,
            "global_ready": False,
            "training": False,
            "truth": False,
        }
        a0_path, i0_path = root / "m33_a0.receipt.json", root / "root17.i0_real.receipt.json"
        write_json(a0_path, a0)
        write_json(i0_path, i0)
        expected["a0_receipt_sha256"] = VERIFY.sha256_file(a0_path)
        expected["i0_receipt_sha256"] = VERIFY.sha256_file(i0_path)
        VERIFY.EXPECTED_ROOTS["root17"] = expected

        raw = {name: VERIFY.sha256_file(path) for name, path in paths.items()}
        semantic = {
            name: VERIFY.artifact_semantic_sha256(VERIFY.ARTIFACTS[name][0], payloads[name])
            for name in paths
        }
        receipt = {
            "stage": VERIFY.STAGE,
            "status": VERIFY.STATUS,
            "schema_id": VERIFY.RECEIPT_SCHEMA,
            "root_label": "root17",
            "root_seed": 17,
            "scientific_evidence": False,
            "consumable": False,
            "truth_read": False,
            "materialize_authorized": False,
            "ready_emitted": False,
            "training_authorized": False,
            "gcs_write": False,
            "append_only": True,
            "reopen_verified": True,
            "write_chmod_rename_probes_failed": True,
            "phase_swap_invariant": True,
            "network_disabled": True,
            "credential_environment_absent": True,
            "f0_anp_only_projection": True,
            "f0_gt_an1_an2_ignored": True,
            "raw_identifiers_exported": False,
            "runner_uid": 65534,
            "runner_euid": 65534,
            "input_sha256_pre": {"source": "d" * 64},
            "input_sha256_post": {"source": "d" * 64},
            "artifact_schema": {name: value[0] for name, value in VERIFY.ARTIFACTS.items()},
            "artifact_raw_sha256": raw,
            "artifact_semantic_sha256": semantic,
            "selected_all_count": 3,
            "selected_incremental_count": 3,
            "selected_overlap_count": 0,
            "incremental_minor_code_0_locus_count": 2,
            "target_count": 2,
            "target_missing_cells": 0,
            "ref_people": 6,
            "ref_people_by_ancestry": {"AFR": 30, "EUR": 30, "ASIA": 30},
            "reference_no_support_loci": 1,
            "flare_marker_count": 2,
            "f0_probability_vectors": 8,
            "target_diploid_dosage_legacy_sha256": expected["target_legacy_sha256"],
            "reference_ac_an_legacy_sha256": expected["ref_legacy_sha256"],
        }
        receipt_path = root / "safe_bridge_technical_kat.receipt.json"
        write_json(receipt_path, receipt)
        return {**paths, "receipt": receipt_path, "a0": a0_path, "i0": i0_path}

    def call(self, paths: dict[str, Path]) -> dict:
        return VERIFY.verify(
            "root17",
            paths["technical_kat_selected_loci_incremental.npz"],
            paths["technical_kat_target_rare_diploid_incremental.npz"],
            paths["technical_kat_reference_rare_summary_incremental.npz"],
            paths["technical_kat_flare_f0_sanitized.npz"],
            paths["receipt"], paths["a0"], paths["i0"],
        )

    def refresh_artifact_receipt(self, paths: dict[str, Path], name: str) -> None:
        receipt = json.loads(paths["receipt"].read_text())
        receipt["artifact_raw_sha256"][name] = VERIFY.sha256_file(paths[name])
        with np.load(paths[name], allow_pickle=False) as loaded:
            arrays = {member: np.array(loaded[member], copy=True) for member in loaded.files}
        receipt["artifact_semantic_sha256"][name] = VERIFY.artifact_semantic_sha256(
            VERIFY.ARTIFACTS[name][0], arrays)
        write_json(paths["receipt"], receipt)

    def test_accepts_exact_independent_known_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.call(self.fixture(Path(temporary) / "case"))
            self.assertEqual(result["status"], VERIFY.STATUS)
            self.assertEqual(result["artifacts_reopened"], 4)
            self.assertEqual(result["legacy_known_answers_verified"], 2)
            self.assertFalse(result["scientific_evidence"])
            self.assertFalse(result["consumable"])
            self.assertRegex(result["a0_receipt_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(set(result["artifact_raw_sha256"]), set(VERIFY.ARTIFACTS))

    def test_independent_receipt_is_exclusive_and_reopenable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.call(self.fixture(root / "case"))
            output = root / "independent.receipt.json"
            VERIFY.write_exclusive_json(output, result)
            self.assertEqual(json.loads(output.read_text()), result)
            with self.assertRaisesRegex(ValueError, "must be new"):
                VERIFY.write_exclusive_json(output, result)

    def test_target_change_fails_legacy_oracle_even_if_artifact_hashes_are_refreshed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary) / "case")
            name = "technical_kat_target_rare_diploid_incremental.npz"
            with np.load(paths[name], allow_pickle=False) as loaded:
                arrays = {member: np.array(loaded[member], copy=True) for member in loaded.files}
            arrays["minor_dosage"][0, 0] = 0
            np.savez(paths[name], **arrays)
            self.refresh_artifact_receipt(paths, name)
            with self.assertRaisesRegex(ValueError, "legacy array semantic hash"):
                self.call(paths)

    def test_forbidden_gt_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary) / "case")
            name = "technical_kat_target_rare_diploid_incremental.npz"
            with np.load(paths[name], allow_pickle=False) as loaded:
                arrays = {member: np.array(loaded[member], copy=True) for member in loaded.files}
            arrays["GT"] = np.zeros((2, 3), dtype="|u1")
            np.savez(paths[name], **arrays)
            receipt = json.loads(paths["receipt"].read_text())
            receipt["artifact_raw_sha256"][name] = VERIFY.sha256_file(paths[name])
            write_json(paths["receipt"], receipt)
            with self.assertRaisesRegex(ValueError, "NPZ member inventory"):
                self.call(paths)

    def test_f0_non_simplex_is_rejected_after_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary) / "case")
            name = "technical_kat_flare_f0_sanitized.npz"
            with np.load(paths[name], allow_pickle=False) as loaded:
                arrays = {member: np.array(loaded[member], copy=True) for member in loaded.files}
            arrays["probabilities"][0, 0, 0] = [0.2, 0.2, 0.2]
            np.savez(paths[name], **arrays)
            self.refresh_artifact_receipt(paths, name)
            with self.assertRaisesRegex(ValueError, "F0 probabilities"):
                self.call(paths)

    def test_scope_or_input_mutability_claim_is_rejected(self) -> None:
        for field, value, message in (
            ("consumable", True, "consumable"),
            ("truth_read", True, "truth_read"),
            ("write_chmod_rename_probes_failed", False, "read-only"),
            ("network_disabled", False, "isolation"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                paths = self.fixture(Path(temporary) / "case")
                receipt = json.loads(paths["receipt"].read_text())
                receipt[field] = value
                write_json(paths["receipt"], receipt)
                with self.assertRaisesRegex(ValueError, message):
                    self.call(paths)

    def test_a0_or_i0_receipt_byte_drift_is_rejected(self) -> None:
        for key_name, message in (("a0", "A0 receipt raw hash"), ("i0", "I0 receipt raw hash")):
            with self.subTest(receipt=key_name), tempfile.TemporaryDirectory() as temporary:
                paths = self.fixture(Path(temporary) / "case")
                payload = json.loads(paths[key_name].read_text())
                payload["unexpected"] = True
                write_json(paths[key_name], payload)
                with self.assertRaisesRegex(ValueError, message):
                    self.call(paths)


if __name__ == "__main__":
    unittest.main()
