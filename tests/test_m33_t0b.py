#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


MODELS = load("m33_t0a_models", "bin/m33_t0a_models.py")
T0A = load("m33_t0a_forward", "bin/m33_t0a_forward.py")
PREFLIGHT = load("m33_t0b_preflight", "bin/m33_t0b_preflight.py")
FORWARD = load("m33_t0b_forward", "bin/m33_t0b_forward.py")
COMPARE = load("m33_t0b_compare", "bin/m33_t0b_compare.py")
AUTH = load("m33_t0b_source_auth", "bin/m33_t0b_source_auth.py")
MATERIALIZE = FORWARD.materialize


COMMIT = "a" * 40
OCI = ("us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:"
       + "c" * 64)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Namespace:
    def __init__(self, **values):
        self.__dict__.update(values)


class M33T0bTests(unittest.TestCase):
    def materialize_fixture(self):
        loci = np.asarray([10, 11, 12], dtype="<u8")
        samples = np.asarray([b"a" * 64, b"b" * 64], dtype="|S64")
        selected = {
            "locus_id": loci, "chrom": np.full(3, 22, dtype="|u1"),
            "pos": np.asarray([100, 200, 300], dtype="<i8"),
            "ref": np.asarray([b"A", b"C", b"G"], dtype="|S1"),
            "alt": np.asarray([b"T", b"G", b"A"], dtype="|S1"),
            "cM": np.asarray([0.05, 0.10, 0.20], dtype="<f8"),
        }
        target = {
            "sample_key_sha256": samples, "locus_id": loci,
            "minor_dosage": np.asarray([[2, 1, 0], [0, 0, 2]], dtype="|i1"),
            "observed_mask": np.asarray([[1, 1, 1], [1, 0, 1]], dtype="|u1"),
        }
        ac = np.asarray([[1, 0, 3], [2, 1, 0], [0, 2, 1]], dtype="<u2")
        an = np.full((3, 3), 4, dtype="<u2")
        reference = {
            "ancestry": np.asarray([b"AFR", b"EUR", b"ASIA"], dtype="|S4"),
            "locus_id": loci, "minor_ac": ac, "callable_an": an,
            "minor_af": ac.astype("<f8") / an,
            "observed_mask": np.ones((3, 3), dtype="|u1"),
            "no_support": (ac == 0).astype("|u1"),
        }
        f0 = {
            "sample_key_sha256": samples, "marker_chrom": np.full(2, 22, dtype="|u1"),
            "marker_pos": np.asarray([150, 250], dtype="<i8"),
            "marker_ref": np.asarray([b"A", b"C"], dtype="|S1"),
            "marker_alt": np.asarray([b"G", b"T"], dtype="|S1"),
            "F0": np.full((2, 2, 2, 3), np.float32(1 / 3), dtype="<f4"),
        }
        return selected, target, reference, f0, np.asarray([0.10, 0.15], dtype="<f8")

    def write_source_auth(self, base: Path, relative: str) -> tuple[Path, Path]:
        staged = base / "staged" / "bin"
        staged.mkdir(parents=True)
        source = ROOT / relative
        target = staged / source.name
        target.write_bytes(source.read_bytes())
        auth = base / "t0b_source_auth.json"
        auth.write_text(json.dumps({
            "stage": "M33_T0B_SOURCE_AUTH",
            "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
            "git_commit": COMMIT,
            "source_sha256": {relative: sha256_file(target)},
        }, sort_keys=True) + "\n", encoding="utf-8")
        return auth, base / "staged"

    def test_contract_is_explicit_and_exact(self):
        contract = json.loads((ROOT / "conf/m33_t0b_contract.json").read_text())
        self.assertEqual(contract["scope"]["marker_count"], 79_791)
        self.assertEqual(contract["scope"]["radii_cM"], [0.05, 0.1, 0.2, 0.5])
        self.assertEqual(contract["execution"]["maximum_padded_tokens_per_batch"], 262_144)
        self.assertEqual(contract["execution"]["process_memory_gib"], 6)
        self.assertEqual(contract["execution"]["minimum_preflight_mem_available_gib"], 26)
        self.assertEqual(contract["execution"]["maximum_parallel_forward_processes"], 3)
        self.assertEqual(len(contract["scope"]["cases"]), 5)

    def test_source_inventory_covers_all_runtime_and_orchestration(self):
        expected = {
            "bin/m31_ordered_linear.py", "bin/m33_m0_contract.py",
            "bin/m33_materialize.py", "bin/m33_m0_factorized_lazy_technical_kat.py",
            "bin/m33_t0a_models.py", "bin/m33_t0a_forward.py",
            "bin/m33_t0b_preflight.py", "bin/m33_t0b_forward.py",
            "bin/m33_t0b_compare.py", "bin/m33_t0b_source_auth.py",
            "conf/m33_pre4_preregistration.json", "conf/m33_t0b_contract.json",
            "conf/m33_t0b.config", "modules/33_T0B_FULL_CHR22.nf",
            "workflows/m33_t0b.nf", "containers/m33-t0a/Dockerfile",
            "tests/test_m33_t0b.py",
        }
        self.assertEqual(AUTH.REQUIRED_SOURCES, expected)

    def test_nextflow_dependency_limits_and_cnn_first_order(self):
        workflow = (ROOT / "workflows/m33_t0b.nf").read_text(encoding="utf-8")
        module = (ROOT / "modules/33_T0B_FULL_CHR22.nf").read_text(encoding="utf-8")
        config = (ROOT / "conf/m33_t0b.config").read_text(encoding="utf-8")
        combined = (workflow + module + config).lower()
        self.assertIn("preflightreceipt", workflow.lower())
        self.assertIn("maxforks 3", module.lower())
        self.assertIn("maxforks = 3", config.lower())
        self.assertIn("memory '6 gb'", module.lower())
        self.assertIn("time '10h'", module.lower())
        self.assertIn("--network none", combined)
        self.assertIn("--memory 6g", combined)
        self.assertIn("path root17_map, stageAs: 'root17-map/*'", module)
        self.assertIn("path root18_map, stageAs: 'root18-map/*'", module)
        cases = workflow.split("cases = Channel.of(", 1)[1].split("\n    )", 1)[0]
        positions = [cases.index(fragment) for fragment in (
            "'small_residual_cnn_1d', 0", "'small_residual_cnn_1d', 1",
            "'root18', 20260818, 'small_residual_cnn_1d', 0",
            "'root17', 20260817, 'local_linear', 0")]
        self.assertEqual(positions, sorted(positions))
        for forbidden in ("optimizer", "backward", "gcloud", "gsutil"):
            self.assertNotIn(forbidden, combined)

    def preflight_fixture(self, base: Path):
        auth, staged = self.write_source_auth(base, "bin/m33_t0b_preflight.py")
        t0a_auth = base / "t0a_source_auth.json"
        t0a_auth.write_text("t0a-auth\n", encoding="utf-8")
        children = []
        inventory = []
        for index in range(12):
            child = base / f"child.{index:02d}.json"
            child.write_text(json.dumps({
                "implementation_commit": "b" * 40,
                "source_auth_sha256": sha256_file(t0a_auth),
                "oci_image": OCI,
                "truth_read": False, "training": False, "gradients": False,
                "optimizer": False, "predictions_persisted": False, "consumable": False,
            }), encoding="utf-8")
            children.append(child)
            inventory.append({"name": child.name, "sha256": sha256_file(child)})
        aggregate = base / "aggregate.json"
        aggregate.write_text(json.dumps({
            "stage": "M33_T0A_CROSS_PROCESS_COMPARISON",
            "status": "PASS_T0A_CROSS_PROCESS_TECHNICAL_ONLY",
            "t0b_open": False, "scientific_evidence": False,
            "implementation_commit": "b" * 40, "oci_image": OCI,
            "child_receipts": inventory,
        }), encoding="utf-8")
        contract = ROOT / "conf/m33_t0b_contract.json"
        meminfo = base / "meminfo"
        meminfo.write_text("MemTotal:       33554432 kB\nMemAvailable:   29360128 kB\n")
        args = Namespace(
            source_auth=auth, source_root=staged, implementation_commit=COMMIT,
            contract=contract, t0a_aggregate=aggregate, t0a_source_auth=t0a_auth,
            t0a_child_receipt=children, root17_technical_dir=base / "root17",
            root18_technical_dir=base / "root18",
            root17_verify=base / "root17.verify", root18_verify=base / "root18.verify",
            root17_map=base / "root17.map", root18_map=base / "root18.map",
            meminfo=meminfo,
        )
        return args, aggregate, t0a_auth

    def test_preflight_accepts_exact_anchors_inventory_markers_and_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            args, aggregate, t0a_auth = self.preflight_fixture(Path(directory))
            expected = json.loads(args.contract.read_text())["expected_inputs"]
            identities = {root: {**value, "marker_count": 79_791}
                          for root, value in expected.items()}
            with mock.patch.object(PREFLIGHT, "EXPECTED_AGGREGATE_SHA256",
                                   sha256_file(aggregate)), \
                 mock.patch.object(PREFLIGHT, "EXPECTED_T0A_SOURCE_AUTH_SHA256",
                                   sha256_file(t0a_auth)), \
                 mock.patch.object(PREFLIGHT, "authenticate_root_inputs",
                                   side_effect=lambda root, *_args: identities[root]):
                receipt = PREFLIGHT.run_preflight(args)
            self.assertEqual(receipt["status"], "PASS_T0B_PREFLIGHT_THREE_WAY_ONLY")
            self.assertGreaterEqual(receipt["mem_available_gib"], 26)

    def write_real_schema_f0(self, directory: Path, *, count: int = 7,
                             wrong_name: bool = False, wrong_axis: bool = False) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        member = "F0" if wrong_name else "probabilities"
        probability_markers = count - 1 if wrong_axis else count
        values = {
            "sample_key_sha256": np.asarray([b"a" * 64], dtype="|S64"),
            "chrom": np.full(count, 22, dtype="|u1"),
            "pos": np.arange(count, dtype="<i8"),
            "ref": np.full(count, b"A", dtype="|S1"),
            "alt": np.full(count, b"C", dtype="|S1"),
            member: np.full((1, 2, probability_markers, 3), np.float32(1 / 3), dtype="<f4"),
        }
        path = directory / "technical_kat_flare_f0_sanitized.npz"
        np.savez(path, **values)
        return path

    def test_marker_count_reads_exact_real_technical_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_real_schema_f0(root, count=7)
            self.assertEqual(PREFLIGHT.marker_count(root), 7)
        for mutation in ("name", "axis"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_real_schema_f0(
                    root, count=7, wrong_name=mutation == "name", wrong_axis=mutation == "axis")
                with self.assertRaises(ValueError):
                    PREFLIGHT.marker_count(root)

    def test_preflight_rejects_missing_marker_low_memory_and_tampered_source(self):
        for mutation in ("marker", "memory", "source"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                args, aggregate, t0a_auth = self.preflight_fixture(base)
                expected = json.loads(args.contract.read_text())["expected_inputs"]
                identities = {root: {**value, "marker_count": 79_791}
                              for root, value in expected.items()}
                if mutation == "marker":
                    identities["root17"]["marker_count"] = 79_790
                if mutation == "memory":
                    args.meminfo.write_text(
                        "MemTotal:       33554432 kB\nMemAvailable:   25165824 kB\n")
                if mutation == "source":
                    (args.source_root / "bin/m33_t0b_preflight.py").write_text("# tampered\n")
                with mock.patch.object(PREFLIGHT, "EXPECTED_AGGREGATE_SHA256",
                                       sha256_file(aggregate)), \
                     mock.patch.object(PREFLIGHT, "EXPECTED_T0A_SOURCE_AUTH_SHA256",
                                       sha256_file(t0a_auth)), \
                     mock.patch.object(PREFLIGHT, "authenticate_root_inputs",
                                       side_effect=lambda root, *_args: identities[root]):
                    with self.assertRaises(ValueError):
                        PREFLIGHT.run_preflight(args)

    def technical_root_fixture(self, base: Path, label: str):
        technical = base / label
        self.write_real_schema_f0(technical, count=7)
        selected = technical / "technical_kat_selected_loci_incremental.npz"
        np.savez(selected, locus_key_sha256=np.asarray([b"a" * 64] * 5, dtype="|S64"))
        target = technical / "technical_kat_target_rare_diploid_incremental.npz"
        np.savez(target, sample_key_sha256=np.asarray([b"b" * 64] * 3, dtype="|S64"))
        reference = technical / "technical_kat_reference_rare_summary_incremental.npz"
        np.savez(reference, ancestry=np.asarray([b"AFR"], dtype="|S4"))
        bridge = technical / "safe_bridge_technical_kat.receipt.json"
        bridge.write_text(f"{label}-bridge\n")
        verify = base / f"{label}.verify.json"; verify.write_text(f"{label}-verify\n")
        genetic_map = base / f"{label}.map"; genetic_map.write_text(f"{label}-map\n")
        expected = {
            "target_count": 3, "rare_locus_count": 5,
            "bridge_receipt_sha256": sha256_file(bridge),
            "independent_verify_receipt_sha256": sha256_file(verify),
            "genetic_map_sha256": sha256_file(genetic_map),
            "technical_npz_sha256": {
                path.name: sha256_file(path) for path in
                (technical / "technical_kat_flare_f0_sanitized.npz", reference,
                 selected, target)
            },
        }
        return technical, verify, genetic_map, expected

    def test_root_input_provenance_accepts_exact_and_rejects_tamper_or_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root17 = self.technical_root_fixture(base, "root17")
            root18 = self.technical_root_fixture(base, "root18")
            with mock.patch.object(PREFLIGHT, "EXPECTED_MARKERS", 7):
                observed = PREFLIGHT.authenticate_root_inputs("root17", *root17)
                self.assertEqual(observed["technical_npz_sha256"],
                                 root17[3]["technical_npz_sha256"])
                with self.assertRaisesRegex(ValueError, "input hash differs"):
                    PREFLIGHT.authenticate_root_inputs(
                        "root17", root17[0], root18[1], root17[2], root17[3])
                target = root17[0] / "technical_kat_target_rare_diploid_incremental.npz"
                target.write_bytes(target.read_bytes() + b"tamper")
                with self.assertRaisesRegex(ValueError, "input hash differs"):
                    PREFLIGHT.authenticate_root_inputs("root17", *root17)

    def test_forward_binds_the_consumed_root_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root17 = self.technical_root_fixture(base, "root17")
            root18 = self.technical_root_fixture(base, "root18")
            args = Namespace(technical_dir=root17[0], independent_verify_receipt=root17[1],
                             genetic_map=root17[2], root_label="root17")
            FORWARD.validate_root_files(args, root17[3])
            args.independent_verify_receipt = root18[1]
            with self.assertRaisesRegex(ValueError, "input provenance differs"):
                FORWARD.validate_root_files(args, root17[3])

    def compare_fixture(self, base: Path):
        auth, staged = self.write_source_auth(base, "bin/m33_t0b_compare.py")
        contract = base / "contract.json"
        contract.write_bytes((ROOT / "conf/m33_t0b_contract.json").read_bytes())
        preflight = base / "preflight.json"
        expected_inputs = json.loads(contract.read_text())["expected_inputs"]
        preflight.write_text(json.dumps({
            "stage": "M33_T0B_PREFLIGHT",
            "status": "PASS_T0B_PREFLIGHT_THREE_WAY_ONLY",
            "contract_sha256": sha256_file(contract),
            "implementation_commit": COMMIT,
            "source_auth_sha256": sha256_file(auth),
            "input_identity_by_root": {
                root: {**identity, "marker_count": 79_791}
                for root, identity in expected_inputs.items()
            },
        }), encoding="utf-8")
        paths = []
        cases = sorted(COMPARE.EXPECTED_CASES)
        sentinels = [{
            "radius_cM": radius, "marker_index": marker,
            "output_semantic_sha256": "1" * 64,
            "feature_semantic_sha256": "2" * 64,
            "valid_tokens": 10, "padded_tokens": 12,
            "row_count": 30, "shard_count": 4,
        } for radius in (0.05, 0.1, 0.2, 0.5) for marker in (0, 39_895, 79_790)]
        for root, family, repetition in cases:
            payload = {
                "stage": "M33_T0B_FULL_CHR22_FORWARD",
                "status": "PASS_T0B_FULL_CHR22_FORWARD_ONLY_NON_CONSUMABLE",
                "root_label": root, "root_seed": 20260817 if root == "root17" else 20260818,
                "model_family": family, "repetition": repetition,
                "marker_count": 79_791, "target_count": 30,
                "radii_cM": [0.05, 0.1, 0.2, 0.5], "channel_count": 13,
                "rare_locus_count": expected_inputs[root]["rare_locus_count"],
                "valid_tokens": 200,
                "padded_tokens": 220, "row_count": 9_574_920, "shard_count": 100,
                "maximum_valid_tokens_per_shard": 150,
                "maximum_padded_tokens_per_batch": 262_144,
                "output_semantic_sha256": "3" * 64,
                "feature_semantic_sha256": "4" * 64,
                "marker_index_semantic_sha256": "5" * 64,
                "technical_locus_key_axis_semantic_sha256": "6" * 64,
                "parameter_count": 90 if family == "local_linear" else 1651,
                "parameter_shape_sha256": "7" * 64,
                "parameter_value_sha256": "8" * 64,
                "zero_residual_F0_max_abs": 0.0, "simplex_max_abs": 0.0,
                "invariance_checks": {"all": 0.0},
                "sentinel_replay": sentinels, "sentinel_replay_exact": True,
                "sentinel_passes": 2, "memory_warning_fraction": 0.70,
                "memory_stop_fraction": 0.80, "peak_rss_fraction": 0.2,
                "memory_warning": False,
                "device": "cpu", "vram_applicable": False, "torch_version": "test",
                "oci_image": OCI, "implementation_commit": COMMIT,
                "source_auth_sha256": sha256_file(auth),
                "contract_sha256": sha256_file(contract),
                "preflight_receipt_sha256": sha256_file(preflight),
                "bridge_receipt_sha256": expected_inputs[root]["bridge_receipt_sha256"],
            }
            payload.update({field: False for field in COMPARE.FALSE_FIELDS})
            path = base / f"{root}.{family}.rep{repetition}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths.append(path)
        return paths, auth, staged, contract, preflight

    def test_comparator_accepts_exact_five_case_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.compare_fixture(Path(directory))
            paths, auth, staged, contract, preflight = args
            receipt = COMPARE.compare_receipts(
                paths, auth, staged, COMMIT, OCI, contract, preflight)
            self.assertEqual(receipt["status"], "PASS_T0B_FULL_CHR22_TECHNICAL_ONLY")
            self.assertTrue(receipt["root17_cnn_repetitions_exact"])

    def test_comparator_rejects_truth_identity_missing_case_sentinel_and_tampering(self):
        for mutation in ("truth", "identity", "missing", "sentinel", "source",
                         "parameter", "hash", "counter", "invariance", "boolean_metric",
                         "root18_bridge", "preflight_identity"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                paths, auth, staged, contract, preflight = self.compare_fixture(Path(directory))
                payload = json.loads(paths[0].read_text())
                if mutation == "truth":
                    payload["truth_read"] = True
                elif mutation == "identity":
                    payload["oci_image"] = OCI[:-1] + "d"
                elif mutation == "missing":
                    paths.pop()
                elif mutation == "sentinel":
                    payload["sentinel_replay"][0]["marker_index"] = 1
                elif mutation == "source":
                    (staged / "bin/m33_t0b_compare.py").write_text("# tampered\n")
                elif mutation == "parameter":
                    payload["parameter_count"] = 91
                elif mutation == "hash":
                    payload["output_semantic_sha256"] = "not-a-hash"
                elif mutation == "counter":
                    payload["row_count"] -= 1
                elif mutation == "invariance":
                    payload["invariance_checks"] = {"padding": 1e-3}
                elif mutation == "boolean_metric":
                    payload["zero_residual_F0_max_abs"] = False
                elif mutation == "root18_bridge":
                    target = next(path for path in paths if
                                  path.name == "root18.local_linear.rep0.json")
                    payload = json.loads(target.read_text())
                    payload["bridge_receipt_sha256"] = "f" * 64
                    target.write_text(json.dumps(payload), encoding="utf-8")
                elif mutation == "preflight_identity":
                    preflight_payload = json.loads(preflight.read_text())
                    preflight_payload["input_identity_by_root"]["root18"]["rare_locus_count"] -= 1
                    preflight.write_text(json.dumps(preflight_payload), encoding="utf-8")
                if mutation in {"truth", "identity", "sentinel", "parameter", "hash",
                                "counter", "invariance", "boolean_metric"}:
                    paths[0].write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    COMPARE.compare_receipts(paths, auth, staged, COMMIT, OCI,
                                             contract, preflight)

    def test_repeat_pair_rejects_full_hash_or_counter_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, auth, staged, contract, preflight = self.compare_fixture(Path(directory))
            target = next(path for path in paths if
                          "root17.small_residual_cnn_1d.rep1" in path.name)
            payload = json.loads(target.read_text())
            payload["valid_tokens"] += 1
            target.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "determinism"):
                COMPARE.compare_receipts(paths, auth, staged, COMMIT, OCI,
                                         contract, preflight)

    def test_inherited_padding_and_haplotype_asymmetry_gate(self):
        torch.manual_seed(4)
        tokens = torch.randn(6, 9, 13)
        mask = torch.ones(6, 9)
        f0 = torch.softmax(torch.randn(6, 2, 3), dim=2)
        self.assertLessEqual(max(T0A.invariant_checks(
            MODELS.build_model("small_residual_cnn_1d"), tokens, mask, f0).values()), 1e-6)

        class Asymmetric(torch.nn.Module):
            def forward_with_features(self, token_rows, row_mask, baseline):
                delta = torch.zeros_like(baseline)
                delta[:, 0, 0] = 0.2
                probability = torch.softmax(torch.log(torch.clamp(baseline, min=1e-7)) + delta,
                                            dim=2)
                return probability, delta, baseline.clone()

        with self.assertRaisesRegex(ValueError, "invariant failed"):
            T0A.invariant_checks(Asymmetric(), tokens, mask, f0)

    def test_full_sentinel_indexes_are_exact(self):
        self.assertEqual(FORWARD.sentinel_indexes(79_791), (0, 39_895, 79_790))
        with self.assertRaises(ValueError):
            FORWARD.sentinel_indexes(79_790)

    def test_interval_table_is_validated_once_and_default_still_fails_closed(self):
        selected, target, reference, f0, marker_cm = self.materialize_fixture()
        normalization = {"AFR": 4, "EUR": 4, "ASIA": 4}
        original = MATERIALIZE.validate_interval_table
        with mock.patch.object(MATERIALIZE, "validate_interval_table", wraps=original) as validator:
            intervals, proof = MATERIALIZE.build_authenticated_interval_table(
                selected["cM"], marker_cm)
            for radius in MATERIALIZE.RADII[:2]:
                MATERIALIZE.build_lazy_packed_shard(
                    selected, target, reference, f0, marker_cm, intervals, normalization,
                    radius, 0, 2, 0, 2, inputs_already_validated=True,
                    interval_validation=proof)
            self.assertEqual(validator.call_count, 1)
        self.assertTrue(all(not value.flags.writeable for value in intervals.values()))
        self.assertFalse(selected["cM"].flags.writeable)
        self.assertFalse(marker_cm.flags.writeable)

        corrupted = MATERIALIZE.build_interval_table(selected["cM"], marker_cm)
        corrupted["context_start"][:] = 0
        corrupted["context_stop"][:] = 0
        with self.assertRaisesRegex(ValueError, "content differs"):
            MATERIALIZE.build_lazy_packed_shard(
                selected, target, reference, f0, marker_cm, corrupted, normalization,
                0.05, 0, 2, 0, 2, inputs_already_validated=True)
        with self.assertRaisesRegex(ValueError, "binding differs"):
            MATERIALIZE.build_lazy_packed_shard(
                selected, target, reference, f0, marker_cm,
                MATERIALIZE.build_interval_table(selected["cM"], marker_cm), normalization,
                0.05, 0, 2, 0, 2, inputs_already_validated=True,
                interval_validation=proof)
        replacement = {name: value for name, value in intervals.items()}
        replacement_start = np.asarray(intervals["context_start"]).copy()
        replacement_start.flags.writeable = False
        replacement["context_start"] = replacement_start
        replacement_proof = proof._replace(intervals_object_id=id(replacement))
        with self.assertRaisesRegex(ValueError, "binding differs"):
            MATERIALIZE.build_lazy_packed_shard(
                selected, target, reference, f0, marker_cm, replacement, normalization,
                0.05, 0, 2, 0, 2, inputs_already_validated=True,
                interval_validation=replacement_proof)

    def test_sentinel_replay_rebuilds_twice_and_detects_counter_change(self):
        selected, target, reference, f0, marker_cm = self.materialize_fixture()
        intervals, proof = MATERIALIZE.build_authenticated_interval_table(
            selected["cM"], marker_cm)
        model = MODELS.build_model("local_linear")
        arguments = (
            model, selected, target, reference, f0, marker_cm, intervals, proof,
            {"AFR": 4, "EUR": 4, "ASIA": 4}, 20260817, "a" * 64,
        )
        with mock.patch.object(FORWARD, "sentinel_indexes", return_value=(0, 1, 1)):
            rows = FORWARD.replay_sentinels_exact(*arguments)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["row_count"] == 2 for row in rows))
        first = [{"row_count": 2}]
        second = [{"row_count": 3}]
        with mock.patch.object(FORWARD, "sentinel_pass", side_effect=(first, second)):
            with self.assertRaisesRegex(ValueError, "sentinel replay differs"):
                FORWARD.replay_sentinels_exact(*arguments)


if __name__ == "__main__":
    unittest.main()
