#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


MODELS = load("m33_t0a_models", "bin/m33_t0a_models.py")
FORWARD = load("m33_t0a_forward", "bin/m33_t0a_forward.py")
AUTH = load("m33_t0a_source_auth", "bin/m33_t0a_source_auth.py")


OCI_IMAGE = ("us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:"
             + "c" * 64)
IMPLEMENTATION_COMMIT = "a" * 40


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M33T0aTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(7)
        tokens = torch.randn(6, 11, 13, dtype=torch.float32)
        mask = torch.ones(6, 11, dtype=torch.float32)
        mask[0] = 0; tokens[0] = 0
        mask[1, 5:] = 0; tokens[1, 5:] = 0
        f0 = torch.softmax(torch.randn(6, 2, 3), dim=2)
        return tokens, mask, f0

    def test_exact_families_counts_and_zero_residual(self):
        tokens, mask, f0 = self.fixture()
        for family, count in (("local_linear", 90), ("small_residual_cnn_1d", 1651)):
            model = MODELS.build_model(family)
            self.assertEqual(MODELS.parameter_count(model), count)
            self.assertFalse(model.training)
            with torch.inference_mode():
                probabilities, delta, _features = model.forward_with_features(tokens, mask, f0)
            self.assertEqual(float(delta.abs().max()), 0.0)
            self.assertLessEqual(float((probabilities - f0).abs().max()), 1e-6)
            MODELS.assert_probabilities(probabilities)

    def test_padding_permutation_chunk_and_haplotype_invariants(self):
        tokens, mask, f0 = self.fixture()
        for family in MODELS.FAMILIES:
            checks = FORWARD.invariant_checks(MODELS.build_model(family), tokens, mask, f0)
            self.assertLessEqual(max(checks.values()), 1e-6)
            probe = MODELS.build_model(family)
            MODELS.activate_deterministic_probe_head(probe)
            probe_checks = FORWARD.invariant_checks(probe, tokens, mask, f0)
            self.assertLessEqual(max(probe_checks.values()), 1e-6)
            with torch.inference_mode():
                _probabilities, delta, _features = probe.forward_with_features(tokens, mask, f0)
            self.assertGreater(float(delta.abs().max()), 0.0)

    def test_asymmetric_model_fails_invariance_gate(self):
        class HaplotypeAsymmetric(torch.nn.Module):
            def forward_with_features(self, tokens, mask, f0):
                delta = torch.zeros_like(f0)
                delta[:, 0, 0] = 0.25
                probabilities = torch.softmax(torch.log(torch.clamp(f0, min=1e-7)) + delta,
                                              dim=2)
                features = f0.clone()
                return probabilities, delta, features

        tokens, mask, f0 = self.fixture()
        with self.assertRaisesRegex(ValueError, "invariant failed"):
            FORWARD.invariant_checks(HaplotypeAsymmetric(), tokens, mask, f0)

    def test_invalid_f0_fails_instead_of_being_clamped_silently(self):
        tokens, mask, f0 = self.fixture()
        f0[0, 0, 0] = -0.1
        with self.assertRaisesRegex(ValueError, "simplex"):
            MODELS.build_model("local_linear")(tokens, mask, f0)

    def test_padded_batches_respect_budget_and_empty_rows(self):
        row_ptr = np.asarray([0, 0, 3, 10, 1000, 2000], dtype="<u8")
        batches = FORWARD.padded_batches(row_ptr, maximum_rows=3)
        self.assertEqual(batches[0][0], 0)
        self.assertEqual(batches[-1][1], 5)
        lengths = np.diff(row_ptr)
        for start, end in batches:
            self.assertLessEqual(int(lengths[start:end].max(initial=0)) * (end - start),
                                 FORWARD.MAX_PADDED_TOKENS)

    def test_source_inventory_covers_runtime_orchestration_and_tests(self):
        expected = {
            "bin/m31_ordered_linear.py", "bin/m33_m0_contract.py", "bin/m33_materialize.py",
            "bin/m33_m0_factorized_lazy_technical_kat.py", "bin/m33_t0a_models.py",
            "bin/m33_t0a_forward.py", "bin/m33_t0a_compare.py", "bin/m33_t0a_source_auth.py",
            "bin/m33_t0a_stress.py",
            "conf/m33_t0a.config", "modules/33_T0A_FORWARD.nf", "workflows/m33_t0a.nf",
            "conf/m33_pre4_preregistration.json",
            "containers/m33-t0a/Dockerfile", "tests/test_m33_t0a.py",
        }
        self.assertEqual(AUTH.REQUIRED_SOURCES, expected)

    def test_nextflow_firewall_and_digest_pin(self):
        combined = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in (
            "conf/m33_t0a.config", "modules/33_T0A_FORWARD.nf", "workflows/m33_t0a.nf"))
        lowered = combined.lower()
        self.assertIn("--network none", lowered)
        self.assertIn("maxforks = 2", lowered)
        self.assertIn("sha256", lowered)
        self.assertNotIn("*root", lowered)
        for required in ("truth_read", "predictions_persisted", "source_auth"):
            self.assertIn(required, (ROOT / "bin/m33_t0a_compare.py").read_text().lower())
        for forbidden in ("truth", "optimizer", "backward", "gcloud", "gsutil"):
            self.assertNotIn(forbidden, lowered)

    def comparison_fixture(self, base: Path):
        staged = base / "staged" / "bin"
        staged.mkdir(parents=True)
        compare_copy = staged / "m33_t0a_compare.py"
        compare_copy.write_bytes((ROOT / "bin/m33_t0a_compare.py").read_bytes())
        source_auth = base / "source_auth.json"
        source_auth.write_text(json.dumps({
            "stage": "M33_T0A_SOURCE_AUTH",
            "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
            "git_commit": IMPLEMENTATION_COMMIT,
            "source_sha256": {"bin/m33_t0a_compare.py": sha256_file(compare_copy)},
        }, sort_keys=True) + "\n", encoding="utf-8")
        auth_sha = sha256_file(source_auth)
        root_paths = []
        for root, seed in (("root17", 20260817), ("root18", 20260818)):
            for family in MODELS.FAMILIES:
                for repetition in (0, 1):
                    payload = {
                        "stage": "M33_T0A_FORWARD_TECHNICAL_ROOT",
                        "status": "PASS_T0A_FORWARD_ONLY_NON_CONSUMABLE",
                        "root_label": root, "root_seed": seed,
                        "model_family": family, "repetition": repetition,
                        "marker_count": 512, "target_count": 30,
                        "radii_cM": [0.05, 0.1, 0.2, 0.5], "channel_count": 13,
                        "rare_locus_count": 100, "valid_tokens": 200,
                        "padded_tokens": 220, "row_count": 30, "shard_count": 2,
                        "maximum_valid_tokens_per_shard": 110,
                        "output_semantic_sha256": "1" * 64,
                        "feature_semantic_sha256": "2" * 64,
                        "marker_index_semantic_sha256": "3" * 64,
                        "technical_locus_key_axis_semantic_sha256": "4" * 64,
                        "parameter_count": 90 if family == "local_linear" else 1651,
                        "parameter_shape_sha256": "5" * 64,
                        "parameter_value_sha256": "6" * 64,
                        "zero_residual_F0_max_abs": 0.0, "simplex_max_abs": 0.0,
                        "invariance_checks": {"all": 0.0},
                        "nonzero_probe_invariance_checks": {"all": 0.0},
                        "nonzero_probe_delta_max_abs": 0.1,
                        "nonzero_probe_output_sha256": "7" * 64,
                        "nonzero_probe_delta_sha256": "8" * 64,
                        "nonzero_probe_feature_sha256": "9" * 64,
                        "torch_version": "test", "oci_image": OCI_IMAGE,
                        "implementation_commit": IMPLEMENTATION_COMMIT,
                        "source_auth_sha256": auth_sha,
                        "bridge_receipt_sha256": "b" * 64,
                        "peak_rss_fraction": 0.1, "elapsed_seconds": 1.0,
                        "device": "cpu", "vram_applicable": False,
                        "model_or_radius_selected": False,
                    }
                    payload.update({field: False for field in (
                        "truth_read", "training", "gradients", "optimizer",
                        "predictions_persisted", "consumable")})
                    path = base / f"{root}.{family}.{repetition}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    root_paths.append(path)
        stress_paths = []
        for family in MODELS.FAMILIES:
            for repetition in (0, 1):
                payload = {
                    "stage": "M33_T0A_SYNTHETIC_STRESS",
                    "status": "PASS_T0A_SYNTHETIC_STRESS_NON_CONSUMABLE",
                    "model_family": family, "repetition": repetition,
                    "stress_people": [30, 256, 1024, 2619], "stress_loci": 512,
                    "synthetic_context_length": 64,
                    "synthetic_people_are_not_individuals": True,
                    "stress_loci_interpretation": "test",
                    "stress_is_streaming_not_full_DNABR_matrix": True,
                    "ragged_geometry_probe": {"padded_tokens": 262144},
                    "parameter_count": 90 if family == "local_linear" else 1651,
                    "parameter_shape_sha256": "5" * 64,
                    "parameter_value_sha256": "6" * 64,
                    "torch_version": "test", "oci_image": OCI_IMAGE,
                    "implementation_commit": IMPLEMENTATION_COMMIT,
                    "source_auth_sha256": auth_sha, "rows": [],
                    "peak_rss_fraction": 0.1, "device": "cpu", "vram_applicable": False,
                    "model_or_radius_selected": False,
                }
                payload.update({field: False for field in (
                    "truth_read", "training", "gradients", "optimizer",
                    "predictions_persisted", "consumable")})
                path = base / f"stress.{family}.{repetition}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                stress_paths.append(path)
        return source_auth, base / "staged", root_paths, stress_paths

    def run_comparison(self, base: Path, source_auth: Path, staged: Path,
                       roots: list[Path], stress: list[Path]):
        command = [sys.executable, str(ROOT / "bin/m33_t0a_compare.py")]
        for path in roots:
            command.extend(("--receipt", str(path)))
        for path in stress:
            command.extend(("--stress-receipt", str(path)))
        command.extend((
            "--source-auth", str(source_auth), "--source-root", str(staged),
            "--implementation-commit", IMPLEMENTATION_COMMIT,
            "--oci-image", OCI_IMAGE, "--output", str(base / "aggregate.json"),
        ))
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def test_comparator_accepts_canonical_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            inputs = self.comparison_fixture(base)
            result = self.run_comparison(base, *inputs)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_comparator_rejects_truth_identity_mix_and_tampering(self):
        mutations = ("truth", "commit", "image", "source")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                source_auth, staged, roots, stress = self.comparison_fixture(base)
                target = roots[0]
                payload = json.loads(target.read_text(encoding="utf-8"))
                if mutation == "truth":
                    payload["truth_read"] = True
                elif mutation == "commit":
                    payload["implementation_commit"] = "d" * 40
                elif mutation == "image":
                    payload["oci_image"] = OCI_IMAGE[:-1] + "d"
                elif mutation == "source":
                    (staged / "bin/m33_t0a_compare.py").write_text(
                        "# tampered\n", encoding="utf-8")
                if mutation != "source":
                    target.write_text(json.dumps(payload), encoding="utf-8")
                result = self.run_comparison(base, source_auth, staged, roots, stress)
                self.assertNotEqual(result.returncode, 0, "mutation unexpectedly passed")


if __name__ == "__main__":
    unittest.main()
