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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))


def load(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


MODELS = load("m33_t0a_models", "bin/m33_t0a_models.py")
PREFLIGHT = load("m33_t1_preflight", "bin/m33_t1_preflight.py")
BACKWARD = load("m33_t1_backward", "bin/m33_t1_backward.py")
COMPARE = load("m33_t1_compare", "bin/m33_t1_compare.py")
AUTH = load("m33_t1_source_auth", "bin/m33_t1_source_auth.py")

COMMIT = "a" * 40
OCI = PREFLIGHT.EXPECTED_OCI


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Namespace:
    def __init__(self, **values):
        self.__dict__.update(values)


class M33T1Tests(unittest.TestCase):
    def small_fixture(self):
        patch = mock.patch.object(BACKWARD, "ROWS", 12)
        patch.start()
        self.addCleanup(patch.stop)
        row = torch.arange(12, dtype=torch.float32)[:, None, None]
        position = torch.arange(8, dtype=torch.float32)[None, :, None]
        channel = torch.arange(13, dtype=torch.float32)[None, None, :]
        tokens = ((row * 17 + position * 13 + channel * 7).remainder(101) - 50) / 50
        lengths = torch.tensor([0, 1, 4, 8], dtype=torch.int64).repeat(3)
        mask = (torch.arange(8)[None, :] < lengths[:, None]).to(torch.float32)
        patterns = torch.tensor([[1.0, 0.0, 0.0], [1e-8, 0.4, 0.59999999],
                                 [1 / 3, 1 / 3, 1 / 3], [0.0, 0.2, 0.8]])
        f0 = patterns[torch.arange(12) % 4][:, None, :].repeat(1, 2, 1)
        f0[:, 1] = f0[:, 1].roll(1, dims=1)
        labels = torch.arange(24, dtype=torch.int64).reshape(12, 2).remainder(3)
        weights = torch.ones((12, 2), dtype=torch.float32)
        weights[1::3] = 2.0
        return tokens.contiguous(), mask.contiguous(), f0.contiguous(), labels, weights

    def test_contract_freezes_only_the_synthetic_backward_gate(self):
        digest, contract = PREFLIGHT.validate_contract(ROOT / "conf/m33_t1_contract.json")
        self.assertEqual(len(digest), 64)
        self.assertEqual(contract["scope"]["subcases"],
                         ["production_zero_head", "private_nonzero_probe_head"])
        self.assertEqual(contract["execution"]["maximum_parallel_processes"], 1)
        self.assertEqual(contract["execution"]["process_memory_gib"], 8)
        self.assertFalse(contract["scope"]["scientific_evidence"])
        self.assertFalse(contract["scope"]["training_or_model_selection"])
        self.assertEqual(contract["synthetic_stress"]["known_answer_boundary_loss_weights"],
                         [0.0, 1.0])

    def test_source_inventory_is_exact(self):
        self.assertEqual(AUTH.REQUIRED_SOURCES, {
            "bin/m33_t0a_models.py", "bin/m33_t1_source_auth.py",
            "bin/m33_t1_preflight.py", "bin/m33_t1_backward.py",
            "bin/m33_t1_compare.py", "conf/m33_pre4_preregistration.json",
            "conf/m33_t1_contract.json", "conf/m33_t1.config",
            "modules/33_T1_BACKWARD_DRY_RUN.nf", "workflows/m33_t1.nf",
            "containers/m33-t0a/Dockerfile", "tests/test_m33_t1.py",
        })

    def test_preflight_source_auth_is_bound_to_executed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "staged"
            (source_root / "bin").mkdir(parents=True)
            (source_root / "conf").mkdir()
            hashes = {}
            for relative in AUTH.REQUIRED_SOURCES:
                hashes[relative] = "0" * 64
            for relative in ("bin/m33_t1_preflight.py", "conf/m33_t1_contract.json"):
                target = source_root / relative
                target.write_bytes((ROOT / relative).read_bytes())
                hashes[relative] = sha256_file(target)
            auth = Path(directory) / "source_auth.json"
            auth.write_text(json.dumps({
                "stage": "M33_T1_SOURCE_AUTH",
                "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
                "git_commit": COMMIT,
                "source_sha256": hashes,
            }) + "\n", encoding="utf-8")
            PREFLIGHT.validate_source_auth(auth, COMMIT, source_root)
            (source_root / "bin/m33_t1_preflight.py").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "preflight source differs"):
                PREFLIGHT.validate_source_auth(auth, COMMIT, source_root)

    def test_nextflow_is_sequential_containerized_and_receipts_only(self):
        workflow = (ROOT / "workflows/m33_t1.nf").read_text(encoding="utf-8")
        module = (ROOT / "modules/33_T1_BACKWARD_DRY_RUN.nf").read_text(encoding="utf-8")
        config = (ROOT / "conf/m33_t1.config").read_text(encoding="utf-8")
        combined = (workflow + module + config).lower()
        self.assertIn("maxforks 1", module.lower())
        self.assertIn("maxforks = 1", config.lower())
        self.assertIn("memory '8 gb'", module.lower())
        self.assertIn("--memory 8g", config.lower())
        self.assertIn("--network none", combined)
        self.assertEqual(workflow.count("tuple('local_linear'"), 2)
        self.assertEqual(workflow.count("tuple('small_residual_cnn_1d'"), 2)
        for forbidden in ("optimizer.step", "torch.save", ".pt'", ".pth'"):
            self.assertNotIn(forbidden, combined)

    def test_fixture_has_explicit_padding_simplex_and_balanced_targets(self):
        tokens, mask, f0, labels, weights = self.small_fixture()
        self.assertEqual(tokens.shape, (12, 8, 13))
        self.assertEqual(mask.shape, (12, 8))
        self.assertTrue(torch.any(mask == 0).item())
        self.assertTrue(torch.any(mask == 1).item())
        self.assertLessEqual(float(torch.max(torch.abs(f0.sum(2) - 1))), 5e-6)
        self.assertTrue(torch.any(f0 == 0).item())
        self.assertEqual(torch.bincount(labels.flatten(), minlength=3).tolist(), [8, 8, 8])
        self.assertGreater(float(weights.sum()), 0)

    def test_full_contract_fixture_is_balanced_and_exact_size(self):
        tokens, mask, f0, labels, weights = BACKWARD.synthetic_fixture()
        self.assertEqual(tokens.shape, (2048, 128, 13))
        self.assertEqual(mask.shape, (2048, 128))
        self.assertEqual(f0.shape, (2048, 2, 3))
        self.assertEqual(labels.shape, weights.shape)
        counts = torch.bincount(labels.flatten(), minlength=3)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        self.assertEqual(int(mask.sum()), 98_816)
        self.assertTrue(torch.all(weights.reshape(8, 256, 2)[:, 0] == 1).item())
        self.assertEqual(int((weights > 1).sum()), 112)
        self.assertEqual(float(weights.sum()), 4208.0)
        transitions = (weights.reshape(8, 256, 2) > 1).sum(dim=1)
        self.assertTrue(torch.all(transitions == 7).item())

    def test_beta_zero_loss_matches_direct_cross_entropy(self):
        _tokens, _mask, f0, labels, _weights = self.small_fixture()
        uniform = torch.ones_like(labels, dtype=torch.float32)
        delta = torch.zeros_like(f0)
        observed = BACKWARD.weighted_cross_entropy(f0, delta, labels, uniform)
        logits = torch.log(f0.clamp_min(1e-7))
        selected = torch.log_softmax(logits, dim=2).gather(
            2, labels.unsqueeze(2)).squeeze(2)
        expected = -selected.mean()
        self.assertTrue(torch.equal(observed, expected))

    def test_beta_one_loss_and_gradient_match_independent_known_answer(self):
        with mock.patch.object(BACKWARD, "ROWS", 1):
            f0 = torch.tensor([[[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]])
            delta = torch.zeros_like(f0, requires_grad=True)
            labels = torch.tensor([[0, 2]])
            weights = torch.tensor([[1.0, 2.0]])
            loss = BACKWARD.weighted_cross_entropy(f0, delta, labels, weights)
            expected = (-torch.log(torch.tensor(0.2)) -
                        2 * torch.log(torch.tensor(0.7))) / 3
            self.assertTrue(torch.allclose(loss, expected, atol=1e-7, rtol=0))
            loss.backward()
            expected_gradient = torch.stack((
                (f0[0, 0] - torch.tensor([1.0, 0.0, 0.0])) / 3,
                2 * (f0[0, 1] - torch.tensor([0.0, 0.0, 1.0])) / 3,
            )).unsqueeze(0)
            self.assertTrue(torch.allclose(delta.grad, expected_gradient, atol=1e-7, rtol=0))

    def test_production_and_probe_gradient_patterns_for_both_models(self):
        tokens, mask, f0, labels, weights = self.small_fixture()
        for family in MODELS.FAMILIES:
            with self.subTest(family=family):
                production = BACKWARD.run_subcase(
                    family, False, tokens, mask, f0, labels, weights)
                probe = BACKWARD.run_subcase(
                    family, True, tokens, mask, f0, labels, weights)
                self.assertGreater(production["global_gradient_norm"], 0)
                self.assertGreater(probe["valid_input_gradient_norm"], 0)
                self.assertEqual(probe["padding_input_gradient_max_abs"], 0)
                self.assertTrue(all(value > 0 for value in
                                    probe["stage_gradient_norms"].values()))
                if family == "small_residual_cnn_1d":
                    for stage in ("stem", "block1", "block2", "head1"):
                        self.assertEqual(production["stage_gradient_norms"][stage], 0)
                for row in (production, probe):
                    self.assertEqual(row["parameter_value_sha256_before"],
                                     row["parameter_value_sha256_after"])

    def test_gradient_validator_rejects_missing_and_nonfinite_gradients(self):
        model = MODELS.build_model("local_linear")
        with self.assertRaisesRegex(ValueError, "missing gradient"):
            BACKWARD.stage_gradient_norms(model)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite gradient"):
            BACKWARD.stage_gradient_norms(model)

    def write_t0b_fixture(self, base: Path):
        children, inventory = [], []
        for index in range(5):
            child = base / f"child.{index}.json"
            child.write_text(json.dumps({"index": index}) + "\n", encoding="utf-8")
            children.append(child)
            inventory.append({"name": child.name, "sha256": sha256_file(child)})
        aggregate = base / "aggregate.json"
        aggregate.write_text(json.dumps({
            "stage": "M33_T0B_FULL_CHR22_COMPARISON",
            "status": "PASS_T0B_FULL_CHR22_TECHNICAL_ONLY",
            "marker_count": 79_791, "t1_open": False, "truth_read": False,
            "training": False, "gradients": False, "optimizer": False,
            "child_receipts": inventory,
        }) + "\n", encoding="utf-8")
        return aggregate, children

    def test_t0b_anchor_rejects_missing_or_tampered_children(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate, children = self.write_t0b_fixture(Path(directory))
            with mock.patch.object(PREFLIGHT, "EXPECTED_T0B_SHA256", sha256_file(aggregate)):
                PREFLIGHT.validate_t0b(aggregate, children)
                with self.assertRaisesRegex(ValueError, "closure differs"):
                    PREFLIGHT.validate_t0b(aggregate, children[:-1])
                children[0].write_text("tampered\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "inventory differs"):
                    PREFLIGHT.validate_t0b(aggregate, children)

    def child_receipt(self, family: str, repetition: int) -> dict:
        stages = ["head"] if family == "local_linear" else [
            "stem", "block1", "block2", "head1", "head2"]
        subcases = []
        for name in ("production_zero_head", "private_nonzero_probe_head"):
            norms = {stage: 1.0 for stage in stages}
            if family == "small_residual_cnn_1d" and name == "production_zero_head":
                norms.update({stage: 0.0 for stage in ("stem", "block1", "block2", "head1")})
            subcases.append({
                "name": name, "loss": 1.0, "probability_sha256": "1" * 64,
                "delta_sha256": "2" * 64, "gradient_sha256": "3" * 64,
                "stage_gradient_norms": norms, "global_gradient_norm": 1.0,
                "valid_input_gradient_norm": 1.0 if name.startswith("private") else 0.0,
                "padding_input_gradient_max_abs": 0.0,
                "peak_rss_fraction_after": 0.5,
                "parameter_value_sha256_before": "4" * 64,
                "parameter_value_sha256_after": "4" * 64,
                "parameter_count": 90 if family == "local_linear" else 1651,
                "parameter_shape_sha256": "8" * 64,
            })
        row = {
            "stage": "M33_T1_BACKWARD_DRY_RUN",
            "status": "PASS_T1_BACKWARD_CASE_TECHNICAL_ONLY_NON_CONSUMABLE",
            "model_family": family, "repetition": repetition,
            "people": 8, "central_markers": 256, "rows": 2048, "channels": 13,
            "maximum_context": 128, "padded_tokens": 262_144,
            "valid_tokens": 98_816, "context_lengths": [0, 1, 64, 128],
            "synthetic_target_class_counts": [1366, 1365, 1365],
            "transition_count": 112, "transition_count_per_person_haplotype": 7,
            "boundary_weight_sum": 4208.0, "first_marker_weight_max": 1.0,
            "boundary_weight_beta": 1.0, "subcases": subcases,
            "memory_limit_gib": 8.0, "peak_rss_fraction": 0.5,
            "memory_warning_fraction": 0.70, "memory_stop_fraction": 0.80,
            "memory_warning": False, "device": "cpu", "vram_applicable": False,
            "torch_version": torch.__version__, "oci_image": OCI,
            "implementation_commit": COMMIT, "source_auth_sha256": "5" * 64,
            "contract_sha256": "6" * 64, "preflight_receipt_sha256": "7" * 64,
            "synthetic_only": True,
        }
        for field in COMPARE.FALSE_FIELDS:
            row[field] = False
        return row

    def test_comparator_requires_exact_process_repetitions(self):
        first = self.child_receipt("local_linear", 0)
        second = self.child_receipt("local_linear", 1)
        self.assertEqual(
            [first[field] for field in COMPARE.EXACT_REPEAT_FIELDS],
            [second[field] for field in COMPARE.EXACT_REPEAT_FIELDS])
        second["subcases"][1]["gradient_sha256"] = "9" * 64
        mismatches = [field for field in COMPARE.EXACT_REPEAT_FIELDS
                      if first[field] != second[field]]
        self.assertEqual(mismatches, [])
        subcase_mismatches = [field for field in COMPARE.EXACT_SUBCASE_FIELDS
                              if first["subcases"][1][field] !=
                              second["subcases"][1][field]]
        self.assertEqual(subcase_mismatches, ["gradient_sha256"])

    def test_comparator_allows_nondeterministic_rss_but_not_gradient(self):
        first = self.child_receipt("small_residual_cnn_1d", 0)
        second = self.child_receipt("small_residual_cnn_1d", 1)
        second["subcases"][0]["peak_rss_fraction_after"] = 0.61
        self.assertTrue(all(first[field] == second[field]
                            for field in COMPARE.EXACT_REPEAT_FIELDS))
        self.assertTrue(all(first["subcases"][0][field] ==
                            second["subcases"][0][field]
                            for field in COMPARE.EXACT_SUBCASE_FIELDS))

    def comparator_fixture(self, base: Path):
        source_root = base / "staged"
        (source_root / "bin").mkdir(parents=True)
        (source_root / "conf").mkdir()
        hashes = {relative: "0" * 64 for relative in AUTH.REQUIRED_SOURCES}
        for relative in ("bin/m33_t1_compare.py", "bin/m33_t1_preflight.py",
                         "conf/m33_t1_contract.json"):
            target = source_root / relative
            target.write_bytes((ROOT / relative).read_bytes())
            hashes[relative] = sha256_file(target)
        source_auth = base / "source_auth.json"
        source_auth.write_text(json.dumps({
            "stage": "M33_T1_SOURCE_AUTH",
            "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
            "git_commit": COMMIT,
            "source_sha256": hashes,
        }) + "\n", encoding="utf-8")
        contract = source_root / "conf/m33_t1_contract.json"
        preflight_receipt = base / "preflight.json"
        preflight_receipt.write_text(json.dumps({
            "status": "PASS_T1_PREFLIGHT_SYNTHETIC_ONLY",
            "contract_sha256": sha256_file(contract),
            "source_auth_sha256": sha256_file(source_auth),
            "implementation_commit": COMMIT,
            "oci_image": OCI,
        }) + "\n", encoding="utf-8")
        receipts = []
        for family in MODELS.FAMILIES:
            for repetition in (0, 1):
                payload = self.child_receipt(family, repetition)
                payload["source_auth_sha256"] = sha256_file(source_auth)
                payload["contract_sha256"] = sha256_file(contract)
                payload["preflight_receipt_sha256"] = sha256_file(preflight_receipt)
                path = base / f"{family}.rep{repetition}.json"
                path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
                receipts.append(path)
        return receipts, source_auth, source_root, contract, preflight_receipt

    def test_comparator_executes_full_closure_and_rejects_memory_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.comparator_fixture(Path(directory))
            receipt = COMPARE.compare_receipts(
                values[0], values[1], values[2], COMMIT, OCI, values[3], values[4])
            self.assertEqual(receipt["status"],
                             "PASS_T1_BACKWARD_DRY_RUN_TECHNICAL_ONLY")
            changed = json.loads(values[0][0].read_text())
            changed["memory_limit_gib"] = 16.0
            values[0][0].write_text(json.dumps(changed) + "\n")
            with self.assertRaisesRegex(ValueError, "memory gate differs"):
                COMPARE.compare_receipts(
                    values[0], values[1], values[2], COMMIT, OCI, values[3], values[4])
            changed["memory_limit_gib"] = 8.0
            changed["subcases"][0].pop("parameter_value_sha256_before")
            values[0][0].write_text(json.dumps(changed) + "\n")
            with self.assertRaisesRegex(ValueError, "gradient or parameter gate differs"):
                COMPARE.compare_receipts(
                    values[0], values[1], values[2], COMMIT, OCI, values[3], values[4])

    def test_comparator_rejects_false_cnn_upstream_pass(self):
        receipt = self.child_receipt("small_residual_cnn_1d", 0)
        receipt["subcases"][0]["stage_gradient_norms"]["stem"] = 1.0
        production = receipt["subcases"][0]["stage_gradient_norms"]
        with self.assertRaisesRegex(ValueError, "zero-head gradient pattern differs"):
            COMPARE.require(
                production["head2"] > 0 and
                all(production[name] == 0.0
                    for name in ("stem", "block1", "block2", "head1")),
                "T1 production CNN zero-head gradient pattern differs")

    def test_preflight_rejects_insufficient_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            path.write_text("MemAvailable: 9437184 kB\n", encoding="ascii")
            self.assertLess(PREFLIGHT.mem_available_gib(path), 10)


if __name__ == "__main__":
    unittest.main()
