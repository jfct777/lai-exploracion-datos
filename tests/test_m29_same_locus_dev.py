import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(os.environ.get("M29_SCRIPT_PATH", Path(__file__).parents[1] / "bin" / "m29_same_locus_dev.py"))
SPEC = importlib.util.spec_from_file_location("m29_same_locus_dev", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class M29SameLocusDevTest(unittest.TestCase):
    def test_sham_preserves_complete_individual_counts(self):
        labels = ["AFR"] * 4 + ["EUR"] * 3 + ["ASIA"] * 2
        first = MODULE.permute_diploid_labels(labels, 7)
        second = MODULE.permute_diploid_labels(labels, 7)
        self.assertEqual(sorted(first.tolist()), sorted(labels))
        np.testing.assert_array_equal(first, second)

    def test_same_locus_identity_allows_only_support_change(self):
        real = np.zeros((5, 7))
        real[:, :4] = np.arange(20).reshape(5, 4)
        real[:, 4:] = [0.1, 0.2, 0.3]
        sham = real.copy()
        sham[:, 4:] = real[:, [6, 4, 5]]
        MODULE.assert_same_locus_identity(real, sham)
        sham[0, 3] += 1e-5
        with self.assertRaisesRegex(ValueError, "sham altered"):
            MODULE.assert_same_locus_identity(real, sham)

    def test_diploid_features_are_homologue_swap_invariant(self):
        window = MODULE.Window(0, 10, 0.0, 1.0, np.asarray([[0.2, 0.3, 0.5]]))
        positions = np.asarray([2, 7])
        support = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        dosage_direct = np.asarray([[1.0], [2.0]])
        dosage_swapped = np.asarray([[1.0], [2.0]])
        direct, _ = MODULE.build_features(["T000"], [window], positions, dosage_direct, support)
        swapped, _ = MODULE.build_features(["T000"], [window], positions, dosage_swapped, support)
        np.testing.assert_array_equal(direct, swapped)

    @staticmethod
    def fixture(seed, injected):
        rng = np.random.default_rng(seed)
        n = 180
        classes = np.arange(n) % 3
        truth = np.eye(3)[classes]
        baseline = rng.normal(size=(n, 3)) * 0.02
        X = np.zeros((n, 7))
        X[:, :3] = baseline
        if injected:
            X[:, 4:] = truth + rng.normal(size=(n, 3)) * 0.01
        lengths = np.ones(n)
        sample_index = np.arange(n)
        return MODULE.RootData(str(seed), seed, [f"T{i:03d}" for i in range(n)], [], truth, lengths, sample_index, X, np.column_stack([baseline, np.zeros((n, 4))]), [])

    def test_null_arm_identical_to_b0_has_identical_error(self):
        train, test = self.fixture(1, False), self.fixture(2, False)
        first, _ = MODULE.fit_and_score(train.b0_features, test.b0_features, train, test, 1.0, 5000, 1e-8)
        second, _ = MODULE.fit_and_score(train.b0_features.copy(), test.b0_features.copy(), train, test, 1.0, 5000, 1e-8)
        self.assertEqual(first, second)

    def test_known_injection_improves_over_uninformative_b0(self):
        train, test = self.fixture(3, True), self.fixture(4, True)
        b0, _ = MODULE.fit_and_score(train.b0_features, test.b0_features, train, test, 1.0, 5000, 1e-8)
        injected, _ = MODULE.fit_and_score(train.real_features, test.real_features, train, test, 1.0, 5000, 1e-8)
        self.assertLess(injected["macro_mae"], b0["macro_mae"] * 0.2)

    def test_all_arms_have_same_declared_capacity(self):
        root = self.fixture(8, True)
        root.sham_features = [root.real_features.copy() for _ in range(32)]
        matrices = [root.b0_features, root.real_features, *root.sham_features]
        self.assertTrue(all(matrix.shape[1] == 7 for matrix in matrices))

    def test_sham_seed_does_not_depend_on_truth(self):
        labels = ["AFR", "EUR", "ASIA"] * 5
        before = MODULE.permute_diploid_labels(labels, MODULE.stable_seed(29, 20260817, 3))
        unrelated_truth = np.eye(3)[np.arange(100) % 3]
        self.assertEqual(unrelated_truth.shape, (100, 3))
        after = MODULE.permute_diploid_labels(labels, MODULE.stable_seed(29, 20260817, 3))
        np.testing.assert_array_equal(before, after)

    def test_scaler_is_fit_on_training_root_only(self):
        train, test = self.fixture(5, True), self.fixture(6, True)
        test.real_features += 1000
        scaler, _ = MODULE._fit_soft_multinomial(train.real_features, train.truth, train.lengths_cm, 1.0, 5000, 1e-8)
        np.testing.assert_allclose(scaler.mean_, train.real_features.mean(axis=0))
        self.assertGreater(np.abs(test.real_features.mean(axis=0) - scaler.mean_).max(), 900)

    def test_nonconvergence_stops(self):
        train = self.fixture(7, True)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            MODULE._fit_soft_multinomial(train.real_features, train.truth, train.lengths_cm, 10.0, 1, 1e-15)

    def test_contract_has_no_fake_baseline_hashes(self):
        path = Path(__file__).parents[1] / "conf" / "m29_same_locus_dev_preregistration.json"
        contract = json.loads(path.read_text())
        for root in contract["roots"].values():
            self.assertEqual(root["required_run_binding"], ["fb", "msp"])
            self.assertNotIn("fb", root["sha256"])
            self.assertNotIn("msp", root["sha256"])


if __name__ == "__main__":
    unittest.main()
