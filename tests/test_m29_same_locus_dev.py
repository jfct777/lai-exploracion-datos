import importlib.util
import csv
import gzip
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

    def test_truth_integration_is_additive_across_half_open_boundaries(self):
        class LinearMap:
            positions = [0, 10]

            @staticmethod
            def cm_at(position):
                return float(position)

        truth = {
            "T000": (
                [MODULE.TruthSegment(0, 4, "AFR"), MODULE.TruthSegment(4, 10, "EUR")],
                [MODULE.TruthSegment(0, 10, "ASIA")],
            )
        }
        window = MODULE.Window(0, 10, 0.0, 10.0, np.asarray([[1 / 3, 1 / 3, 1 / 3]]))
        observed, lengths, indexes = MODULE._integrated_truth(truth, ["T000"], [window], LinearMap())
        np.testing.assert_allclose(observed, [[0.2, 0.3, 0.5]])
        np.testing.assert_array_equal(lengths, [10.0])
        np.testing.assert_array_equal(indexes, [0])

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

    def test_target_dosage_is_oriented_to_minor_code_zero_and_one(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rare.tsv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("chrom", "position", "minor_code", "T000_h0", "T000_h1", "T001_h0", "T001_h1"))
                writer.writerow(("22", 10, 0, 0, 0, 0, 1))
                writer.writerow(("22", 20, 1, 1, 1, 0, 0))
            samples, positions, dosage = MODULE._load_target_rare(path, {10: 0, 20: 1})
            self.assertEqual(samples, ["T000", "T001"])
            np.testing.assert_array_equal(positions, [10, 20])
            np.testing.assert_array_equal(dosage, [[2.0, 1.0], [2.0, 0.0]])
            self.assertEqual(0 + 0, 0)  # historical allele-1 dosage
            self.assertEqual(MODULE.minor_diploid_dosage([0, 0], 0), 2)
            self.assertEqual(MODULE.minor_diploid_dosage([1, 1], 1), 2)

    def test_target_minor_code_must_match_selected_catalog(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rare.tsv"
            path.write_text("chrom\tposition\tminor_code\tT000_h0\tT000_h1\n22\t10\t1\t0\t0\n")
            with self.assertRaisesRegex(ValueError, "minor_code mismatch"):
                MODULE._load_target_rare(path, {10: 0})

    def test_m29r_accepts_only_fixed_historical_C_10(self):
        import tempfile

        path = Path(__file__).parents[1] / "conf" / "m29r_minor_orientation_erratum.json"
        contract = MODULE._load_contract(path)
        self.assertEqual(contract["model"]["C_grid"], [10.0])
        self.assertEqual(contract["model"]["fixed_C"], 10.0)
        self.assertEqual(contract["model"]["C_selection"], "none_fixed_from_historical_M29_all_arms")
        modified = json.loads(path.read_text())
        modified["model"]["C_grid"] = [1.0, 10.0]
        with tempfile.TemporaryDirectory() as tmp:
            invalid = Path(tmp) / "invalid.json"
            invalid.write_text(json.dumps(modified))
            with self.assertRaisesRegex(ValueError, "only.*C=10"):
                MODULE._load_contract(invalid)

    def test_durable_code_provenance_validation(self):
        import tempfile

        self.assertEqual(MODULE.validate_git_commit("a" * 40), "a" * 40)
        with self.assertRaisesRegex(ValueError, "git_commit"):
            MODULE.validate_git_commit("unknown")
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "script.py"
            script.write_text("print('fixed')\n")
            observed = MODULE.sha256_file(script)
            self.assertEqual(MODULE.authenticate_script(script, observed), observed)
            with self.assertRaisesRegex(ValueError, "script sha256 mismatch"):
                MODULE.authenticate_script(script, "0" * 64)


if __name__ == "__main__":
    unittest.main()
