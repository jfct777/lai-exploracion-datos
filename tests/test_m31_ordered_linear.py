import copy
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
MODULE_PATH = Path(os.environ.get("M31_ORDERED_LINEAR_SCRIPT_PATH", ROOT / "bin" / "m31_ordered_linear.py"))
CONTRACT_PATH = ROOT / "conf" / "m31_ordered_linear_preregistration.json"
SPEC = importlib.util.spec_from_file_location("m31_ordered_linear", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_frozen_contract_parses_to_typed_values(self):
        parsed = MODULE.load_contract(CONTRACT_PATH)
        self.assertEqual(parsed.experiment_id, "M31_ORDERED_LINEAR_DEV")
        self.assertEqual(parsed.rings_cm, MODULE.EXPECTED_RINGS)
        self.assertEqual(parsed.alphas, MODULE.EXPECTED_ALPHAS)
        self.assertEqual({(d.train_seed, d.evaluation_seed) for d in parsed.directions},
                         {(20260817, 20260818), (20260818, 20260817)})

    def test_rejects_scope_root_and_selector_drift(self):
        mutations = (
            (("scope",), "validation"),
            (("roots_are_not_independent_validation",), False),
            (("roots", "train_root17_test_root18"), [20260817, 20260817]),
            (("rare_universe", "selector"), "TARGET"),
            (("rare_universe", "minor_presence"), "I(state == 1)"),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                payload = copy.deepcopy(self.payload)
                cursor = payload
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                with self.assertRaises(MODULE.ContractError):
                    MODULE.parse_contract(payload)

    def test_rejects_model_leakage_and_endpoint_drift(self):
        mutations = (
            (("ordered_representation", "signed_half_open_rings_cM"), [[0.0, 0.2], [0.2, 1.0]]),
            (("ordered_representation", "sides"), ["right", "left"]),
            (("ordered_representation", "edge_masks"), False),
            (("model", "posthoc_smoothing"), True),
            (("model", "feature_standardization"), "fit_globally"),
            (("model", "inner_cv"), "random rows"),
            (("evaluation", "primary"), "MAE"),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                payload = copy.deepcopy(self.payload)
                cursor = payload
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                with self.assertRaises(MODULE.ContractError):
                    MODULE.parse_contract(payload)

    def test_rejects_definition_support_decision_and_stop_rule_drift(self):
        mutations = (
            (("rare_universe", "definition"), "MAF only"),
            (("rare_universe", "unsupported_in_REF_LAI"), "drop unsupported"),
            (("decision", "GO_NEW_ROOTS"), "one direction is enough"),
            (("stop_rules",), self.payload["stop_rules"][:-1]),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                payload = copy.deepcopy(self.payload)
                cursor = payload
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                with self.assertRaises(MODULE.ContractError):
                    MODULE.parse_contract(payload)

    def test_ring_parser_rejects_gaps_overlap_and_bad_numbers(self):
        for rings in ([[0.1, 0.2]], [[0.0, 0.2], [0.1, 0.3]],
                      [[0.0, 0.1], [0.2, 0.3]], [[0.0, float("inf")]], [[0.1]]):
            with self.subTest(rings=rings), self.assertRaises(MODULE.ContractError):
                MODULE.parse_rings(rings)

    def test_cli_writes_authenticated_selftest_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "report.json"
            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), "--contract", str(CONTRACT_PATH),
                 "--selftest", "--output", str(output)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["contract"]["status"], "PASS")
            self.assertEqual(report["contract"]["sha256"], MODULE.sha256_file(CONTRACT_PATH))
            self.assertTrue(all(value == "PASS" for value in report["known_answers"].values()))

    def test_cli_rejects_partial_frozen_input_bundle(self):
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--genetic-map", str(CONTRACT_PATH)],
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("requires the map and all seven files for both roots", completed.stderr)


class SimplexTest(unittest.TestCase):
    def test_known_vectors_and_matrix_axis(self):
        actual = MODULE.project_simplex([[0.2, 0.3, 0.5], [-1.0, 0.5, 2.0]])
        np.testing.assert_array_equal(actual[0], [0.2, 0.3, 0.5])
        np.testing.assert_array_equal(actual[1], [0.0, 0.0, 1.0])
        by_column = MODULE.project_simplex(np.array([[2.0, 0.2], [-1.0, 0.8]]), axis=0)
        np.testing.assert_allclose(by_column.sum(axis=0), 1.0, atol=0.0, rtol=0.0)
        self.assertTrue(np.all(by_column >= 0.0))

    def test_projection_is_idempotent_and_sums_exactly(self):
        rng = np.random.default_rng(31)
        projected = MODULE.project_simplex(rng.normal(size=(20, 7)))
        np.testing.assert_allclose(projected.sum(axis=1), np.ones(20), atol=1e-15, rtol=0.0)
        np.testing.assert_allclose(MODULE.project_simplex(projected), projected, atol=2e-16, rtol=0.0)

    def test_rejects_scalar_empty_nonfinite_and_bad_axis(self):
        invalid = (1.0, np.empty((2, 0)), [0.0, np.nan])
        for values in invalid:
            with self.subTest(values=np.shape(values)), self.assertRaises(ValueError):
                MODULE.project_simplex(values)
        with self.assertRaises(ValueError):
            MODULE.project_simplex([1.0, 2.0], axis=2)


class RingAggregationTest(unittest.TestCase):
    def test_signed_half_open_boundary_membership_is_exact(self):
        result = MODULE.aggregate_signed_half_open_rings(
            [1.0], [0.5, 0.75, 1.0, 1.25, 1.5], [1, 2, 4, 8, 16],
            ((0.0, 0.25), (0.25, 0.5)), domain_cm=(0.5, 1.5),
        )
        np.testing.assert_array_equal(result.sums, [[[2.0, 1.0], [4.0, 8.0]]])
        np.testing.assert_array_equal(result.observed_site_count, np.ones((1, 2, 2), dtype=int))
        np.testing.assert_array_equal(result.by_observed_site_count, result.sums)
        np.testing.assert_array_equal(result.by_genetic_length, result.sums / 0.25)
        self.assertFalse(result.edge_mask.any())
        self.assertFalse(result.sums.flags.writeable)

    def test_channels_missingness_counts_and_zero_denominator(self):
        values = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
        result = MODULE.aggregate_signed_rings(
            [0.5], [0.1, 0.2, 0.7], values, ((0.0, 0.25),), domain_cm=(0.0, 1.0)
        )
        self.assertEqual(result.sums.shape, (1, 2, 1, 2))
        np.testing.assert_array_equal(result.sums[0, 0, 0], [0.0, 0.0])
        np.testing.assert_array_equal(result.observed_site_count[0, 0, 0], [0, 0])
        np.testing.assert_array_equal(result.by_observed_site_count[0, 0, 0], [0.0, 0.0])
        np.testing.assert_array_equal(result.sums[0, 1, 0], [5.0, 4.0])

    def test_edge_lengths_are_clipped_and_exposed(self):
        result = MODULE.aggregate_signed_rings(
            [0.05], [0.05, 0.2], [2.0, 3.0], ((0.0, 0.1),), domain_cm=(0.0, 0.3)
        )
        self.assertAlmostEqual(result.genetic_length_cm[0, 0, 0], 0.05)
        self.assertAlmostEqual(result.genetic_length_cm[0, 1, 0], 0.1)
        self.assertTrue(result.edge_mask[0, 0, 0])
        self.assertFalse(result.edge_mask[0, 1, 0])

    def test_rejects_order_shape_domain_and_observed_errors(self):
        bad_calls = (
            lambda: MODULE.aggregate_signed_rings([1], [1, 0], [1, 2]),
            lambda: MODULE.aggregate_signed_rings([1], [0, 1], [1]),
            lambda: MODULE.aggregate_signed_rings([1], [0, 1], [1, 2], domain_cm=(1, 1)),
            lambda: MODULE.aggregate_signed_rings([1], [0, 1], [1, np.nan], observed=[True, True]),
        )
        for call in bad_calls:
            with self.subTest(call=call), self.assertRaises((ValueError, MODULE.ContractError)):
                call()


class GroupingAndWeightTest(unittest.TestCase):
    def test_grouping_is_row_order_invariant_and_never_splits_sample(self):
        ids = [f"S{index:02d}" for index in range(10) for _ in range(4)]
        assignment = MODULE.grouped_three_fold_ids(ids, seed=71)
        mapping = {sample: int(assignment[index]) for index, sample in enumerate(ids)}
        reversed_ids = list(reversed(ids))
        reversed_assignment = MODULE.grouped_three_fold_ids(reversed_ids, seed=71)
        self.assertEqual(mapping, {sample: int(fold) for sample, fold in zip(reversed_ids, reversed_assignment)})
        counts = np.bincount([mapping[sample] for sample in sorted(set(ids))], minlength=3)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        for split in MODULE.grouped_three_fold_split(ids, seed=71):
            train_samples = {ids[index] for index in split.train_indices}
            valid_samples = {ids[index] for index in split.validation_indices}
            self.assertFalse(train_samples & valid_samples)

    def test_grouping_rejects_too_few_groups_and_bad_ids(self):
        with self.assertRaises(ValueError):
            MODULE.grouped_three_fold_split(["A", "A", "B"])
        with self.assertRaises(ValueError):
            MODULE.grouped_three_fold_ids(["A", "B", ""])

    def test_weights_sum_to_unique_diploid_individuals(self):
        result = MODULE.normalize_weights([1, 5, 20, 1, 5, 20], ["A", "A", "A", "B", "B", "B"])
        self.assertAlmostEqual(float(result.sum()), 2.0)
        np.testing.assert_allclose(result[2] / result[0], 20.0)
        custom = MODULE.normalize_weights([1, 3], ["A", "B"], target_total=8)
        np.testing.assert_array_equal(custom, [2.0, 6.0])

    def test_weights_reject_invalid_inputs(self):
        for weights, ids in (([-1, 1], ["A", "B"]), ([0, 0], ["A", "B"]),
                             ([1, np.inf], ["A", "B"]), ([1], ["A", "B"])):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                MODULE.normalize_weights(weights, ids)


class RidgeTest(unittest.TestCase):
    def test_exact_known_answer_and_constant_feature(self):
        model = MODULE.fit_weighted_standardized_ridge_residual(
            [[-1.0, 7.0], [1.0, 7.0]], [[-2.0, 2.0], [2.0, -2.0]], ["A", "B"], alpha=2.0
        )
        np.testing.assert_array_equal(model.feature_mean, [0.0, 7.0])
        np.testing.assert_array_equal(model.feature_scale, [1.0, 1.0])
        np.testing.assert_array_equal(model.coefficients, [[1.0, -1.0], [0.0, 0.0]])
        np.testing.assert_array_equal(model.predict_residual([[-1.0, 7.0], [1.0, 7.0]]),
                                      [[-1.0, 1.0], [1.0, -1.0]])

    def test_weighted_training_statistics_and_probability_projection(self):
        features = np.array([[0.0], [2.0], [10.0]])
        residual = np.array([[0.0, 0.0], [2.0, -2.0], [9.0, -9.0]])
        model = MODULE.fit_weighted_standardized_ridge_residual(
            features, residual, ["A", "B", "C"], weights=[1, 1, 0], alpha=1.0
        )
        self.assertAlmostEqual(model.feature_mean[0], 1.0)
        self.assertAlmostEqual(model.residual_intercept[0], 1.0)
        predicted = model.predict([[0.0], [2.0]], [[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_allclose(predicted.sum(axis=1), 1.0, atol=0.0, rtol=0.0)
        self.assertTrue(np.all(predicted >= 0.0))
        self.assertEqual(model.normalized_weight_sum, 3.0)

    def test_corrector_forms_residual_and_rejects_invalid_probabilities(self):
        truth = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        baseline = np.full((3, 2), 0.5)
        model = MODULE.fit_ridge_corrector([[0], [1], [2]], truth, baseline,
                                           ["A", "B", "C"], alpha=0.1)
        self.assertEqual(model.coefficients.shape, (1, 2))
        with self.assertRaises(ValueError):
            MODULE.fit_ridge_corrector([[0], [1], [2]], truth * 2, baseline,
                                       ["A", "B", "C"], alpha=0.1)

    def test_fit_and_predict_guards(self):
        calls = (
            lambda: MODULE.fit_weighted_standardized_ridge_residual([[1]], [[1]], ["A"], alpha=0),
            lambda: MODULE.fit_weighted_standardized_ridge_residual([[np.nan]], [[1]], ["A"]),
            lambda: MODULE.fit_weighted_standardized_ridge_residual([[1], [2]], [[1]], ["A", "B"]),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_complete_known_answer_selftest(self):
        report = MODULE.run_known_answer_selftest()
        self.assertEqual(set(report), {
            "simplex_projection", "signed_half_open_rings", "exact_anchor_is_right",
            "exact_label_aware_boundary_matching",
            "grouped_three_fold", "normalized_weights", "weighted_multivariate_ridge",
            "unequal_panel_frequency_support", "diploid_sham_invariants",
            "synthetic_end_to_end",
        })
        self.assertEqual(set(report.values()), {"PASS"})


class MaterializationAndEndToEndTest(unittest.TestCase):
    def test_file_fixture_crosses_all_loaders_features_and_32_shams(self):
        import tskit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genetic_map_path = root / "map.tsv"
            genetic_map_path.write_text("22\t0\t0.0\n22\t100\t1.0\n", encoding="utf-8")
            sites_path = root / "sites.tsv"
            with sites_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("root_seed", "locus_index", "chrom", "position", "minor_code", "mac", "an", "maf", "freq_carrier_individuals"))
                writer.writerow((20260817, 0, 22, 20, 1, 2, 300, format(2 / 300, ".17g"), 2))
                writer.writerow((20260817, 1, 22, 50, 1, 2, 300, format(2 / 300, ".17g"), 2))
            target_path = root / "target.tsv"
            target_rows = (
                (0, "S0", 1, 0), (0, "S1", 0, 1),
                (1, "S0", 1, 1), (1, "S1", 0, 0),
            )
            with target_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("root_seed", "sample_id", "locus_index", "chrom", "position", "minor_code", "h0_minor_presence", "h1_minor_presence", "minor_dosage", "missing_haplotypes"))
                for locus, sample, h0, h1 in target_rows:
                    writer.writerow((20260817, sample, locus, 22, (20, 50)[locus], 1, h0, h1, h0 + h1, 0))

            flare_path = root / "flare.vcf"
            flare_header = "\n".join((
                "##fileformat=VCFv4.2",
                '##FORMAT=<ID=AN1,Number=1,Type=Integer,Description="Ancestry of first haplotype">',
                '##FORMAT=<ID=AN2,Number=1,Type=Integer,Description="Ancestry of second haplotype">',
                '##FORMAT=<ID=ANP1,Number=3,Type=Float,Description="Ancestry probabilities for first haplotype">',
                '##FORMAT=<ID=ANP2,Number=3,Type=Float,Description="Ancestry probabilities for second haplotype">',
                "##ANCESTRY=<AFR=0,EUR=1,ASIA=2>",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1",
            )) + "\n"
            flare_rows = (
                (10, "0:2:1,0,0:0,0,1", "1:0:0,1,0:1,0,0"),
                (50, "1:2:0,1,0:0,0,1", "1:2:0,1,0:0,0,1"),
                (90, "1:2:0,1,0:0,0,1", "1:2:0,1,0:0,0,1"),
            )
            flare_path.write_text(
                flare_header + "".join(
                    f"22\t{position}\tm{position}\tA\tG\t.\tPASS\t.\tAN1:AN2:ANP1:ANP2\t{s0}\t{s1}\n"
                    for position, s0, s1 in flare_rows
                ),
                encoding="utf-8",
            )
            truth_path = root / "truth.tsv"
            with truth_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("target_haplotype", "chrom", "start_bp", "end_bp_exclusive", "ancestry"))
                writer.writerows((
                    ("S0_h0", 22, 10, 50, "AFR"), ("S0_h0", 22, 50, 91, "EUR"),
                    ("S0_h1", 22, 10, 91, "ASIA"),
                    ("S1_h0", 22, 10, 91, "EUR"),
                    ("S1_h1", 22, 10, 50, "AFR"), ("S1_h1", 22, 50, 91, "ASIA"),
                ))
            pools_path = root / "pools.tsv"
            with pools_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("role", "ancestry", "individual_id", "node_id"))
                for person, ancestry, nodes in (("R0", "AFR", (0, 1)), ("R1", "EUR", (2, 3)), ("R2", "ASIA", (4, 5))):
                    for node in nodes:
                        writer.writerow(("REF_LAI", ancestry, person, node))
            tables = tskit.TableCollection(sequence_length=101)
            for _ in range(6):
                tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
            ancestor = tables.nodes.add_row(time=1)
            for child in range(6):
                tables.edges.add_row(0, 101, ancestor, child)
            for position, mutated_nodes in ((20, (0, 1, 2)), (50, (4,))):
                site = tables.sites.add_row(position, ancestral_state="0")
                for node in mutated_nodes:
                    tables.mutations.add_row(site, node, derived_state="1")
            tables.sort()
            tree_path = root / "fixture.trees"
            tables.tree_sequence().dump(tree_path)

            genetic_map = MODULE.load_genetic_map(genetic_map_path)
            rare = MODULE.load_ordered_rare(sites_path, target_path, 20260817)
            flare = MODULE.load_flare(flare_path)
            truth = MODULE.load_truth(truth_path, flare.samples, 10, 91)
            binding = MODULE.validate_phase_binding(rare, flare, truth)
            self.assertEqual(binding["FLARE_ANP1_AN1"], "truth_h0")
            self.assertEqual(binding["post_truth_haplotype_swap"], "forbidden")
            marker_positions = np.asarray([locus[1] for locus in flare.loci])
            marker_cm = genetic_map.cm_at(marker_positions)
            rare_cm = genetic_map.cm_at(rare.positions)
            truth_probabilities = MODULE.truth_at_markers(truth, flare.samples, marker_positions)
            ref_dosage, people, labels = MODULE.load_ref_minor_dosage(tree_path, pools_path, rare, genetic_map)
            support, no_support = MODULE.ancestry_support(ref_dosage, labels)
            target_before = rare.hap_presence.copy()
            real = MODULE.materialize_sample_features(
                marker_cm, rare_cm, flare.probabilities[:, 0], truth_probabilities[:, 0],
                rare.hap_presence[:, 0], support, no_support,
            )
            for replicate in range(32):
                sham_labels = MODULE.permute_diploid_labels(labels, 20260817, replicate)
                self.assertCountEqual(sham_labels, labels)
                sham_support, sham_no_support = MODULE.ancestry_support(ref_dosage, sham_labels)
                sham = MODULE.materialize_sample_features(
                    marker_cm, rare_cm, flare.probabilities[:, 0], truth_probabilities[:, 0],
                    rare.hap_presence[:, 0], sham_support, sham_no_support,
                )
                np.testing.assert_array_equal(sham.arms["C"], real.arms["C"])
                np.testing.assert_array_equal(sham.arms["L"], real.arms["L"])
                self.assertEqual(sham.arms["D"].shape, real.arms["D"].shape)
                self.assertEqual(sham.arms["H"].shape, real.arms["H"].shape)
            np.testing.assert_array_equal(rare.hap_presence, target_before)
            self.assertEqual(people, ("R0", "R1", "R2"))

    def test_flare_header_must_bind_anp1_and_anp2_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.vcf"
            path.write_text(
                "##fileformat=VCFv4.2\n"
                '##FORMAT=<ID=AN1,Number=1,Type=Integer,Description="Ancestry of first haplotype">\n'
                '##FORMAT=<ID=AN2,Number=1,Type=Integer,Description="Ancestry of second haplotype">\n'
                '##FORMAT=<ID=ANP1,Number=3,Type=Float,Description="Ancestry probabilities">\n'
                '##FORMAT=<ID=ANP2,Number=3,Type=Float,Description="Ancestry probabilities for second haplotype">\n'
                "##ANCESTRY=<AFR=0,EUR=1,ASIA=2>\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "bind first/second haplotypes exactly"):
                MODULE.load_flare(path)

    def test_boundary_dp_beats_greedy_and_is_transition_label_aware(self):
        truth = [MODULE.Boundary(0.10, 0, 1), MODULE.Boundary(0.20, 0, 1)]
        predicted = [MODULE.Boundary(0.00, 0, 1), MODULE.Boundary(0.11, 0, 1)]
        pairs = MODULE.ordered_boundary_pairs(truth, predicted, 0.11)
        self.assertEqual([(left, right) for left, right, _ in pairs], [(0, 0), (1, 1)])
        self.assertAlmostEqual(sum(distance for _, _, distance in pairs), 0.19)
        minimum = MODULE.ordered_boundary_pairs(
            [MODULE.Boundary(0.10, 0, 1), MODULE.Boundary(0.30, 0, 1)],
            [MODULE.Boundary(0.00, 0, 1), MODULE.Boundary(0.11, 0, 1), MODULE.Boundary(0.31, 0, 1)],
            0.25,
        )
        self.assertEqual([(left, right) for left, right, _ in minimum], [(0, 1), (1, 2)])
        self.assertEqual(
            MODULE.ordered_boundary_pairs(
                [MODULE.Boundary(0.1, 0, 1)], [MODULE.Boundary(0.1, 1, 0)], 0.2
            ),
            [],
        )

    def test_exact_anchor_is_in_right_ring_and_haplotype_binding_is_not_swapped(self):
        marker_cm = np.array([0.0, 0.1, 0.2])
        rare_cm = np.array([0.1])
        baseline = np.array([
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
            [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]],
            [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]],
        ])
        truth = np.eye(3)[[[0, 1], [0, 1], [1, 0]]]
        support = np.array([[1.0, 0.0, 0.0]])
        result = MODULE.materialize_sample_features(
            marker_cm, rare_cm, baseline, truth, np.array([[1.0, 0.0]]),
            support, np.array([False]), ((0.0, 0.1),),
        )
        h_names = result.feature_names["H"]
        afr_right = h_names.index("haplotype_support.per_observed_site.right.r0.AFR")
        # The exact marker 0.1 is in its own right ring, never its left ring.
        self.assertEqual(result.arms["H"][1, 0, afr_right], 1.0)
        self.assertEqual(result.arms["H"][1, 1, afr_right], 0.0)
        self.assertEqual(result.baseline[0, 0, 0], 0.8)  # ANP1 -> h0
        self.assertEqual(result.baseline[0, 1, 1], 0.8)  # ANP2 -> h1

    def test_ancestry_support_uses_within_ancestry_frequency(self):
        dosage = np.array([[2, 0, 1, 0]])
        support, unsupported = MODULE.ancestry_support(dosage, ["AFR", "EUR", "EUR", "ASIA"])
        np.testing.assert_allclose(support, [[0.8, 0.2, 0]])
        self.assertFalse(unsupported[0])

    def test_sham_is_diploid_deterministic_and_preserves_counts(self):
        labels = ["AFR", "AFR", "EUR", "EUR", "ASIA", "ASIA"]
        first = MODULE.permute_diploid_labels(labels, 20260817, 3)
        second = MODULE.permute_diploid_labels(labels, 20260817, 3)
        self.assertEqual(first, second)
        self.assertCountEqual(first, labels)
        self.assertNotEqual(first, MODULE.permute_diploid_labels(labels, 20260817, 4))

    def test_grouped_cv_and_held_individual_known_answer(self):
        report = MODULE.run_synthetic_end_to_end()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["split_unit"], "complete_diploid_individual")
        self.assertLess(report["corrected_haplotype_brier"], report["baseline_haplotype_brier"])
        self.assertEqual(report["boundary_f1_0.2cM"], 1.0)

    def test_all_preregistered_metrics_and_deterministic_individual_bootstrap(self):
        marker_cm = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        truth_labels = np.array([0, 0, 1, 1, 1, 1])
        predicted_labels = np.array([0, 0, 0, 0, 1, 1])
        truth_rows = []
        predicted_rows = []
        ids = []
        for sample in ("A", "B"):
            for truth_label, predicted_label in zip(truth_labels, predicted_labels):
                for _hap in (0, 1):
                    truth_rows.append(np.eye(3)[truth_label])
                    predicted_rows.append(np.eye(3)[predicted_label])
                    ids.append(sample)
        summary, individual = MODULE.evaluate_haplotype_predictions(
            np.asarray(predicted_rows), np.asarray(truth_rows), marker_cm, ids
        )
        for tolerance in ("0.1cM", "0.2cM", "0.5cM"):
            self.assertIn(f"boundary_f1_{tolerance}", summary)
            self.assertIn(f"false_transitions_per_cM_{tolerance}", summary)
            self.assertIn(f"matched_boundary_median_{tolerance}", summary)
            self.assertIn(f"matched_boundary_p90_{tolerance}", summary)
        self.assertEqual(summary["boundary_f1_0.1cM"], 0.0)
        self.assertEqual(summary["boundary_f1_0.2cM"], 1.0)
        self.assertGreater(summary["macro_ancestry_dose_mae"], 0.0)
        self.assertEqual(summary["ancestry_dose_mae_ASIA"], 0.0)
        self.assertIn("diploid_macro_f1_fixed_six", summary)
        first = MODULE.bootstrap_individual_metrics(individual)
        second = MODULE.bootstrap_individual_metrics(individual)
        self.assertEqual(first, second)
        self.assertEqual(first["replicates"], 10000)
        self.assertEqual(first["unit"], "complete_diploid_individual")

    def test_frozen_input_authentication_rejects_any_hash_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map"
            path.write_text("not the frozen map\n", encoding="utf-8")
            roots = {
                root: {key: path for key in ("sites", "target", "tree", "pools", "truth", "flare_vcf", "flare_audit")}
                for root in MODULE.ROOTS
            }
            with self.assertRaisesRegex(MODULE.ContractError, "SHA-256 mismatch"):
                MODULE.authenticate_frozen_run_inputs(path, roots)


if __name__ == "__main__":
    unittest.main()
