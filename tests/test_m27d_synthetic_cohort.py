"""Known-answer tests for the cohort where endogamy and pedigree are separate facts.

A synthetic cohort is only worth running a pipeline over if its own truth is right.  The
checks below are the ones that would silently invalidate every downstream number: children
that do not actually inherit from their parents, a composition formula that overstates the
truth by the size of the contrast, a scenario that claims no drift while drawing drifted
frequencies, and a seed that does not reproduce.

The scoring is tested against hand-built tables rather than against pipeline output, so a
regression in the scorer cannot hide behind a regression in the estimator.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))

from m27d_coancestry_screen import SCENARIOS, detectability_margin  # noqa: E402
from m27d_fixture_scoring import (  # noqa: E402
    NEGATIVE_CLASSES,
    POSITIVE_CLASSES,
    score_pass,
    truth_class,
)
from m27d_pipeline_chain import (  # noqa: E402
    FULL_AUDIT,
    THROUGH_CONFIGURATIONS,
    chain,
)
from m27d_synthetic_cohort import (  # noqa: E402
    CohortLayout,
    Scenario,
    build,
    compose,
    offspring_of,
    pedigree_units,
    truth_pairs,
)

PREREG = REPO / "conf" / "m27d_donor_kinship_preregistration.json"
# Small enough to build in a moment, large enough to keep every structural property.
SMALL = CohortLayout(
    n_background_per_group=20,
    n_deme_members=6,
    n_pedigree_units_in_background=2,
    n_markers_per_chromosome=8,
    n_chromosomes=2,
)
ISOLATE = Scenario("isolate", 0.05, 0.04, 0.12)


def read_vcf_dosages(path: Path) -> tuple[list[str], list[list[int]]]:
    calls = {"0/0": 0, "0/1": 1, "1/1": 2}
    rows: list[list[int]] = []
    samples: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
        elif not line.startswith("#"):
            rows.append([calls[value] for value in line.split("\t")[9:]])
    return samples, rows


class TestPedigreeTruth(unittest.TestCase):
    def test_each_unit_declares_four_first_degree_and_one_second_degree_pair(self):
        rows = truth_pairs(SMALL, ISOLATE)
        units = pedigree_units(SMALL)
        first = [row for row in rows if row["true_degree"] == 1]
        second = [row for row in rows if row["true_degree"] == 2]
        self.assertEqual(len(first), 4 * len(units))
        self.assertEqual(len(second), len(units))

    def test_the_pedigree_lives_both_inside_a_deme_and_in_the_background(self):
        """Without both, a failure cannot be attributed to the deme or to the estimator."""
        locations = {unit["location"] for unit in pedigree_units(SMALL)}
        self.assertEqual(locations, {"deme", "background"})

    def test_children_inherit_one_allele_from_each_declared_parent(self):
        """The genotypes have to carry the pedigree, not only the labels."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            build(root, PREREG, ISOLATE, SMALL, seed=5)
            samples, rows = read_vcf_dosages(root / "panel" / "panel.1.vcf")
        index = {sample: position for position, sample in enumerate(samples)}
        for child, (father, mother) in offspring_of(SMALL).items():
            for row in rows:
                paternal = {0: {0}, 1: {0, 1}, 2: {1}}[row[index[father]]]
                maternal = {0: {0}, 1: {0, 1}, 2: {1}}[row[index[mother]]]
                achievable = {a + b for a in paternal for b in maternal}
                self.assertIn(
                    row[index[child]], achievable,
                    msg=f"{child} carries a dosage its parents cannot transmit",
                )

    def test_composition_is_not_addition(self):
        """theta + phi overstates the truth by theta*phi, which at theta=0.2 is 0.05."""
        self.assertAlmostEqual(compose(0.20, 0.25), 0.40, places=9)
        self.assertNotAlmostEqual(compose(0.20, 0.25), 0.45, places=3)
        self.assertAlmostEqual(compose(0.0, 0.25), 0.25, places=9)

    def test_the_total_truth_uses_the_composition(self):
        rows = truth_pairs(SMALL, ISOLATE)
        inside = next(
            row for row in rows
            if row["true_degree"] == 1 and row["pedigree_location"] == "deme"
        )
        self.assertAlmostEqual(
            float(inside["total_phi"]),
            compose(ISOLATE.within_deme_coancestry, 0.25),
            places=5,
        )


class TestNonKinshipTruth(unittest.TestCase):
    def test_deme_members_without_a_pedigree_carry_coancestry_and_no_kinship(self):
        rows = truth_pairs(SMALL, ISOLATE)
        within = [
            row for row in rows
            if row["coancestry_class"] == "within_deme" and not row["has_recent_kinship"]
        ]
        self.assertTrue(within)
        for row in within:
            self.assertEqual(row["pedigree_phi"], 0.0)
            self.assertAlmostEqual(
                float(row["coancestry_phi"]), ISOLATE.within_deme_coancestry, places=5
            )

    def test_pairs_across_demes_carry_only_the_shared_branch(self):
        rows = truth_pairs(SMALL, ISOLATE)
        between = [row for row in rows if row["coancestry_class"] == "between_demes"]
        self.assertTrue(between)
        for row in between:
            self.assertAlmostEqual(
                float(row["coancestry_phi"]), ISOLATE.f_intermediate, places=5
            )

    def test_the_null_scenario_gives_the_demes_no_genetic_content(self):
        null = next(s for s in SCENARIOS if s.name == "null_demes")
        self.assertEqual(null.within_deme_coancestry, 0.0)
        self.assertEqual(null.between_deme_coancestry, 0.0)
        rows = truth_pairs(SMALL, null)
        self.assertTrue(all(float(row["coancestry_phi"]) == 0.0 for row in rows))

    def test_the_screen_spans_both_sides_of_the_graph_threshold(self):
        """A screen entirely on one side of 0.0442 would answer only half the question."""
        within = [s.within_deme_coancestry for s in SCENARIOS]
        self.assertTrue(any(value < 0.0442 for value in within))
        self.assertTrue(any(value > 0.0442 for value in within))
        self.assertIn(0.0, within)

    def test_a_pure_panmictic_arm_exists_so_estimator_bias_is_separable(self):
        pure = next(s for s in SCENARIOS if s.name == "panmictic_pure")
        self.assertEqual((pure.f_background, pure.f_intermediate, pure.f_deme), (0.0, 0.0, 0.0))


class TestReproducibility(unittest.TestCase):
    def test_the_same_seed_reproduces_the_cohort_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            build(root / "a", PREREG, ISOLATE, SMALL, seed=7)
            build(root / "b", PREREG, ISOLATE, SMALL, seed=7)
            build(root / "c", PREREG, ISOLATE, SMALL, seed=8)
            first = (root / "a" / "panel" / "panel.1.vcf").read_bytes()
            same = (root / "b" / "panel" / "panel.1.vcf").read_bytes()
            other = (root / "c" / "panel" / "panel.1.vcf").read_bytes()
        self.assertEqual(first, same)
        self.assertNotEqual(first, other)

    def test_a_different_seed_changes_every_chromosome(self):
        """A seed that only moved the first chromosome would fake independent replicates."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            build(root / "a", PREREG, ISOLATE, SMALL, seed=7)
            build(root / "b", PREREG, ISOLATE, SMALL, seed=8)
            for chromosome in range(1, SMALL.n_chromosomes + 1):
                self.assertNotEqual(
                    (root / "a" / "panel" / f"panel.{chromosome}.vcf").read_bytes(),
                    (root / "b" / "panel" / f"panel.{chromosome}.vcf").read_bytes(),
                    msg=f"chromosome {chromosome} did not change with the seed",
                )


class TestCohortShape(unittest.TestCase):
    def test_the_sample_and_pair_counts_are_the_declared_ones(self):
        with tempfile.TemporaryDirectory() as name:
            truth = build(Path(name), PREREG, ISOLATE, SMALL, seed=3)
        expected_samples = (
            2 * SMALL.n_background_per_group
            + 2 * SMALL.n_deme_members
            + SMALL.n_pedigree_deme_members
        )
        self.assertEqual(truth["n_samples"], expected_samples)
        self.assertEqual(truth["n_pairs"], expected_samples * (expected_samples - 1) // 2)

    def test_the_contract_checkpoint_matches_the_cohort_it_will_run_on(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            truth = build(root, PREREG, ISOLATE, SMALL, seed=3)
            contract = json.loads((root / "prereg.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["pass0_checkpoint"]["expected_pairs"], truth["n_pairs"])
        self.assertEqual(
            contract["pass0_checkpoint"]["expected_eligible_samples"], truth["n_samples"]
        )

    def test_the_scientific_parameters_are_inherited_unchanged(self):
        """The fixture may restate its own counts; it must not restate the science."""
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            build(root, PREREG, ISOLATE, SMALL, seed=3)
            fixture_contract = json.loads((root / "prereg.json").read_text(encoding="utf-8"))
        real = json.loads(PREREG.read_text(encoding="utf-8"))
        for key in ("pcrelate", "configurations", "pca_axis_contract", "stop_rules"):
            self.assertEqual(fixture_contract[key], real[key], msg=key)


class TestScoringKnownAnswers(unittest.TestCase):
    TRUTH = [
        {"ID1": "A", "ID2": "B", "true_relationship": "parent_offspring", "true_degree": "1",
         "pedigree_location": "deme", "pedigree_phi": "0.25", "coancestry_class": "within_deme",
         "coancestry_phi": "0.1", "total_phi": "0.325", "has_recent_kinship": "True"},
        {"ID1": "C", "ID2": "D", "true_relationship": "unrelated", "true_degree": "0",
         "pedigree_location": "none", "pedigree_phi": "0.0", "coancestry_class": "within_deme",
         "coancestry_phi": "0.1", "total_phi": "0.1", "has_recent_kinship": "False"},
        {"ID1": "E", "ID2": "F", "true_relationship": "unrelated", "true_degree": "0",
         "pedigree_location": "none", "pedigree_phi": "0.0", "coancestry_class": "none",
         "coancestry_phi": "0.0", "total_phi": "0.0", "has_recent_kinship": "False"},
    ]

    def test_a_relative_that_vanished_is_counted_rather_than_read_as_zero(self):
        scored = score_pass({}, self.TRUTH, 0.0442, 0.0221)
        block = scored["first_degree_in_deme"]
        self.assertEqual(block["n_absent_from_pair_table"], 1)
        self.assertEqual(block["sensitivity"], 0.0)

    def test_the_two_truths_are_reported_separately(self):
        scored = score_pass({("A", "B"): 0.25}, self.TRUTH, 0.0442, 0.0221)
        block = scored["first_degree_in_deme"]
        self.assertAlmostEqual(block["expected_pedigree_phi"], 0.25, places=6)
        self.assertAlmostEqual(block["expected_total_phi"], 0.325, places=6)
        self.assertAlmostEqual(block["absolute_error_vs_pedigree_median"], 0.0, places=6)
        self.assertAlmostEqual(block["absolute_error_vs_total_median"], 0.075, places=6)

    def test_a_retained_coancestry_pair_is_a_false_positive_with_its_denominator(self):
        scored = score_pass(
            {("A", "B"): 0.25, ("C", "D"): 0.09}, self.TRUTH, 0.0442, 0.0221
        )
        block = scored["coancestry_within_deme"]
        self.assertEqual(block["n_pairs"], 1)
        self.assertEqual(block["false_positive_rate"], 1.0)
        self.assertEqual(scored["overall"]["sensitivity"], 1.0)
        self.assertEqual(scored["overall"]["n_false_positives"], 1)

    def test_every_class_is_positive_or_negative_and_never_both(self):
        classes = {truth_class(row) for row in self.TRUTH}
        self.assertTrue(classes)
        self.assertFalse(set(POSITIVE_CLASSES) & set(NEGATIVE_CLASSES))
        for name in classes:
            self.assertIn(name, set(POSITIVE_CLASSES) | set(NEGATIVE_CLASSES))

    def test_training_membership_is_recorded_because_the_scale_differs(self):
        """Residuals of the fitted set sum to zero; projected individuals are not bound."""
        scored = score_pass(
            {("A", "B"): 0.25}, self.TRUTH, 0.0442, 0.0221, training={"A", "C", "D"}
        )
        self.assertEqual(scored["first_degree_in_deme"]["membership"], {"one_outside": 1})
        self.assertEqual(scored["coancestry_within_deme"]["membership"], {"both_in_training": 1})


class TestPipelineChainSafety(unittest.TestCase):
    FULL = chain(FULL_AUDIT)
    TRIMMED = chain(THROUGH_CONFIGURATIONS)
    SOURCES = tuple(
        (REPO / "bin" / name).read_text(encoding="utf-8")
        for name in ("m27d_synthetic_cohort.py", "m27d_fixture_scoring.py",
                     "m27d_coancestry_screen.py", "m27d_pipeline_chain.py")
    )

    def test_the_screen_stops_before_baseline_identity_and_selection(self):
        self.assertNotIn("m27d_baseline_identity.R", self.TRIMMED)
        self.assertNotIn("m27d_candidate_selection.py", self.TRIMMED)
        self.assertIn("m27d_baseline_identity.R", self.FULL)
        self.assertIn("m27d_candidate_selection.py", self.FULL)

    def test_pass0_precedes_the_training_set_which_precedes_the_refit(self):
        """The order is the design; reordering it would test a different procedure."""
        self.assertLess(
            self.TRIMMED.index("m27d_pass0_pcrelate.R"),
            self.TRIMMED.index("m27d_kinship_graph.py"),
        )
        self.assertLess(
            self.TRIMMED.index("m27d_kinship_graph.py"),
            self.TRIMMED.index("m27d_pca_projection.R"),
        )
        self.assertLess(
            self.TRIMMED.index("m27d_pca_projection.R"),
            self.TRIMMED.index("m27d_pcrelate_configuration.R"),
        )

    def test_no_stage_reaches_object_storage_or_a_frozen_holdout(self):
        for text in (self.FULL, self.TRIMMED) + self.SOURCES:
            lowered = text.lower()
            for forbidden in ("gs://", "gnomix", "gcloud", "boto3", "gsutil"):
                self.assertNotIn(forbidden, lowered)

    def test_the_container_runs_without_a_network(self):
        source = (REPO / "bin" / "m27d_pipeline_chain.py").read_text(encoding="utf-8")
        self.assertIn('"--network", "none"', source)

    def test_king_and_pcair_appear_nowhere_in_the_new_path(self):
        for text in (self.FULL, self.TRIMMED) + self.SOURCES:
            lowered = text.lower()
            for forbidden in ("snpgdsibdking", "pcair(", "pcairpartition", "king-robust"):
                self.assertNotIn(forbidden, lowered)


class TestScreenDeclaresItsLimits(unittest.TestCase):
    def test_the_detectability_margin_is_published_rather_than_assumed(self):
        """Below one the deme is not separable, so a null result is about the cohort."""
        strong = next(s for s in SCENARIOS if s.name == "isolate")
        margin = detectability_margin(strong, n_deme=6, n_samples=118, n_markers=15400)
        self.assertGreater(margin, 1.0)
        null = next(s for s in SCENARIOS if s.name == "null_demes")
        self.assertEqual(detectability_margin(null, 6, 118, 15400), 0.0)

    def test_every_scenario_states_why_its_values_were_chosen(self):
        for scenario in SCENARIOS:
            self.assertTrue(scenario.rationale.strip(), msg=scenario.name)
            self.assertGreater(len(scenario.rationale), 80, msg=scenario.name)


if __name__ == "__main__":
    unittest.main()
