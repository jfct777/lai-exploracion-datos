import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m27d_representation_control import (  # noqa: E402
    ARM_BOTH_REPRESENTED,
    ARM_PCA_REPRESENTED,
    ARM_STRICT,
    derive_confirmation_seeds,
    intervention_supported,
    pair_false_positive_units,
    prepare_context,
    run_arm,
)
from m27d_pipeline_chain import image_available  # noqa: E402
from m27d_synthetic_cohort import CohortLayout, Scenario, truth_pairs  # noqa: E402


LAYOUT = CohortLayout(
    n_background_per_group=20,
    n_pedigree_units_in_background=2,
    n_markers_per_chromosome=2,
    n_chromosomes=1,
)
SCENARIO = Scenario("isolate", 0.05, 0.04, 0.12)
INTEGRATION_SCENARIO = Scenario(
    "representation_integration", 0.0, 0.04, 0.12
)


def unit(value):
    return {
        "false_positive_fraction": value,
        "n_recent_pedigree_false_positives": round(value * 15),
        "n_pairs_descriptive_only": 15,
    }


def point(seed, control, represented, both=None):
    arms = {
        ARM_STRICT: {"primary_units": {"DEME_A": unit(control), "DEME_B": unit(control)}},
        ARM_PCA_REPRESENTED: {
            "primary_units": {"DEME_A": unit(represented), "DEME_B": unit(represented)}
        },
    }
    if both is not None:
        arms[ARM_BOTH_REPRESENTED] = {
            "primary_units": {"DEME_A": unit(both), "DEME_B": unit(both)}
        }
    return {"seed": seed, "arms": arms}


class TestRepresentationDecision(unittest.TestCase):
    def test_confirmation_seeds_are_deterministic_unique_and_not_pilot_seeds(self):
        first = derive_confirmation_seeds("run-a", 7, {11, 23, 37})
        repeat = derive_confirmation_seeds("run-a", 7, {11, 23, 37})
        self.assertEqual(first, repeat)
        self.assertEqual(len(first), len(set(first)))
        self.assertFalse(set(first) & {11, 23, 37})
        self.assertNotEqual(first, derive_confirmation_seeds("run-b", 7, {11, 23, 37}))

    def test_pca_mechanism_requires_near_floor_and_pairwise_nonworsening(self):
        passing = [point(seed, 0.6, 0.0) for seed in (11, 23, 37)]
        self.assertTrue(intervention_supported(passing)["supported"])
        one_bad = passing + [point(41, 0.6, 2 / 15)]
        self.assertFalse(intervention_supported(one_bad)["supported"])

    def test_the_same_gate_can_adjudicate_the_adaptive_both_represented_arm(self):
        rows = [point(seed, 0.6, 0.4, 0.0) for seed in (11, 23, 37)]
        self.assertFalse(intervention_supported(rows)["supported"])
        self.assertTrue(
            intervention_supported(rows, ARM_BOTH_REPRESENTED)["supported"]
        )


class TestPairScoringByExperimentalUnit(unittest.TestCase):
    def test_pairs_are_counted_inside_deme_units_not_as_independent_replicates(self):
        rows = truth_pairs(LAYOUT, SCENARIO)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth_path = root / "truth.tsv"
            with truth_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)

            pair_path = root / "pairs.tsv.gz"
            a_pairs = [
                row for row in rows
                if row["true_relationship"] == "unrelated"
                and row["deme_1"] == row["deme_2"] == "DEME_A"
            ]
            with gzip.open(pair_path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("ID1", "ID2", "kin"), delimiter="\t")
                writer.writeheader()
                for row in a_pairs:
                    writer.writerow({"ID1": row["ID1"], "ID2": row["ID2"], "kin": 0.08})

            scored = pair_false_positive_units(pair_path, truth_path, 0.0442)
            self.assertEqual(set(scored), {"DEME_A", "DEME_B"})
            self.assertEqual(scored["DEME_A"]["n_pairs_descriptive_only"], 15)
            self.assertEqual(scored["DEME_A"]["false_positive_fraction"], 1.0)
            self.assertEqual(scored["DEME_B"]["false_positive_fraction"], 0.0)
            self.assertEqual(scored["DEME_B"]["n_below_reporting_threshold"], 15)
            self.assertEqual(scored["DEME_A"]["repeated_subunit"], "deme=DEME_A")
            self.assertNotIn("experimental_unit", scored["DEME_A"])

    def test_decision_receipt_names_counts_and_fractions_explicitly(self):
        checked = intervention_supported([point(11, 1.0, 0.0)])
        row = checked["deme_subunits"][0]
        self.assertEqual(row["control_false_positive_count"], 15)
        self.assertEqual(row["intervention_false_positive_count"], 0)
        self.assertEqual(row["control_false_positive_fraction"], 1.0)
        self.assertEqual(row["intervention_false_positive_fraction"], 0.0)
        self.assertNotIn(ARM_STRICT, row)


@unittest.skipUnless(image_available(), "pinned M27D analysis container is not available")
class TestSeparatedTrainingSetsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = prepare_context(
            INTEGRATION_SCENARIO,
            11,
            CohortLayout(
                n_background_per_group=20,
                n_pedigree_units_in_background=2,
                # At 440 markers, estimator noise alone creates many pass0 edges and
                # collapses the independent set.  Four thousand four hundred markers
                # keep this integration test below the production fixture (15,400)
                # while making the phi=0.0442 boundary technically resolvable.
                n_markers_per_chromosome=200,
            ),
            ROOT,
            ROOT / "conf" / "m27d_donor_kinship_preregistration.json",
            threads=2,
            point_timeout=600,
        )
        cls.control = run_arm(cls.context, ARM_STRICT, ROOT, 2, 600)
        cls.represented = run_arm(
            cls.context, ARM_PCA_REPRESENTED, ROOT, 2, 600
        )

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.context.workspace, ignore_errors=True)

    def test_only_the_pca_set_changes_between_the_two_arms(self):
        self.assertEqual(
            self.control["pcrelate_training_set_sha256"],
            self.represented["pcrelate_training_set_sha256"],
        )
        self.assertNotEqual(
            self.control["pca_training_set_sha256"],
            self.represented["pca_training_set_sha256"],
        )
        control_sets = self.control["scored"]["training_sets"]
        represented_sets = self.represented["scored"]["training_sets"]
        self.assertTrue(control_sets["pca_equals_strict"])
        self.assertFalse(represented_sets["pca_equals_strict"])
        self.assertTrue(represented_sets["pcrelate_equals_strict"])
        representation = self.represented["scored"]["configurations"][
            "anchor_pc8_r2_020"
        ]["representation"]
        self.assertFalse(representation["algorithmic_control_applicable"])
        self.assertIsNone(representation["algorithmic_control"])
        self.assertIsNone(
            representation["training_set_identity_jaccard_vs_alternate"]
        )

    def test_the_r_receipts_name_the_training_set_each_stage_consumed(self):
        represented_out = self.context.workspace / ARM_PCA_REPRESENTED
        pca = json.loads((represented_out / "m27d_pca_anchor.json").read_text())
        pcrelate = json.loads(
            (represented_out / "m27d_pcrelate_anchor_pc8_r2_020.json").read_text()
        )
        self.assertEqual(pca["training_set_input_basename"], "pca_training_set.txt")
        self.assertEqual(
            pcrelate["training_set_input_basename"], "training_set.txt"
        )
        self.assertTrue(pcrelate["training_set_reused_from_pass0"])


if __name__ == "__main__":
    unittest.main()
