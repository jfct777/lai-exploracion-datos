import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m27d_synthetic_cohort import (  # noqa: E402
    CohortLayout,
    Scenario,
    first_cousin_units,
    metadata_rows,
    truth_pairs,
)
from m27d_training_set_intervention import (  # noqa: E402
    count_internal_pairs,
    proportional_quotas,
    recent_pairs,
    represented_training_set,
)


ISOLATE = Scenario("isolate", 0.05, 0.04, 0.12)
LAYOUT = CohortLayout(
    n_background_per_group=20,
    n_deme_members=6,
    n_pedigree_deme_members=6,
    n_pedigree_units_in_deme=1,
    n_pedigree_units_in_background=2,
    n_markers_per_chromosome=2,
    n_chromosomes=1,
)
COUSIN_LAYOUT = CohortLayout(
    n_background_per_group=20,
    n_deme_members=6,
    n_pedigree_deme_members=8,
    n_pedigree_units_in_deme=1,
    n_pedigree_units_in_background=2,
    n_first_cousin_pairs_in_deme=1,
    n_first_cousin_pairs_in_background=1,
    n_markers_per_chromosome=2,
    n_chromosomes=1,
)


def truth_json():
    from m27d_synthetic_cohort import deme_ids, pedigree_units

    return {
        "demes": {deme: deme_ids(LAYOUT, deme) for deme in LAYOUT.demes},
        "pedigree_units": pedigree_units(LAYOUT),
    }


def population():
    return {row["IID"]: row["Population"] for row in metadata_rows(LAYOUT)}


class TestM27DTrainingSetIntervention(unittest.TestCase):
    def setUp(self):
        # The strict set represents DEME_A poorly and contains one offspring from DEME_C,
        # but no complete known pedigree pair.
        self.strict = (
            [f"BG{i:03d}" for i in range(20)]
            + ["DA00"]
            + [f"DB{i:02d}" for i in range(6)]
            + ["DC03", "DC05"]
        )
        self.rows = truth_pairs(LAYOUT, ISOLATE)

    def build(self, seed=11):
        return represented_training_set(
            self.strict, truth_json(), self.rows, population(), seed
        )

    def test_the_intervention_is_size_matched_and_represents_pure_demes(self):
        values, summary = self.build()
        self.assertEqual(len(values), len(self.strict))
        self.assertEqual(set(f"DA{i:02d}" for i in range(6)) & set(values),
                         set(f"DA{i:02d}" for i in range(6)))
        self.assertEqual(set(f"DB{i:02d}" for i in range(6)) & set(values),
                         set(f"DB{i:02d}" for i in range(6)))
        self.assertTrue(summary["same_size_as_strict"])

    def test_offspring_are_not_added_to_the_pca_fitting_set(self):
        values, summary = self.build()
        self.assertNotIn("DC03", values)
        self.assertNotIn("DC04", values)
        for founder in ("DC00", "DC01", "DC02", "DC05"):
            self.assertIn(founder, values)
        self.assertEqual(summary["n_recent_pairs_both_in_represented"], 0)

    def test_no_known_recent_pair_has_both_members_inside(self):
        values, _ = self.build()
        self.assertEqual(
            count_internal_pairs(set(values), recent_pairs(self.rows)), []
        )

    def test_the_same_seed_is_byte_stable_and_another_seed_changes_only_removals(self):
        first, first_summary = self.build(seed=11)
        repeat, repeat_summary = self.build(seed=11)
        other, other_summary = self.build(seed=23)
        self.assertEqual(first, repeat)
        self.assertEqual(first_summary, repeat_summary)
        self.assertNotEqual(first_summary["removed_for_size_matching"],
                            other_summary["removed_for_size_matching"])
        self.assertEqual(first_summary["required_deme_members"],
                         other_summary["required_deme_members"])

    def test_removal_quotas_use_largest_remainder(self):
        self.assertEqual(proportional_quotas({"A": 50, "B": 35}, 8), {"A": 5, "B": 3})
        self.assertEqual(proportional_quotas({"A": 4, "B": 4}, 3), {"A": 2, "B": 1})

    def test_summary_declares_that_final_estimates_are_not_inputs(self):
        _, summary = self.build()
        self.assertFalse(summary["uses_final_pcrelate_estimates"])
        self.assertNotIn("estimated_phi", " ".join(summary["selection_inputs"]))

    def test_first_cousin_endpoints_are_excluded_and_size_is_preserved(self):
        from m27d_synthetic_cohort import deme_ids, pedigree_units

        truth = {
            "demes": {
                deme: deme_ids(COUSIN_LAYOUT, deme) for deme in COUSIN_LAYOUT.demes
            },
            "pedigree_units": pedigree_units(COUSIN_LAYOUT),
            "always_excluded_from_training": sorted(
                unit[key]
                for unit in first_cousin_units(COUSIN_LAYOUT)
                for key in ("cousin_1", "cousin_2")
            ),
        }
        pop = {
            row["IID"]: row["Population"] for row in metadata_rows(COUSIN_LAYOUT)
        }
        strict = list(dict.fromkeys(self.strict + truth["always_excluded_from_training"]))
        values, summary = represented_training_set(
            strict,
            truth,
            truth_pairs(COUSIN_LAYOUT, ISOLATE),
            pop,
            seed=11,
        )
        self.assertEqual(len(values), len(strict))
        self.assertFalse(set(truth["always_excluded_from_training"]) & set(values))
        self.assertEqual(
            summary["always_excluded_from_training"],
            truth["always_excluded_from_training"],
        )


if __name__ == "__main__":
    unittest.main()
