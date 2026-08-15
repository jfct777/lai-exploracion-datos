"""End-to-end M27D audit over a synthetic cohort with known relatives.

The chain needs R, GENESIS and SNPRelate, so it runs inside the pinned analysis
container and skips when Docker or the image is unavailable.  Everything it asserts is
a property that has a right answer by construction: the fixture builds the trios, the
exclusions and the baseline overlap itself, so a silent regression in the estimator or
in the selection order shows up as a wrong number rather than as a passing test.

Two of the assertions are the ones worth keeping honest about:

* the pair count must equal n(n-1)/2 over the *eligible* universe, which is the same
  invariant the production stop-rule uses on the real panel;
* the refit pass must estimate a parent-offspring pair closer to 0.25 than pass0 does.
  Pass0 fits allele frequencies on a set that still contains relatives, so it is biased
  upward by construction.  If the refit ever stopped correcting that, the two-pass
  design would be pure ceremony and the test should fail.
"""

from __future__ import annotations

import csv
import glob
import gzip
import json
import shutil
import statistics
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "bin"))

import make_m27d_synthetic_fixture as fixture  # noqa: E402
from m27d_pipeline_chain import (  # noqa: E402
    FULL_AUDIT,
    chain,
    image_available,
    run as run_chain,
)

PRIMARY_PHI = 0.0442
CHAIN = chain(FULL_AUDIT)  # kept: two tests assert on the chain text itself


def read_pairs(path: Path) -> dict[tuple[str, str], float]:
    pairs: dict[tuple[str, str], float] = {}
    with gzip.open(path, "rt") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            pairs[tuple(sorted((record["ID1"], record["ID2"])))] = float(record["kin"])
    return pairs


@unittest.skipUnless(image_available(), "pinned M27D analysis container is not available")
class TestM27DAuditIntegration(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.fixture_dir = root / "fx"
        cls.out = root / "out"
        cls.out.mkdir()
        cls.expected = fixture.build(
            cls.fixture_dir, REPO / "conf" / "m27d_donor_kinship_preregistration.json"
        )
        completed = run_chain(cls.fixture_dir, cls.out, REPO, FULL_AUDIT, timeout=3600)
        cls.completed = completed
        if completed.returncode != 0:
            raise AssertionError(f"M27D chain failed:\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def summary(self, name):
        return json.loads((self.out / name).read_text(encoding="utf-8"))

    def test_alias_collisions_resolve_and_orphans_survive_without_metadata(self):
        strata = self.summary("strata_summary.json")
        self.assertEqual(strata["n_ambiguous"], 0)
        self.assertEqual(strata["n_unmatched"], self.expected["n_unmatched"])
        self.assertEqual(
            strata["resolution_methods"]["RESOLVED_ACTIVE_GENOTYPED_IID"],
            self.expected["n_colliding"],
        )
        self.assertEqual(strata["resolution_methods"]["AMBIGUOUS_FAIL_CLOSED"], 0)

    def test_only_metadata_excluded_samples_leave_the_kinship_universe(self):
        pass0 = self.summary("m27d_pass0_pcrelate.json")
        self.assertEqual(pass0["n_eligible_samples"], self.expected["n_eligible"])
        self.assertEqual(pass0["n_excluded_by_metadata"], self.expected["n_excluded"])

    def test_pair_count_matches_the_closed_form_in_every_stage(self):
        expected = self.expected["n_expected_pairs"]
        self.assertEqual(self.summary("m27d_pass0_pcrelate.json")["n_pairs_total"], expected)
        for path in glob.glob(str(self.out / "m27d_pcrelate_*.json")):
            summary = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(summary["n_pairs_total"], expected, msg=path)

    def test_known_relatives_are_recovered_by_every_configuration(self):
        known = {tuple(sorted(pair)) for pair in self.expected["related_pairs"]}
        for path in glob.glob(str(self.out / "m27d_pcrelate_*_pairs.private.tsv.gz")):
            pairs = read_pairs(Path(path))
            recovered = sum(1 for pair in known if pairs.get(pair, 0.0) >= PRIMARY_PHI)
            self.assertEqual(recovered, len(known), msg=path)

    def test_refit_reduces_the_upward_bias_of_pass0(self):
        known = {tuple(sorted(pair)) for pair in self.expected["related_pairs"]}
        pass0 = read_pairs(self.out / "m27d_pass0_related_pairs.private.tsv.gz")
        final = read_pairs(self.out / "m27d_pcrelate_anchor_pc8_r2_020_pairs.private.tsv.gz")
        pass0_median = statistics.median(pass0[pair] for pair in known)
        final_median = statistics.median(final[pair] for pair in known)
        # Parent-offspring kinship is 0.25; pass0 fits on a set that still holds
        # relatives, so it overshoots and the refit has to close part of that gap.
        self.assertLess(abs(final_median - 0.25), abs(pass0_median - 0.25))
        self.assertGreater(pass0_median, final_median)

    def test_no_related_pair_survives_inside_the_training_set(self):
        training = set(
            (self.out / "training_set.txt").read_text(encoding="utf-8").split()
        )
        for left, right in self.expected["related_pairs"]:
            self.assertFalse(
                left in training and right in training,
                msg="a known related pair reached the training set",
            )
        self.assertEqual(self.summary("training_set.json")["internal_edges_in_primary_set"], 0)

    def test_every_configuration_reuses_the_same_training_set(self):
        sizes = set()
        for path in glob.glob(str(self.out / "m27d_pcrelate_*.json")):
            summary = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertTrue(summary["training_set_reused_from_pass0"], msg=path)
            sizes.add(summary["n_training_samples"])
        self.assertEqual(len(sizes), 1, msg=f"training set sizes diverged: {sizes}")
        self.assertEqual(sizes.pop(), self.summary("training_set.json")["n_independent_primary_order"])

    def test_pca_is_fitted_only_on_the_training_set(self):
        for marker_set in ("anchor", "strict"):
            summary = self.summary(f"m27d_pca_{marker_set}.json")
            self.assertTrue(summary["pca_fitted_only_on_training_set"])
            self.assertEqual(
                summary["n_training_samples"] + summary["n_projected_samples"],
                self.expected["n_eligible"],
            )
            self.assertGreater(summary["n_projected_samples"], 0)

    def test_baseline_identity_is_confirmed_by_genotype_and_flags_the_absent_donor(self):
        summary = self.summary("m27d_baseline_identity.json")
        self.assertEqual(summary["n_identities_confirmed"], self.expected["n_baseline_shared"])
        self.assertEqual(
            summary["n_baseline_donors_absent_from_panel"], self.expected["n_baseline_absent"]
        )
        self.assertTrue(summary["unmatched_baseline_donor_blocks_full_kinship_disjointness"])
        # The true twin must stand clear of the best impostor, otherwise the identity
        # claim rests on a threshold rather than on the genotypes.
        self.assertGreater(
            summary["min_confirmed_dosage_concordance"], summary["max_runner_up_concordance"]
        )

    def test_selection_excludes_baseline_and_uninterpretable_samples(self):
        summary = self.summary("candidate_selection.json")
        self.assertEqual(
            summary["n_excluded_baseline_identity"], self.expected["n_baseline_shared"]
        )
        self.assertEqual(
            summary["n_excluded_metadata_unresolved"], self.expected["n_unmatched"]
        )
        self.assertEqual(summary["n_universe"], self.expected["n_eligible"])
        # No gate may FAIL.  NOT_EVALUATED is allowed and expected for the gates another
        # stage owns, and PASS_WITH_BLIND_SPOT is the honest verdict when a baseline
        # donor has no panel twin to be compared against.
        self.assertNotIn("FAIL", set(summary["gates"].values()))

    def test_every_preregistered_gate_appears_with_a_status(self):
        """A gates file that silently omits gates reads as 'everything passed'."""
        contract = json.loads(
            (REPO / "conf" / "m27d_donor_kinship_preregistration.json").read_text(
                encoding="utf-8"
            )
        )
        emitted = self.summary("candidate_selection.json")["gates"]
        for gate in contract["gates"]:
            self.assertIn(gate, emitted, msg=f"{gate} produced no row")
        with (self.out / "gates.tsv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertTrue(all(row["detail"] for row in rows))

    def test_g2a_is_a_verdict_and_g2b_is_a_measurement(self):
        """Only the technical check can fail; localisation is reported, never condemned."""
        for marker_set in ("anchor", "strict"):
            summary = self.summary(f"m27d_pca_{marker_set}.json")
            self.assertIn(summary["g2a_technical_integrity_status"], {"PASS", "FAIL"})
            self.assertTrue(all(summary["g2a_checks"].values()), msg=summary["g2a_failed_checks"])
            for entry in summary["g2b_status_by_preregistered_n_pcs"].values():
                self.assertIn(entry["status"], {"PASS", "REVIEW", "NOT_EVALUATED"})

    def test_the_localization_bound_scales_with_cohort_size(self):
        """A bound fixed as an absolute count would fire on any smaller cohort."""
        for marker_set in ("anchor", "strict"):
            summary = self.summary(f"m27d_pca_{marker_set}.json")
            expected_bound = summary["g2b_review_fraction"] * summary["n_eligible_samples"]
            self.assertAlmostEqual(summary["g2b_review_bound"], expected_bound, places=6)
            self.assertEqual(
                len(summary["g2b_effective_individuals_by_axis"]), summary["n_pcs"]
            )

    def test_every_configuration_is_judged_on_its_own_prefix(self):
        """The old aggregate over max_pcs condemned four configurations for one axis."""
        contract = json.loads(
            (REPO / "conf" / "m27d_donor_kinship_preregistration.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {f"n_pcs_{c['n_pcs']}" for c in contract["configurations"]}
        for marker_set in ("anchor", "strict"):
            summary = self.summary(f"m27d_pca_{marker_set}.json")
            by_prefix = summary["g2b_status_by_preregistered_n_pcs"]
            self.assertEqual(set(by_prefix), expected)
            for key, entry in by_prefix.items():
                used = summary["g2b_effective_individuals_by_axis"][: entry["n_pcs"]]
                self.assertAlmostEqual(entry["min_effective_individuals"], min(used), places=6)

    def test_small_strata_are_suppressed_in_the_public_json_too(self):
        summary = self.summary("candidate_selection.json")
        threshold = summary["candidate_strata_suppressed_below"]
        for label, count in summary["candidate_counts_by_stratum"].items():
            if count is None:
                self.assertIn(label, summary["candidate_strata_suppressed"])
            else:
                self.assertTrue(
                    count == 0 or count >= threshold,
                    msg=f"{label} published a count of {count} below the {threshold} floor",
                )

    def test_union_rule_is_at_least_as_strict_as_any_single_configuration(self):
        summary = self.summary("candidate_selection.json")
        per_configuration = summary["n_related_edges_by_configuration"]
        self.assertEqual(len(per_configuration), 4)
        self.assertGreaterEqual(summary["n_union_related_edges"], max(per_configuration.values()))

    def test_alternative_tie_break_order_is_reported_not_optimised(self):
        summary = self.summary("candidate_selection.json")
        self.assertIn("n_candidates_alternate_order", summary)
        self.assertFalse(summary["configuration_chosen_after_seeing_counts"])
        self.assertFalse(summary["selection_used_ancestry_or_population"])
        self.assertFalse(summary["selection_used_historical_unrelated_flags"])

    def test_no_stage_reports_a_king_execution(self):
        for name in (
            "m27d_pass0_pcrelate.json",
            "m27d_baseline_identity.json",
            "candidate_selection.json",
        ):
            self.assertFalse(self.summary(name)["king_executed"], msg=name)
        for path in glob.glob(str(self.out / "m27d_pcrelate_*.json")):
            self.assertFalse(json.loads(Path(path).read_text())["king_executed"], msg=path)
        combined = (self.completed.stdout + self.completed.stderr).lower()
        for token in ("snpgdsibdking", "kingtomatrix", "pc-air", "pcair"):
            self.assertNotIn(token, combined)

    def test_public_outputs_carry_no_sample_identifiers(self):
        private = self.out / "candidates.private.tsv"
        with private.open(encoding="utf-8") as handle:
            identifiers = [row["sample_id"] for row in csv.DictReader(handle, delimiter="\t")]
        self.assertGreater(len(identifiers), 0)
        for name in ("candidate_counts.tsv", "gates.tsv", "candidate_selection.json",
                     "strata_summary.json", "m27d_pass0_pcrelate.json"):
            text = (self.out / name).read_text(encoding="utf-8")
            for identifier in identifiers:
                self.assertNotIn(identifier, text, msg=f"{identifier} leaked into {name}")


if __name__ == "__main__":
    unittest.main()
