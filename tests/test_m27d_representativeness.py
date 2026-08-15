"""Known-answer tests for the training-set representativeness survey.

The survey exists because M27D only ever enforced independence, and a set can be
independent under the kinship graph while no longer representing the populations it was
drawn from.  Every fixture below has a hand-computable answer: a population that is a
complete clique collapses to one member, a population with no edges survives whole, and a
population with no survivor is named rather than silently missing from the table.

The degree bands are asserted at their cut points because the whole reason to report them
is to tell a cohort whose close edges are duplicated samples from one whose close edges are
siblings.  Merging those two bands would hide exactly the distinction the survey is for.
"""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bin" / "m27d_training_set_representativeness.py"
CONTRACT_PATH = REPO / "conf" / "m27d_donor_kinship_preregistration.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


class Fixture:
    """A cohort whose every answer is known before the script runs."""

    def __init__(self, root: Path):
        self.root = root
        # CLIQUE: four members, all mutually related, one survives.
        # FREE: three members, no edges, all survive.
        # GONE: two members related to each other and both dropped by the caller.
        # ORPHAN: two members with no metadata row, one duplicate-band edge between them.
        # Identifiers are deliberately long: a short one like "C1" is a substring of the
        # "PC1" column name, and the leak test would then fail on its own fixture.
        self.clique = [f"XCLQ{i}" for i in range(4)]
        self.free = [f"XFRE{i}" for i in range(3)]
        self.gone = [f"XGON{i}" for i in range(2)]
        self.orphans = ["ONG-1_ONG-1", "JAR-2_JAR-2"]
        self.samples = self.clique + self.free + self.gone + self.orphans
        self.training = [self.clique[0]] + self.free + [self.orphans[0]]

        edges = [(a, b, 0.06) for i, a in enumerate(self.clique) for b in self.clique[i + 1:]]
        edges.append((self.gone[0], self.gone[1], 0.20))
        # Just above 2^(-3/2): a duplicate-band edge, which must not land in first degree.
        edges.append((self.orphans[0], self.orphans[1], 0.40))
        with gzip.open(root / "pairs.tsv.gz", "wt", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["ID1", "ID2", "kin", "k0", "k2", "nsnp"])
            for left, right, kinship in edges:
                writer.writerow([left, right, kinship, 1 - 4 * kinship + 0.01, 0.01, 1000])

        (root / "universe.txt").write_text("\n".join(self.samples) + "\n", encoding="utf-8")
        (root / "training.txt").write_text("\n".join(self.training) + "\n", encoding="utf-8")
        with (root / "strata.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["sample_id", "match_status", "population_interpretable",
                             "Source", "Ancestry", "Population", "Country", "Exclude"])
            for sample in self.samples:
                if sample in self.orphans:
                    writer.writerow([sample, "UNMATCHED", "FALSE", "", "", "", "", ""])
                    continue
                label = {"XCLQ": "CLIQUE", "XFRE": "FREE", "XGON": "GONE"}[sample[:4]]
                writer.writerow([sample, "MATCHED", "TRUE", "SRC", "ANC", label, "X", "FALSE"])

        with gzip.open(root / "scores.tsv.gz", "wt", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["sample_id", "PC1", "PC2"])
            for sample in self.samples:
                # PC2 is carried entirely by the two samples without a metadata row.
                writer.writerow([sample, 1.0, 1.0 if sample in self.orphans else 0.0])

    def run(self, **flags) -> tuple[subprocess.CompletedProcess, dict | None]:
        out = self.root / "out"
        out.mkdir(exist_ok=True)
        command = [
            sys.executable, str(SCRIPT),
            "--universe", str(flags.get("universe", self.root / "universe.txt")),
            "--training-set", str(flags.get("training", self.root / "training.txt")),
            "--strata", str(self.root / "strata.tsv"),
            "--pairs", str(self.root / "pairs.tsv.gz"),
            "--preregistration", str(CONTRACT_PATH),
            "--pca-scores", str(self.root / "scores.tsv.gz"),
            "--out-summary", str(out / "summary.json"),
            "--out-table", str(out / "table.tsv"),
            # The fixture's groups are all smaller than the publication floor, so the
            # per-group assertions ask for every group and the floor gets its own test.
            "--suppress-below", str(flags.get("suppress_below", 1)),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        payload = None
        if result.returncode == 0:
            payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        return result, payload


class TestRepresentativenessSurvey(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.fixture = Fixture(Path(cls._tmp.name))
        cls.result, cls.payload = cls.fixture.run()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.assertEqual(self.result.returncode, 0, msg=self.result.stderr)
        self.groups = self.payload["by_population"]["by_group"]

    def test_a_complete_clique_survives_as_one_member(self):
        row = self.groups["CLIQUE"]
        self.assertEqual(row["n_eligible"], 4)
        self.assertEqual(row["n_in_training_set"], 1)
        self.assertTrue(row["is_complete_clique"])
        self.assertEqual(row["alpha_exact"], 1)
        self.assertEqual(row["status"], "SINGLETON")

    def test_a_population_without_edges_survives_whole(self):
        row = self.groups["FREE"]
        self.assertEqual(row["n_in_training_set"], row["n_eligible"])
        self.assertEqual(row["n_internal_edges"], 0)
        self.assertEqual(row["alpha_exact"], 3)
        self.assertEqual(row["status"], "INTACT")

    def test_a_population_with_no_survivor_is_named(self):
        self.assertEqual(self.groups["GONE"]["status"], "ABSENT")
        self.assertIn("GONE", self.payload["by_population"]["groups_absent_from_training_set"])

    def test_singletons_and_absences_share_one_denominator(self):
        """A population represented by one person contributes no internal variation."""
        block = self.payload["by_population"]
        self.assertEqual(block["n_groups_absent"], 1)
        self.assertEqual(
            block["n_groups_at_or_below_one_member"],
            block["n_groups_absent"] + block["n_groups_reduced_to_a_single_member"],
        )

    def test_the_duplicate_band_is_not_folded_into_first_degree(self):
        bands = self.payload["samples_without_metadata"]["internal_edges"]["edges_by_degree_band"]
        self.assertEqual(bands["duplicate_or_monozygotic"], 1)
        self.assertEqual(bands["first_degree"], 0)

    def test_the_bands_cut_at_the_half_powers_of_two(self):
        """These are the cut points GENESIS itself branches on; approximations drift."""
        source = SCRIPT.read_text(encoding="utf-8")
        for exponent in ("-1.5", "-2.5", "-3.5", "-4.5"):
            self.assertIn(f"2.0**{exponent}", source)

    def test_samples_without_metadata_are_a_group_and_not_a_deletion(self):
        block = self.payload["samples_without_metadata"]
        self.assertEqual(block["n_samples"], 2)
        self.assertEqual(block["interpretation_status"], "BLIND_SPOT_POPULATION_UNRESOLVED")
        self.assertEqual(block["identifier_prefix_counts"], {"JAR": 1, "ONG": 1})
        # The eligible total still counts them: the denominator must not shrink quietly.
        self.assertEqual(self.payload["by_population"]["n_eligible"], len(self.fixture.samples))

    def test_the_axis_share_of_that_group_is_measured(self):
        share = self.payload["samples_without_metadata"]["axis_share"]
        self.assertAlmostEqual(share["PC2"], 1.0, places=6)
        self.assertLess(share["PC1"], 0.5)

    def test_the_mass_each_axis_keeps_inside_the_training_set_is_reported(self):
        """The refit sees only the training set; an axis can be mostly outside it."""
        axes = self.payload["axis_support"]["by_axis"]
        # PC2 is carried entirely by the two samples without metadata, and only one of
        # them is in the training set, so exactly half of that axis is fitted.
        self.assertAlmostEqual(axes["PC2"]["mass_retained_by_training_set"], 0.5, places=6)
        self.assertAlmostEqual(axes["PC2"]["mass_projected_not_fitted"], 0.5, places=6)
        # PC1 is flat across the cohort, so it keeps the training-set fraction.
        expected = len(self.fixture.training) / len(self.fixture.samples)
        self.assertAlmostEqual(
            axes["PC1"]["mass_retained_by_training_set"], expected, places=6
        )

    def test_the_retained_mass_is_measured_and_not_gated(self):
        """No operating value for it has been justified, so it must not act as a gate."""
        self.assertIn("not gated", self.payload["axis_support"]["note"])
        text = json.dumps(self.payload["axis_support"])
        for verdict in ("PASS", "FAIL", "REVIEW"):
            self.assertNotIn(verdict, text)

    def test_g2c_is_reported_but_not_adjudicated(self):
        self.assertFalse(self.payload["g2c_adjudicated_here"])
        self.assertEqual(
            self.payload["g2c_status"],
            CONTRACT["pca_axis_contract"]["g2c_ancestry_representativeness"]["status"],
        )

    def test_nothing_new_was_computed(self):
        self.assertFalse(self.payload["new_pcrelate_pass_executed"])
        self.assertFalse(self.payload["king_executed"])
        self.assertFalse(self.payload["pcair_used"])

    def test_no_sample_identifier_reaches_the_summary(self):
        text = json.dumps(self.payload)
        for sample in self.fixture.samples:
            self.assertNotIn(sample, text)


class TestSuppressionFloor(unittest.TestCase):
    """Publishing "one eligible, zero retained" for a named population is a size-one cell.

    The rest of the module already suppresses strata below the same floor before writing a
    public table; this stage was writing population names with counts of one and two.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.fixture = Fixture(Path(cls._tmp.name))
        cls.result, cls.payload = cls.fixture.run(suppress_below=5)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_small_groups_are_counted_but_not_named(self):
        block = self.payload["by_population"]
        self.assertEqual(block["by_group"], {})
        self.assertEqual(block["n_groups_published"], 0)
        self.assertEqual(block["n_groups_total"], 4)
        self.assertEqual(block["suppression"]["n_groups_suppressed"], 4)

    def test_the_denominator_does_not_shrink_to_what_was_published(self):
        block = self.payload["by_population"]
        self.assertEqual(block["n_eligible"], len(self.fixture.samples))
        self.assertEqual(
            block["suppression"]["n_eligible_in_suppressed_groups"], len(self.fixture.samples)
        )
        self.assertEqual(
            block["suppression"]["n_in_training_set_in_suppressed_groups"],
            len(self.fixture.training),
        )
        # The loss is still reported in aggregate: GONE has no survivor.
        self.assertEqual(block["suppression"]["n_suppressed_groups_absent_from_training_set"], 1)

    def test_no_suppressed_population_name_reaches_the_output(self):
        text = json.dumps(self.payload["by_population"])
        for label in ("CLIQUE", "FREE", "GONE"):
            self.assertNotIn(label, text)


class TestFailClosed(unittest.TestCase):
    def test_a_training_member_outside_the_universe_aborts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            fixture = Fixture(root)
            stray = root / "stray.txt"
            stray.write_text("\n".join(fixture.training + ["NOT_IN_PANEL"]) + "\n", encoding="utf-8")
            result, _ = fixture.run(training=stray)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside the eligible universe", result.stderr)


if __name__ == "__main__":
    unittest.main()
