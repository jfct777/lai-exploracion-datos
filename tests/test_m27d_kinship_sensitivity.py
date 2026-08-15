"""Known-answer tests for the threshold sensitivity and for the exact independent set.

The sensitivity analysis exists to price two choices that the retained donor count
depends on: the relatedness threshold and the greedy construction.  Both of its engines
therefore need answers that are right by construction rather than by agreement with a
previous run.  Graphs with a hand-computable maximum independent set do that job: a path
of three vertices has one, a triangle has one, and a greedy walk that starts at the wrong
vertex provably misses it.

The guard that matters most is the one against a truncated graph.  The pair table is
written at a reporting threshold, so evaluating any lower value would build the graph
from edges that were never recorded and report a *larger* retained set, which reads as
good news.  That path must fail closed, not succeed quietly.
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
sys.path.insert(0, str(REPO / "bin"))

from m27d_kinship_graph import (  # noqa: E402
    connected_components,
    adjacency,
    maximal_independent_set,
    maximum_independent_set_size,
)


def edges_of(*pairs: tuple[str, str]) -> set[tuple[str, str]]:
    return {tuple(sorted(pair)) for pair in pairs}


class TestExactMaximumIndependentSet(unittest.TestCase):
    def test_isolated_vertices_are_all_retained(self):
        result = maximum_independent_set_size(["a", "b", "c"], set())
        self.assertEqual(result["size"], 3)
        self.assertTrue(result["exact"])
        self.assertEqual(result["n_components"], 0)

    def test_path_of_three_keeps_the_two_ends(self):
        result = maximum_independent_set_size(["a", "b", "c"], edges_of(("a", "b"), ("b", "c")))
        self.assertEqual(result["size"], 2)

    def test_triangle_keeps_exactly_one(self):
        result = maximum_independent_set_size(
            ["a", "b", "c"], edges_of(("a", "b"), ("b", "c"), ("a", "c"))
        )
        self.assertEqual(result["size"], 1)

    def test_five_cycle_keeps_two(self):
        cycle = edges_of(("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "a"))
        self.assertEqual(maximum_independent_set_size(list("abcde"), cycle)["size"], 2)

    def test_complete_bipartite_keeps_one_whole_side(self):
        left, right = ["l1", "l2", "l3"], ["r1", "r2", "r3"]
        edges = edges_of(*[(a, b) for a in left for b in right])
        self.assertEqual(maximum_independent_set_size(left + right, edges)["size"], 3)

    def test_star_keeps_every_leaf(self):
        edges = edges_of(*[("hub", f"leaf{i}") for i in range(7)])
        nodes = ["hub"] + [f"leaf{i}" for i in range(7)]
        self.assertEqual(maximum_independent_set_size(nodes, edges)["size"], 7)

    def test_the_greedy_set_can_be_strictly_smaller(self):
        """The centre wins on call rate, so greedy takes it and loses both ends."""
        nodes = ["a", "b", "c"]
        edges = edges_of(("a", "b"), ("b", "c"))
        call_rate = {"a": 0.90, "b": 0.99, "c": 0.90}
        greedy = maximal_independent_set(nodes, edges, call_rate)
        self.assertEqual(len(greedy), 1)
        self.assertEqual(maximum_independent_set_size(nodes, edges)["size"], 2)

    def test_the_exact_answer_is_never_below_the_greedy_one(self):
        nodes = [f"s{i}" for i in range(24)]
        edges = edges_of(*[(f"s{i}", f"s{(i * 7 + 3) % 24}") for i in range(24) if (i * 7 + 3) % 24 != i])
        call_rate = {node: 0.9 + (index % 5) / 100 for index, node in enumerate(nodes)}
        greedy = maximal_independent_set(nodes, edges, call_rate)
        result = maximum_independent_set_size(nodes, edges)
        self.assertGreaterEqual(result["size"], len(greedy))
        self.assertTrue(result["exact"])

    def test_a_component_above_the_cap_is_declared_not_exact(self):
        """An NP-hard component must be reported as unsolved, never approximated silently."""
        nodes = [f"n{i}" for i in range(12)]
        edges = edges_of(*[(f"n{i}", f"n{i + 1}") for i in range(11)])
        result = maximum_independent_set_size(nodes, edges, max_component_nodes=5)
        self.assertFalse(result["exact"])
        self.assertEqual(result["n_components_not_solved_exactly"], 1)
        self.assertEqual(result["largest_unsolved_component_nodes"], 12)
        # The reported size stays a usable lower bound rather than dropping the component.
        greedy = maximal_independent_set(nodes, edges, {})
        self.assertGreaterEqual(result["size"], len(greedy))

    def test_components_are_split_and_solved_independently(self):
        edges = edges_of(("a", "b"), ("c", "d"), ("d", "e"))
        nodes = list("abcdef")
        result = maximum_independent_set_size(nodes, edges)
        self.assertEqual(result["n_components"], 2)
        self.assertEqual(result["largest_component_nodes"], 3)
        # one from {a,b}, two from the path c-d-e, plus the isolated f
        self.assertEqual(result["size"], 4)

    def test_connected_components_ignore_vertices_without_edges(self):
        components = connected_components(list("abc"), adjacency(edges_of(("a", "b"))))
        self.assertEqual([sorted(component) for component in components], [["a", "b"]])


PREREGISTRATION = {
    "pcrelate": {
        "king_allowed": False,
        "primary_phi_threshold": 0.0442,
        "descriptive_phi_thresholds": [0.0221, 0.0884, 0.177],
    }
}


class TestSensitivityCli(unittest.TestCase):
    """A tiny cohort whose relatedness structure is decided by the fixture, not measured."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.samples = [f"S{i:03d}" for i in range(20)]
        # Two populations. FAMILY holds a chain of strong relatives, SPREAD holds a
        # single weak pair, so the retained composition must move with the threshold.
        family = cls.samples[:8]
        pairs = [(family[i], family[i + 1], 0.20) for i in range(7)]
        pairs += [(cls.samples[10], cls.samples[11], 0.03)]

        (cls.root / "universe.txt").write_text("\n".join(cls.samples) + "\n", encoding="utf-8")
        with gzip.open(cls.root / "pairs.tsv.gz", "wt", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["ID1", "ID2", "kin", "k0", "k2", "nsnp"])
            for left, right, kinship in pairs:
                writer.writerow([left, right, kinship, 0.5, 0.0, 1000])
        with (cls.root / "call_rate.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["sample_id", "call_rate"])
            for index, sample in enumerate(cls.samples):
                writer.writerow([sample, 0.99 - index / 1000])
        with (cls.root / "inbreeding.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["ID", "f", "nsnp"])
            for sample in cls.samples:
                writer.writerow([sample, 0.10 if sample in family else 0.001, 1000])
        with (cls.root / "strata.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                ["sample_id", "match_status", "resolution_method", "n_candidate_rows",
                 "population_interpretable", "Source", "Ancestry", "Population", "Country",
                 "Exclude", "N_genotypes", "Maximum_unrelated_dataset"]
            )
            for sample in cls.samples:
                isolate = sample in family
                writer.writerow(
                    [sample, "MATCHED", "DIRECT_UNIQUE", 1, "TRUE", "SRC",
                     "Isolate" if isolate else "Wide", "FAMILY" if isolate else "SPREAD",
                     "X", "FALSE", 1000, ""]
                )
        (cls.root / "prereg.json").write_text(json.dumps(PREREGISTRATION), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def invoke(self, *extra: str, reported: str = "0.0221"):
        out = self.root / "out"
        out.mkdir(exist_ok=True)
        result = subprocess.run(
            [
                sys.executable, str(REPO / "bin" / "m27d_kinship_sensitivity.py"),
                "--pairs", str(self.root / "pairs.tsv.gz"),
                "--samples", str(self.root / "universe.txt"),
                "--call-rates", str(self.root / "call_rate.tsv"),
                "--strata", str(self.root / "strata.tsv"),
                "--inbreeding", str(self.root / "inbreeding.tsv"),
                "--preregistration", str(self.root / "prereg.json"),
                "--reported-threshold", reported,
                "--out-summary", str(out / "summary.json"),
                "--out-thresholds", str(out / "thresholds.tsv"),
                "--out-coverage", str(out / "coverage.tsv"),
                *extra,
            ],
            capture_output=True, text=True, check=False,
        )
        payload = None
        if (out / "summary.json").exists():
            payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        return result, payload, out

    def test_thresholds_come_from_the_preregistration(self):
        _, payload, _ = self.invoke()
        self.assertEqual(payload["phi_thresholds"], [0.0221, 0.0442, 0.0884, 0.177])
        self.assertFalse(payload["threshold_selected_by_this_analysis"])

    def test_a_threshold_below_the_reported_one_fails_closed(self):
        """Otherwise the graph is truncated and every retained count is overstated."""
        result, _, _ = self.invoke(reported="0.05")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot be evaluated", result.stderr)

    def test_edges_fall_and_retention_rises_with_the_threshold(self):
        _, payload, _ = self.invoke()
        rows = payload["by_threshold"]
        edges = [row["n_edges"] for row in rows]
        retained = [row["n_retained_primary_order"] for row in rows]
        self.assertEqual(edges, sorted(edges, reverse=True))
        self.assertEqual(retained, sorted(retained))

    def test_the_weak_pair_disappears_above_its_own_kinship(self):
        _, payload, _ = self.invoke()
        by_phi = {row["phi_threshold"]: row for row in payload["by_threshold"]}
        self.assertEqual(by_phi[0.0221]["n_edges"], 8)
        self.assertEqual(by_phi[0.0442]["n_edges"], 7)

    def test_the_family_chain_keeps_alternating_members(self):
        """A path of eight has a maximum independent set of four, by construction."""
        _, payload, _ = self.invoke()
        by_phi = {row["phi_threshold"]: row for row in payload["by_threshold"]}
        top = by_phi[0.177]
        self.assertEqual(top["maximum_independent_set"]["size"], 12 + 4)
        self.assertTrue(top["maximum_independent_set"]["exact"])

    def test_the_exact_set_is_never_smaller_than_the_greedy_one(self):
        _, payload, _ = self.invoke()
        for row in payload["by_threshold"]:
            self.assertGreaterEqual(
                row["maximum_independent_set"]["size"], row["n_retained_primary_order"]
            )
            self.assertGreaterEqual(row["greedy_shortfall_vs_maximum"], 0)

    def test_coverage_denominators_add_up_to_the_cohort(self):
        _, payload, _ = self.invoke()
        for row in payload["by_threshold"]:
            total = sum(c["available"] for c in row["stratum_coverage"]["Population"].values())
            self.assertEqual(total, len(self.samples))

    def test_the_losing_stratum_is_named_and_the_other_is_not(self):
        _, payload, _ = self.invoke()
        by_phi = {row["phi_threshold"]: row for row in payload["by_threshold"]}
        coverage = by_phi[0.0442]["stratum_coverage"]["Population"]
        self.assertEqual(coverage["SPREAD"]["retained"], coverage["SPREAD"]["available"])
        self.assertLess(coverage["FAMILY"]["retained"], coverage["FAMILY"]["available"])

    def test_inbreeding_is_reported_for_both_groups(self):
        _, payload, _ = self.invoke()
        row = [r for r in payload["by_threshold"] if r["phi_threshold"] == 0.0442][0]
        self.assertGreater(row["inbreeding_removed"]["median"], row["inbreeding_retained"]["median"])
        self.assertEqual(
            row["inbreeding_retained"]["n"] + row["inbreeding_removed"]["n"], len(self.samples)
        )

    def test_edge_homogeneity_carries_its_own_denominator(self):
        _, payload, _ = self.invoke()
        row = [r for r in payload["by_threshold"] if r["phi_threshold"] == 0.0442][0]
        homogeneity = row["edge_homogeneity"]["Population"]
        self.assertEqual(homogeneity["n_edges_classifiable"], row["n_edges"])
        self.assertEqual(homogeneity["fraction_within_same_label"], 1.0)

    def test_no_output_carries_a_sample_identifier(self):
        _, _, out = self.invoke()
        for name in ("summary.json", "thresholds.tsv", "coverage.tsv"):
            text = (out / name).read_text(encoding="utf-8")
            for sample in self.samples:
                self.assertNotIn(sample, text, msg=f"{sample} leaked into {name}")

    def test_the_summary_states_that_no_new_kinship_pass_ran(self):
        _, payload, _ = self.invoke()
        self.assertFalse(payload["new_pcrelate_pass_executed"])
        self.assertFalse(payload["king_executed"])
        self.assertFalse(payload["pcair_used"])
        self.assertTrue(payload["derived_from_existing_pairs_only"])
        self.assertTrue(payload["interpretation_limits"])

    def test_a_contract_allowing_king_is_refused(self):
        broken = self.root / "king.json"
        broken.write_text(
            json.dumps({"pcrelate": dict(PREREGISTRATION["pcrelate"], king_allowed=True)}),
            encoding="utf-8",
        )
        result, _, _ = self.invoke("--preregistration", str(broken))
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
