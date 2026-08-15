"""Tests for the component-count comparison, the attribution diagnostics and the phase DAG.

Two defects motivate most of what is asserted here.

The first was live: `pc_sensitivity` was declared in the preregistration before it had a
branch in the workflow, so it passed the launch gate and fell through into the audit
block.  It would have re-run pass0, audited baseline identity over twenty-two VCFs and
executed all four configurations, while the provenance record described a narrow technical
phase.  A declared phase without a branch must now be an error, and the phase must not be
able to reach the stages it does not belong to.

The second is the reason the comparison exists at all: a one-factor claim nobody checks is
just a claim.  The comparator refuses to report unless the two arms agree on every held
factor and differ only in the component count.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))

CONTRACT = json.loads(
    (REPO / "conf" / "m27d_donor_kinship_preregistration.json").read_text(encoding="utf-8")
)
WORKFLOW = (REPO / "workflows" / "m27d_donor_kinship_audit.nf").read_text(encoding="utf-8")
MODULE = (REPO / "modules" / "27D_DONOR_KINSHIP_AUDIT.nf").read_text(encoding="utf-8")


class TestPhaseCoverage(unittest.TestCase):
    def test_every_declared_phase_is_handled(self):
        """A phase in the contract with no branch used to fall into the audit block."""
        for phase in CONTRACT["authorization"]["phases"]:
            self.assertIn(
                f"'{phase}'", WORKFLOW, msg=f"phase {phase} is declared but never named"
            )

    def test_a_phase_without_a_branch_cannot_reach_the_dag(self):
        """The workflow refuses to start if the contract and the branches disagree."""
        self.assertIn("IMPLEMENTED_PHASES", WORKFLOW)
        self.assertIn("phases drifted between the preregistration and the workflow", WORKFLOW)
        implemented = re.search(r"IMPLEMENTED_PHASES = \[([^\]]+)\]", WORKFLOW)
        self.assertIsNotNone(implemented)
        members = set(re.findall(r"'([^']+)'", implemented.group(1)))
        self.assertEqual(members, set(CONTRACT["authorization"]["phases"]))

    def test_pc_sensitivity_returns_before_pass0(self):
        """It reuses the pass0 training set; recomputing it would cost 37 minutes."""
        start = WORKFLOW.index("if( phase == 'pc_sensitivity' )")
        end = WORKFLOW.index("RUN_DONOR_KINSHIP_PASS0(")
        branch = WORKFLOW[start:end]
        self.assertIn("return", branch)
        for forbidden in (
            "RUN_DONOR_KINSHIP_PASS0(",
            "AUDIT_BASELINE_DONOR_IDENTITY(",
            "SELECT_DONOR_KINSHIP_CANDIDATES(",
        ):
            self.assertNotIn(forbidden, branch, msg=f"pc_sensitivity invokes {forbidden}")

    def test_pc_sensitivity_demands_the_reused_training_set_and_its_hash(self):
        self.assertIn("donor_kinship_pass0_training_set_sha256", WORKFLOW)
        self.assertIn("--expected-training-set-sha256", MODULE)

    def test_the_marker_set_lookup_is_defined_once(self):
        self.assertEqual(WORKFLOW.count("def PREPARED_MARKER_SETS"), 1)

    def test_threads_are_decoupled_from_the_machine_size(self):
        """64 GiB forces at least eight vCPU; the thread count must not follow it."""
        config = (REPO / "nextflow.config").read_text(encoding="utf-8")
        self.assertIn("donor_kinship_pcrelate_threads", config)
        self.assertNotIn("--threads ${params.donor_kinship_pcrelate_cpus}", MODULE)

    def test_the_comparison_stage_is_pinned_and_labeled_in_the_cloud_profile(self):
        cloud = (REPO / "conf" / "google_batch.config").read_text(encoding="utf-8")
        line = next(
            row for row in cloud.splitlines() if "withName: 'COMPARE_DONOR_KINSHIP_PC_COUNT'" in row
        )
        self.assertIn("container", line)
        self.assertIn("resourceLabels = [team: 'frank']", cloud)


class TestG2IsReportedPerConfiguration(unittest.TestCase):
    SCRIPT = (REPO / "bin" / "m27d_pca_projection.R").read_text(encoding="utf-8")

    def test_the_gate_reports_a_status_per_preregistered_component_count(self):
        """A twelfth degenerate axis must not silently condemn an eight-component run."""
        self.assertIn("g2_status_by_preregistered_n_pcs", self.SCRIPT)
        self.assertIn("g2_axis_carriers", self.SCRIPT)

    def test_enforcement_is_named_rather_than_implied(self):
        self.assertIn("abort_if_any_preregistered_prefix_fails", self.SCRIPT)

    def test_the_failure_message_names_the_axes_and_the_blocked_configurations(self):
        self.assertIn("Degenerate axes", self.SCRIPT)
        self.assertIn("Blocked configurations", self.SCRIPT)


def build_arm(root: Path, name: str, pairs, f_values, n_pcs: int, **overrides) -> None:
    with gzip.open(root / f"m27d_pcrelate_{name}_pairs.private.tsv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["ID1", "ID2", "kin", "k0", "k2", "nsnp"])
        for left, right, kinship in pairs:
            writer.writerow([left, right, kinship, 1 - 4 * kinship, 0.0, 1000])
    with (root / f"m27d_pcrelate_{name}_inbreeding.private.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["ID", "f", "nsnp"])
        for sample, value in f_values.items():
            writer.writerow([sample, value, 1000])
    summary = {
        "configuration_id": name,
        "n_pcs": n_pcs,
        "n_eligible_samples": 10,
        "n_markers": 5000,
        "n_training_samples": 8,
        "n_pairs_total": 45,
        "random_seed": 20260814,
        "threads": 4,
        "ld_r2_max": 0.2,
        "report_threshold": 0.0221,
        "king_executed": False,
        "pcair_used": False,
        "training_set_reused_from_pass0": True,
        "pair_counts_by_threshold": {"phi_ge_0.0442": len([p for p in pairs if p[2] >= 0.0442])},
    }
    summary.update(overrides)
    (root / f"m27d_pcrelate_{name}.json").write_text(json.dumps(summary), encoding="utf-8")


class TestComparatorOneFactorGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.samples = [f"S{i:02d}" for i in range(10)]
        (cls.root / "universe.txt").write_text("\n".join(cls.samples) + "\n", encoding="utf-8")
        with (cls.root / "call_rate.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["sample_id", "call_rate"])
            for index, sample in enumerate(cls.samples):
                writer.writerow([sample, 0.99 - index / 1000])
        with (cls.root / "strata.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["sample_id", "population_interpretable", "Source", "Ancestry",
                             "Population", "Country", "Exclude"])
            for sample in cls.samples:
                writer.writerow([sample, "TRUE", "SRC", "A", "P", "X", "FALSE"])
        cls.f_values = {sample: 0.01 for sample in cls.samples}
        cls.pairs_eight = [("S00", "S01", 0.20), ("S02", "S03", 0.05)]
        cls.pairs_twelve = [("S00", "S01", 0.19), ("S02", "S03", 0.03)]
        (cls.root / "prereg.json").write_text(
            json.dumps(
                {
                    "pcrelate": {
                        "king_allowed": False,
                        "primary_phi_threshold": 0.0442,
                        "descriptive_phi_thresholds": [0.0221, 0.0884],
                    }
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def invoke(self, **overrides):
        build_arm(self.root, "arm8", self.pairs_eight, self.f_values, 8)
        build_arm(
            self.root, "arm12", self.pairs_twelve, self.f_values,
            overrides.pop("n_pcs", 12), **overrides,
        )
        out = self.root / "out"
        out.mkdir(exist_ok=True)
        result = subprocess.run(
            [
                sys.executable, str(REPO / "bin" / "m27d_pc_comparison.py"),
                "--pairs",
                f"arm8={self.root / 'm27d_pcrelate_arm8_pairs.private.tsv.gz'}",
                f"arm12={self.root / 'm27d_pcrelate_arm12_pairs.private.tsv.gz'}",
                "--inbreeding",
                f"arm8={self.root / 'm27d_pcrelate_arm8_inbreeding.private.tsv'}",
                f"arm12={self.root / 'm27d_pcrelate_arm12_inbreeding.private.tsv'}",
                "--summaries",
                str(self.root / "m27d_pcrelate_arm8.json"),
                str(self.root / "m27d_pcrelate_arm12.json"),
                "--reference-configuration", "arm8",
                "--samples", str(self.root / "universe.txt"),
                "--call-rates", str(self.root / "call_rate.tsv"),
                "--strata", str(self.root / "strata.tsv"),
                "--preregistration", str(self.root / "prereg.json"),
                "--out-summary", str(out / "summary.json"),
                "--out-table", str(out / "table.tsv"),
            ],
            capture_output=True, text=True, check=False,
        )
        payload = None
        if (out / "summary.json").exists() and result.returncode == 0:
            payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        return result, payload

    def test_a_clean_pair_is_compared(self):
        result, payload = self.invoke()
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(payload["one_factor_guard"]["one_factor_verified"])
        self.assertEqual(payload["one_factor_guard"]["n_pcs"], {"arm8": 8, "arm12": 12})

    def test_a_different_marker_count_aborts_the_comparison(self):
        result, _ = self.invoke(n_markers=4999)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("more than the component count", result.stderr)

    def test_a_different_seed_aborts_the_comparison(self):
        result, _ = self.invoke(random_seed=1)
        self.assertNotEqual(result.returncode, 0)

    def test_a_different_thread_count_aborts_the_comparison(self):
        """Thread count changes the reduction order, so it must not vary silently."""
        result, _ = self.invoke(threads=8)
        self.assertNotEqual(result.returncode, 0)

    def test_a_reported_king_execution_aborts_the_comparison(self):
        result, _ = self.invoke(king_executed=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbidden", result.stderr)

    def test_an_arm_that_refit_the_training_set_aborts_the_comparison(self):
        result, _ = self.invoke(training_set_reused_from_pass0=False)
        self.assertNotEqual(result.returncode, 0)

    def test_the_same_component_count_in_both_arms_aborts(self):
        result, _ = self.invoke(n_pcs=8)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("same number of components", result.stderr)

    def test_threshold_crossings_are_reported_in_both_directions(self):
        _, payload = self.invoke()
        row = next(r for r in payload["by_threshold"] if r["phi_threshold"] == 0.0442)
        # S02-S03 falls from 0.05 to 0.03, so it is an edge only in the reference arm.
        self.assertEqual(row["n_edges_only_in_reference"], 1)
        self.assertEqual(row["n_edges_only_in_other"], 0)
        self.assertEqual(row["n_retained_primary_order"]["arm12"], 9)

    def test_the_administrative_nam_denominator_is_declared_unevaluated(self):
        """Reconstructing it from metadata would substitute a different denominator."""
        _, payload = self.invoke()
        status = payload["administrative_nam_candidates"]["status"]
        self.assertEqual(status, "NOT_EVALUATED_MISSING_BASELINE_IDENTITY")

    def test_no_sample_identifier_reaches_the_outputs(self):
        _, payload = self.invoke()
        text = json.dumps(payload)
        for sample in self.samples:
            self.assertNotIn(sample, text)


class TestAttributionDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        # CLIQUE is a complete graph on four members: no algorithm retains more than one.
        # CHAIN is a path on four: the ceiling is two, and a greedy walk can find it.
        clique = [f"C{i}" for i in range(4)]
        chain = [f"H{i}" for i in range(4)]
        loose = [f"L{i}" for i in range(4)]
        cls.samples = clique + chain + loose
        pairs = [(a, b, 0.06) for i, a in enumerate(clique) for b in clique[i + 1:]]
        pairs += [(chain[i], chain[i + 1], 0.06) for i in range(3)]
        with gzip.open(cls.root / "pairs.tsv.gz", "wt", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["ID1", "ID2", "kin", "k0", "k2", "nsnp"])
            for left, right, kinship in pairs:
                writer.writerow([left, right, kinship, 1 - 4 * kinship, 0.0, 1000])
        (cls.root / "universe.txt").write_text("\n".join(cls.samples) + "\n", encoding="utf-8")
        with (cls.root / "call_rate.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["sample_id", "call_rate"])
            for index, sample in enumerate(cls.samples):
                writer.writerow([sample, 0.99 - index / 1000])
        with (cls.root / "strata.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["sample_id", "population_interpretable", "Source", "Ancestry",
                             "Population", "Country", "Exclude"])
            for sample in cls.samples:
                label = {"C": "CLIQUE", "H": "CHAIN", "L": "LOOSE"}[sample[0]]
                writer.writerow([sample, "TRUE", "SRC", "A", label, "X", "FALSE"])
        (cls.root / "prereg.json").write_text(
            json.dumps(
                {
                    "pcrelate": {"king_allowed": False, "primary_phi_threshold": 0.0442,
                                 "descriptive_phi_thresholds": [0.0221]},
                    "pca_axis_contract": {"min_effective_individual_fraction_per_axis": 0.02},
                }
            ),
            encoding="utf-8",
        )
        out = cls.root / "out"
        out.mkdir()
        cls.result = subprocess.run(
            [
                sys.executable, str(REPO / "bin" / "m27d_kinship_attribution.py"),
                "--pairs", str(cls.root / "pairs.tsv.gz"),
                "--samples", str(cls.root / "universe.txt"),
                "--call-rates", str(cls.root / "call_rate.tsv"),
                "--strata", str(cls.root / "strata.tsv"),
                "--preregistration", str(cls.root / "prereg.json"),
                "--min-population-members", "3",
                "--out-summary", str(out / "summary.json"),
                "--out-table", str(out / "table.tsv"),
            ],
            capture_output=True, text=True, check=False,
        )
        cls.payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_it_ran(self):
        self.assertEqual(self.result.returncode, 0, msg=self.result.stderr)

    def test_a_complete_clique_has_a_ceiling_of_one(self):
        populations = self.payload["structural_ceiling"]["by_population"]
        self.assertEqual(populations["CLIQUE"]["alpha_exact"], 1)
        self.assertTrue(populations["CLIQUE"]["is_complete_clique"])
        self.assertEqual(populations["CLIQUE"]["shortfall_vs_local_ceiling"], 0)

    def test_a_chain_of_four_has_a_ceiling_of_two(self):
        populations = self.payload["structural_ceiling"]["by_population"]
        self.assertEqual(populations["CHAIN"]["alpha_exact"], 2)
        self.assertFalse(populations["CHAIN"]["is_complete_clique"])

    def test_a_population_without_edges_is_not_reported_as_a_loss(self):
        self.assertNotIn("LOOSE", self.payload["structural_ceiling"]["by_population"])

    def test_edges_on_the_pedigree_line_produce_a_zero_residual(self):
        """The fixture writes k0 = 1 - 4*phi, so the residual must vanish."""
        groups = self.payload["pedigree_locus"]["by_group"]
        self.assertAlmostEqual(groups["within_population"]["residual_median"], 0.0, places=6)
        self.assertEqual(groups["within_population"]["fraction_residual_above_0_10"], 0.0)

    def test_a_pair_table_without_ibd_probabilities_fails_closed(self):
        thin = self.root / "thin.tsv.gz"
        with gzip.open(thin, "wt", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["ID1", "ID2", "kin"])
            writer.writerow(["C0", "C1", 0.2])
        result = subprocess.run(
            [
                sys.executable, str(REPO / "bin" / "m27d_kinship_attribution.py"),
                "--pairs", str(thin),
                "--samples", str(self.root / "universe.txt"),
                "--call-rates", str(self.root / "call_rate.tsv"),
                "--strata", str(self.root / "strata.tsv"),
                "--preregistration", str(self.root / "prereg.json"),
                "--out-summary", str(self.root / "out" / "x.json"),
                "--out-table", str(self.root / "out" / "x.tsv"),
            ],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ibd.probs", result.stderr)

    def test_the_summary_states_what_it_cannot_decide(self):
        self.assertIn("what_this_cannot_decide", self.payload["pedigree_locus"])
        self.assertFalse(self.payload["new_pcrelate_pass_executed"])

    def test_no_sample_identifier_reaches_the_summary(self):
        text = json.dumps(self.payload)
        for sample in self.samples:
            self.assertNotIn(sample, text)


if __name__ == "__main__":
    unittest.main()
