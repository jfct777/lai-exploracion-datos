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


class TestG2IsThreeQuestionsNotOne(unittest.TestCase):
    """G2 mixed a technical check, a localisation measurement and a representativeness
    question into one verdict, and aborted runs on the third under the name of the first.
    """

    SCRIPT = (REPO / "bin" / "m27d_pca_projection.R").read_text(encoding="utf-8")
    ATTRIBUTION = (REPO / "bin" / "m27d_kinship_attribution.py").read_text(encoding="utf-8")
    AXIS_CONTRACT = CONTRACT["pca_axis_contract"]

    def test_the_contract_declares_the_three_parts_separately(self):
        for part in (
            "g2a_technical_integrity",
            "g2b_axis_localization",
            "g2c_ancestry_representativeness",
        ):
            self.assertIn(part, self.AXIS_CONTRACT)
        for gate in ("G2A_pca_technical_integrity", "G2B_axis_localization",
                     "G2C_ancestry_representativeness"):
            self.assertIn(gate, CONTRACT["gates"])
        self.assertNotIn("G2_pca_ancestry_not_family_or_source", CONTRACT["gates"])

    def test_only_the_technical_check_can_abort(self):
        self.assertEqual(
            self.AXIS_CONTRACT["g2a_technical_integrity"]["enforcement"],
            "PASS_REQUIRED_ABORTS_ON_FAIL",
        )
        self.assertEqual(
            self.AXIS_CONTRACT["g2b_axis_localization"]["enforcement"],
            "REPORT_ONLY_REVIEW_DOES_NOT_ABORT",
        )
        self.assertIn('identical(g2a_status, "FAIL")', self.SCRIPT)
        # The old unconditional stop on localisation must not survive anywhere.
        self.assertNotIn('identical(g2_status, "FAIL")', self.SCRIPT)

    def test_a_localised_axis_is_reviewed_and_not_condemned(self):
        localization = self.AXIS_CONTRACT["g2b_axis_localization"]
        self.assertEqual(localization["status_below_fraction"], "REVIEW")
        self.assertEqual(localization["review_blocks"], ["donor_certification"])
        # Five explanations compete for a low ratio; the metric picks none of them.
        self.assertGreaterEqual(len(localization["what_a_low_ratio_cannot_separate"]), 5)

    def test_representativeness_is_left_unadjudicated(self):
        block = self.AXIS_CONTRACT["g2c_ancestry_representativeness"]
        self.assertEqual(block["status"], "NOT_ADJUDICATED_REQUIRES_CALIBRATION")
        self.assertNotIn("review_fraction_per_axis", block)

    def test_each_configuration_is_judged_on_the_prefix_it_reads(self):
        """A four-component run must not be blocked by an eleventh axis it never sees."""
        self.assertIn("g2b_status_by_preregistered_n_pcs", self.SCRIPT)
        self.assertIn("seq_len(min(k, length(effective_individuals)))", self.SCRIPT)
        self.assertIn("status_by_leading_prefix", self.ATTRIBUTION)

    def test_the_review_message_names_the_axes_and_refuses_to_diagnose_them(self):
        self.assertIn("G2B marks", self.SCRIPT)
        self.assertIn("This is not a failure", self.SCRIPT)
        self.assertIn("the ratio alone does not separate them", self.SCRIPT)

    def test_the_certification_stop_is_wired_to_the_contract_and_not_hardcoded(self):
        """The only new brake in this change had no coverage at all.

        The integration fixture cannot reach it: its cohort is small enough that the 2 per
        cent bound lands near one effective individual, so no axis is ever REVIEW there.
        These assert the wiring instead of the arithmetic.
        """
        selection = (REPO / "bin" / "m27d_candidate_selection.py").read_text(encoding="utf-8")
        self.assertIn(
            'contract["pca_axis_contract"]["g2b_axis_localization"]["review_blocks"]',
            selection,
        )
        self.assertIn('"donor_certification" in blocking_review', selection)
        self.assertIn('status == "REVIEW"', selection)
        self.assertIn("if (failed or reviewed)", selection)

    def test_certification_does_not_promise_per_prefix_relief(self):
        """The union rule exposes a certified donor to every prefix, not only its own."""
        selection = (REPO / "bin" / "m27d_candidate_selection.py").read_text(encoding="utf-8")
        self.assertIn("union", selection.lower())
        localization = self.AXIS_CONTRACT["g2b_axis_localization"]
        self.assertEqual(localization["evaluated_on"].startswith("the leading prefix"), True)
        amendment = next(
            a for a in CONTRACT["operational_amendments"] if a["date"] == "2026-08-15"
        )
        self.assertEqual(amendment["per_prefix_relief_applies_to"], "the PCA stage only")
        self.assertIn("union", amendment["why_certification_still_aggregates"])

    def test_the_bound_has_one_definition_shared_by_r_and_python(self):
        """A number repeated in two languages is a number that will drift."""
        path = 'contract["pca_axis_contract"]["g2b_axis_localization"]["review_fraction_per_axis"]'
        self.assertIn(path, self.ATTRIBUTION)
        self.assertIn("localization_contract$review_fraction_per_axis", self.SCRIPT)
        for source in (self.SCRIPT, self.ATTRIBUTION):
            self.assertNotIn("0.02", source)


def build_arm(root: Path, name: str, pairs, f_values, n_pcs: int, **overrides) -> None:
    with gzip.open(root / f"m27d_pcrelate_{name}_pairs.private.tsv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["ID1", "ID2", "kin", "k0", "k2", "nsnp"])
        for left, right, kinship in pairs:
            # Below 2^(-5/2) GENESIS derives k0 from kin and k2; the fixture follows the
            # same rule so no test can read a vanishing residual as an observation.
            writer.writerow([left, right, kinship, 1 - 4 * kinship + 0.01, 0.01, 1000])
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
        # Two members sit just above 2^(-5/2) but still inside the reported band: that is
        # the only place where k0 is the estimator's own and not the substituted value.
        above = [f"A{i}" for i in range(2)]
        cls.samples = clique + chain + loose + above
        pairs = [(a, b, 0.06) for i, a in enumerate(clique) for b in clique[i + 1:]]
        pairs += [(chain[i], chain[i + 1], 0.06) for i in range(3)]
        with gzip.open(cls.root / "pairs.tsv.gz", "wt", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["ID1", "ID2", "kin", "k0", "k2", "nsnp"])
            for left, right, kinship in pairs:
                # k0 is written the way GENESIS writes it below the cutoff, from kin and
                # k2, so the fixture reproduces the substitution instead of inventing an
                # independent k0 whose vanishing residual would prove nothing.
                writer.writerow([left, right, kinship, 1 - 4 * kinship + 0.01, 0.01, 1000])
            writer.writerow([above[0], above[1], 0.1769, 0.42, 0.03, 1000])
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
                # LOOSE members keep a row but no resolvable population, which is the real
                # shape of an unmatched sample: the resolver never omits the row.
                if sample.startswith("L"):
                    writer.writerow([sample, "FALSE", "", "", "", "", ""])
                    continue
                label = {"C": "CLIQUE", "H": "CHAIN", "A": "ABOVE"}[sample[0]]
                writer.writerow([sample, "TRUE", "SRC", "A", label, "X", "FALSE"])
        (cls.root / "prereg.json").write_text(
            json.dumps(
                {
                    "pcrelate": {"king_allowed": False, "primary_phi_threshold": 0.0442,
                                 "descriptive_phi_thresholds": [0.0221]},
                    "pca_axis_contract": {
                        "g2b_axis_localization": {"review_fraction_per_axis": 0.02}
                    },
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

    def test_the_report_warns_that_genesis_derived_k0(self):
        """A reader must meet the warning where the numbers are, not in a commit message."""
        block = self.payload["k0_algebraic_identity"]
        self.assertEqual(block["diagnostic_class"], "NON_INDEPENDENT_DIAGNOSTIC")
        self.assertEqual(block["evidence_status"], "NOT_EVIDENCE_OF_IBD")
        self.assertIn("correctK0", block["warning"] + block["genesis_rule_source"])
        for token in ("kin", "k2"):
            self.assertIn(token, block["warning"])
        self.assertAlmostEqual(block["genesis_rule_cutoff"], 2.0**-2.5, places=12)

    def test_the_substitution_is_reproduced_and_reported_as_k2(self):
        block = self.payload["k0_algebraic_identity"]
        self.assertTrue(block["substitution_reproduced"])
        group = block["by_group"]["within_population"]
        self.assertAlmostEqual(group["k2_median"], 0.01, places=6)
        # The quantity is published under its own name; calling it a residual against the
        # pedigree line is the error this block exists to prevent.
        self.assertFalse([key for key in group if "residual" in key])

    def test_edges_above_the_cutoff_are_counted_apart(self):
        block = self.payload["k0_algebraic_identity"]
        self.assertEqual(block["n_edges_in_band_with_independently_estimated_k0"], 1)
        self.assertEqual(
            block["n_edges_in_band"],
            block["n_edges_in_band_with_k0_substituted"]
            + block["n_edges_in_band_with_independently_estimated_k0"],
        )

    def test_no_output_claims_the_edges_are_descent(self):
        text = json.dumps(self.payload).lower()
        for claim in ("shared descent", "genuine sharing", "real ibd", "identity-by-descent sharing"):
            self.assertNotIn(claim, text)

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
        self.assertIn("what_this_cannot_decide", self.payload["k0_algebraic_identity"])
        self.assertFalse(self.payload["new_pcrelate_pass_executed"])

    def test_no_sample_identifier_reaches_the_summary(self):
        text = json.dumps(self.payload)
        for sample in self.samples:
            self.assertNotIn(sample, text)


class TestTheCircularDiagnosticStaysRetired(unittest.TestCase):
    """Below 2^(-5/2) GENESIS writes k0 from kin and k2, so the old residual was k2.

    The claim it supported — that the edges are shared descent rather than an artifact of
    the fitted allele frequencies — was never testable with these numbers.  These are
    regression guards so it cannot come back under a new name.
    """

    ATTRIBUTION = (REPO / "bin" / "m27d_kinship_attribution.py").read_text(encoding="utf-8")
    PASS0_R = (REPO / "bin" / "m27d_pass0_pcrelate.R").read_text(encoding="utf-8")
    # Assembled at runtime so this file can be scanned along with the rest.
    RETIRED_NAME = "pedigree" + "_locus"

    def test_the_old_diagnostic_name_is_gone(self):
        sources = sorted((REPO / "bin").glob("m27d_*"))
        sources += sorted((REPO / "tests").glob("test_m27d_*.py"))
        for path in sources:
            if self.RETIRED_NAME in path.read_text(encoding="utf-8"):
                self.fail(f"{path} still refers to the retired diagnostic")

    def test_the_module_names_the_rule_that_makes_it_circular(self):
        self.assertIn("correctK0", self.ATTRIBUTION)
        self.assertIn("2.0**-2.5", self.ATTRIBUTION)

    def test_the_inbreeding_coefficient_is_not_offered_as_an_arbiter(self):
        """f comes from the same fit and re-enters the pair table through correctK2."""
        self.assertIn("cannot arbitrate", self.PASS0_R)
        self.assertIn("correctK2", self.PASS0_R)


if __name__ == "__main__":
    unittest.main()
