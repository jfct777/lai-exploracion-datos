import csv
import re
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

import m27d_prepare_sample_strata as strata  # noqa: E402


def metadata_row(**overrides):
    row = {
        "IID": "",
        "Sample_ID(Aliases)": "",
        "Illumina_ID": "",
        "original_IID": "",
        "Exclude": "FALSE",
        "N_genotypes": "1000",
        "Source": "NatWGS",
        "Ancestry": "Native_American",
        "Population": "P",
        "Country": "Brazil",
        "Maximum_unrelated_dataset": "",
    }
    row.update(overrides)
    return row


def resolve_one(sample, rows):
    return strata.resolve_rows([sample], rows)[0]


class TestM27DResolutionPolicy(unittest.TestCase):
    """Each case pins one branch of the documented resolution policy."""

    def test_single_reachable_row_resolves_directly(self):
        rows = [metadata_row(IID="A")]
        result = resolve_one("A", rows)
        self.assertEqual(result.method, strata.DIRECT_UNIQUE)
        self.assertEqual(result.match_status, "MATCHED")
        self.assertEqual(result.n_candidate_rows, 1)

    def test_alias_only_match_still_resolves(self):
        rows = [metadata_row(IID="OTHER", **{"Sample_ID(Aliases)": "A"})]
        result = resolve_one("A", rows)
        self.assertEqual(result.method, strata.DIRECT_UNIQUE)
        self.assertTrue(result.resolved)

    def test_collision_against_excluded_row_prefers_the_active_row(self):
        rows = [
            metadata_row(IID="A", Exclude="TRUE"),
            metadata_row(IID="A", Exclude="FALSE"),
        ]
        result = resolve_one("A", rows)
        self.assertEqual(result.method, strata.RESOLVED_ACTIVE_GENOTYPED_IID)
        self.assertEqual(result.row["Exclude"], "FALSE")

    def test_collision_against_ungenotyped_row_prefers_the_genotyped_row(self):
        rows = [
            metadata_row(IID="A", N_genotypes="0"),
            metadata_row(IID="A", N_genotypes="5000"),
        ]
        result = resolve_one("A", rows)
        self.assertEqual(result.row["N_genotypes"], "5000")

    def test_direct_iid_wins_over_alias_only_row(self):
        rows = [
            metadata_row(IID="OTHER", **{"Sample_ID(Aliases)": "A"}),
            metadata_row(IID="A"),
        ]
        result = resolve_one("A", rows)
        self.assertEqual(result.method, strata.RESOLVED_ACTIVE_GENOTYPED_IID)
        self.assertEqual(result.row["IID"], "A")

    def test_two_equally_valid_active_rows_fail_closed(self):
        rows = [metadata_row(IID="A"), metadata_row(IID="A")]
        result = resolve_one("A", rows)
        self.assertEqual(result.method, strata.AMBIGUOUS_FAIL_CLOSED)
        self.assertIsNone(result.row)
        self.assertFalse(result.resolved)

    def test_two_alias_rows_without_direct_iid_fail_closed(self):
        rows = [
            metadata_row(IID="B", **{"Sample_ID(Aliases)": "A"}),
            metadata_row(IID="C", **{"Sample_ID(Aliases)": "A"}),
        ]
        result = resolve_one("A", rows)
        self.assertEqual(result.method, strata.AMBIGUOUS_FAIL_CLOSED)

    def test_collision_where_no_row_has_genotypes_fails_closed(self):
        rows = [
            metadata_row(IID="A", N_genotypes="0"),
            metadata_row(IID="A", N_genotypes="0"),
        ]
        self.assertEqual(resolve_one("A", rows).method, strata.AMBIGUOUS_FAIL_CLOSED)

    def test_missing_n_genotypes_counts_as_no_genotypes(self):
        rows = [
            metadata_row(IID="A", N_genotypes=""),
            metadata_row(IID="A", N_genotypes="4000"),
        ]
        result = resolve_one("A", rows)
        self.assertEqual(result.row["N_genotypes"], "4000")

    def test_missing_exclude_is_not_treated_as_excluded(self):
        rows = [metadata_row(IID="A", Exclude=""), metadata_row(IID="A", Exclude="TRUE")]
        result = resolve_one("A", rows)
        self.assertEqual(result.row["Exclude"], "")

    def test_collision_of_only_excluded_rows_still_reaches_the_iid_rule(self):
        rows = [
            metadata_row(IID="OTHER", Exclude="TRUE", **{"Sample_ID(Aliases)": "A"}),
            metadata_row(IID="A", Exclude="TRUE"),
        ]
        result = resolve_one("A", rows)
        self.assertEqual(result.method, strata.RESOLVED_ACTIVE_GENOTYPED_IID)
        self.assertEqual(result.row["IID"], "A")

    def test_sample_without_any_reachable_row_is_unmatched(self):
        rows = [metadata_row(IID="B")]
        result = resolve_one("A", rows)
        self.assertEqual(result.method, strata.UNMATCHED)
        self.assertEqual(result.match_status, "UNMATCHED")
        self.assertFalse(result.resolved)

    def test_duplicated_panel_prefix_is_normalised_before_matching(self):
        rows = [metadata_row(IID="A")]
        self.assertTrue(resolve_one("A_A", rows).resolved)

    def test_doubled_identifier_containing_an_underscore_is_normalised(self):
        """The four-field case ``A_B_A_B`` is what inflated the orphan count."""
        rows = [metadata_row(IID="ABC_123")]
        self.assertTrue(resolve_one("ABC_123_ABC_123", rows).resolved)

    def test_dedoubling_only_collapses_a_true_mirror(self):
        self.assertEqual(strata.dedoubled_panel_id("A_B_A_B"), "A_B")
        self.assertEqual(strata.dedoubled_panel_id("A_A"), "A")
        # Not a mirror: the halves differ, so the identifier must survive intact.
        self.assertEqual(strata.dedoubled_panel_id("A_B_C_D"), "A_B_C_D")
        # Odd field counts cannot be mirrors.
        self.assertEqual(strata.dedoubled_panel_id("A_B_A"), "A_B_A")
        self.assertEqual(strata.dedoubled_panel_id("PLAIN"), "PLAIN")

    def test_dedoubling_does_not_invent_a_match_for_a_distinct_sample(self):
        rows = [metadata_row(IID="A_B")]
        self.assertEqual(resolve_one("A_B_C_D", rows).method, strata.UNMATCHED)

    def test_separator_variants_inside_an_alias_cell_are_matched(self):
        rows = [metadata_row(IID="B", **{"Sample_ID(Aliases)": "A; other"})]
        self.assertTrue(resolve_one("A", rows).resolved)

    def test_case_difference_does_not_match(self):
        """Identifier matching stays case sensitive; a case-only hit is not evidence."""
        rows = [metadata_row(IID="a")]
        self.assertEqual(resolve_one("A", rows).method, strata.UNMATCHED)

    def test_resolution_does_not_depend_on_row_order(self):
        first = [metadata_row(IID="A", Exclude="TRUE"), metadata_row(IID="A")]
        second = list(reversed(first))
        self.assertEqual(
            resolve_one("A", first).row["Exclude"], resolve_one("A", second).row["Exclude"]
        )


class TestM27DStrataOutputs(unittest.TestCase):
    def _write_inputs(self, root, rows, samples):
        vcf = root / "panel.vcf"
        header = "\t".join(
            ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *samples]
        )
        vcf.write_text(f"##fileformat=VCFv4.2\n{header}\n", encoding="utf-8")
        metadata = root / "metadata.tsv"
        fields = list(rows[0].keys())
        with metadata.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return vcf, metadata

    def _args(self, root, vcf, metadata, suppress_below=1):
        return type(
            "Args",
            (),
            {
                "panel_vcf": vcf,
                "metadata": metadata,
                "private_out": root / "private.tsv",
                "summary_out": root / "summary.json",
                "suppress_below": suppress_below,
            },
        )()

    def test_public_summary_hides_identifiers_and_flags_interpretability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [metadata_row(IID="S1"), metadata_row(IID="S2", Country="Kenya")]
            vcf, metadata = self._write_inputs(root, rows, ["S1_S1", "S2", "GHOST"])
            args = self._args(root, vcf, metadata)

            summary = strata.run(args)

            self.assertEqual(summary["n_panel_samples"], 3)
            self.assertEqual(summary["n_matched"], 2)
            self.assertEqual(summary["n_unmatched"], 1)
            self.assertEqual(summary["n_population_not_interpretable"], 1)
            public = args.summary_out.read_text(encoding="utf-8")
            for identifier in ("S1_S1", "S1", "S2", "GHOST"):
                self.assertNotIn(identifier, public)

            with args.private_out.open(encoding="utf-8") as handle:
                private = list(csv.DictReader(handle, delimiter="\t"))
            by_id = {row["sample_id"]: row for row in private}
            self.assertEqual(by_id["S1_S1"]["population_interpretable"], "TRUE")
            self.assertEqual(by_id["GHOST"]["population_interpretable"], "FALSE")
            self.assertEqual(by_id["GHOST"]["Country"], "")
            self.assertEqual(by_id["GHOST"]["resolution_method"], strata.UNMATCHED)

    def test_unresolved_collision_fails_the_stage_and_keeps_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [metadata_row(IID="S1"), metadata_row(IID="S1")]
            vcf, metadata = self._write_inputs(root, rows, ["S1"])
            args = self._args(root, vcf, metadata)

            with self.assertRaises(SystemExit):
                strata.run(args)

            summary = json.loads(args.summary_out.read_text(encoding="utf-8"))
            self.assertEqual(summary["resolution_methods"][strata.AMBIGUOUS_FAIL_CLOSED], 1)

    def test_unmatched_samples_are_absent_from_aggregate_strata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [metadata_row(IID="S1", Population="ONLY")]
            vcf, metadata = self._write_inputs(root, rows, ["S1", "GHOST"])
            summary = strata.run(self._args(root, vcf, metadata))
            total = sum(entry["n"] for entry in summary["strata_counts_suppressed"])
            self.assertEqual(total, 1)


class TestM27DNoKing(unittest.TestCase):
    SCIENTIFIC_SCRIPTS = (
        "m27d_prepare_genotype_resources.R",
        "m27d_resource_smoke.R",
        "m27d_prepare_sample_strata.py",
        "m27d_pass0_pcrelate.R",
        "m27d_pca_projection.R",
        "m27d_pcrelate_configuration.R",
        "m27d_kinship_graph.py",
        "m27d_candidate_selection.py",
        "m27d_baseline_identity.R",
        "m27d_common.R",
        "verify_m27d_prepared_inputs.py",
    )

    def test_scientific_scripts_do_not_call_king_or_pcair(self):
        root = Path(__file__).resolve().parents[1]
        forbidden = ("snpgdsIBDKING", "pcair(", "kingToMatrix", "king2mat", "KINGmat")
        for name in self.SCIENTIFIC_SCRIPTS:
            path = root / "bin" / name
            self.assertTrue(path.exists(), msg=f"missing scientific script {name}")
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, msg=f"{token} found in {name}")

    def test_module_and_workflow_never_reference_the_legacy_king_script(self):
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "modules/27D_DONOR_KINSHIP_AUDIT.nf",
            "workflows/m27d_donor_kinship_audit.nf",
        ):
            text = (root / relative).read_text(encoding="utf-8")
            self.assertNotIn("pcrelate_kinship.R", text, msg=relative)

    def test_cloud_stages_are_pinned_and_labeled(self):
        root = Path(__file__).resolve().parents[1]
        cloud = (root / "conf" / "google_batch.config").read_text(encoding="utf-8")
        module = (root / "modules" / "27D_DONOR_KINSHIP_AUDIT.nf").read_text(encoding="utf-8")
        digest = "3a4661e41f7e397e986472bb8039671f85b1e8f7b86fc26af83a9837ef83d954"
        self.assertIn("resourceLabels = [team: 'frank']", cloud)
        self.assertIn(f"dnabr-qc@sha256:{digest}", cloud)
        self.assertIn("m27d_prepared_input_verification.json", module)
        self.assertIn("time = '75m'", cloud)

    AUDIT_PROCESSES = (
        "VERIFY_DONOR_KINSHIP_PREPARED_INPUTS",
        "RESOLVE_DONOR_KINSHIP_STRATA",
        "AUDIT_BASELINE_DONOR_IDENTITY",
        "RUN_DONOR_KINSHIP_PASS0",
        "FIT_DONOR_KINSHIP_PCA",
        "RUN_DONOR_KINSHIP_CONFIGURATION",
        "SELECT_DONOR_KINSHIP_CANDIDATES",
        "COMPARE_DONOR_KINSHIP_PC_COUNT",
        "WRITE_DONOR_KINSHIP_RUN_PROVENANCE",
    )

    def test_every_audit_process_is_pinned_to_the_analysis_container(self):
        """A process without an explicit container would silently inherit another one."""
        root = Path(__file__).resolve().parents[1]
        cloud = (root / "conf" / "google_batch.config").read_text(encoding="utf-8")
        for process in self.AUDIT_PROCESSES:
            line = next(
                (row for row in cloud.splitlines() if f"withName: '{process}'" in row), None
            )
            self.assertIsNotNone(line, msg=f"{process} has no cloud block")
            for directive in ("container", "disk", "maxForks"):
                self.assertIn(directive, line, msg=f"{process} does not declare {directive}")

    def test_cpu_memory_and_time_have_a_single_source_of_truth(self):
        """The module owns them; restating them in the cloud config could drift.

        The cloud config also cannot resolve a param declared in another configuration
        file, so referencing them there aborts the run before any task is created.
        """
        root = Path(__file__).resolve().parents[1]
        cloud = (root / "conf" / "google_batch.config").read_text(encoding="utf-8")
        module = (root / "modules" / "27D_DONOR_KINSHIP_AUDIT.nf").read_text(encoding="utf-8")
        for process in self.AUDIT_PROCESSES:
            line = next(row for row in cloud.splitlines() if f"withName: '{process}'" in row)
            for directive in ("cpus", "memory", "time"):
                self.assertNotIn(directive, line, msg=f"{process} restates {directive} in the cloud config")
        for directive in ("cpus", "memory", "time"):
            self.assertIn(directive, module, msg=f"the module never declares {directive}")

    def test_cloud_config_never_reads_a_param_it_does_not_define(self):
        """Nextflow aborts on a cross-file params reference inside a process block."""
        root = Path(__file__).resolve().parents[1]
        cloud = (root / "conf" / "google_batch.config").read_text(encoding="utf-8")
        defined = set(re.findall(r"^\s{2}(\w+)\s*=", cloud, flags=re.MULTILINE))
        referenced = set(re.findall(r"params\.(\w+)", cloud))
        missing = sorted(referenced - defined)
        self.assertEqual(missing, [], msg=f"referenced but not defined here: {missing}")

    def test_full_audit_is_gated_behind_an_explicit_authorization(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / "workflows" / "m27d_donor_kinship_audit.nf").read_text(
            encoding="utf-8"
        )
        config = (root / "nextflow.config").read_text(encoding="utf-8")
        self.assertIn("donor_kinship_full_run_authorized", workflow)
        self.assertIn("donor_kinship_full_run_authorized = false", config)
        self.assertIn("is missing prepared inputs", workflow)
        self.assertIn("donor_kinship_preparation_manifest_sha256", workflow)
        self.assertIn("PREPARE_DONOR_KINSHIP_RESOURCES", workflow)

    def test_configurations_are_read_from_the_preregistration_not_restated(self):
        """Hardcoding the four configurations in Groovy would let code and contract drift."""
        root = Path(__file__).resolve().parents[1]
        workflow = (root / "workflows" / "m27d_donor_kinship_audit.nf").read_text(
            encoding="utf-8"
        )
        contract = json.loads(
            (root / "conf" / "m27d_donor_kinship_preregistration.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("contract.configurations", workflow)
        for configuration in contract["configurations"]:
            self.assertNotIn(configuration["id"], workflow)
        self.assertEqual(len(contract["configurations"]), 4)
        # Exactly one primary and three one-factor sensitivities.
        roles = Counter(config["role"] for config in contract["configurations"])
        self.assertEqual(roles["primary"], 1)
        self.assertEqual(roles["sensitivity_one_factor"], 3)

    def test_each_sensitivity_changes_exactly_one_factor(self):
        root = Path(__file__).resolve().parents[1]
        contract = json.loads(
            (root / "conf" / "m27d_donor_kinship_preregistration.json").read_text(
                encoding="utf-8"
            )
        )
        configurations = contract["configurations"]
        primary = next(c for c in configurations if c["role"] == "primary")
        for configuration in configurations:
            if configuration["role"] == "primary":
                continue
            differing = sum(
                1
                for factor in ("n_pcs", "ld_r2_max")
                if configuration[factor] != primary[factor]
            )
            self.assertEqual(
                differing, 1, msg=f"{configuration['id']} changes {differing} factors"
            )

    def test_batch_style_script_staging_resolves_helper_from_workdir(self):
        root = Path(__file__).resolve().parents[1]
        module = (root / "modules" / "27D_DONOR_KINSHIP_AUDIT.nf").read_text(encoding="utf-8")
        self.assertIn("PYTHONPATH=. python3 ${sample_strata_py}", module)

        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            script_dir = temp_root / "batch-script"
            work_dir = temp_root / "work"
            script_dir.mkdir()
            work_dir.mkdir()
            script = script_dir / "m27d_prepare_sample_strata.py"
            script.write_bytes((root / "bin" / script.name).read_bytes())
            helper = work_dir / "audit_rare_scaffold_bridge.py"
            helper.write_bytes((root / "bin" / helper.name).read_bytes())
            (work_dir / "panel.vcf").write_text(
                "##fileformat=VCFv4.2\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n",
                encoding="utf-8",
            )
            (work_dir / "metadata.tsv").write_text(
                "IID\tSample_ID(Aliases)\tIllumina_ID\toriginal_IID\tExclude\tN_genotypes\t"
                "Source\tAncestry\tPopulation\tCountry\tMaximum_unrelated_dataset\n"
                "S1\tS1\tS1\tS1\tFALSE\t1000\tNatWGS\tNAM\tP1\tBrazil\t\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "."
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--panel-vcf",
                    "panel.vcf",
                    "--metadata",
                    "metadata.tsv",
                    "--private-out",
                    "private.tsv",
                    "--summary-out",
                    "summary.json",
                ],
                cwd=work_dir,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
