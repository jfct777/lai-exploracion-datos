"""Tiny synthetic contracts only: no real subjects, cloud or analysis jobs."""
import csv
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"bin"))
import r02_biological_evaluation as evaluation


def write_tsv(path, rows, fields):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class BiologicalEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.samples = self.root/"samples.txt"
        self.samples.write_text("B\nA\nC\nD\n")  # Not lexical; use exact analytical order.
        self.kin = self.root/"kin.tsv"
        self.kin.write_text("ID1\tID2\tkin\nB\tA\t0.05\nA\tC\tNA\nD\tC\t0.01\n")
        self.full, self.strict, self.empty = "L10_G10_N2", "L20_G10_N3", "L100_G20_N6"
        self.configs = [dict(zip(evaluation.CONFIG_FIELDS, row)) for row in
                        ((self.full, 10, 10, 2), (self.strict, 10, 20, 3), (self.empty, 20, 100, 6))]
        self.rows = [self.segment("B", "A", 1, 10, 2, 4, 5, 6, 1),
                     self.segment("B", "A", 21, 40, 3, 6, 8, 10, 3),
                     self.segment("A", "C", 101, 110, 2, 2, 2, 2, 0)]
        self.rows[0].update(O=0, Q_C=3, common_status="OK", length_cm=.1, map_status="OK",
                            callable_bp=8, ibd_union_bp=4, ibd_fraction=.5, ibd_status="OK")
        self.rows[1].update(O=1, Q_C=4, common_status="OK")
        self.rows[2].update(length_cm=.05, map_status="OK")
        self.bundle = self.root/"chr22"
        self.bundle.mkdir()
        self.manifest = self.bundle/"manifest.json"
        self.write_bundle()

    def segment(self, a, b, start, end, shared, union, called, catalog, missing):
        return dict(chain_id="chain_"+hashlib.sha256(f"22/{a}/{b}/{start}/{end}/10".encode()).hexdigest(),
            chrom="chr22", sample_a=a, sample_b=b, max_gap_bp=10, start_pos=start, end_pos=end,
            length_bp=end-start+1, n_shared_variants=shared, I=shared, U=union, Q=called,
            rare_catalog_sites=catalog, rare_missing_gt_count=missing, J=shared/union, rare_status="OK",
            O="NA", Q_C="NA", common_status="NO_EVALUABLE:common_input_absent",
            length_cm="NA", map_status="NO_EVALUABLE:outside_map",
            callable_bp="NA", ibd_union_bp="NA", ibd_fraction="NA", ibd_status="NO_EVALUABLE:IBD_input_absent")

    def write_bundle(self, links=None):
        if links is None:
            links = [dict(chain_id=row["chain_id"], config_id=config["config_id"])
                     for row in self.rows for config in self.configs
                     if row["max_gap_bp"] == config["max_gap_bp"] and
                     row["length_bp"] >= config["min_length_bp"] and
                     row["n_shared_variants"] >= config["min_shared_effective"]]
        write_tsv(self.bundle/evaluation.FILES[0], self.rows, evaluation.SEGMENT_FIELDS)
        write_tsv(self.bundle/evaluation.FILES[1], links, ("chain_id", "config_id"))
        write_tsv(self.bundle/evaluation.FILES[2], self.configs, evaluation.CONFIG_FIELDS)
        self.territory_fields = ("chrom", "sample_a", "sample_b", "callable_bp", "ibd_union_bp", "ibd_status")
        self.territory = [dict(zip(self.territory_fields, row)) for row in
                          (("22", "B", "A", 100, 50, "OK"), ("22", "B", "D", 100, 20, "OK"))]
        write_tsv(self.bundle/evaluation.IBD_TERRITORY, self.territory, self.territory_fields)
        self.record = dict(schema="r02_segment_evidence_v1", status="COMPLETE_DESCRIPTIVE_NOT_VALIDATED",
            sample_ids_sha256=evaluation.sha256(self.samples), n_samples=4, chrom="22", genome_build="GRCh38",
            coordinate_system="1-based-inclusive",
            source_rare_contract=dict(dnabr_original_alleles="v1", dnabr_rare_contract="minor_v1",
                                     dnabr_rare_cohort_n_samples="5", dnabr_rare_cohort_sha256="a"*64),
            evidence_criteria=dict(rare_allele="fixed_source_minor_minor_v1",
                rare_gt="complete_diploid_GT_RD_validated", rare_missingness="joint_complete_calls_no_imputation",
                common_criteria=dict(min_maf=.05, max_missing=.02),
                ibd_quantity="any_copy_ibd_territory", coordinate_system="1-based-inclusive",
                ibd_criteria=dict(caller=dict(name="synthetic", version="1"), absence_means_no_called_ibd_within_callable=True),
                quality_filters="new_DP_GQ_filters_not_applied"),
            outputs_sha256={name: evaluation.sha256(self.bundle/name) for name in (*evaluation.FILES, evaluation.IBD_TERRITORY)})
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.record))

    def args(self):
        return dict(segment_manifests=[self.manifest], sample_ids=self.samples, expected_samples=4,
                    pcrelate_file=self.kin, expected_pcrelate_sha256=evaluation.sha256(self.kin),
                    expected_configurations=3, output_dir=self.root/"result", edge_thresholds_bp=(0, 25),
                    scratch_dir=self.root, min_free_disk_mb=0)

    def study_fixture(self):
        folder = self.root/"study"
        folder.mkdir(exist_ok=True)
        people = [dict(sample_id=sid, role="UNASSIGNED", Cohort="ONE", Batch="UNKNOWN",
                       component_phi_0_0221=component, component_phi_0_0442=component)
                  for sid, component in zip(("B", "A", "C", "D"), ("depAB", "depAB", "depC", "depD"))]
        people = [{key.replace("phi_0_", "phi_0."): value for key, value in person.items()} for person in people]
        write_tsv(folder/"persons.private.tsv", people, tuple(people[0]))
        record = dict(schema="r02_study_design_v1", sample_ids_sha256=evaluation.sha256(self.samples),
            status="COMPLETE_DESCRIPTIVE_AUDIT", n_samples=4, thresholds=[.0221, .0442], roles_supplied=False,
            nominal_columns=["Cohort"],
            kinship_audit=dict(sha256=evaluation.sha256(self.kin)),
            outputs_sha256={"persons.private.tsv": evaluation.sha256(folder/"persons.private.tsv")},
            training_status="NOT_AUTHORIZED")
        path = folder/"study_contract.json"
        path.write_text(json.dumps(record))
        return path

    def run_evaluation(self, **changes):
        args = self.args()
        args.update(changes)
        return evaluation.run(**args)

    def other_chromosome(self, chrom="21", preserve_ids=False):
        other = self.root/("chr"+chrom)
        shutil.copytree(self.bundle, other)
        translated = {row["chain_id"]: row["chain_id"] if preserve_ids else
            "chain_"+hashlib.sha256((chrom+"/"+row["chain_id"]).encode()).hexdigest() for row in self.rows}
        write_tsv(other/evaluation.FILES[0], [dict(row, chain_id=translated[row["chain_id"]], chrom=chrom)
            for row in self.rows], evaluation.SEGMENT_FIELDS)
        links = [dict(row, chain_id=translated[row["chain_id"]]) for row in
            evaluation.read_table(self.bundle/evaluation.FILES[1], ("chain_id", "config_id"))]
        write_tsv(other/evaluation.FILES[1], links, ("chain_id", "config_id"))
        write_tsv(other/evaluation.IBD_TERRITORY, [dict(row, chrom=chrom) for row in self.territory],
                  self.territory_fields)
        record = dict(self.record, chrom=chrom, outputs_sha256={name: evaluation.sha256(other/name)
            for name in self.record["outputs_sha256"]})
        (other/"manifest.json").write_text(json.dumps(record))
        return other/"manifest.json"

    def output_rows(self, name="configuration_metrics.tsv"):
        with (self.root/"result"/name).open() as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def metric(self, name, config=None, edge=0):
        return next(row for row in self.output_rows() if row["config_id"] == (config or self.full)
                    and row["metric"] == name and int(row["min_edge_bp"]) == edge)

    def test_exact_aggregate_denominators_not_mean_ratios_and_partial_evidence(self):
        result = self.run_evaluation()
        j = self.metric("rare_local_J")
        self.assertEqual((j["numerator"], j["denominator"]), ("7", "12"))
        self.assertAlmostEqual(float(j["value"]), 7/12)
        self.assertEqual(j["status"], "OBSERVED")
        quality = self.metric("rare_missing_GT_fraction")
        self.assertEqual((quality["numerator"], quality["denominator"]), ("4", "36"))
        common = self.metric("common_opposite_homozygote_fraction")
        self.assertEqual((common["numerator"], common["denominator"]), ("1", "7"))
        self.assertEqual(common["status"], "OBSERVED_PARTIAL")
        ibd = self.metric("ibd_union_coverage_fraction")
        self.assertEqual((ibd["numerator"], ibd["denominator"]), ("4", "8"))
        self.assertEqual(ibd["n_segments_evaluable"], "1")
        self.assertEqual(ibd["n_segments_total"], "3")
        self.assertEqual(result["status"], evaluation.STATUS)
        self.assertTrue(result["no_pvalues"] and result["no_winner_selected"] and result["no_population_labels"])
        self.assertFalse(result["contains_individual_identifiers"])
        self.assertEqual(result["n_cards"], 6)

    def test_empty_configs_and_zero_denominators_are_not_biological_negatives(self):
        self.run_evaluation()
        self.assertEqual(self.metric("people_represented_fraction", self.empty)["status"], "OBSERVED_ZERO")
        self.assertEqual(self.metric("rare_local_J", self.empty)["value"], "NA")
        self.assertEqual(self.metric("rare_local_J", self.empty)["status"], "NO_EVALUABLE")
        self.assertEqual(self.metric("ibd_union_coverage_fraction", self.strict)["status"], "NO_EVALUABLE")
        self.assertEqual(self.metric("biological_negative_result")["status"], "NO_EVALUABLE")
        self.assertAlmostEqual(float(self.metric("ibd_recall_all_evaluable_pairs")["value"]), 4/70)
        self.assertEqual(self.metric("ibd_recall_all_evaluable_pairs", self.empty)["status"], "OBSERVED_ZERO")

    def test_ibd_recall_denominator_includes_pairs_without_M14_and_complement(self):
        self.run_evaluation()
        recalled = self.metric("ibd_recall_all_evaluable_pairs")
        outside = self.metric("ibd_fraction_outside_M14")
        self.assertEqual((recalled["numerator"], recalled["denominator"]), ("4", "70"))
        self.assertEqual((outside["numerator"], outside["denominator"]), ("66", "70"))
        self.assertEqual(self.metric("ibd_recall_all_evaluable_pairs", self.strict)["denominator"], "70")

    def test_absent_ibd_contract_is_NA_and_not_zero_recall(self):
        self.rows[0].update(callable_bp="NA", ibd_union_bp="NA", ibd_fraction="NA", ibd_status="NO_EVALUABLE:no_IBD")
        self.write_bundle()
        del self.record["outputs_sha256"][evaluation.IBD_TERRITORY]
        self.record["evidence_criteria"]["ibd_quantity"] = None
        self.write_manifest()
        self.run_evaluation()
        self.assertEqual(self.metric("ibd_recall_all_evaluable_pairs")["status"], "NO_EVALUABLE")

    def test_local_ibd_overlap_cannot_exceed_global_pair_union(self):
        self.territory[0]["ibd_union_bp"] = 2
        write_tsv(self.bundle/evaluation.IBD_TERRITORY, self.territory, self.territory_fields)
        self.record["outputs_sha256"][evaluation.IBD_TERRITORY] = evaluation.sha256(self.bundle/evaluation.IBD_TERRITORY)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "exceed authenticated pair territory"):
            self.run_evaluation()

    def test_observed_zero_not_missing_and_map_zero_is_measured(self):
        self.rows[0].update(ibd_union_bp=0, ibd_fraction=0, length_cm=0)
        self.rows[1].update(O=0)
        self.write_bundle()
        self.run_evaluation()
        self.assertEqual(self.metric("ibd_union_coverage_fraction")["status"], "OBSERVED_ZERO_PARTIAL")
        self.assertEqual(self.metric("ibd_union_coverage_fraction")["value"], "0.0")
        self.assertEqual(self.metric("common_opposite_homozygote_fraction")["status"], "OBSERVED_ZERO_PARTIAL")
        self.assertEqual(self.metric("map_evaluable_interval_fraction")["numerator"], "2")

    def test_pair_threshold_is_applied_after_sum_and_kinship_keeps_nonfinite(self):
        self.run_evaluation()
        self.assertEqual(self.metric("rare_local_J", edge=25)["numerator"], "5")
        self.assertEqual(self.metric("rare_local_J", edge=25)["denominator"], "10")
        row = next(row for row in self.output_rows("configuration_kinship.tsv")
                   if row["config_id"] == self.full and row["min_edge_bp"] == "0")
        self.assertEqual(row["n_pairs"], "2")
        self.assertEqual(row["n_kin_ge_threshold"], "1")
        self.assertEqual(row["n_missing_nonfinite"], "1")
        self.assertEqual(row["n_missing_absent"], "0")

    def test_inter_configuration_overlap_is_not_pooled(self):
        self.run_evaluation()
        self.assertEqual(self.metric("rare_local_J", self.strict)["numerator"], "3")
        self.assertEqual(self.metric("rare_local_J", self.full)["numerator"], "7")
        cards = json.loads((self.root/"result/configuration_cards.json").read_text())["cards"]
        self.assertTrue(all(card["decision_status"] == "DESCRIPTIVE_ONLY_NO_SELECTION" for card in cards))
        self.assertTrue(all("own selected territory" in card["comparison_scope"] for card in cards))

    def test_overlapping_or_duplicate_geometry_inside_config_fails(self):
        self.rows.append(self.segment("B", "A", 5, 14, 2, 3, 4, 4, 0))
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "Overlapping intervals"):
            self.run_evaluation()
        self.assertFalse((self.root/"result").exists())

    def test_duplicate_chain_and_duplicate_memberships_fail(self):
        self.rows.append(dict(self.rows[0]))
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "Duplicate chain"):
            self.run_evaluation()
        self.rows.pop()
        links = [dict(chain_id=self.rows[0]["chain_id"], config_id=self.full)]*2
        self.write_bundle(links)
        with self.assertRaisesRegex(ValueError, "Duplicate chain"):
            self.run_evaluation()

    def test_memberships_must_be_valid_and_complete(self):
        self.write_bundle([])
        with self.assertRaisesRegex(ValueError, "Incomplete chain/configuration"):
            self.run_evaluation()
        self.write_bundle([dict(chain_id=self.rows[0]["chain_id"], config_id=self.empty)])
        with self.assertRaisesRegex(ValueError, "Invalid chain/configuration"):
            self.run_evaluation()
        self.write_bundle([dict(chain_id="absent", config_id=self.full)])
        with self.assertRaisesRegex(ValueError, "unknown/wrong-chromosome"):
            self.run_evaluation()

    def test_hash_cohort_order_source_status_and_empty_ledger_fail(self):
        self.record["outputs_sha256"][evaluation.FILES[0]] = "f"*64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_evaluation()
        self.write_bundle()
        self.samples.write_text("A\nB\nC\nD\n")
        with self.assertRaisesRegex(ValueError, "cohort/order mismatch"):
            self.run_evaluation()
        self.samples.write_text("B\nA\nC\nD\n")
        self.record["status"] = "RUNNING"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Incomplete/unsupported"):
            self.run_evaluation()
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "Incomplete configuration"):
            self.run_evaluation(expected_configurations=74)

    def test_invalid_counts_nonfinite_ratios_and_territory_fail(self):
        for patch, message in ((dict(I=9), "Invalid rare"), (dict(J="NaN"), "Nonfinite"),
                               (dict(J=.99), "Inconsistent J"), (dict(rare_missing_gt_count=6), "GT missingness"),
                               (dict(O=4), "Opposite homozygotes"),
                               (dict(callable_bp=11), "Callable territory"),
                               (dict(ibd_union_bp=9), "IBD union"),
                               (dict(ibd_status="NO_EVALUABLE:missing"), "IBD status")):
            with self.subTest(patch=patch):
                original = dict(self.rows[0])
                self.rows[0].update(patch)
                self.write_bundle()
                with self.assertRaisesRegex(ValueError, message):
                    self.run_evaluation()
                self.rows[0] = original

    def test_undefined_ratio_requires_NA_not_zero(self):
        evaluation.check_ratio("NA", 0, 0, "J")
        with self.assertRaisesRegex(ValueError, "must be NA"):
            evaluation.check_ratio("0", 0, 0, "J")

    def test_duplicate_chromosome_and_incompatible_criteria_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate or unsupported chromosome"):
            self.run_evaluation(segment_manifests=[self.manifest, self.manifest])
        other = self.root/"chr21"
        shutil.copytree(self.bundle, other)
        record = json.loads((other/"manifest.json").read_text())
        record["chrom"] = "21"
        record["genome_build"] = "GRCh37"
        (other/"manifest.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Incompatible chromosome"):
            self.run_evaluation(segment_manifests=[self.manifest, other/"manifest.json"])

    def test_autosomal_pair_threshold_uses_all_chromosomes_and_fixed_IBD_universe(self):
        other = self.root/"chr21"
        shutil.copytree(self.bundle, other)
        translated = {row["chain_id"]: "chain_"+hashlib.sha256(("21/"+row["chain_id"]).encode()).hexdigest() for row in self.rows}
        rows = [dict(row, chain_id=translated[row["chain_id"]], chrom="21") for row in self.rows]
        write_tsv(other/evaluation.FILES[0], rows, evaluation.SEGMENT_FIELDS)
        links = [dict(row, chain_id=translated[row["chain_id"]]) for row in evaluation.read_table(self.bundle/evaluation.FILES[1], ("chain_id", "config_id"))]
        write_tsv(other/evaluation.FILES[1], links, ("chain_id", "config_id"))
        write_tsv(other/evaluation.IBD_TERRITORY, [dict(row, chrom="21") for row in self.territory], self.territory_fields)
        record = dict(self.record, chrom="21", outputs_sha256={name: evaluation.sha256(other/name) for name in self.record["outputs_sha256"]})
        (other/"manifest.json").write_text(json.dumps(record))
        result = self.run_evaluation(segment_manifests=[self.manifest, other/"manifest.json"], edge_thresholds_bp=(0, 50))
        self.assertEqual(result["chromosomes"], ["21", "22"])
        self.assertEqual(self.metric("rare_local_J", edge=50)["numerator"], "10")
        self.assertEqual(self.metric("rare_local_J", edge=50)["denominator"], "20")
        self.assertEqual(self.metric("ibd_recall_all_evaluable_pairs", edge=50)["denominator"], "140")
        self.assertEqual(self.metric("ibd_recall_all_evaluable_pairs", edge=50)["numerator"], "8")

    def test_chromosomes_cannot_mix_IBD_callers_or_absence_semantics(self):
        other = self.root/"chr21"
        shutil.copytree(self.bundle, other)
        record = json.loads((other/"manifest.json").read_text())
        record["chrom"] = "21"
        record["evidence_criteria"]["ibd_criteria"]["caller"]["version"] = "different"
        (other/"manifest.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Incompatible optional evidence criteria"):
            self.run_evaluation(segment_manifests=[self.manifest, other/"manifest.json"])

    def test_optional_evidence_definition_cannot_be_missing_when_measured(self):
        self.record["evidence_criteria"]["common_criteria"] = None
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Measured commons lack"):
            self.run_evaluation()

    def test_existing_study_contract_is_linked_not_an_authorization(self):
        study = self.study_fixture()
        result = self.run_evaluation(study_contract=study)
        self.assertEqual(result["study_contract"]["sha256"], evaluation.sha256(study))
        self.assertEqual(result["inference_status"], "NO_EVALUABLE_DESCRIPTIVE_ONLY")
        rows = self.output_rows("configuration_components.tsv")
        row = next(row for row in rows if row["config_id"] == self.full and row["min_edge_bp"] == "0")
        self.assertEqual(row["n_components_total"], "3")
        self.assertEqual(row["n_dependence_components_represented"], "2")
        self.assertEqual(row["n_components_in_cross_edges"], "2")
        self.assertEqual(row["n_within_component_pairs"], "1")
        self.assertEqual(row["n_between_component_pairs"], "1")
        self.assertAlmostEqual(float(row["within_component_weight"]), math.log1p(30))
        self.assertAlmostEqual(float(row["total_edge_weight"]), math.log1p(30)+math.log1p(10))
        self.assertEqual(float(row["max_component_incident_weight_fraction"]), 1)
        metadata = self.output_rows("configuration_metadata_coverage.tsv")
        self.assertTrue(all(row["n_measured"] == "0" for row in metadata if row["field"] in ("role", "Batch")))
        known = next(row for row in metadata if row["config_id"] == self.full and row["min_edge_bp"] == "0"
                     and row["field"] == "Cohort" and row["stratum"] == "represented")
        self.assertEqual((known["n_measured"], known["n_people"]), ("3", "3"))
        categories = self.output_rows("configuration_metadata_categories.private.tsv")
        self.assertTrue(all(row["field"] == "Cohort" and row["category"] == "ONE" for row in categories))
        nominal = next(row for row in categories if row["config_id"] == self.full and row["min_edge_bp"] == "0"
                       and row["stratum"] == "represented")
        self.assertEqual((nominal["n_people_category"], nominal["n_people"]), ("3", "3"))
        self.assertIn("configuration_metadata_categories.private.tsv", result["privacy"]["private_outputs"])

    def test_study_tamper_and_mismatched_kinship_criteria_fail(self):
        study = self.study_fixture()
        (study.parent/"persons.private.tsv").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Study persons SHA256"):
            self.run_evaluation(study_contract=study)
        study = self.study_fixture()
        record = json.loads(study.read_text())
        record["kinship_audit"]["sha256"] = "f"*64
        study.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "kinship criteria differ"):
            self.run_evaluation(study_contract=study)

    def test_no_study_is_not_fabricated_component_independence(self):
        self.run_evaluation()
        rows = self.output_rows("configuration_components.tsv")
        self.assertTrue(all(row["status"] == "NO_EVALUABLE" for row in rows))
        self.assertTrue(all(row["n_dependence_components_represented"] == "NA" for row in rows))

    def test_cli_no_overwrite_and_resource_failure_have_no_completion(self):
        with self.assertRaisesRegex(ValueError, "database resource limit"):
            self.run_evaluation(max_database_mb=.00001)
        self.assertFalse((self.root/"result").exists())
        command = [sys.executable, str(ROOT/"bin/r02_biological_evaluation.py"),
            "--segment-manifest", str(self.manifest), "--sample-ids", str(self.samples),
            "--expected-samples", "4", "--expected-configurations", "3", "--pcrelate-file", str(self.kin),
            "--expected-pcrelate-sha256", evaluation.sha256(self.kin), "--output-dir", str(self.root/"result"),
            "--edge-thresholds-bp", "0,25", "--min-free-disk-mb", "0"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], evaluation.STATUS)
        with self.assertRaisesRegex(ValueError, "no overwrite"):
            self.run_evaluation()

    def test_legacy_all_scientific_outputs_exact_on_reference_fixture(self):
        # Captured by executing the pre-streaming evaluator, SHA256
        # b8ebfc857ed858a3afe635466948443671f03d8d648fcdd6c871867bd05e0723,
        # on this synthetic fixture with study_fixture(). Not production data.
        expected = {
            "configuration_cards.json": "fee91efed4bddeac0d50355924e8f618a9024bf1ab2f5ebca9a72ebb18ed8764",
            "configuration_chromosomes.tsv": "cd8ce8d464620dd2ce8461b4c2131240deb126914ecf02d785fb19ae4bebb0f2",
            "configuration_components.tsv": "d327fd19f47754df69d656830dee29e171d484710dad89d47645291231dfd165",
            "configuration_kinship.tsv": "156b33d60ea5f8fc0da8ed5d76ae525eec074aef5d8cf01133a5db1d269f9aaf",
            "configuration_metadata_categories.private.tsv": "ebb21a5961537dd944cee622b27675114e5242a33e0717bfbe91ea1c6f0c1530",
            "configuration_metadata_coverage.tsv": "4cae5bc81bbd6abe1581dbadb06ecdb4d8c76163f52a10f6025a84bf6c5e3708",
            "configuration_metrics.tsv": "f00e4e0851b74a9ab257376307cf1dd3b18090f7460847b7d2932ca40ffb39b9"}
        result = self.run_evaluation(study_contract=self.study_fixture())
        self.assertEqual(result["outputs_sha256"], expected)

    def test_chromosome_order_does_not_change_scientific_outputs(self):
        other = self.other_chromosome()
        study = self.study_fixture()
        first = self.run_evaluation(segment_manifests=[self.manifest, other], study_contract=study,
                                    edge_thresholds_bp=(0, 50))
        second = self.run_evaluation(segment_manifests=[other, self.manifest], study_contract=study,
                                     edge_thresholds_bp=(50, 0), output_dir=self.root/"reversed")
        self.assertEqual(first["outputs_sha256"], second["outputs_sha256"])

    def test_cross_chromosome_duplicate_chain_ids_are_still_rejected(self):
        other = self.other_chromosome(preserve_ids=True)
        with self.assertRaisesRegex(ValueError, "Duplicate chain ID"):
            self.run_evaluation(segment_manifests=[self.manifest, other])
        self.assertFalse((self.root/"result").exists())
        self.assertFalse(list(self.root.glob("r02-biological-evaluation-*")))

    def test_unsorted_source_rows_preserve_all_results(self):
        first = self.run_evaluation()
        self.rows.reverse()
        self.write_bundle()
        second = self.run_evaluation(output_dir=self.root/"unsorted")
        self.assertEqual(first["outputs_sha256"], second["outputs_sha256"])

    def test_sqlite_holds_one_chromosome_and_scratch_is_removed(self):
        other = self.other_chromosome()
        original = evaluation.load_bundle
        calls = []
        def checked_load(db, *args, **kwargs):
            calls.append(db.execute("SELECT count(*) FROM chains").fetchone()[0])
            return original(db, *args, **kwargs)
        with mock.patch.object(evaluation, "load_bundle", side_effect=checked_load):
            result = self.run_evaluation(segment_manifests=[self.manifest, other])
        self.assertEqual(calls, [0, 0])
        self.assertFalse(list(self.root.glob("r02-biological-evaluation-*")))
        self.assertGreaterEqual(result["resources"]["peak_observed_scratch_bytes"],
                                result["resources"]["peak_observed_sqlite_bytes"])
        self.assertIn("no internal resume", result["resources"]["restart"])

    def test_reserve_and_nonfinite_resource_limits_fail_closed(self):
        usage = shutil.disk_usage(self.root)
        with mock.patch.object(evaluation.shutil, "disk_usage", return_value=usage._replace(free=0)):
            with self.assertRaisesRegex(ValueError, "free-space reserve"):
                self.run_evaluation(min_free_disk_mb=1)
        for field in ("max_database_mb", "min_free_disk_mb"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Invalid scratch"):
                self.run_evaluation(**{field: math.inf})
        self.assertFalse((self.root/"result").exists())

    def test_pair_aggregation_does_not_buffer_interval_rows(self):
        # Weak references test the structural bound, not a machine-specific RSS.
        alive, peak = weakref.WeakSet(), [0]
        class TrackedRow(dict):
            __hash__ = object.__hash__
        base = evaluation.parse_segment({k: str(v) for k, v in self.rows[0].items()},
                                        "22", {"B": 0, "A": 1, "C": 2, "D": 3})
        def stream(_):
            for index in range(10000):
                row = TrackedRow(base, start_pos=1+20*index, end_pos=10+20*index)
                alive.add(row)
                peak[0] = max(peak[0], len(alive))
                yield row
        configs = {row["config_id"]: row for row in self.configs}
        with mock.patch.object(evaluation, "stream_segments", side_effect=stream):
            result = evaluation.aggregate_pairs([Path("synthetic")], configs, 4, {}, (.0221,), (0, 25),
                                                None, lambda: None)
        accumulator, totals, _ = result[self.full, 25]
        self.assertEqual((accumulator.n_pairs, totals.n, accumulator.bp), (1, 10000, 100000))
        self.assertLessEqual(peak[0], 3)  # Current/lookahead rows, never the whole pair.


if __name__ == "__main__":
    unittest.main()
