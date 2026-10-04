"""Synthetic L1 genotype/quality tests; no real people, cloud or inference."""
import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "tests"))
import r02_lai_allele_support as l1
import r02_lai_genotype_support as gt
from test_r02_lai_allele_support import CatalogueFixture, write_json


class GenotypeFixture:
    """Self-contained authenticated synthetic inputs, also used by Nextflow."""
    def __init__(self, root):
        self.root = Path(root)
        self.catalogue = CatalogueFixture(root)
        self.samples = [f"fictional_ref_{i}" for i in range(8)]
        self.panel = self.root / "genotype_panel.vcf"
        self.quality_path = self.root / "genotype_contract.json"
        self.source_proof = self.root / "source_build_proof.json"
        self.panel_proof = self.root / "panel_build_proof.json"
        self.hashes = self.root / "hashes.json"
        self.contract = dict(schema="r02_lai_genotype_contract_v1", panel_role="REF_TRAIN",
                             panel_sample_order_sha256=l1.sample_hash(self.samples),
                             panel_sample_members_sha256=l1.sample_hash(sorted(self.samples)),
                             allele_representation=dict(status="ORIGINAL_UNSPLIT", panel_vcf_sha256="",
                                                        evidence="Synthetic unsplit records constructed in this fixture"),
                             site_filter_policy="PASS_ONLY",
                             quality=dict(contract_id="synthetic_dp10_gq20", mode="DP_GQ",
                                          min_dp=10, min_gq=20, missing_policy="exclude"))
        self.rows = [self.row(10, "A", "C", ["0|1:20:40", "1/1:20:40", "0/0:5:40", "0/.:20:40",
                                              "./.:20:40", "0:20:40", "0/1/1:20:40", "0/0:20:."]),
                     self.row(20, "G", "T", ["0/1:20:40"] * 8),
                     self.row(30, "A", "T", ["0/1:20:40"] * 8)]
        self.catalogue.roles.write_text("sample_id\trole\tancestry\n" + "".join(
            f"{sample}\tREF_TRAIN\t{'AFR' if i < 4 else 'EUR'}\n" for i, sample in enumerate(self.samples)))
        self.save()

    def row(self, pos, ref, alt, calls, site_filter="PASS", format_fields="GT:DP:GQ"):
        return f"chr22\t{pos}\t.\t{ref}\t{alt}\t.\t{site_filter}\t.\t{format_fields}\t" + "\t".join(calls) + "\n"

    def save(self):
        header = ('##fileformat=VCFv4.2\n##contig=<ID=chr22,length=50818468>\n'
                  '##FILTER=<ID=q10,Description="Synthetic filter">\n'
                  '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                  '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
                  '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Quality">\n'
                  '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + "\t".join(self.samples) + "\n")
        self.panel.write_text(header + "".join(self.rows))
        self.contract["allele_representation"]["panel_vcf_sha256"] = l1.sha256(self.panel)
        write_json(self.quality_path, self.contract)
        builds = dict(schema="r02_lai_build_evidence_v1")
        for name, vcf, proof in (("source", self.catalogue.rare, self.source_proof), ("panel", self.panel, self.panel_proof)):
            write_json(proof, dict(schema="r02_lai_build_verification_v1", status="VERIFIED", build="GRCh38",
                                  vcf_sha256=l1.sha256(vcf), fasta_sha256="a" * 64,
                                  method="bcftools_norm_check_ref_e_all_records_v1",
                                  reference_provenance=dict(schema="r02_lai_reference_identity_v1", source_uri="synthetic://fixture",
                                                            assembly="GRCh38", assembly_accession="synthetic_accession",
                                                            identity_evidence="Synthetic reference declaration, not a real assembly",
                                                            fasta_sha256="a" * 64, fai_sha256="b" * 64, contigs={"chr22":50818468}),
                                  reference_audit=dict(chromosome="22", records_checked=4 if name == "source" else len(self.rows),
                                                       mismatches=0, bcftools_exit_code=0, scope="ALL_VCF_RECORDS",
                                                       reference_contig="chr22", vcf_contigs=["chr22"])))
            builds[name] = dict(build="GRCh38", status="VERIFIED", fasta_sha256="a" * 64,
                                evidence="Synthetic authenticated build fixture",
                                verification_receipt=dict(path=str(proof), sha256=l1.sha256(proof)))
        write_json(self.catalogue.builds, builds)
        self.authenticate()

    def inputs(self):
        return dict(rare_vcf=self.catalogue.rare, rare_contract=self.catalogue.contract,
                    panel_vcf=self.panel, roles=self.catalogue.roles, build_evidence=self.catalogue.builds,
                    contig_map=self.catalogue.contigs, genotype_contract=self.quality_path,
                    source_build_verification=self.source_proof, panel_build_verification=self.panel_proof)

    def authenticate(self):
        write_json(self.hashes, {name: l1.sha256(path) for name, path in self.inputs().items()})

    def argv(self, output="output"):
        argv = self.catalogue.argv(output)
        argv[argv.index("--panel-vcf") + 1] = str(self.panel)
        return argv + ["--mode", "genotype_support", "--input-hashes", str(self.hashes),
                       "--genotype-contract", str(self.quality_path), "--source-build-verification", str(self.source_proof),
                       "--panel-build-verification", str(self.panel_proof)]

    def run(self, output="output", **updates):
        args = l1.parser().parse_args(self.argv(output))
        for name, value in updates.items():
            setattr(args, name, value)
        return l1.run(args)

    def output_rows(self, output="output"):
        with gzip.open(self.root / output / "soporte_alelos_dnabr.tsv.gz", "rt") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def get(self, position=10, ancestry="ALL", output="output"):
        return next(row for row in self.output_rows(output)
                    if row["position_bp"] == str(position) and row["panel_ancestry"] == ancestry)


class GenotypeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.fx = GenotypeFixture(temp.name)

    def test_ref_orientation_gt_quality_and_ancestry_denominators(self):
        result = self.fx.run()
        row = self.fx.get()
        expected = dict(counted_allele="A", panel_allele_index="0", source_rare_ac="2", source_rare_an="202",
                        n_role_members="8", n_gt_complete="4", n_gt_missing="2", n_gt_non_diploid="2",
                        ac_counted_called="5", an_called="8", n_quality_failed="1", n_quality_missing="1",
                        n_quality_evaluable="2", ac_counted_evaluable="1", an_evaluable="4", n_unresolved="6",
                        n_carriers_evaluable="1", called_call_rate="0.5", quality_call_rate="0.25",
                        af_counted_called="0.625", af_counted_evaluable="0.25")
        for name, value in expected.items():
            self.assertEqual(row[name], value, name)
        self.assertEqual(self.fx.get(20)["ac_counted_evaluable"], "8")
        self.assertEqual(self.fx.get(20)["counted_allele"], "T")
        afr, eur = self.fx.get(ancestry="AFR"), self.fx.get(ancestry="EUR")
        self.assertEqual(afr["an_evaluable"], "4")
        self.assertEqual(eur["an_evaluable"], "0")
        self.assertEqual(eur["ac_counted_evaluable"], "NA")
        self.assertEqual(eur["af_counted_evaluable"], "NA")
        self.assertEqual(eur["support_status"], "NO_QUALITY_EVALUABLE_GENOTYPES")
        self.assertEqual(result["output_rows"], 12)
        self.assertEqual(result["counts"]["source_rows"], 4)
        self.assertEqual(result["ancestry_strata"], {"ALL": 8, "AFR": 4, "EUR": 4})

    def test_missing_record_and_swap_are_unknown_not_zero(self):
        self.fx.run()
        for position, status in ((30, "POSITION_PRESENT_ALLELE_MISMATCH"), (40, "NO_RECORD")):
            row = self.fx.get(position)
            self.assertEqual(row["correspondence_status"], status)
            self.assertEqual(row["support_status"], "UNKNOWN_NO_COMPATIBLE_RECORD")
            for name in ("ac_counted_called", "an_called", "ac_counted_evaluable", "an_evaluable", "quality_call_rate"):
                self.assertEqual(row[name], "NA")
            self.assertEqual(row["n_unresolved"], "8")

    def test_phase_is_observed_not_certified(self):
        self.fx.run()
        row = self.fx.get()
        self.assertEqual(row["n_phased_gt_complete"], "1")
        self.assertEqual(row["n_phased_carriers_evaluable"], "1")
        self.assertEqual(row["n_phase_usable_carriers"], "NA")
        self.assertEqual(row["phase_contract_id"], "NA")
        self.assertEqual(row["phase_status"], "OBSERVED_PHASE_NOT_VALIDATED")

    def test_quality_failures_excluded_not_partial_genotype_counted(self):
        self.fx.rows[0] = self.fx.row(10, "A", "C", ["0/1:20:19"] * 8)
        self.fx.save()
        self.fx.run()
        row = self.fx.get()
        self.assertEqual((row["n_quality_failed"], row["an_evaluable"], row["ac_counted_evaluable"]), ("8", "0", "NA"))
        self.assertEqual((row["ac_counted_called"], row["an_called"]), ("8", "16"))

    def test_all_missing_has_no_called_zero_claim(self):
        self.fx.rows[0] = self.fx.row(10, "A", "C", ["./.:.:."] * 8)
        self.fx.save()
        self.fx.run()
        self.assertEqual(self.fx.get()["ac_counted_called"], "NA")
        self.assertEqual(self.fx.get()["an_called"], "0")
        self.assertEqual(self.fx.get()["n_gt_missing"], "8")

    def test_missing_quality_error_or_explicit_gt_only(self):
        self.fx.contract["quality"]["missing_policy"] = "error"
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "quality field is missing"):
            self.fx.run()
        self.assertFalse((self.fx.root / "output/manifest.json").exists())
        self.fx.contract["quality"] = dict(mode="GT_ONLY", contract_id="synthetic_unfiltered_calls")
        self.fx.save()
        self.fx.run("gt_only")
        row = self.fx.get(output="gt_only")
        self.assertEqual(row["ac_counted_called"], "5")
        for name in (*gt.QC_COUNTERS, "quality_call_rate", "af_counted_evaluable"):
            self.assertEqual(row[name], "NA", name)
        self.assertEqual(row["support_status"], "CALLED_GT_ONLY_NOT_QUALITY_VALIDATED")

    def test_inert_or_missing_quality_parameters_rejected(self):
        self.fx.contract["quality"] = dict(mode="GT_ONLY", contract_id="synthetic", min_dp=10)
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "inert quality"):
            self.fx.run()
        self.fx.contract["quality"] = dict(mode="DP_GQ", contract_id="synthetic", min_dp=None, min_gq=None, missing_policy="exclude")
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "explicit threshold"):
            self.fx.run()

    def test_one_explicit_quality_threshold_and_format_order(self):
        self.fx.rows[0] = self.fx.row(10, "A", "C", ["0/0:40:."] * 8, format_fields="GT:GQ:DP")
        self.fx.contract["quality"]["min_dp"] = None
        self.fx.save()
        self.fx.run()
        self.assertEqual(self.fx.get()["an_evaluable"], "16")
        self.assertEqual(self.fx.get()["ac_counted_evaluable"], "16")
        self.assertEqual(self.fx.get()["n_homozygous_carriers_evaluable"], "8")

    def test_site_filter_policy_is_explicit_and_effective(self):
        self.fx.rows[0] = self.fx.row(10, "A", "C", ["0/0:20:40"] * 8, site_filter="q10")
        self.fx.save()
        self.fx.run()
        self.assertEqual(self.fx.get()["n_site_filter_failed"], "8")
        self.assertEqual(self.fx.get()["an_evaluable"], "0")
        self.fx.contract["site_filter_policy"] = "IGNORE"
        self.fx.save()
        self.fx.run("ignore")
        self.assertEqual(self.fx.get(output="ignore")["ac_counted_evaluable"], "16")
        self.fx.rows[0] = self.fx.row(10, "A", "C", ["0/0:20:40"] * 8, site_filter=".")
        self.fx.contract["site_filter_policy"] = "PASS_OR_UNFILTERED"
        self.fx.save()
        self.fx.run("unfiltered")
        self.assertEqual(self.fx.get(output="unfiltered")["an_evaluable"], "16")

    def test_multiallelic_indices_preserve_other_alt_denominator(self):
        self.fx.rows[0] = self.fx.row(10, "A", "C,G", ["2/2:20:40"] * 8)
        self.fx.rows[1] = self.fx.row(20, "G", "A,T", ["2/0:20:40"] * 8)
        self.fx.save()
        self.fx.run()
        first, second = self.fx.get(), self.fx.get(20)
        self.assertEqual(first["correspondence_status"], "PANEL_MULTIALLELIC_ALLELE_MAPPED")
        self.assertEqual((first["panel_allele_index"], first["ac_counted_evaluable"], first["an_evaluable"]), ("0", "0", "16"))
        self.assertEqual(first["support_status"], "ZERO_OBSERVED_ALL_ROLE_MEMBERS_EVALUABLE")
        self.assertEqual((second["panel_allele_index"], second["ac_counted_evaluable"], second["an_evaluable"]), ("2", "8", "16"))

    def test_selected_ref_requires_source_alt_and_gvcf_is_not_evidence(self):
        self.fx.rows[0] = self.fx.row(10, "A", "G", ["0/0:20:40"] * 8)
        self.fx.rows[1] = self.fx.row(20, "G", "T,<NON_REF>", ["0/1:20:40"] * 8)
        self.fx.save()
        self.fx.run()
        self.assertEqual(self.fx.get()["correspondence_status"], "POSITION_PRESENT_ALLELE_MISMATCH")
        self.assertEqual(self.fx.get()["ac_counted_evaluable"], "NA")
        self.assertEqual(self.fx.get(20)["correspondence_status"], "UNSUPPORTED_PANEL_ALLELES")

    def test_zero_observed_partial_evaluability_not_absence(self):
        self.fx.rows[0] = self.fx.row(10, "A", "C", ["1/1:20:40"] * 7 + ["./.:.:."])
        self.fx.save()
        self.fx.run()
        self.assertEqual(self.fx.get()["support_status"], "ZERO_OBSERVED_PARTIAL_EVALUABILITY")
        self.assertEqual(self.fx.get()["an_evaluable"], "14")

    def test_ref_from_split_other_alt_to_zero_is_unknown(self):
        # True C/G was split into A/C=0/1 and A/G=0/1. Neither 0 is
        # trustworthy evidence of an A copy without decomposition provenance.
        self.fx.rows[0] = self.fx.row(10, "A", "C", ["0/1:20:40"] * 8)
        self.fx.rows.insert(1, self.fx.row(10, "A", "G", ["0/1:20:40"] * 8))
        self.fx.contract["allele_representation"]["status"] = "UNKNOWN"
        self.fx.contract["allele_representation"]["evidence"] = "Other-ALT handling not certified"
        self.fx.save()
        self.fx.run()
        self.assertEqual(self.fx.get()["correspondence_status"], "UNKNOWN_REF_DOSAGE_ALLELE_REPRESENTATION")
        self.assertEqual(self.fx.get()["ac_counted_called"], "NA")
        # Dropping the second split row does not establish safe REF semantics.
        self.fx.rows.pop(1)
        self.fx.save()
        self.fx.run("single_surviving_split")
        self.assertEqual(self.fx.get(output="single_surviving_split")["ac_counted_called"], "NA")
        self.assertEqual(self.fx.get(20, output="single_surviving_split")["ac_counted_evaluable"], "8")

    def test_split_other_alt_missing_excludes_the_partial_gt(self):
        self.fx.rows[0] = self.fx.row(10, "A", "C", ["./1:20:40"] * 8)
        self.fx.rows.insert(1, self.fx.row(10, "A", "G", ["./1:20:40"] * 8))
        self.fx.contract["allele_representation"]["status"] = "DECOMPOSED_OTHER_ALT_MISSING"
        self.fx.contract["allele_representation"]["evidence"] = "Synthetic decomposition retains other ALT as missing"
        self.fx.save()
        self.fx.run()
        self.assertEqual(self.fx.get()["an_evaluable"], "0")
        self.assertEqual(self.fx.get()["ac_counted_evaluable"], "NA")

    def test_duplicate_and_ambiguous_compatible_records_fail(self):
        self.fx.rows.insert(1, self.fx.rows[0])
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Duplicate panel"):
            self.fx.run()
        self.fx.rows[1] = self.fx.row(10, "A", "C,G", ["0/1:20:40"] * 8)
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Multiple compatible"):
            self.fx.run()
        self.assertFalse((self.fx.root / "output/manifest.json").exists())

    def test_decreasing_panel_and_position_memory_bound(self):
        self.fx.rows = self.fx.rows[::-1]
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Decreasing panel"):
            self.fx.run()
        self.fx.rows = self.fx.rows[::-1]
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "position buffer"):
            self.fx.run(max_panel_index_bytes=1)

    def test_sample_order_membership_and_duplicate_roles_are_authenticated(self):
        self.fx.samples.reverse()
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "order hash"):
            self.fx.run()
        self.fx.samples.reverse()
        self.fx.save()
        path = self.fx.catalogue.roles
        path.write_text(path.read_text() + "fictional_ref_0\tSOURCE_VALID\tAFR\n")
        self.fx.authenticate()
        with self.assertRaisesRegex(ValueError, "Duplicate identity in roles"):
            self.fx.run()

    def test_hash_and_verified_build_required_even_for_gt_only(self):
        self.fx.contract["quality"] = dict(mode="GT_ONLY", contract_id="synthetic")
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "requires hashes"):
            self.fx.run(input_hashes=None)
        builds = json.loads(self.fx.catalogue.builds.read_text())
        builds["panel"]["status"] = "DECLARED"
        write_json(self.fx.catalogue.builds, builds)
        self.fx.authenticate()
        with self.assertRaisesRegex(ValueError, "verified source and panel builds"):
            self.fx.run()

    def test_build_proof_binds_fasta_and_actual_vcf(self):
        proof = json.loads(self.fx.panel_proof.read_text())
        proof["vcf_sha256"] = "b" * 64
        write_json(self.fx.panel_proof, proof)
        builds = json.loads(self.fx.catalogue.builds.read_text())
        builds["panel"]["verification_receipt"]["sha256"] = l1.sha256(self.fx.panel_proof)
        write_json(self.fx.catalogue.builds, builds)
        self.fx.authenticate()
        with self.assertRaisesRegex(ValueError, "declared build to this VCF"):
            self.fx.run()
        self.fx.save()
        proof = json.loads(self.fx.panel_proof.read_text())
        proof["fasta_sha256"] = "b" * 64
        write_json(self.fx.panel_proof, proof)
        builds = json.loads(self.fx.catalogue.builds.read_text())
        builds["panel"]["verification_receipt"]["sha256"] = l1.sha256(self.fx.panel_proof)
        write_json(self.fx.catalogue.builds, builds)
        self.fx.authenticate()
        with self.assertRaisesRegex(ValueError, "FASTA digest"):
            self.fx.run()

    def test_build_hashes_without_audit_or_with_partial_scope_rejected(self):
        proof = json.loads(self.fx.panel_proof.read_text())
        proof.pop("reference_audit")
        write_json(self.fx.panel_proof, proof)
        builds = json.loads(self.fx.catalogue.builds.read_text())
        builds["panel"]["verification_receipt"]["sha256"] = l1.sha256(self.fx.panel_proof)
        write_json(self.fx.catalogue.builds, builds)
        self.fx.authenticate()
        with self.assertRaisesRegex(ValueError, "hashes alone are insufficient"):
            self.fx.run()
        self.fx.save()
        proof = json.loads(self.fx.panel_proof.read_text())
        proof["reference_audit"]["records_checked"] = 1
        write_json(self.fx.panel_proof, proof)
        builds = json.loads(self.fx.catalogue.builds.read_text())
        builds["panel"]["verification_receipt"]["sha256"] = l1.sha256(self.fx.panel_proof)
        write_json(self.fx.catalogue.builds, builds)
        self.fx.authenticate()
        with self.assertRaisesRegex(ValueError, "cover this entire input"):
            self.fx.run()

    def test_cli_and_outputs_do_not_export_samples(self):
        import subprocess
        completed = subprocess.run([sys.executable, str(ROOT / "bin/r02_lai_allele_support.py"), *self.fx.argv()],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        text = json.dumps(self.fx.output_rows()) + (self.fx.root / "output/manifest.json").read_text()
        self.assertNotIn("fictional_ref_", text)
        self.assertNotIn("fictional_source_", text)
        self.assertIn("NOT_PHASE_OR_BIOLOGICAL_VALIDATION", text)
        with self.assertRaisesRegex(ValueError, "Output exists"):
            self.fx.run()


if __name__ == "__main__":
    unittest.main()
