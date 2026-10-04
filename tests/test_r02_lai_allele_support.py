"""Synthetic catalogue-only L1 tests; no real genotypes or cloud execution."""
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import r02_lai_allele_support as l1


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


class CatalogueFixture:
    """Small fixture usable by the Nextflow integration test."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples = [f"fictional_source_{i}" for i in range(101)]
        self.panel_samples = ["fictional_ref_0", "fictional_ref_1"]
        self.rare = self.root / "rare.vcf.gz"
        self.panel = self.root / "panel.vcf.gz"
        self.contract = self.root / "rare.contract.json"
        self.roles = self.root / "roles.tsv"
        self.builds = self.root / "build.json"
        self.contigs = self.root / "contigs.json"
        self.rare_rows = [self.rare_row(10, "A", "C", 0), self.rare_row(20, "G", "T", 1),
                          self.rare_row(30, "T", "A", 1), self.rare_row(40, "C", "G", 1)]
        self.panel_rows = [self.panel_row(10, "A", "C"), self.panel_row(20, "G", "T"),
                           self.panel_row(30, "A", "T")]
        self.source_contract = dict(schema="m02_1_minor_v1", chromosome="22", cohort_samples_before=101,
                                    cohort_samples_after=101, cohort_sha256=l1.sample_hash(self.samples),
                                    counts=dict(rare=4, rare_ref=1, rare_alt=3),
                                    criteria=dict(min_mac_inclusive=2, max_maf_inclusive="0.01", original_allele_count=2,
                                                  frequency_denominator="called_alleles", GT_REF_ALT_preserved=True))
        self.build_contract = dict(schema="r02_lai_build_evidence_v1",
                                  source=dict(build="GRCh38", status="DECLARED", evidence="Synthetic source contract"),
                                  panel=dict(build="GRCh38", status="UNVERIFIED", evidence="Synthetic panel lacks assembly proof"))
        self.roles.write_text("sample_id\trole\tancestry\nfictional_ref_0\tREF_TRAIN\tAFR\nfictional_ref_1\tREF_TRAIN\tEUR\nfictional_other\tSOURCE_VALID\tNAM\n")
        write_json(self.contigs, {"22": "22", "chr22": "22"})
        self.save()

    def rare_row(self, pos, ref, alt, selected, info_extra=""):
        info = f"ORIG_NALLELES=2;RARE_ALLELE={selected};RARE_AC=2;RARE_AN=202{info_extra}"
        return f"22\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\tGT:RD\t".encode() + b"\xffNOT_DECODED_GENOTYPE\n"

    def panel_row(self, pos, ref, alt, contig="chr22"):
        return f"{contig}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\tAC=999;AN=1000\tDP:GT\t".encode() + b"\xffNOT_DECODED_GENOTYPE\n"

    def header(self, samples, rare=False):
        lines = ["##fileformat=VCFv4.2"]
        if rare:
            lines.extend(["##dnabr_original_alleles=v1", "##dnabr_rare_contract=minor_v1",
                          f"##dnabr_rare_cohort_n_samples={len(samples)}",
                          f"##dnabr_rare_cohort_sha256={l1.sample_hash(samples)}"])
            for key in ("ORIG_NALLELES", "RARE_ALLELE", "RARE_AC", "RARE_AN"):
                lines.append(f'##INFO=<ID={key},Number=1,Type=Integer,Description="Synthetic">')
            lines.append('##FORMAT=<ID=RD,Number=1,Type=Integer,Description="Synthetic">')
        lines.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Synthetic">')
        lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples))
        return ("\n".join(lines) + "\n").encode()

    def save(self):
        for path, rows, samples, rare in ((self.rare, self.rare_rows, self.samples, True),
                                        (self.panel, self.panel_rows, self.panel_samples, False)):
            with gzip.open(path, "wb") as handle:
                handle.write(self.header(samples, rare=rare))
                handle.writelines(rows)
        write_json(self.contract, self.source_contract)
        write_json(self.builds, self.build_contract)

    def argv(self, output="output"):
        return ["--rare-vcf", str(self.rare), "--rare-contract", str(self.contract), "--panel-vcf", str(self.panel),
                "--roles", str(self.roles), "--build-evidence", str(self.builds), "--contig-map", str(self.contigs),
                "--panel-role", "REF_TRAIN", "--chromosome", "22", "--expected-source-samples", "101",
                "--expected-rare-sites", "4", "--outdir", str(self.root / output), "--min-free-gib", "0",
                "--resource-check-rows", "1"]

    def run(self, output="output", **updates):
        args = l1.parser().parse_args(self.argv(output))
        for name, value in updates.items():
            setattr(args, name, value)
        return l1.run(args)

    def rows(self, output="output"):
        with (self.root / output / "correspondencia_alelos_dnabr.tsv").open() as handle:
            return list(csv.DictReader(handle, delimiter="\t"))


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.fx = CatalogueFixture(temp.name)

    def test_preserve_all_rare_alleles_without_gt_decode(self):
        result = self.fx.run()
        rows = self.fx.rows()
        self.assertEqual(len(rows), 4)
        self.assertEqual(result["status"], l1.STATUS)
        self.assertEqual(result["genotype_fields_decoded"], 0)
        self.assertEqual([row["counted_allele"] for row in rows], ["A", "T", "A", "G"])
        self.assertEqual(rows[0]["source_rare_allele_index"], "0")
        self.assertEqual([row["correspondence_status"] for row in rows],
                         ["LEXICAL_MATCH_BUILD_UNVERIFIED"] * 2 + ["POSITION_PRESENT_ALLELE_MISMATCH", "NO_RECORD"])
        self.assertEqual(rows[3]["genotype_evidence_status"], "NO_GENOTYPE_EVIDENCE_IN_THIS_SOURCE")
        for row in rows:
            self.assertEqual(row["source_rare_ac"], "2")
            self.assertTrue(all(row[name] == "NA" for name in l1.SUPPORT_FIELDS))
            self.assertEqual(row["support_status"], "NOT_EVALUATED_CATALOGUE_ONLY")
        self.assertEqual(result["role_validation"]["n_role_members"], 2)
        self.assertEqual(result["role_validation"]["role_ancestry_counts"], {"AFR": 1, "EUR": 1})
        self.assertEqual(result["counts"]["source_minor_ref"], 1)
        self.assertEqual(result["counts"]["rows"], 4)
        self.assertFalse(result["public_distribution_allowed"])
        for path in (self.fx.root / "output").iterdir():
            self.assertNotIn("fictional_ref_0", path.read_text())
            self.assertNotIn("fictional_source_0", path.read_text())

    def test_verified_build_requires_evidence_for_both_inputs(self):
        for entry in self.fx.build_contract.values():
            if isinstance(entry, dict):
                entry["status"] = "VERIFIED"
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "authenticated verification receipt"):
            self.fx.run()
        for name, vcf in (("source", self.fx.rare), ("panel", self.fx.panel)):
            path = self.fx.root / f"{name}_build_receipt.json"
            write_json(path, dict(schema="r02_lai_build_verification_v1", status="VERIFIED", build="GRCh38",
                                 vcf_sha256=l1.sha256(vcf), method="Synthetic build verification for this fixture"))
            self.fx.build_contract[name]["verification_receipt"] = {"path": str(path), "sha256": l1.sha256(path)}
        write_json(self.fx.builds, self.fx.build_contract)
        self.fx.run()
        self.assertEqual(self.fx.rows()[0]["correspondence_status"], "EXACT_KEY")

    def test_panel_build_unknown_is_lexical_only(self):
        self.fx.build_contract["panel"]["build"] = None
        self.fx.save()
        self.fx.run()
        self.assertEqual(self.fx.rows()[0]["build_status"], "BUILD_UNVERIFIED")

    def test_build_mismatch_fails_before_outputs(self):
        self.fx.build_contract["panel"]["build"] = "GRCh37"
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "builds differ"):
            self.fx.run()
        self.assertFalse((self.fx.root / "output").exists())

    def test_fasta_mismatch_and_empty_evidence_fail(self):
        self.fx.build_contract["source"]["fasta_sha256"] = "a" * 64
        self.fx.build_contract["panel"]["fasta_sha256"] = "b" * 64
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "FASTA digests differ"):
            self.fx.run()
        self.fx.build_contract["panel"]["evidence"] = ""
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Build evidence"):
            self.fx.run()

    def test_ref_alt_swap_not_automatically_repaired(self):
        self.fx.run()
        row = self.fx.rows()[2]
        self.assertEqual((row["REF"], row["ALT"], row["counted_allele"]), ("T", "A", "A"))
        self.assertEqual(row["correspondence_status"], "POSITION_PRESENT_ALLELE_MISMATCH")

    def test_multiallelic_panel_is_not_split(self):
        self.fx.panel_rows[0] = self.fx.panel_row(10, "A", "C,G")
        self.fx.save()
        result = self.fx.run()
        self.assertEqual(self.fx.rows()[0]["correspondence_status"], "POSITION_PRESENT_ALLELE_MISMATCH")
        self.assertEqual(result["panel_counts"]["other_record_types_not_split"], 1)

    def test_duplicate_panel_key_rejected(self):
        self.fx.panel_rows.append(self.fx.panel_rows[0])
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Duplicate panel allele key"):
            self.fx.run()
        self.assertFalse((self.fx.root / "output").exists())

    def test_source_duplicate_position_and_decreasing_position_fail_partial(self):
        self.fx.rare_rows[1] = self.fx.rare_row(10, "A", "G", 1)
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Duplicate or decreasing"):
            self.fx.run()
        self.assertFalse((self.fx.root / "output/manifest.json").exists())
        self.assertTrue((self.fx.root / "output/correspondencia_alelos_dnabr.tsv.partial").exists())

    def test_source_original_multiallelic_not_rescued(self):
        self.fx.rare_rows[0] = self.fx.rare_rows[0].replace(b"ORIG_NALLELES=2", b"ORIG_NALLELES=3")
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Originally multiallelic"):
            self.fx.run()

    def test_source_rarity_and_diploid_denominator_checked(self):
        self.fx.rare_rows[0] = self.fx.rare_rows[0].replace(b"RARE_AN=202", b"RARE_AN=100")
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "MAC>=2"):
            self.fx.run()
        self.fx.rare_rows[0] = self.fx.rare_row(10, "A", "C", 0).replace(b"RARE_AN=202", b"RARE_AN=204")
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "denominator exceeds"):
            self.fx.run("second")

    def test_wrong_rare_identity_index_rejected(self):
        self.fx.rare_rows[0] = self.fx.rare_row(10, "A", "C", 2)
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Invalid selected"):
            self.fx.run()

    def test_source_contract_counts_and_cohort_hash_checked(self):
        self.fx.source_contract["counts"]["rare_ref"] = 0
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Selected REF/ALT count"):
            self.fx.run()
        self.fx.source_contract["cohort_sha256"] = "0" * 64
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "cohort hash mismatch"):
            self.fx.run("second")

    def test_role_membership_mismatch_and_duplicate_identifiers_fail(self):
        self.fx.roles.write_text(self.fx.roles.read_text().replace("fictional_ref_1\tREF_TRAIN", "fictional_ref_1\tSOURCE_VALID"))
        with self.assertRaisesRegex(ValueError, "exactly the selected role"):
            self.fx.run()
        self.fx.roles.write_text(self.fx.roles.read_text() + "fictional_ref_0\tREF_TRAIN\tAFR\n")
        with self.assertRaisesRegex(ValueError, "Duplicate identity"):
            self.fx.run()

    def test_role_change_is_not_silent(self):
        with self.assertRaisesRegex(ValueError, "exactly the selected role"):
            self.fx.run(panel_role="SOURCE_TEST")

    def test_unknown_contig_rejected_not_heuristically_normalized(self):
        write_json(self.fx.contigs, {"22": "22"})
        with self.assertRaisesRegex(ValueError, "absent from explicit mapping"):
            self.fx.run()

    def test_duplicate_json_and_info_fail(self):
        self.fx.contigs.write_text('{"22":"22","22":"22"}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            self.fx.run()
        write_json(self.fx.contigs, {"22": "22", "chr22": "22"})
        self.fx.rare_rows[0] = self.fx.rare_row(10, "A", "C", 0, ";RARE_AC=2")
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "Duplicate INFO"):
            self.fx.run()

    def test_limits_and_low_disk_fail(self):
        with self.assertRaisesRegex(ValueError, "max-panel-records"):
            self.fx.run(max_panel_records=1)
        with self.assertRaisesRegex(ValueError, "max-panel-bytes"):
            self.fx.run(max_panel_bytes=1)
        with self.assertRaisesRegex(ValueError, "max-panel-index-bytes"):
            self.fx.run(max_panel_index_bytes=1)
        with self.assertRaisesRegex(ValueError, "max-line-bytes"):
            self.fx.run(max_line_bytes=10)
        with mock.patch.object(l1.shutil, "disk_usage", return_value=type("Usage", (), {"free": 0})()):
            with self.assertRaisesRegex(ValueError, "Free disk"):
                self.fx.run(min_free_gib=1)

    def test_no_overwrite_and_hash_receipts(self):
        result = self.fx.run()
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            self.fx.run()
        for key, receipt in result["input_files"].items():
            self.assertEqual(receipt["sha256"], l1.sha256(receipt["path"]))
        for name, digest in result["outputs_sha256"].items():
            self.assertEqual(digest, l1.sha256(self.fx.root / "output" / name))
        expected = {key: receipt["sha256"] for key, receipt in result["input_files"].items()}
        hashes = self.fx.root / "hashes.json"
        write_json(hashes, expected)
        self.fx.run("authenticated", input_hashes=hashes)
        expected["panel_vcf"] = "0" * 64
        write_json(hashes, expected)
        with self.assertRaisesRegex(ValueError, "Input SHA256 mismatch: panel_vcf"):
            self.fx.run("tampered", input_hashes=hashes)

    def test_midrun_input_modification_detected(self):
        original = l1.verify_inputs_unchanged
        def mutate_then_check(receipts):
            self.fx.roles.write_text(self.fx.roles.read_text() + "fictional_added\tEXCLUDED\tNA\n")
            original(receipts)
        with mock.patch.object(l1, "verify_inputs_unchanged", side_effect=mutate_then_check):
            with self.assertRaisesRegex(ValueError, "Input changed"):
                self.fx.run()
        self.assertFalse((self.fx.root / "output/manifest.json").exists())

    def test_cli_and_same_table_for_same_inputs(self):
        first = self.fx.run()
        result = subprocess.run([sys.executable, str(ROOT / "bin/r02_lai_allele_support.py"), *self.fx.argv("cli")],
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        second = json.loads((self.fx.root / "cli/manifest.json").read_text())
        self.assertEqual(first["outputs_sha256"], second["outputs_sha256"])


if __name__ == "__main__":
    unittest.main()
