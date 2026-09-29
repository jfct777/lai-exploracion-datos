#!/usr/bin/env python3
"""M14 consumes the source-cohort allele contract without subset reorientation."""
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
try:
    import rare_allele_sharing_painter as painter
except ModuleNotFoundError:
    painter = None


SOURCE_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=22,length=1000>\n"
    "##dnabr_rare_contract=minor_v1\n"
    "##dnabr_rare_cohort_sha256=" + "a" * 64 + "\n"
    "##dnabr_rare_cohort_n_samples=3\n"
    '##INFO=<ID=RARE_ALLELE,Number=1,Type=Integer,Description="Source minor code">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    '##FORMAT=<ID=RD,Number=1,Type=Integer,Description="Source minor dosage">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\tc\n"
)


class FakeQuery:
    def __init__(self, payload):
        self.stdout = io.BytesIO(payload)
        self.stderr = io.BytesIO()

    def wait(self):
        return 0


@unittest.skipIf(painter is None, "requires the M14 scientific Python stack")
class SourceMinorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vcf = Path(self.temp.name) / "input.vcf"
        self.vcf.write_text(SOURCE_HEADER, encoding="ascii")

    def parse(self, payload, mode="source_minor", samples=None):
        with patch.object(painter.subprocess, "Popen", return_value=FakeQuery(payload)) as query:
            result = painter.parse_genotypes_carrier_sets(
                self.vcf, "22", samples or ["a", "b", "c"],
                carrier_allele_mode=mode, return_orientation_qc=True,
            )
        return result, query.call_args.args[0]

    def test_ref_minor_stays_ref_when_subset_is_ref_major(self):
        result, command = self.parse(b"22\t100\tC\t0\t0/0:2\t0/0:2\n", samples=["a", "b"])
        self.assertEqual(result[1], [(100, frozenset({0, 1}))])
        self.assertEqual(result[-1]["source_ref_sites"], 1)
        self.assertIn("%INFO/RARE_ALLELE", command[3])
        self.assertIn("%GT:%RD", command[3])

    def test_source_tie_in_selected_subset_does_not_drop_locus(self):
        result, _ = self.parse(b"22\t100\tC\t1\t0/1:1\t1|0:1\t./.:.\n")
        self.assertEqual(result[1], [(100, frozenset({0, 1}))])
        self.assertEqual(result[-1]["incomplete_genotypes_excluded"], 1)

    def test_partial_genotypes_have_missing_rd_and_are_not_carriers(self):
        result, _ = self.parse(b"22\t100\tC\t0\t0/1:1\t0|0:2\t0/.:.\n")
        self.assertEqual(result[1], [(100, frozenset({0, 1}))])
        self.assertEqual(result[-1]["partially_missing_genotypes"], 1)

    def test_bad_or_missing_dosage_and_allele_codes_fail_closed(self):
        invalid = [
            b"22\t100\tC\t0\t0/0:0\t0/1:1\t1/1:0\n",
            b"22\t100\tC\t0\t0/0:.\t0/1:1\t1/1:0\n",
            b"22\t100\tC\t0\t0/.:1\t0/1:1\t1/1:0\n",
            b"22\t100\tC\t2\t0/0:2\t0/1:1\t1/1:0\n",
            b"22\t100\tC\t.\t0/0:2\t0/1:1\t1/1:0\n",
            b"22\t100\tC\t0\t0/2:1\t0/1:1\t1/1:0\n",
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(SystemExit):
                self.parse(payload)

    def test_source_minor_rejects_repeated_position(self):
        payload = (b"22\t100\tC\t1\t0/1:1\t0/1:1\t0/0:0\n"
                   b"22\t100\tG\t1\t0/1:1\t0/1:1\t0/0:0\n")
        with self.assertRaises(SystemExit):
            self.parse(payload)

    def test_annotated_input_rejects_legacy_modes_before_query(self):
        for mode in ("historical_alt", "minor_allele"):
            with self.subTest(mode=mode), patch.object(painter.subprocess, "Popen") as query:
                with self.assertRaises(SystemExit):
                    painter.parse_genotypes_carrier_sets(self.vcf, "22", ["a"], mode)
                query.assert_not_called()

    def test_contract_metadata_contains_no_sample_ids(self):
        self.assertEqual(painter.read_source_minor_contract(self.vcf, "source_minor"), {
            "contract": "minor_v1", "cohort_sha256": "a" * 64, "cohort_n_samples": 3,
        })

    def test_required_header_fields_and_version_fail_closed(self):
        invalid_headers = [
            SOURCE_HEADER.replace("##dnabr_rare_contract=minor_v1\n", ""),
            SOURCE_HEADER.replace("minor_v1", "minor_v2"),
            SOURCE_HEADER.replace("a" * 64, "not-a-hash"),
            SOURCE_HEADER.replace("ID=RD,Number=1", "ID=RD,Number=2"),
            SOURCE_HEADER.replace("ID=RARE_ALLELE,Number=1", "ID=RARE_ALLELE,Number=A"),
            SOURCE_HEADER + "##dnabr_rare_contract=minor_v1\n",
        ]
        for header in invalid_headers:
            with self.subTest(header=header):
                self.vcf.write_text(header, encoding="ascii")
                with self.assertRaises(SystemExit):
                    painter.read_source_minor_contract(self.vcf, "source_minor")

    def test_legacy_modes_retain_existing_unannotated_semantics(self):
        header = "\n".join(line for line in SOURCE_HEADER.splitlines()
                           if not line.startswith("##dnabr_rare_")) + "\n"
        self.vcf.write_text(header, encoding="ascii")
        payload = b"22\t100\tC\t1/1\t1/1\t0/0\n"
        historical, _ = self.parse(payload, "historical_alt")
        minor, _ = self.parse(payload, "minor_allele")
        self.assertEqual(historical[1], [(100, frozenset({0, 1}))])
        self.assertEqual(minor[1], [])
        with self.assertRaises(SystemExit):
            painter.read_source_minor_contract(self.vcf, "source_minor")

    @unittest.skipUnless(shutil.which("bcftools"), "requires bcftools")
    def test_actual_bcftools_query_source_minor(self):
        self.vcf.write_text(
            SOURCE_HEADER + "22\t100\t.\tA\tC\t.\tPASS\tRARE_ALLELE=0\tGT:RD\t0/0:2\t0/1:1\t0/.:.\n",
            encoding="ascii",
        )
        result = painter.parse_genotypes_carrier_sets(
            self.vcf, "22", ["a", "b", "c"], carrier_allele_mode="source_minor",
        )
        self.assertEqual(result[1], [(100, frozenset({0, 1}))])


if __name__ == "__main__":
    unittest.main()
