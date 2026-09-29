"""Synthetic tests for pre-normalization site provenance; no human data used."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "mark_original_alleles.py"
SPEC = importlib.util.spec_from_file_location("mark_original_alleles", SCRIPT)
MARKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MARKER)
PYSAM_AVAILABLE = importlib.util.find_spec("pysam") is not None
if PYSAM_AVAILABLE:
    import pysam


def record(pos, alts=("T",), ref="C", chrom="chr22"):
    return SimpleNamespace(contig=chrom, pos=pos, ref=ref, alts=alts)


class SiteGroupingTests(unittest.TestCase):
    def test_union_across_split_and_multiallelic_rows(self):
        rows = [record(10), record(10, ("A",)), record(10, ("T", "G")), record(20)]
        groups = list(MARKER.iter_site_groups(rows))
        self.assertEqual([len(group) for group in groups], [3, 1])
        self.assertEqual([MARKER.site_allele_count(group) for group in groups], [4, 2])

    def test_duplicate_allele_does_not_increase_union(self):
        self.assertEqual(MARKER.site_allele_count([record(10), record(10)]), 2)

    def test_symbolic_and_star_alleles_are_counted(self):
        self.assertEqual(MARKER.site_allele_count([record(10, ("T", "*", "<DEL>"))]), 4)

    def test_inconsistent_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Inconsistent REF"):
            MARKER.site_allele_count([record(10), record(10, ref="G")])

    def test_missing_alleles_are_rejected(self):
        for alts in (None, (), (".",), (None,)):
            with self.subTest(alts=alts), self.assertRaisesRegex(ValueError, "Missing ALT"):
                MARKER.site_allele_count([record(10, alts)])
        with self.assertRaisesRegex(ValueError, "Missing REF"):
            MARKER.site_allele_count([record(10, ref=".")])

    def test_group_must_have_one_coordinate(self):
        with self.assertRaisesRegex(ValueError, "multiple CHROM/POS"):
            MARKER.site_allele_count([record(10), record(11)])
        with self.assertRaisesRegex(ValueError, "empty"):
            MARKER.site_allele_count([])

    def test_descending_positions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not sorted"):
            list(MARKER.iter_site_groups([record(20), record(10)]))

    def test_chromosomes_cannot_be_revisited(self):
        with self.assertRaisesRegex(ValueError, "revisited"):
            list(MARKER.iter_site_groups([record(10), record(1, chrom="chr1"), record(20)]))

    def test_chromosome_order_need_not_be_lexicographic(self):
        groups = list(MARKER.iter_site_groups([record(20), record(1, chrom="chr1")]))
        self.assertEqual(len(groups), 2)
        self.assertEqual(list(MARKER.iter_site_groups([])), [])

    def test_nonpositive_position_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive POS"):
            list(MARKER.iter_site_groups([record(0)]))

    def test_output_formats_are_explicit(self):
        for suffix, mode in ((".bcf", "wb"), (".vcf.gz", "wz"), (".vcf", "w")):
            self.assertEqual(MARKER.output_mode(Path("result" + suffix)), mode)
        with self.assertRaisesRegex(ValueError, "Output must end"):
            MARKER.output_mode(Path("result.txt"))


@unittest.skipUnless(PYSAM_AVAILABLE, "pysam is required for real VCF/BCF round-trip tests")
class VariantFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def header(self):
        header = pysam.VariantHeader()
        header.contigs.add("chr22", length=1000)
        header.contigs.add("chr1", length=1000)
        header.info.add("AC", "A", "Integer", "Synthetic ALT allele count")
        header.info.add("END", 1, "Integer", "End position of a symbolic variant")
        header.formats.add("GT", 1, "String", "Genotype")
        header.formats.add("DP", 1, "Integer", "Read depth")
        header.formats.add("AD", "R", "Integer", "Allele depths")
        header.add_sample("synthetic_a")
        header.add_sample("synthetic_b")
        return header

    def write_fixture(self, path, rows=None, header=None):
        header = header if header is not None else self.header()
        rows = rows if rows is not None else [
            ("chr22", 10, ("C", "T")),
            ("chr22", 10, ("C", "A")),
            ("chr22", 20, ("A", "G", "T")),
            ("chr22", 30, ("G", "C")),
            ("chr22", 40, ("C", "*", "<DEL>")),
        ]
        with pysam.VariantFile(str(path), MARKER.output_mode(path), header=header) as writer:
            for chrom, pos, alleles in rows:
                row = writer.new_record(contig=chrom, start=pos - 1, stop=pos,
                                        alleles=alleles, qual=50)
                row.filter.add("PASS")
                row.info["AC"] = tuple(1 for _ in alleles[1:])
                for name, gt in (("synthetic_a", (0, 1)), ("synthetic_b", (None, 0))):
                    row.samples[name]["GT"] = gt
                    row.samples[name].phased = name == "synthetic_a"
                    row.samples[name]["DP"] = 10
                    row.samples[name]["AD"] = tuple(5 for _ in alleles)
                writer.write(row)

    def test_round_trip_all_input_and_output_formats_preserves_records(self):
        for input_suffix in (".vcf", ".vcf.gz", ".bcf"):
            source = self.root / ("source" + input_suffix)
            self.write_fixture(source)
            with pysam.VariantFile(str(source)) as reader:
                original_rows = [str(row).rstrip().split("\t") for row in reader]
                original_samples = list(reader.header.samples)
            for output_suffix in (".vcf", ".vcf.gz", ".bcf"):
                output = self.root / (input_suffix.replace(".", "_") + "-out" + output_suffix)
                with self.subTest(input=input_suffix, output=output_suffix):
                    self.assertEqual(MARKER.annotate(source, output), (5, 4))
                    with pysam.VariantFile(str(output)) as reader:
                        self.assertEqual(list(reader.header.samples), original_samples)
                        self.assertIn("##dnabr_original_alleles=v1", str(reader.header))
                        self.assertEqual(reader.header.info["ORIG_NALLELES"].number, 1)
                        self.assertEqual(reader.header.info["ORIG_NALLELES"].type, "Integer")
                        self.assertEqual(reader.header.info["ORIG_SITE"].number, 1)
                        self.assertEqual(reader.header.info["ORIG_SITE"].type, "String")
                        observed = list(reader)
                    self.assertEqual([r.info["ORIG_NALLELES"] for r in observed], [3, 3, 3, 2, 3])
                    self.assertEqual([r.info["ORIG_SITE"] for r in observed],
                                     ["chr22|10", "chr22|10", "chr22|20", "chr22|30", "chr22|40"])
                    for original, row in zip(original_rows, observed):
                        fields = str(row).rstrip().split("\t")
                        self.assertEqual(fields[:7], original[:7])
                        self.assertEqual(fields[8:], original[8:])
                        filtered_info = ";".join(value for value in fields[7].split(";")
                                                 if not value.startswith(("ORIG_NALLELES=", "ORIG_SITE=")))
                        self.assertEqual(filtered_info, original[7])
                    self.assertFalse(Path(str(output) + ".tbi").exists())
                    self.assertFalse(Path(str(output) + ".csi").exists())

    def test_existing_annotation_definitions_or_marker_are_rejected(self):
        for tag in ("ORIG_NALLELES", "ORIG_SITE", "dnabr_original_alleles"):
            with self.subTest(tag=tag):
                header = self.header()
                if tag == "dnabr_original_alleles":
                    header.add_meta(tag, value="unknown_version")
                else:
                    header.info.add(tag, 1, "String", "Already annotated")
                source, output = self.root / (tag + ".vcf"), self.root / (tag + "-out.bcf")
                self.write_fixture(source, header=header)
                with self.assertRaisesRegex(ValueError, "already declares"):
                    MARKER.annotate(source, output)
                self.assertFalse(output.exists())

    def test_annotation_cannot_be_applied_twice(self):
        source, once, twice = (self.root / name for name in ("source.vcf", "once.bcf", "twice.vcf"))
        self.write_fixture(source)
        MARKER.annotate(source, once)
        with self.assertRaisesRegex(ValueError, "already declares"):
            MARKER.annotate(once, twice)
        self.assertFalse(twice.exists())

    def test_existing_output_and_input_are_never_overwritten(self):
        source, output = self.root / "source.vcf", self.root / "existing.vcf"
        self.write_fixture(source)
        output.write_text("do not overwrite\n")
        with self.assertRaises(FileExistsError):
            MARKER.annotate(source, output)
        self.assertEqual(output.read_text(), "do not overwrite\n")
        original = source.read_bytes()
        with self.assertRaisesRegex(ValueError, "different files"):
            MARKER.annotate(source, source)
        self.assertEqual(source.read_bytes(), original)

    def test_ordering_or_reference_failure_leaves_no_partial_output(self):
        cases = [
            [("chr22", 20, ("C", "T")), ("chr22", 10, ("C", "A"))],
            [("chr22", 10, ("C", "T")), ("chr22", 10, ("G", "A"))],
            [("chr22", 10, ("C", "T")), ("chr1", 10, ("C", "A")),
             ("chr22", 20, ("C", "A"))],
        ]
        for index, rows in enumerate(cases):
            with self.subTest(case=index):
                source, output = self.root / f"bad{index}.vcf", self.root / f"bad{index}-out.bcf"
                self.write_fixture(source, rows=rows)
                with self.assertRaises(ValueError):
                    MARKER.annotate(source, output)
                self.assertFalse(output.exists())

    def test_undeclared_header_fields_fail_without_partial_output(self):
        header = (
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=chr22,length=1000>\n"
            "##FILTER=<ID=PASS,Description=\"All filters passed\">\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        )
        for tag, alt in (("UNKNOWN=3", "T"), ("END=20", "<DEL>")):
            for suffix in (".vcf", ".vcf.gz", ".bcf"):
                with self.subTest(tag=tag, output=suffix):
                    source = self.root / "undeclared.vcf"
                    source.write_text(header + f"chr22\t10\t.\tC\t{alt}\t.\tPASS\t{tag}\n")
                    output = self.root / ("undeclared-out" + suffix)
                    with self.assertRaisesRegex(ValueError, "undeclared info"):
                        MARKER.annotate(source, output)
                    self.assertFalse(output.exists())

    def test_cli_writes_bcf(self):
        source, output = self.root / "source.vcf", self.root / "out.bcf"
        self.write_fixture(source)
        result = subprocess.run([sys.executable, str(SCRIPT), "--input", str(source),
                                 "--output", str(output)], capture_output=True, text=True, check=True)
        self.assertIn("Annotated 5 records at 4 supplied-input sites", result.stdout)
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
