import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

from m34_audit_rare_loci import audit  # noqa: E402


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    split = tmp_path / "split.tsv"
    split.write_text(
        "sample_id\tancestry\tcanonical_population\tatomic_unit_id\trole\n"
        "a1\tAfrican\tAFR|A\tu1\tREF_TRAIN\n"
        "a2\tAfrican\tAFR|B\tu2\tREF_TRAIN\n"
        "e1\tEuropean\tEUR|A\tu3\tREF_TRAIN\n"
        "n1\tNative_American\tNAM|A\tu4\tREF_TRAIN\n"
        "n2\tNative_American\tNAM|B\tu5\tREF_TRAIN\n",
        encoding="utf-8",
    )
    panel = tmp_path / "panel.vcf.gz"
    with gzip.open(panel, "wt", encoding="utf-8", newline="") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=Genotype>\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                     "a1\ta2\te1\tn1\tn2\n")
        handle.write("22\t10\t.\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|0\t0|0\t0|1\t0|1\n")
        handle.write("22\t20\t.\tC\tT\t.\tPASS\t.\tGT\t1|1\t1|1\t1|1\t1|0\t1|0\n")
        handle.write("22\t30\t.\tG\tA\t.\tPASS\t.\tGT\t0|0\t0|0\t0|0\t0|0\t0|1\n")
    return panel, split


class RareLocusAuditTest(unittest.TestCase):
    def test_audit_orients_minor_allele_and_counts_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            panel, split = write_fixture(tmp_path)
            per_locus = tmp_path / "loci.tsv"
            summary_path = tmp_path / "summary.json"
            result = audit(
                panel_vcf=panel, split_tsv=split, per_locus_path=per_locus,
                summary_path=summary_path, chromosome="22", min_mac=2,
                max_maf_exclusive=0.25, expected_loci=2,
            )
            self.assertEqual(result["selection"]["minor_alt_loci"], 1)
            self.assertEqual(result["selection"]["minor_ref_loci"], 1)
            with per_locus.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["minor_code"] for row in rows], ["1", "0"])
            self.assertEqual([row["NAM_carrier_units"] for row in rows], ["2", "2"])
            self.assertTrue(all(row["NAM_minor_af"] == "0.5" for row in rows))
            self.assertFalse(json.loads(summary_path.read_text())["scope"]["king_used"])

    def test_expected_locus_count_is_a_hard_identity_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            panel, split = write_fixture(tmp_path)
            with self.assertRaisesRegex(ValueError, "selected locus count differs"):
                audit(
                    panel_vcf=panel, split_tsv=split,
                    per_locus_path=tmp_path / "loci.tsv",
                    summary_path=tmp_path / "summary.json", chromosome="22",
                    min_mac=2, max_maf_exclusive=0.25, expected_loci=3,
                )


if __name__ == "__main__":
    unittest.main()
