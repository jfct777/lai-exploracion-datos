import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

from m27c_gvcf_core import (  # noqa: E402
    STATE_ALLELE_INCOMPATIBLE,
    STATE_EXPLICIT_EXACT,
    STATE_EXPLICIT_OTHER_ALT_HOMREF,
    STATE_MISSING_GENOTYPE,
    STATE_REFERENCE_BLOCK,
    STATE_UNCOVERED,
    classify_record,
    is_high_quality,
    parse_header_contract,
    parse_targeted_lines,
)


KEY = ("22", 100, "A", "G")


def fields(position, ref, alt, info, fmt, sample):
    return ["chr22", str(position), ".", ref, alt, ".", ".", info, fmt, sample]


class TestM27CGvcfCore(unittest.TestCase):
    def test_reference_block_uses_min_dp_at_both_boundaries(self):
        row = fields(90, "C", "<NON_REF>", "END=100", "GT:DP:GQ:MIN_DP", "0/0:21:60:12")
        call = classify_record(row, KEY, 100)
        self.assertEqual(call.state, STATE_REFERENCE_BLOCK)
        self.assertEqual(call.dosage, 0)
        self.assertEqual(call.depth, 12)
        self.assertTrue(is_high_quality(call, 10, 20))

    def test_uncovered_gap_remains_distinct_from_reference(self):
        calls = parse_targeted_lines([], [KEY], {100: [0]})
        self.assertEqual(calls[0].state, STATE_UNCOVERED)

    def test_exact_biallelic_variant_is_remapped_to_model_dosage(self):
        row = fields(100, "A", "G,<NON_REF>", ".", "GT:DP:GQ", "1/0:18:45")
        call = classify_record(row, KEY, 100)
        self.assertEqual(call.state, STATE_EXPLICIT_EXACT)
        self.assertEqual(call.dosage, 1)

    def test_multiallelic_model_alt_is_remapped_by_index(self):
        row = fields(100, "A", "C,G,<NON_REF>", ".", "GT:DP:GQ", "2|2:22:50")
        call = classify_record(row, KEY, 100)
        self.assertEqual(call.state, STATE_EXPLICIT_EXACT)
        self.assertEqual(call.dosage, 2)
        self.assertTrue(call.phased)

    def test_different_carried_alt_is_incompatible(self):
        row = fields(100, "A", "C,<NON_REF>", ".", "GT:DP:GQ", "0/1:20:40")
        call = classify_record(row, KEY, 100)
        self.assertEqual(call.state, STATE_ALLELE_INCOMPATIBLE)

    def test_different_alt_with_homref_call_is_compatible(self):
        row = fields(100, "A", "C,<NON_REF>", ".", "GT:DP:GQ", "0/0:20:40")
        call = classify_record(row, KEY, 100)
        self.assertEqual(call.state, STATE_EXPLICIT_OTHER_ALT_HOMREF)
        self.assertEqual(call.dosage, 0)

    def test_missing_gt_is_not_homref(self):
        row = fields(90, "C", "<NON_REF>", "END=100", "GT:DP:GQ:MIN_DP", "./.:21:60:12")
        call = classify_record(row, KEY, 100)
        self.assertEqual(call.state, STATE_MISSING_GENOTYPE)

    def test_low_quality_is_a_property_not_a_new_structural_state(self):
        row = fields(90, "C", "<NON_REF>", "END=100", "GT:DP:GQ:MIN_DP", "0/0:21:19:9")
        call = classify_record(row, KEY, 100)
        self.assertEqual(call.state, STATE_REFERENCE_BLOCK)
        self.assertFalse(is_high_quality(call, 10, 20))

    def test_exact_record_has_precedence_over_reference_block(self):
        lines = [
            "\t".join(fields(90, "C", "<NON_REF>", "END=100", "GT:DP:GQ:MIN_DP", "0/0:21:60:12")),
            "\t".join(fields(100, "A", "G,<NON_REF>", ".", "GT:DP:GQ", "0/1:18:45")),
        ]
        calls = parse_targeted_lines(lines, [KEY], {100: [0]})
        self.assertEqual(calls[0].state, STATE_EXPLICIT_EXACT)
        self.assertEqual(calls[0].dosage, 1)

    def test_header_contract_requires_one_sample_and_expected_fields(self):
        header = "\n".join(
            [
                "##source=HaplotypeCaller",
                "##reference=file:///GRCh38.fa",
                "##contig=<ID=chr22,length=50818468>",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"GT\">",
                "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"DP\">",
                "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"GQ\">",
                "##FORMAT=<ID=MIN_DP,Number=1,Type=Integer,Description=\"MIN_DP\">",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            ]
        )
        contract = parse_header_contract(header)
        self.assertEqual(contract["samples"], ["S1"])
        self.assertEqual(contract["chr22_length"], 50818468)
        self.assertTrue(contract["has_required_fields"])


if __name__ == "__main__":
    unittest.main()
