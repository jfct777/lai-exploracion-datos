#!/usr/bin/env python3
"""Synthetic M02.1 contract tests, including real M01 -> M02 -> M14 I/O.

All sample names and genotypes here are invented. Integration tests use local
bcftools and a temporary FASTA; they never open project datasets or launch jobs.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))
try:
    import pysam
    from mark_original_alleles import annotate
    from select_rare_minor import DEFAULT_MIN_MAC, count_alleles, minor_dosage, select_rare
except ModuleNotFoundError:
    pysam = None

try:
    import rare_allele_sharing_painter as painter
except ModuleNotFoundError:
    painter = None


@unittest.skipUnless(pysam is not None, "requires pysam")
class SelectRareMinorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = [f"synthetic_{i:02d}" for i in range(50)]
        self.serial = 0

    def paths(self):
        self.serial += 1
        stem = self.root / f"selection_{self.serial}"
        return (stem.with_suffix(".rare.minor.vcf.gz"),
                stem.with_suffix(".report.json"), stem.with_suffix(".counts.tsv"))

    def write_vcf(self, records, *, name="input.vcf", samples=None,
                  certified=True, metadata=()):
        sample_names = self.samples if samples is None else samples
        header = pysam.VariantHeader()
        for chrom in ("22", "chr22", "23"):
            header.contigs.add(chrom, length=2000)
        header.filters.add("q10", None, None, "Synthetic low quality")
        header.info.add("ORIG_NALLELES", 1, "Integer", "Alleles before normalization")
        header.info.add("ORIG_SITE", 1, "String", "Original chromosome and position")
        header.info.add("AC", "A", "Integer", "Possibly stale ALT count")
        header.info.add("AN", 1, "Integer", "Possibly stale called denominator")
        header.info.add("AF", "A", "Float", "Possibly stale ALT frequency")
        header.info.add("TEST_SCORE", 1, "Integer", "Synthetic optional field")
        header.info.add("END", 1, "Integer", "Synthetic symbolic variant end")
        header.formats.add("GT", 1, "String", "Genotype")
        header.formats.add("DP", 1, "Integer", "Synthetic depth")
        if certified:
            header.add_line("##dnabr_original_alleles=v1")
        for line in metadata:
            header.add_line(line)
        for sample in sample_names:
            header.add_sample(sample)
        target = self.root / name
        with pysam.VariantFile(str(target), "w", header=header) as output:
            for specification in records:
                pos = specification.get("pos", 100)
                chrom = specification.get("chrom", "22")
                alleles = specification.get("alleles", ("A", "C"))
                record = output.new_record(contig=chrom, start=pos - 1,
                                           stop=pos - 1 + len(alleles[0]), alleles=alleles)
                record.filter.add(specification.get("filter", "PASS"))
                orig_n = specification.get("orig_n", 2)
                orig_site = specification.get("orig_site", f"{chrom}|{pos}")
                if certified and orig_n is not None:
                    record.info["ORIG_NALLELES"] = orig_n
                if certified and orig_site is not None:
                    record.info["ORIG_SITE"] = orig_site
                # Intentionally incorrect: selection must recount GT, not trust INFO.
                record.info["AC"] = tuple(9 for _ in alleles[1:])
                record.info["AN"] = 18
                record.info["AF"] = tuple(0.5 for _ in alleles[1:])
                record.info["TEST_SCORE"] = 7
                genotypes = specification.get("gts", [(0, 1)] + [(0, 0)] * (len(sample_names) - 1))
                self.assertEqual(len(genotypes), len(sample_names))
                for i, (sample, gt) in enumerate(zip(sample_names, genotypes)):
                    record.samples[sample]["GT"] = gt
                    record.samples[sample]["DP"] = 10 + i
                    record.samples[sample].phased = i in specification.get("phased", ())
                output.write(record)
        return target

    def select(self, source, **kwargs):
        output, report_path, counts_path = self.paths()
        report = select_rare(source, output, report_path, counts_path, chrom="22", **kwargs)
        pysam.tabix_index(str(output), preset="vcf")
        with pysam.VariantFile(str(output)) as result:
            header = result.header.copy()
            records = [record.copy() for record in result]
        self.assertEqual(json.loads(report_path.read_text()), report)
        self.assertFalse(output.with_name(output.name + ".partial").exists())
        return records, report, header, (output, report_path, counts_path)

    def assert_rejected(self, source, error=ValueError, **kwargs):
        targets = self.paths()
        with self.assertRaises(error):
            select_rare(source, *targets, chrom="22", **kwargs)
        for target in targets:
            self.assertFalse(target.exists(), f"Failed selection published {target.name}")
        self.assertFalse(targets[0].with_name(targets[0].name + ".partial").exists())

    def write_mac_boundary_vcf(self):
        # AN=200: doubletons meet the unchanged default MAF <= 0.01.
        samples = [f"synthetic_{i:03d}" for i in range(100)]
        return self.write_vcf([
            {"pos": 100, "gts": [(0, 1)] + [(0, 0)] * 99},
            {"pos": 200, "gts": [(0, 1)] + [(1, 1)] * 99},
            {"pos": 300, "gts": [(0, 1)] * 2 + [(0, 0)] * 98},
            {"pos": 400, "gts": [(0, 1)] * 2 + [(1, 1)] * 98},
        ], samples=samples)

    def test_function_default_mac_two_keeps_alt_ref_doubletons_not_singletons(self):
        records, report, _, _ = self.select(self.write_mac_boundary_vcf())
        self.assertEqual([(r.pos, r.info["RARE_ALLELE"]) for r in records],
                         [(300, 1), (400, 0)])
        self.assertEqual(report["criteria"]["min_mac_inclusive"], 2)
        self.assertEqual(report["criteria"]["max_maf_inclusive"], "0.01")
        self.assertEqual(report["counts"]["excluded_mac"], 2)
        self.assertEqual(report["counts"]["rare"], 2)
        for record in records:
            self.assertEqual((record.info["RARE_AC"], record.info["RARE_AN"]), (2, 200))
            self.assertAlmostEqual(record.info["RARE_AF"], 0.01)

    def test_function_override_mac_one_still_keeps_alt_ref_singletons(self):
        records, report, _, _ = self.select(self.write_mac_boundary_vcf(), min_mac=1)
        self.assertEqual([(r.pos, r.info["RARE_ALLELE"], r.info["RARE_AC"])
                          for r in records],
                         [(100, 1, 1), (200, 0, 1), (300, 1, 2), (400, 0, 2)])
        self.assertEqual(report["criteria"]["min_mac_inclusive"], 1)
        self.assertEqual(report["counts"].get("excluded_mac", 0), 0)

    def test_real_cli_default_mac_two_and_explicit_one_override(self):
        source = self.write_mac_boundary_vcf()
        for options, expected_mac, expected_positions in (
            ([], 2, [300, 400]),
            (["--min-mac", "1"], 1, [100, 200, 300, 400]),
        ):
            with self.subTest(options=options):
                output, report_path, counts_path = self.paths()
                subprocess.run([
                    sys.executable, str(REPO / "bin" / "select_rare_minor.py"),
                    "--input", str(source), "--output", str(output),
                    "--report", str(report_path), "--counts", str(counts_path),
                    "--chrom", "22", *options,
                ], check=True, capture_output=True, text=True)
                pysam.tabix_index(str(output), preset="vcf")
                with pysam.VariantFile(str(output)) as selected:
                    records = list(selected)
                self.assertEqual([r.pos for r in records], expected_positions)
                self.assertEqual([r.info["RARE_ALLELE"] for r in records],
                                 [1, 0] if expected_mac == 2 else [1, 0, 1, 0])
                report = json.loads(report_path.read_text())
                self.assertEqual(report["criteria"]["min_mac_inclusive"], expected_mac)
                self.assertEqual(report["criteria"]["max_maf_inclusive"], "0.01")
                self.assertEqual(report["counts"]["rare"], len(expected_positions))
                self.assertTrue(counts_path.exists())

    def test_nextflow_config_and_python_default_mac_remain_consistent(self):
        configured_values = re.findall(
            r"^\s*lai_rare_min_mac\s*=\s*(\d+)\s*(?://[^\n]*)?$",
            (REPO / "nextflow.config").read_text(), flags=re.MULTILINE,
        )
        self.assertEqual(configured_values, ["2"])
        self.assertEqual(DEFAULT_MIN_MAC, int(configured_values[0]))
        self.assertEqual(inspect.signature(select_rare).parameters["min_mac"].default,
                         DEFAULT_MIN_MAC)

    def test_override_mac_one_alt_singleton_is_inclusive_at_one_percent(self):
        source = self.write_vcf([{}])
        records, report, header, paths = self.select(source, min_mac=1)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual((record.info["RARE_ALLELE"], record.info["RARE_AC"], record.info["RARE_AN"]), (1, 1, 100))
        self.assertAlmostEqual(record.info["RARE_AF"], 0.01)
        self.assertEqual((record.info["AC"], record.info["AN"]), ((1,), 100))
        self.assertAlmostEqual(record.info["AF"][0], 0.01)
        self.assertEqual([s["RD"] for s in record.samples.values()], [1] + [0] * 49)
        self.assertEqual(report["counts"]["rare_alt"], 1)
        self.assertIn("##dnabr_rare_contract=minor_v1", str(header))
        self.assertEqual(header.info["RARE_ALLELE"].type, "Integer")
        self.assertEqual(header.formats["RD"].number, 1)
        self.assertIn("22\trare_ref\t0\n", paths[2].read_text())

    def test_ref_singleton_preserves_ref_alt_gt_and_phase(self):
        genotypes = [(1, 0)] + [(1, 1)] * 49
        source = self.write_vcf([{"gts": genotypes, "phased": (0, 1)}])
        records, report, _, _ = self.select(source, min_mac=1)
        record = records[0]
        self.assertEqual(record.alleles, ("A", "C"))
        self.assertEqual([s["GT"] for s in record.samples.values()], genotypes)
        self.assertTrue(record.samples[self.samples[0]].phased)
        self.assertTrue(record.samples[self.samples[1]].phased)
        self.assertFalse(record.samples[self.samples[2]].phased)
        self.assertEqual((record.info["RARE_ALLELE"], record.info["RARE_AC"], record.info["RARE_AN"]), (0, 1, 100))
        self.assertEqual(record.info["AC"], (99,))
        self.assertAlmostEqual(record.info["AF"][0], 0.99)
        self.assertEqual([s["RD"] for s in record.samples.values()], [1] + [0] * 49)
        self.assertEqual(report["counts"]["rare_ref"], 1)

    def test_missing_monomorphic_tie_and_above_threshold_are_excluded(self):
        source = self.write_vcf([
            {"pos": 100, "gts": [(None, None)] * 50},
            {"pos": 200, "gts": [(0, 0)] * 50},
            {"pos": 300, "gts": [(1, 1)] * 50},
            {"pos": 400, "gts": [(0, 1)] * 50},
            {"pos": 500, "gts": [(0, 1), (0, None)] + [(0, 0)] * 48},
            {"pos": 600},
        ])
        records, report, _, _ = self.select(source, min_mac=1)
        self.assertEqual([r.pos for r in records], [600])
        self.assertEqual(report["counts"], {
            "input_filtered": 6, "excluded_no_called_alleles": 1,
            "excluded_mac": 2, "excluded_tie": 1, "excluded_maf": 1,
            "rare": 1, "rare_alt": 1,
        })

    def test_configured_mac_and_maf_are_both_inclusive(self):
        source = self.write_vcf([
            {"pos": 100},
            {"pos": 200, "gts": [(0, 1)] * 2 + [(0, 0)] * 48},
            {"pos": 300, "gts": [(0, 1)] * 3 + [(0, 0)] * 47},
        ])
        records, report, _, _ = self.select(source, min_mac=2, max_maf="0.02")
        self.assertEqual([r.pos for r in records], [200])
        self.assertEqual(report["counts"]["excluded_mac"], 1)
        self.assertEqual(report["counts"]["excluded_maf"], 1)
        self.assertEqual(report["criteria"]["min_mac_inclusive"], 2)

    def test_disabled_maf_does_not_admit_frequency_ties(self):
        source = self.write_vcf([
            {"pos": 100, "gts": [(0, 1)] * 50},
            {"pos": 200, "gts": [(0, 1)] * 49 + [(0, 0)]},
        ])
        records, report, _, _ = self.select(source, max_maf=None)
        self.assertEqual([r.pos for r in records], [200])
        self.assertEqual(report["counts"]["excluded_tie"], 1)

    def test_partial_genotype_contributes_known_allele_but_rd_is_missing(self):
        for minor, complete, partial, background in (
            (1, (0, 1), (1, None), (0, 0)),
            (0, (0, 1), (0, None), (1, 1)),
        ):
            with self.subTest(minor=minor):
                source = self.write_vcf([{"gts": [complete, partial] + [background] * 48}])
                records, _, _, _ = self.select(source, max_maf="0.021")
                record = records[0]
                self.assertEqual((record.info["RARE_ALLELE"], record.info["RARE_AC"], record.info["RARE_AN"]), (minor, 2, 99))
                self.assertEqual(record.samples[self.samples[0]]["RD"], 1)
                self.assertIsNone(record.samples[self.samples[1]]["RD"])
                self.assertEqual(record.samples[self.samples[1]]["GT"], partial)
                # Known alleles in partial GT enter AC/AN, but partial samples
                # cannot be assigned a complete dosage or an M14 carrier call.
                observed_dosage_sum = sum(s["RD"] for s in record.samples.values()
                                         if s["RD"] is not None)
                self.assertEqual(observed_dosage_sum, 1)
                self.assertNotEqual(observed_dosage_sum, record.info["RARE_AC"])

    def test_haploid_counts_follow_called_ploidy_not_two_times_n(self):
        source = self.write_vcf([{"gts": [(1,)] + [(0,)] * 49}])
        records, _, _, _ = self.select(source, min_mac=1, max_maf="0.02")
        self.assertEqual((records[0].info["RARE_AC"], records[0].info["RARE_AN"]), (1, 50))
        self.assertEqual(records[0].samples[self.samples[0]]["RD"], 1)

    def test_subset_recounts_and_flips_minor_from_alt_to_ref(self):
        genotypes = [(0, 1)] + [(1, 1)] * 24 + [(0, 0)] * 25
        source = self.write_vcf([{"gts": genotypes}])
        subset = self.root / "subset.txt"
        subset.write_text("\n".join(reversed(self.samples[:25])) + "\n")
        full, _, _, _ = self.select(source, max_maf=None)
        self.assertEqual(full[0].info["RARE_ALLELE"], 1)
        records, report, header, _ = self.select(source, samples_path=subset,
                                                min_mac=1, max_maf="0.02")
        record = records[0]
        self.assertEqual((record.info["RARE_ALLELE"], record.info["RARE_AC"], record.info["RARE_AN"]), (0, 1, 50))
        self.assertEqual((record.info["AC"], record.info["AN"]), ((49,), 50))
        self.assertAlmostEqual(record.info["AF"][0], 0.98)
        output_samples = list(header.samples)
        self.assertEqual(set(output_samples), set(self.samples[:25]))
        expected_hash = hashlib.sha256(("\n".join(output_samples) + "\n").encode()).hexdigest()
        self.assertEqual(report["cohort_sha256"], expected_hash)
        self.assertIn(f"##dnabr_rare_cohort_sha256={expected_hash}", str(header))
        self.assertEqual((report["cohort_samples_before"], report["cohort_samples_after"]), (50, 25))

    def test_subset_removing_all_minor_copies_excludes_monomorphic_site(self):
        source = self.write_vcf([{}])
        subset = self.root / "subset.txt"
        subset.write_text("\n".join(self.samples[1:]))
        records, report, _, _ = self.select(source, samples_path=subset)
        self.assertEqual(records, [])
        self.assertEqual(report["counts"]["excluded_mac"], 1)

    def test_empty_duplicate_and_unknown_subset_samples_fail_closed(self):
        source = self.write_vcf([{}])
        subset = self.root / "subset.txt"
        for content in ("", "synthetic_00\nsynthetic_00\n", "synthetic_absent\n"):
            with self.subTest(content=content):
                subset.write_text(content)
                self.assert_rejected(source, samples_path=subset)

    def test_no_sample_input_is_rejected(self):
        source = self.write_vcf([], samples=[])
        self.assert_rejected(source)

    def test_original_multiallelic_and_non_snv_records_are_excluded(self):
        source = self.write_vcf([
            {"pos": 100, "orig_n": 3},
            {"pos": 100, "orig_n": 3, "alleles": ("A", "G")},
            {"pos": 200, "orig_n": 3, "alleles": ("A", "C", "G")},
            {"pos": 300, "alleles": ("A", "AC")},
            {"pos": 400, "alleles": ("A", "*")},
            {"pos": 500, "alleles": ("A", "<DEL>")},
            {"pos": 600, "alleles": ("A", "N")},
            {"pos": 700},
        ])
        records, report, _, _ = self.select(source, min_mac=1)
        self.assertEqual([r.pos for r in records], [700])
        self.assertEqual(report["counts"]["excluded_original_multiallelic"], 3)
        self.assertEqual(report["counts"]["excluded_non_biallelic_snv"], 4)

    def test_missing_wrong_or_conflicting_provenance_marker_is_rejected(self):
        source = self.write_vcf([{}])
        original = source.read_text()
        for marker in ("", "##dnabr_original_alleles=v2\n",
                       "##dnabr_original_alleles=v1\n##dnabr_original_alleles=v2\n"):
            with self.subTest(marker=marker):
                source.write_text(original.replace("##dnabr_original_alleles=v1\n", marker))
                self.assert_rejected(source)

    def test_missing_or_invalid_record_provenance_is_rejected(self):
        for fields in ({"orig_n": None}, {"orig_site": None}, {"orig_n": 0}, {"orig_n": 1}):
            with self.subTest(fields=fields):
                self.assert_rejected(self.write_vcf([fields]))

    def test_any_existing_rare_contract_or_annotation_is_rejected(self):
        declarations = [
            "##dnabr_rare_contract=minor_v1",
            '##FORMAT=<ID=RD,Number=1,Type=Integer,Description="Existing dosage">',
        ] + [f'##INFO=<ID={name},Number=1,Type=Integer,Description="Existing annotation">'
             for name in ("RARE_ALLELE", "RARE_AC", "RARE_AN", "RARE_AF")]
        for declaration in declarations:
            with self.subTest(declaration=declaration):
                source = self.write_vcf([{}], metadata=[declaration])
                self.assert_rejected(source)

    def test_same_or_decreasing_certified_positions_fail_without_outputs(self):
        for second in (100, 99):
            with self.subTest(second=second):
                self.assert_rejected(self.write_vcf([{"pos": 100}, {"pos": second}]))

    def test_wrong_chromosome_is_rejected_but_chr_prefix_is_supported(self):
        self.assert_rejected(self.write_vcf([{"chrom": "23"}]))
        records, _, _, _ = self.select(self.write_vcf([{"chrom": "chr22"}]), min_mac=1)
        self.assertEqual(records[0].contig, "chr22")

    def test_remove_info_and_format_selection_preserve_contract_and_gt(self):
        source = self.write_vcf([{}])
        records, _, _, _ = self.select(source, min_mac=1, remove_info=True)
        self.assertEqual(set(records[0].info), {
            "ORIG_NALLELES", "ORIG_SITE", "AC", "AN", "AF",
            "RARE_ALLELE", "RARE_AC", "RARE_AN", "RARE_AF",
        })
        self.assertEqual(set(records[0].format), {"GT", "RD"})
        self.assertEqual(records[0].samples[self.samples[0]]["GT"], (0, 1))
        kept, _, _, _ = self.select(source, min_mac=1, keep_format="GT,DP")
        self.assertEqual(set(kept[0].format), {"GT", "DP", "RD"})
        self.assertEqual(kept[0].info["TEST_SCORE"], 7)
        self.assertEqual(kept[0].samples[self.samples[0]]["DP"], 10)
        all_formats, _, _, _ = self.select(source, min_mac=1, keep_format="")
        self.assertIn("DP", all_formats[0].format)
        self.assert_rejected(source, keep_format="DP")

    def test_invalid_thresholds_fail_closed_and_zero_maf_is_valid_empty(self):
        source = self.write_vcf([{}])
        for value in ("-0.01", "0.5", "1", "nan", "inf"):
            with self.subTest(max_maf=value):
                self.assert_rejected(source, max_maf=value)
        for value in (0, -1):
            with self.subTest(min_mac=value):
                self.assert_rejected(source, min_mac=value)
        records, _, _, _ = self.select(source, min_mac=1, max_maf="0")
        self.assertEqual(records, [])

    def test_existing_targets_and_partial_file_are_never_overwritten(self):
        source = self.write_vcf([{}])
        for index in range(4):
            with self.subTest(target=index):
                paths = self.paths()
                candidates = (*paths, paths[0].with_name(paths[0].name + ".partial"))
                protected = candidates[index]
                protected.write_bytes(b"preserve synthetic sentinel\n")
                with self.assertRaises(FileExistsError):
                    select_rare(source, *paths, chrom="22")
                self.assertEqual(protected.read_bytes(), b"preserve synthetic sentinel\n")
                for candidate in candidates:
                    if candidate != protected:
                        self.assertFalse(candidate.exists())

    def test_allele_helpers_reject_invalid_codes_and_do_not_impute(self):
        self.assertEqual(count_alleles([(0, 1), (1, None), (0,), ()]), (2, 4))
        self.assertEqual(minor_dosage((0, 0), 0), 2)
        self.assertEqual(minor_dosage((1,), 1), 1)
        self.assertIsNone(minor_dosage((0, None), 0))
        self.assertIsNone(minor_dosage((), 0))
        for gt in ((0, 2), (-1, 1)):
            with self.subTest(gt=gt), self.assertRaises(ValueError):
                count_alleles([gt])

    @unittest.skipUnless(shutil.which("bcftools"), "requires local bcftools")
    def test_real_m01_norm_m02_selection_excludes_original_multiallelic_sites(self):
        # M01 accepts raw input without existing provenance declarations.
        raw = self.write_vcf([
            {"pos": 100, "alleles": ("A", "C", "G"), "gts": [(1, 2)] + [(0, 0)] * 49},
            {"pos": 200, "alleles": ("A", "C")},
            {"pos": 200, "alleles": ("A", "G"), "filter": "q10"},
            {"pos": 300},
            {"pos": 400, "gts": [(0, 1)] + [(1, 1)] * 49},
            {"pos": 500, "filter": "q10"},
            {"pos": 600, "alleles": ("A", "AC")},
        ], certified=False)
        raw.write_text("\n".join(line for line in raw.read_text().splitlines()
                                 if not line.startswith("##INFO=<ID=ORIG_")) + "\n")
        original = self.root / "original.vcf.gz"
        self.assertEqual(annotate(raw, original), (7, 6))
        reference = self.root / "synthetic.fa"
        reference.write_text(">22\n" + "A" * 2000 + "\n")
        pysam.faidx(str(reference))
        normalized = self.root / "normalized.vcf.gz"
        subprocess.run(["bcftools", "norm", "-f", str(reference), "-m", "-any",
                        "-Oz", "-o", str(normalized), str(original)], check=True, capture_output=True)
        filtered = self.root / "filtered.vcf.gz"
        subprocess.run(["bcftools", "view", "-m2", "-M2", "-v", "snps", "-f", "PASS",
                        "-Oz", "-o", str(filtered), str(normalized)], check=True, capture_output=True)
        pysam.tabix_index(str(filtered), preset="vcf")
        with pysam.VariantFile(str(filtered)) as source:
            observed = [(r.pos, r.info["ORIG_NALLELES"]) for r in source]
        self.assertEqual(observed, [(100, 3), (100, 3), (200, 3), (300, 2), (400, 2)])
        records, report, _, _ = self.select(filtered, min_mac=1)
        self.assertEqual([(r.pos, r.info["RARE_ALLELE"]) for r in records], [(300, 1), (400, 0)])
        self.assertEqual(report["counts"]["excluded_original_multiallelic"], 3)

    @unittest.skipUnless(shutil.which("bcftools") and painter is not None,
                         "requires local bcftools and M14 Python stack")
    def test_real_selected_cohort_and_m14_preserve_source_ref_through_subset_flip(self):
        # Full input: ALT minor (47/100). M02.1's 25 samples: REF minor
        # (3/50). M14's two samples: REF major (3/4), but source code stays 0.
        genotypes = [(0, 0), (0, 1)] + [(1, 1)] * 23 + [(0, 0)] * 25
        raw = self.write_vcf([{"gts": genotypes, "phased": (0, 1)}], certified=False)
        raw.write_text("\n".join(line for line in raw.read_text().splitlines()
                                 if not line.startswith("##INFO=<ID=ORIG_")) + "\n")
        certified = self.root / "certified.vcf.gz"
        annotate(raw, certified)
        reference = self.root / "synthetic.fa"
        reference.write_text(">22\n" + "A" * 2000 + "\n")
        pysam.faidx(str(reference))
        normalized = self.root / "normalized.vcf.gz"
        subprocess.run(["bcftools", "norm", "-f", str(reference), "-m", "-any",
                        "-Oz", "-o", str(normalized), str(certified)], check=True, capture_output=True)
        filtered = self.root / "filtered.vcf.gz"
        subprocess.run(["bcftools", "view", "-m2", "-M2", "-v", "snps", "-f", "PASS",
                        "-Oz", "-o", str(filtered), str(normalized)], check=True, capture_output=True)
        pysam.tabix_index(str(filtered), preset="vcf")
        subset = self.root / "subset.txt"
        subset.write_text("\n".join(self.samples[:25]) + "\n")
        records, report, _, paths = self.select(filtered, samples_path=subset, max_maf="0.06")
        self.assertEqual((records[0].info["RARE_ALLELE"], records[0].info["RARE_AC"], records[0].info["RARE_AN"]), (0, 3, 50))
        self.assertEqual([records[0].samples[name]["RD"] for name in self.samples[:2]], [2, 1])
        result = painter.parse_genotypes_carrier_sets(
            paths[0], "22", self.samples[:2], carrier_allele_mode="source_minor",
            return_orientation_qc=True,
        )
        self.assertEqual(result[1], [(100, frozenset({0, 1}))])
        self.assertEqual(result[-1]["source_ref_sites"], 1)
        self.assertEqual(painter.read_source_minor_contract(paths[0], "source_minor")["cohort_sha256"], report["cohort_sha256"])
        for legacy_mode in ("historical_alt", "minor_allele"):
            with self.subTest(legacy_mode=legacy_mode), self.assertRaises(SystemExit):
                painter.parse_genotypes_carrier_sets(paths[0], "22", self.samples[:2], legacy_mode)


if __name__ == "__main__":
    unittest.main()
