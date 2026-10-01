"""Synthetic exact-count, missingness, allele and GRM aggregation checks."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import shutil
import subprocess

import numpy as np
import pysam

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import r02_genomic_pair_evidence as evidence


class PairEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = ["a", "b", "c"]

    def counts(self, carrier, measured, chunk=2):
        accumulator = evidence.RareCounts(carrier.shape[0])
        for start in range(0, carrier.shape[1], chunk):
            c, m = carrier[:, start:start + chunk], measured[:, start:start + chunk]
            cr, cc = np.where(c)
            mr, mc = np.where(~m)
            accumulator.add_sparse(cr, cc, mr, mc, c.shape[1])
        return accumulator.finish()

    def test_sparse_masked_counts_match_bruteforce(self):
        random = np.random.default_rng(11)
        m = random.random((9, 103)) > 0.2
        c = (random.random(m.shape) < 0.3) & m
        actual = self.counts(c, m, 17)
        for i in range(9):
            for j in range(9):
                q = m[i] & m[j]
                self.assertEqual(actual["Q"][i, j], q.sum())
                self.assertEqual(actual["I"][i, j], (q & c[i] & c[j]).sum())
                self.assertEqual(actual["U"][i, j], (q & (c[i] | c[j])).sum())
        np.testing.assert_array_equal(actual["individual_callable"], m.sum(axis=1))

    def test_zero_union_is_unknown_not_perfect_or_zero(self):
        actual = self.counts(np.zeros((3, 3), bool), np.ones((3, 3), bool))
        j = evidence.masked_jaccard(actual["I"], actual["U"])
        self.assertTrue(np.isnan(j).all())
        self.assertTrue(np.all(actual["Q"] == 3))

    def test_single_analytical_carrier_contributes_union(self):
        c = np.array([[1, 1], [1, 0]], dtype=bool)
        actual = self.counts(c, np.ones(c.shape, dtype=bool))
        self.assertEqual(actual["I"][0, 1], 1)
        self.assertEqual(actual["U"][0, 1], 2)
        self.assertEqual(evidence.masked_jaccard(actual["I"], actual["U"])[0, 1], .5)

    def test_no_uint8_overflow_and_chunks_do_not_change_counts(self):
        c = np.ones((3, 900), dtype=bool)
        a = self.counts(c, c, 900)
        b = self.counts(c, c, 13)
        self.assertEqual(a["I"][0, 1], 900)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def write_vcf(self, bad_rd=False, orig=2):
        samples = self.samples + [f"other_{i}" for i in range(98)]
        header = pysam.VariantHeader()
        header.contigs.add("chr22", length=10000)
        for key, typ in (("ORIG_NALLELES", "Integer"), ("RARE_ALLELE", "Integer"),
                         ("RARE_AC", "Integer"), ("RARE_AN", "Integer")):
            header.info.add(key, 1, typ, "Synthetic field")
        header.formats.add("GT", 1, "String", "Genotype")
        header.formats.add("RD", 1, "Integer", "Selected allele dosage")
        for key, val in (("dnabr_original_alleles", "v1"), ("dnabr_rare_contract", "minor_v1"),
                         ("dnabr_rare_cohort_n_samples", len(samples)),
                         ("dnabr_rare_cohort_sha256", evidence.sample_hash(samples))):
            header.add_line(f"##{key}={val}")
        for sample in samples:
            header.add_sample(sample)
        # One shared ALT; a singleton after analytical subsetting; one REF homozygote;
        # incomplete GT; and a site with no carrier remaining in analytical people.
        cases = [
            (1, [(0, 1), (0, 1), (0, 0)], []),
            (1, [(0, 1), (0, 0), (None, None)], [3]),
            (0, [(1, 1), (1, 1), (0, 0)], []),
            (1, [(0, 0), (None, 0), (0, 1)], [3]),
            (1, [(0, 0), (0, 0), (0, 0)], [3, 4]),
        ]
        path = self.root / "rare.vcf"
        with pysam.VariantFile(str(path), "w", header=header) as sink:
            for index, (allele, first_calls, outside_carriers) in enumerate(cases, 1):
                rec = sink.new_record(contig="chr22", start=index * 100 - 1, stop=index * 100,
                                      alleles=("A", "C"))
                rec.info["ORIG_NALLELES"] = orig
                rec.info["RARE_ALLELE"] = allele
                gts = first_calls + [(1 - allele, 1 - allele)] * (len(samples) - 3)
                for i in outside_carriers:
                    gts[i] = (0, 1)
                rec.info["RARE_AC"] = sum(a == allele for gt in gts for a in gt if a is not None)
                rec.info["RARE_AN"] = sum(a is not None for gt in gts for a in gt)
                for sample, gt in zip(samples, gts):
                    rec.samples[sample]["GT"] = gt
                    rec.samples[sample]["RD"] = None if None in gt else sum(a == allele for a in gt)
                if bad_rd and index == 1:
                    rec.samples["a"]["RD"] = 0
                sink.write(rec)
        return path

    def test_vcf_preserves_source_ref_orientation_missing_and_zero_carriers(self):
        path = self.write_vcf()
        # Reversed analysis order must actually reorder all count matrices.
        arrays, info = evidence.rare_evidence(path, ["c", "b", "a"], "22", 101, 2)
        self.assertEqual(info["counts"]["sites"], 5)
        self.assertEqual(info["counts"]["sites_zero_analytical_carriers"], 1)
        self.assertEqual(info["counts"]["sites_one_analytical_carriers"], 3)
        self.assertEqual(info["counts"]["source_minor_ref"], 1)
        np.testing.assert_array_equal(arrays["individual_carrier"], [2, 1, 2])
        np.testing.assert_array_equal(arrays["individual_callable"], [4, 4, 5])
        self.assertEqual(arrays["I"][1, 2], 1)
        self.assertEqual(arrays["U"][1, 2], 2)

    def test_wrong_rd_and_original_multiallelic_fail(self):
        with self.assertRaisesRegex(ValueError, "RD disagrees"):
            evidence.rare_evidence(self.write_vcf(bad_rd=True), self.samples, "22", 101)
        with self.assertRaisesRegex(ValueError, "Originally multiallelic"):
            evidence.rare_evidence(self.write_vcf(orig=3), self.samples, "22", 101)

    def grm(self, stem, k, n, ids=None):
        ids = self.samples if ids is None else ids
        prefix = self.root / stem
        indices = np.tril_indices(len(ids))
        np.asarray(k, dtype="<f4")[indices].tofile(str(prefix) + ".grm.bin")
        np.asarray(n, dtype="<f4")[indices].tofile(str(prefix) + ".grm.N.bin")
        Path(str(prefix) + ".grm.id").write_text("".join(f"0\t{s}\n" for s in ids))
        return prefix

    def test_import_common_reorders_and_retains_unknown(self):
        k = np.array([[1, .2, .3], [.2, 1, np.nan], [.3, np.nan, 1]])
        n = np.array([[10, 10, 10], [10, 10, 0], [10, 0, 10]])
        arrays, meta = evidence.import_common(self.grm("common", k, n), ["c", "a", "b"])
        self.assertTrue(meta["reordered_to_analytical_samples"])
        self.assertTrue(np.isnan(arrays["K"][0, 2]))
        self.assertEqual(arrays["common_numerator"][0, 2], 0)
        self.assertAlmostEqual(arrays["K"][0, 1], .3)

    def make_bundles(self):
        rare_paths, common_paths = [], []
        for chrom, k_value, n_value in (("1", .2, 10), ("2", .8, 30)):
            c = np.array([[1, 1], [1, 0], [0, 0]], dtype=bool)
            rare = self.counts(c, np.ones(c.shape, dtype=bool))
            common, meta = evidence.import_common(
                self.grm("grm" + chrom, np.full((3, 3), k_value), np.full((3, 3), n_value)), self.samples)
            rprefix, cprefix = self.root / ("rare" + chrom), self.root / ("common" + chrom)
            rare_meta = dict(source_cohort_n=3, source_cohort_members_sha256=evidence.sample_hash(self.samples),
                             mask="synthetic", allele="synthetic fixed allele")
            meta["producer"] = dict(tool="PLINK2", version="synthetic", frequency_scope="synthetic", missingness="synthetic", chromosome=chrom,
                marker_selection=dict(original_biallelic=True, PASS=True, min_maf=.05, max_missing=.02,
                                      window_kb=500, step_variants=1, r2=.2, indep_order=2))
            evidence.write_bundle(rprefix, rare, self.samples, [chrom], "rare", rare_meta)
            evidence.write_bundle(cprefix, common, self.samples, [chrom], "common", meta)
            rare_paths.append(str(rprefix) + ".npz")
            common_paths.append(str(cprefix) + ".npz")
        return rare_paths, common_paths

    def test_aggregation_weights_common_by_opportunity_not_chromosomes(self):
        rare, common = self.make_bundles()
        arrays, _ = evidence.aggregate_evidence(rare, common, self.samples, ["1", "2"])
        self.assertAlmostEqual(arrays["K"][0, 1], .65, places=6)
        self.assertEqual(arrays["J"][0, 1], .5)
        self.assertTrue(np.isnan(arrays["J"][2, 2]))
        self.assertEqual(arrays["n_sites"], 4)

    def test_missing_or_repeated_chromosome_rejected(self):
        rare, common = self.make_bundles()
        with self.assertRaisesRegex(ValueError, "Incomplete rare"):
            evidence.aggregate_evidence(rare[:1], common, self.samples, ["1", "2"])
        with self.assertRaisesRegex(ValueError, "Duplicated chromosome"):
            evidence.aggregate_evidence([rare[0], rare[0]], common, self.samples, ["1", "2"])

    def test_mixed_ld_configurations_fail(self):
        rare, common = self.make_bundles()
        manifest_path = Path(common[1]).with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["producer"]["marker_selection"]["r2"] = .5
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Inconsistent common"):
            evidence.aggregate_evidence(rare, common, self.samples, ["1", "2"])

    def test_hash_tampering_and_overwrite_rejected(self):
        rare, common = self.make_bundles()
        with Path(rare[0]).open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            evidence.aggregate_evidence(rare, common, self.samples, ["1", "2"])
        with self.assertRaisesRegex(ValueError, "Output exists"):
            evidence.write_bundle(self.root / "rare2", {}, self.samples, ["2"], "rare", {})

    @unittest.skipUnless(shutil.which("plink2") and shutil.which("bcftools"), "requires PLINK2/bcftools")
    def test_real_common_preparation_and_import(self):
        source_samples = [f"person_{i:03d}" for i in range(80)]
        selected = source_samples[:64]
        sample_file = self.root / "samples.txt"
        sample_file.write_text("\n".join(selected) + "\n")
        path = self.root / "common_source.vcf"
        header = pysam.VariantHeader()
        header.contigs.add("chr22", length=1000000)
        header.filters.add("q10", None, None, "Synthetic failure")
        header.info.add("ORIG_NALLELES", 1, "Integer", "Original count")
        header.formats.add("GT", 1, "String", "Genotype")
        header.add_line("##dnabr_original_alleles=v1")
        for sample in source_samples:
            header.add_sample(sample)
        random = np.random.default_rng(671)
        with pysam.VariantFile(str(path), "w", header=header) as sink:
            for i in range(45):
                rec = sink.new_record(contig="chr22", start=(i + 1) * 1000 - 1,
                                      stop=(i + 1) * 1000, alleles=("A", "C"))
                rec.info["ORIG_NALLELES"] = 3 if i == 40 else 2
                rec.filter.add("q10" if i == 41 else "PASS")
                values = random.binomial(2, .3, size=80)
                if i == 42:
                    values[:] = 0
                    values[0] = 1
                for j, sample in enumerate(source_samples):
                    dosage = values[j]
                    rec.samples[sample]["GT"] = (int(dosage > 0), int(dosage == 2))
                if i == 43:
                    for sample in selected[:2]:
                        rec.samples[sample]["GT"] = (None, None)
                if i == 44:
                    rec.samples[selected[0]]["GT"] = (0, None)
                sink.write(rec)
        output = self.root / "prepared"
        command = ["bash", str(Path(__file__).resolve().parents[1] / "bin/r02_common_grm.sh"),
                   "--vcf", str(path), "--samples", str(sample_file), "--chrom", "22",
                   "--expected-source-samples", "80", "--expected-samples", "64",
                   "--output-dir", str(output), "--threads", "1", "--memory-mb", "640"]
        result = subprocess.run(command, text=True, capture_output=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        pvar_rows = [line.split() for line in (output / "common.pvar").read_text().splitlines()
                     if line and not line.startswith("#")]
        positions = {int(row[1]) for row in pvar_rows}
        self.assertEqual(len(positions), 41)
        self.assertFalse({41000, 42000, 43000, 44000} & positions)
        self.assertIn(45000, positions)
        for name in ("primary", "sensitivity"):
            arrays, meta = evidence.import_common(output / name / "grm", selected)
            self.assertEqual(arrays["K"].shape, (64, 64))
            self.assertTrue(np.all(np.isfinite(arrays["K"])))
            self.assertTrue(np.all(arrays["common_N"] > 0))
            manifest = json.loads((output / name / "method.json").read_text())
            self.assertGreater(manifest["marker_selection"]["retained_snps"], 0)
            self.assertEqual(manifest["analytical_samples_sha256"], evidence.sample_hash(selected))


if __name__ == "__main__":
    unittest.main()
