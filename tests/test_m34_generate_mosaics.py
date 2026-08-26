#!/usr/bin/env python3
"""Known-answer and failure tests for the M34 phased mosaic generator."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))
import m34_generate_mosaics as m34  # noqa: E402


SAMPLES = ("afr_v", "eur_v", "nam_v", "afr_r", "eur_t", "nam_t")


def write_fixture(root: Path, *, unphased: bool = False, missing_sample: bool = False):
    split = root / "split.tsv"
    rows = [
        ("afr_v", "African", "u_afr_v", "SOURCE_VALID"),
        ("eur_v", "European", "u_eur_v", "SOURCE_VALID"),
        ("nam_v", "Native_American", "u_nam_v", "SOURCE_VALID"),
        ("afr_r", "African", "u_afr_r", "REF_TRAIN"),
        ("eur_t", "European", "u_eur_t", "SOURCE_TEST"),
        ("nam_t", "Native_American", "u_nam_t", "SOURCE_TEST"),
    ]
    if missing_sample:
        rows[2] = ("nam_absent", "Native_American", "u_nam_v", "SOURCE_VALID")
    with split.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("sample_id", "ancestry", "atomic_unit_id", "role"))
        writer.writerows(rows)

    genetic_map = root / "chr22.map"
    genetic_map.write_text(
        "chrom\tbp\tcm\n"
        "chr22\t100\t0.0\n"
        "chr22\t300\t0.5\n"
        "chr22\t500\t1.0\n",
        encoding="utf-8",
    )

    phased = "0/1" if unphased else "0|1"
    vcf = root / "panel.vcf.gz"
    with m34.deterministic_gzip_text(vcf) as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(SAMPLES)
            + "\n"
        )
        records = [
            (100, (phased, "1|1", "0|0", "0|0", "1|0", "0|1")),
            (200, ("1|0", "0|1", "1|1", "0|0", "0|0", "1|0")),
            (300, (".|.", ".|.", ".|.", "0|0", "0|1", "1|1")),
            (400, ("1|1", "0|0", "0|1", "1|0", "0|1", "0|0")),
            (500, ("0|1", "1|0", "1|1", "0|0", "1|1", "0|1")),
        ]
        for position, genotypes in records:
            handle.write(
                f"chr22\t{position}\tv{position}\tA\tG\t.\tPASS\t.\tGT\t"
                + "\t".join(genotypes)
                + "\n"
            )
    return split, genetic_map, vcf


def make_args(root: Path, outdir: Path, **overrides) -> argparse.Namespace:
    split, genetic_map, vcf = write_fixture(
        root,
        unphased=overrides.pop("unphased", False),
        missing_sample=overrides.pop("missing_sample", False),
    )
    values = {
        "phased_vcf": vcf,
        "split_tsv": split,
        "genetic_map": genetic_map,
        "outdir": outdir,
        "chromosome": "chr22",
        "donor_role": "SOURCE_VALID",
        "forbidden_role": None,
        "donor_unit_partition": "all",
        "rotation": 0,
        "seed": 271828,
        "target_individuals": 4,
        "target_prefix": "M34_TARGET",
        "mixture_proportions": {"AFR": 0.34, "EUR": 0.33, "NAM": 0.33},
        "admixture_generations": None,
        "transitions_per_morgan": 250.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def read_vcf(path: Path):
    samples = []
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
            elif not line.startswith("#"):
                fields = line.rstrip("\n").split("\t")
                rows.append((int(fields[1]), fields[9:]))
    return samples, rows


def read_source_haplotypes(path: Path):
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        samples = []
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
            elif not line.startswith("#"):
                fields = line.rstrip("\n").split("\t")
                position = int(fields[1])
                for sample, genotype in zip(samples, fields[9:]):
                    left, right = genotype.split(":", 1)[0].replace("/", "|").split("|")
                    result[(sample, 0, position)] = left
                    result[(sample, 1, position)] = right
    return result


def read_audit(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class TestM34GenerateMosaics(unittest.TestCase):
    def test_output_is_phased_and_each_allele_matches_the_audited_donor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root, root / "out")
            receipt = m34.run(args)
            self.assertEqual(receipt["decision"], "PASS_EXPLORATORY_MOSAICS_WITH_LOCAL_TRUTH")
            self.assertFalse(receipt["scope"]["confirmatory_validation"])
            self.assertEqual(receipt["role_audit"]["atomic_units_crossing_forbidden_roles"], 0)

            target_samples, target_rows = read_vcf(args.outdir / "m34_target.chr22.vcf.gz")
            source = read_source_haplotypes(args.phased_vcf)
            audit = read_audit(args.outdir / "m34_donor_audit.private.tsv")
            self.assertEqual(len(target_samples), 4)
            for position, genotypes in target_rows:
                for target_id, genotype in zip(target_samples, genotypes):
                    self.assertEqual(genotype.count("|"), 1)
                    copied = genotype.split("|")
                    for haplotype in (0, 1):
                        segment = next(
                            row
                            for row in audit
                            if row["target_id"] == target_id
                            and int(row["haplotype"]) == haplotype
                            and int(row["start_bp"]) <= position
                            < int(row["end_bp_exclusive"])
                        )
                        expected = source[
                            (
                                segment["donor_sample_id"],
                                int(segment["donor_haplotype"]),
                                position,
                            )
                        ]
                        self.assertEqual(copied[haplotype], expected)

            at_300 = dict(target_rows)[300]
            self.assertTrue(all(genotype == ".|." for genotype in at_300))
            self.assertEqual(receipt["counts"]["copied_missing_alleles"], 8)

    def test_same_seed_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args_a = make_args(root, root / "out_a")
            receipt_a = m34.run(args_a)
            args_b = argparse.Namespace(**{**vars(args_a), "outdir": root / "out_b"})
            receipt_b = m34.run(args_b)
            self.assertEqual(receipt_a["outputs"], receipt_b["outputs"])
            self.assertEqual(
                (args_a.outdir / "m34_target.chr22.vcf.gz").read_bytes(),
                (args_b.outdir / "m34_target.chr22.vcf.gz").read_bytes(),
            )
            self.assertEqual(
                (args_a.outdir / "m34_truth.chr22.tsv.gz").read_bytes(),
                (args_b.outdir / "m34_truth.chr22.tsv.gz").read_bytes(),
            )

    def test_target_prefix_separates_fit_and_valid_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit = make_args(root, root / "fit", target_prefix="M34_R0_FIT")
            valid = argparse.Namespace(
                **{**vars(fit), "outdir": root / "valid", "target_prefix": "M34_R0_VALID"}
            )
            m34.run(fit)
            m34.run(valid)
            fit_samples, _ = read_vcf(fit.outdir / "m34_target.chr22.vcf.gz")
            valid_samples, _ = read_vcf(valid.outdir / "m34_target.chr22.vcf.gz")
            self.assertTrue(set(fit_samples).isdisjoint(valid_samples))
            self.assertTrue(all(sample.startswith("M34_R0_FIT_") for sample in fit_samples))
            self.assertTrue(all(sample.startswith("M34_R0_VALID_") for sample in valid_samples))

    def test_unit_partition_is_disjoint_and_unit_balanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split, _, _ = write_fixture(root)
            text = split.read_text(encoding="utf-8")
            text += (
                "afr_v2\tAfrican\tu_afr_v2\tSOURCE_VALID\n"
                "eur_v2\tEuropean\tu_eur_v2\tSOURCE_VALID\n"
                "nam_v2\tNative_American\tu_nam_v2\tSOURCE_VALID\n"
            )
            split.write_text(text, encoding="utf-8")
            fit, fit_audit = m34.load_split(
                split, "SOURCE_VALID", m34.DEFAULT_FORBIDDEN_ROLES, "fit", 0
            )
            valid, valid_audit = m34.load_split(
                split, "SOURCE_VALID", m34.DEFAULT_FORBIDDEN_ROLES, "valid", 0
            )
            for ancestry in m34.ANCESTRIES:
                fit_units = {donor.atomic_unit_id for donor in fit[ancestry]}
                valid_units = {donor.atomic_unit_id for donor in valid[ancestry]}
                self.assertTrue(fit_units)
                self.assertTrue(valid_units)
                self.assertTrue(fit_units.isdisjoint(valid_units))
            self.assertEqual(fit_audit["unit_partition"], "fit")
            self.assertEqual(valid_audit["unit_partition"], "valid")

    def test_truth_covers_each_haplotype_without_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root, root / "out", transitions_per_morgan=500.0)
            receipt = m34.run(args)
            with gzip.open(
                args.outdir / "m34_truth.chr22.tsv.gz", "rt", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            grouped = {}
            for row in rows:
                grouped.setdefault((row["target_id"], row["haplotype"]), []).append(row)
            self.assertEqual(len(grouped), 2 * receipt["counts"]["target_individuals"])
            for segments in grouped.values():
                self.assertEqual(int(segments[0]["start_bp"]), 100)
                self.assertEqual(int(segments[-1]["end_bp_exclusive"]), 501)
                for left, right in zip(segments, segments[1:]):
                    self.assertEqual(left["end_bp_exclusive"], right["start_bp"])
                    self.assertNotEqual(left["ancestry"], right["ancestry"])

    def test_unphased_selected_donor_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root, root / "out", unphased=True)
            with self.assertRaisesRegex(ValueError, "Unphased"):
                m34.run(args)

    def test_selected_split_sample_absent_from_vcf_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root, root / "out", missing_sample=True)
            with self.assertRaisesRegex(ValueError, "absent from VCF"):
                m34.run(args)

    def test_donor_atomic_unit_crossing_forbidden_role_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(root, root / "out")
            text = args.split_tsv.read_text(encoding="utf-8")
            args.split_tsv.write_text(
                text.replace("afr_r\tAfrican\tu_afr_r", "afr_r\tAfrican\tu_afr_v"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cross a forbidden role"):
                m34.run(args)

    def test_cli_requires_an_explicit_transition_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(
                root,
                root / "out",
                admixture_generations=None,
                transitions_per_morgan=None,
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                m34.run(args)

    def test_receipt_hashes_all_outputs_and_keeps_scope_exploratory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_args(
                root,
                root / "out",
                transitions_per_morgan=None,
                admixture_generations=8.0,
            )
            receipt = m34.run(args)
            reopened = json.loads(
                (args.outdir / "m34_mosaic.receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reopened, receipt)
            self.assertEqual(receipt["parameters"]["transition_parameterization"], "pulse_generations")
            self.assertFalse(receipt["scope"]["holdout_independent"])
            for name, descriptor in receipt["outputs"].items():
                path = args.outdir / name
                self.assertEqual(descriptor["sha256"], m34.sha256_file(path))
                self.assertEqual(descriptor["bytes"], path.stat().st_size)


if __name__ == "__main__":
    unittest.main()
