#!/usr/bin/env python3
"""Known-answer tests for separate M34 F0 and truth conversion."""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))
import m33_safe_bridge_core as core  # noqa: E402
import m34_materialize as materialize  # noqa: E402
import m34_parse_flare_truth as subject  # noqa: E402


ANCESTRY_ORDER = "AFR,EUR,NAM"
FLARE_ID_MAP = "0=EUR,1=AFR,2=NAM"


def write_map(path: Path) -> None:
    path.write_text(
        "chrom\tbp\tcm\n"
        "chr22\t50\t0.0\n"
        "chr22\t250\t2.0\n"
        "chr22\t450\t4.0\n",
        encoding="utf-8",
    )


def write_flare(
    path: Path,
    *,
    positions=(100, 200, 300, 400),
    rounded=False,
    bad_mass=False,
) -> None:
    import io

    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
                handle.write("##fileformat=VCFv4.2\n")
                handle.write("##ANCESTRY=<EUR=0,AFR=1,NAM=2>\n")
                for name in ("AN1", "AN2", "ANP1", "ANP2"):
                    handle.write(
                        f'##FORMAT=<ID={name},Number=.,Type=String,Description="{name}">\n'
                    )
                handle.write(
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT0\tT1\n"
                )
                for index, position in enumerate(positions):
                    if bad_mass and index == 1:
                        p1 = "0.10,0.10,0.10"
                    elif rounded and index == 1:
                        p1 = "0.33,0.33,0.33"
                    else:
                        p1 = "0.60,0.30,0.10"
                    p2 = "0.20,0.70,0.10"
                    sample = f"0:1:{p1}:{p2}"
                    handle.write(
                        f"chr22\t{position}\tv{position}\tA\tG\t.\tPASS\t.\t"
                        f"AN1:AN2:ANP1:ANP2\t{sample}\t{sample}\n"
                    )


def truth_rows():
    return [
        ("T0", 0, "22", 100, 200, "AFR"),
        ("T0", 0, "22", 200, 401, "EUR"),
        ("T0", 1, "22", 100, 301, "NAM"),
        ("T0", 1, "22", 301, 401, "AFR"),
        ("T1", 0, "22", 100, 401, "NAM"),
        ("T1", 1, "22", 100, 400, "EUR"),
        ("T1", 1, "22", 400, 401, "NAM"),
    ]


def write_truth(path: Path, rows=None) -> None:
    rows = truth_rows() if rows is None else rows
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ("target_id", "haplotype", "chrom", "start_bp", "end_bp_exclusive", "ancestry")
        )
        writer.writerows(rows)


def f0_args(root: Path, **overrides):
    flare = root / "flare.anc.vcf.gz"
    genetic_map = root / "chr22.map"
    write_flare(flare, **overrides)
    write_map(genetic_map)
    return argparse.Namespace(
        flare_anc=flare,
        genetic_map=genetic_map,
        ancestry_order=ANCESTRY_ORDER,
        flare_id_map=FLARE_ID_MAP,
        outdir=root / "f0",
    )


def truth_args(root: Path, f0_directory: Path, rows=None, role="FIT"):
    truth = root / "m34_truth.chr22.tsv.gz"
    write_truth(truth, rows)
    return argparse.Namespace(
        truth_segments=truth,
        f0=f0_directory / "m34_f0.npz",
        marker_cm=f0_directory / "marker_cM.npz",
        ancestry_order=ANCESTRY_ORDER,
        role=role,
        outdir=root / f"truth-{role.lower()}",
    )


class TestM34ParseFlareTruth(unittest.TestCase):
    def test_f0_is_materializer_compatible_and_reorders_flare_ids(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = f0_args(root)
            receipt = subject.run_f0(args)
            self.assertEqual(receipt["decision"], "PASS_F0_TRUTH_BLIND")
            self.assertFalse(receipt["truth_opened"])
            with np.load(args.outdir / "m34_f0.npz", allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), materialize.PRODUCTIVE_MEMBERS["f0"])
                self.assertEqual(archive["sample_key_sha256"].dtype, np.dtype("|S64"))
                self.assertEqual(archive["marker_chrom"].dtype, np.dtype("|u1"))
                self.assertEqual(archive["marker_pos"].dtype, np.dtype("<i8"))
                self.assertEqual(archive["F0"].dtype, np.dtype("<f4"))
                self.assertEqual(archive["F0"].shape, (2, 2, 4, 3))
                # FLARE ID 0 is EUR and ID 1 is AFR; output is AFR/EUR/NAM.
                np.testing.assert_allclose(
                    archive["F0"][0, 0, 0], [0.30, 0.60, 0.10], rtol=0, atol=1e-6
                )
                np.testing.assert_allclose(archive["F0"].sum(axis=3), 1.0, rtol=0, atol=5e-6)
            with np.load(args.outdir / "marker_cM.npz", allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"marker_cM"})
                np.testing.assert_allclose(archive["marker_cM"], [0.5, 1.5, 2.5, 3.5])

    def test_rounded_probability_vector_is_renormalized(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = f0_args(root, rounded=True)
            receipt = subject.run_f0(args)
            self.assertAlmostEqual(receipt["raw_probability_sum_min"], 0.99)
            with np.load(args.outdir / "m34_f0.npz", allow_pickle=False) as archive:
                np.testing.assert_allclose(
                    archive["F0"][0, 0, 1], [1 / 3, 1 / 3, 1 / 3], rtol=0, atol=1e-6
                )

    def test_truth_known_answer_includes_half_open_boundaries(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            f0 = f0_args(root)
            subject.run_f0(f0)
            args = truth_args(root, f0.outdir)
            receipt = subject.run_truth(args)
            self.assertEqual(receipt["decision"], "PASS_TRUTH_ALIGNED_FIT")
            self.assertFalse(receipt["contains_probabilities"])
            self.assertFalse(receipt["f0_probability_values_loaded"])
            with np.load(args.outdir / "truth.npz", allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"sample_key_sha256", "marker_pos", "labels"})
                np.testing.assert_array_equal(
                    archive["sample_key_sha256"],
                    np.asarray([core.sample_key("T0"), core.sample_key("T1")], dtype="|S64"),
                )
                expected = np.asarray(
                    [
                        [[0, 1, 1, 1], [2, 2, 2, 0]],
                        [[2, 2, 2, 2], [1, 1, 1, 2]],
                    ],
                    dtype=np.int8,
                )
                np.testing.assert_array_equal(archive["labels"], expected)

    def test_segment_gap_and_overlap_fail(self):
        for replacement, expected in ((("T0", 0, "22", 100, 199, "AFR"), "gap or overlap"),
                                      (("T0", 0, "22", 100, 201, "AFR"), "gap or overlap")):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                f0 = f0_args(root)
                subject.run_f0(f0)
                rows = truth_rows()
                rows[0] = replacement
                args = truth_args(root, f0.outdir, rows=rows)
                with self.assertRaisesRegex(ValueError, expected):
                    subject.run_truth(args)

    def test_disordered_markers_and_segments_fail(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = f0_args(root, positions=(100, 300, 200, 400))
            with self.assertRaisesRegex(ValueError, "out of order"):
                subject.run_f0(args)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            f0 = f0_args(root)
            subject.run_f0(f0)
            rows = truth_rows()
            rows[0], rows[1] = rows[1], rows[0]
            args = truth_args(root, f0.outdir, rows=rows)
            with self.assertRaisesRegex(ValueError, "out of order"):
                subject.run_truth(args)

    def test_truth_axis_must_match_every_f0_sample_and_haplotype(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            f0 = f0_args(root)
            subject.run_f0(f0)
            rows = [row for row in truth_rows() if row[0] != "T1"]
            args = truth_args(root, f0.outdir, rows=rows)
            with self.assertRaisesRegex(ValueError, "sample axes"):
                subject.run_truth(args)

    def test_markers_must_lie_inside_complete_truth(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            f0 = f0_args(root)
            subject.run_f0(f0)
            rows = truth_rows()
            rows = [tuple(400 if index == 4 and value == 401 else value
                          for index, value in enumerate(row)) for row in rows]
            rows[-2] = ("T1", 1, "22", 100, 399, "EUR")
            rows[-1] = ("T1", 1, "22", 399, 400, "NAM")
            args = truth_args(root, f0.outdir, rows=rows)
            with self.assertRaisesRegex(ValueError, "outside truth"):
                subject.run_truth(args)

    def test_bad_probability_mass_and_truth_in_f0_directory_fail(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = f0_args(root, bad_mass=True)
            with self.assertRaisesRegex(ValueError, "mass"):
                subject.run_f0(args)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            f0 = f0_args(root)
            subject.run_f0(f0)
            args = truth_args(root, f0.outdir)
            args.outdir = f0.outdir
            with self.assertRaisesRegex(ValueError, "physically separate"):
                subject.run_truth(args)


if __name__ == "__main__":
    unittest.main()
