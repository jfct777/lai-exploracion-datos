#!/usr/bin/env python3

from __future__ import annotations

import gzip
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import m34_materialize as MATERIALIZE  # noqa: E402
import m34_prepare_panel_factors as BRIDGE  # noqa: E402


def write_gzip_text(path: Path, value: str) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
                handle.write(value)


def read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [line.rstrip("\r\n") for line in handle]


def fixture(
    root: Path,
    *,
    axis_mismatch: bool = False,
    unphased_reference: bool = False,
    target_collision: bool = False,
    crossed_population: bool = False,
    crossed_unit: bool = False,
    complete_monomorphic: bool = False,
    ancestry_enriched: bool = False,
) -> tuple[dict[str, Path], tuple[str, ...], tuple[str, ...]]:
    ancestries = ("AFR", "EUR", "NAM")
    reference_counts = (
        {"AFR": 70, "EUR": 70, "NAM": 2}
        if ancestry_enriched else {ancestry: 34 for ancestry in ancestries}
    )
    by_ancestry = {
        ancestry: tuple(
            f"ref_{ancestry}_{index:02d}"
            for index in range(reference_counts[ancestry])
        )
        for ancestry in ancestries
    }
    reference_samples = tuple(
        by_ancestry[ancestry][index]
        for index in range(max(reference_counts.values()))
        for ancestry in ancestries
        if index < reference_counts[ancestry]
    )
    other_samples = ("valid_0", "valid_1", "valid_2", "test_0", "discovery_0")
    panel_samples = reference_samples + other_samples
    target_samples = (("valid_0", "target_B") if target_collision
                      else ("target_A", "target_B"))

    split_path = root / "m27f_split.tsv"
    header = (
        "sample_id\tsource\tancestry\tpopulation\tcanonical_population\t"
        "atomic_unit_id\trole\texclusion_reason\n"
    )
    rows: list[str] = []
    for ancestry in ancestries:
        ancestry_long = {
            "AFR": "African", "EUR": "European", "NAM": "Native_American"
        }[ancestry]
        for index, sample in enumerate(by_ancestry[ancestry]):
            rows.append(
                f"{sample}\tPANEL\t{ancestry_long}\tpop_{sample}\tpop_{sample}\t"
                f"unit_{sample}\tREF_TRAIN\t\n"
            )
    role_rows = (
        ("valid_0", "African", "SOURCE_VALID"),
        ("valid_1", "European", "SOURCE_VALID"),
        ("valid_2", "Native_American", "SOURCE_VALID"),
        ("test_0", "African", "SOURCE_TEST"),
        ("discovery_0", "African", "DISCOVERY"),
    )
    for sample, ancestry, role in role_rows:
        population = "pop_ref_AFR_00" if crossed_population and sample == "valid_0" else f"pop_{sample}"
        unit = "unit_ref_AFR_00" if crossed_unit and sample == "valid_0" else f"unit_{sample}"
        rows.append(
            f"{sample}\tPANEL\t{ancestry}\t{population}\t{population}\t"
            f"{unit}\t{role}\t\n"
        )
    split_path.write_text(header + "".join(rows), encoding="utf-8")

    map_path = root / "chr22.map"
    map_path.write_text("22\t100\t0.0\n22\t300\t0.5\n22\t500\t1.0\n",
                        encoding="utf-8")

    metadata = (
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    )
    column_prefix = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
    panel_lines = [metadata, column_prefix + "\t" + "\t".join(panel_samples) + "\n"]
    mosaic_lines = [metadata, column_prefix + "\t" + "\t".join(target_samples) + "\n"]

    def panel_genotypes(position: int) -> list[str]:
        values: dict[str, str]
        if position == 100:
            values = {sample: "0|0" for sample in reference_samples}
            values["ref_AFR_00"] = "0/1" if unphased_reference else "1|1"
            if "ref_NAM_33" in values:
                values["ref_NAM_33"] = ".|0"
        elif position == 200:
            values = {sample: "1|1" for sample in reference_samples}
            values["ref_AFR_00"] = "0|0"
            values["ref_EUR_33"] = ".|1"
        elif position == 300:
            values = {sample: "0|0" for sample in reference_samples}
            values["ref_AFR_00"] = "0|1"
        elif position == 450:
            values = {sample: "0|0" for sample in reference_samples}
        elif position == 475:
            values = {sample: "0|0" for sample in reference_samples}
            values["ref_NAM_00"] = "0|1"
            values["ref_NAM_01"] = "0|1"
        else:
            values = {sample: "0|1" for sample in reference_samples}
        return [values[sample] + ":9" if sample in values else "BROKEN"
                for sample in panel_samples]

    axes = [
        (100, "A", "G"),
        (200, "C", "T"),
        (300, "G", "A"),
        (350, "A", "AT"),
        (400, "A", "C,G"),
    ]
    if complete_monomorphic:
        axes.append((450, "T", "C"))
    if ancestry_enriched:
        axes.append((475, "G", "T"))
    target_gts = {
        100: ("0|1:8", ".|1:8"),
        200: ("0|1:8", "1|1:8"),
        300: ("0|0:8", "0|1:8"),
        350: ("0|0:8", "0|1:8"),
        400: ("BROKEN", "BROKEN"),
        450: ("0|0:8", "0|0:8"),
        475: ("0|1:8", "0|0:8"),
    }
    for position, ref, alt in axes:
        fixed = f"22\t{position}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT:DP"
        panel_lines.append(fixed + "\t" + "\t".join(panel_genotypes(position)) + "\n")
        mosaic_alt = "C" if axis_mismatch and position == 200 else alt
        mosaic_fixed = f"22\t{position}\t.\t{ref}\t{mosaic_alt}\t.\tPASS\t.\tGT:DP"
        mosaic_lines.append(mosaic_fixed + "\t" + "\t".join(target_gts[position]) + "\n")

    panel_path = root / "panel.vcf.gz"
    mosaic_path = root / "mosaics.vcf.gz"
    write_gzip_text(panel_path, "".join(panel_lines))
    write_gzip_text(mosaic_path, "".join(mosaic_lines))
    return ({
        "panel_vcf": panel_path,
        "mosaic_vcf": mosaic_path,
        "split_tsv": split_path,
        "genetic_map_path": map_path,
    }, reference_samples, target_samples)


def run_bridge(inputs: dict[str, Path], outdir: Path, **kwargs):
    return BRIDGE.prepare_panel_factors(outdir=outdir, **inputs, **kwargs)


class M34PreparePanelFactorsTests(unittest.TestCase):
    def test_known_answer_ref_minor_missing_and_materializer_compatibility(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs, reference_input_order, target_samples = fixture(root)
            outdir = root / "bridge"
            receipt = run_bridge(inputs, outdir)

            with np.load(outdir / BRIDGE.OUTPUT_NAMES["selected"], allow_pickle=False) as archive:
                selected = {name: archive[name] for name in archive.files}
            with np.load(outdir / BRIDGE.OUTPUT_NAMES["target"], allow_pickle=False) as archive:
                target = {name: archive[name] for name in archive.files}
            with np.load(outdir / BRIDGE.OUTPUT_NAMES["reference"], allow_pickle=False) as archive:
                reference = {name: archive[name] for name in archive.files}

            np.testing.assert_array_equal(selected["pos"], np.asarray([100, 200], dtype="<i8"))
            np.testing.assert_array_equal(selected["ref"], np.asarray([b"A", b"C"]))
            np.testing.assert_array_equal(selected["alt"], np.asarray([b"G", b"T"]))
            np.testing.assert_allclose(selected["cM"], [0.0, 0.25], rtol=0, atol=0)
            np.testing.assert_array_equal(
                selected["locus_id"],
                np.asarray([
                    BRIDGE._locus_id("22", 100, "A", "G"),
                    BRIDGE._locus_id("22", 200, "C", "T"),
                ], dtype="<u8"),
            )
            np.testing.assert_array_equal(target["minor_dosage"], [[1, 1], [0, 0]])
            np.testing.assert_array_equal(target["observed_mask"], [[1, 1], [0, 1]])
            np.testing.assert_array_equal(
                target["sample_key_sha256"],
                np.asarray([BRIDGE.sample_key(sample) for sample in target_samples], dtype="|S64"),
            )
            np.testing.assert_array_equal(reference["ancestry"], [b"AFR", b"EUR", b"NAM"])
            np.testing.assert_array_equal(reference["minor_ac"], [[2, 2], [0, 0], [0, 0]])
            np.testing.assert_array_equal(reference["callable_an"], [[68, 68], [68, 67], [67, 68]])
            np.testing.assert_array_equal(reference["observed_mask"], np.ones((3, 2), dtype="|u1"))
            np.testing.assert_array_equal(reference["no_support"], [[0, 0], [1, 1], [1, 1]])

            marker_cm = np.asarray([0.5], dtype="<f8")
            f0 = {
                "sample_key_sha256": target["sample_key_sha256"],
                "marker_chrom": np.full(1, 22, dtype="|u1"),
                "marker_pos": np.asarray([300], dtype="<i8"),
                "marker_ref": np.asarray([b"G"], dtype="|S1"),
                "marker_alt": np.asarray([b"A"], dtype="|S1"),
                "F0": np.full((2, 2, 1, 3), np.float32(1 / 3), dtype="<f4"),
            }
            dimensions = MATERIALIZE.validate_inputs(
                selected, target, reference, f0, marker_cm, BRIDGE.ANCESTRIES
            )
            self.assertEqual(dimensions, {
                "sample_count": 2, "locus_count": 2, "marker_count": 1,
                "ancestry_count": 3, "channel_count": 13,
            })
            self.assertEqual(receipt["counts"]["minor_is_alt_loci"], 1)
            self.assertEqual(receipt["counts"]["minor_is_ref_loci"], 1)
            self.assertEqual(receipt["counts"]["target_missing_genotypes_on_selected_axis"], 1)
            self.assertEqual(receipt["roles"]["frequency_role"], "REF_TRAIN")
            self.assertFalse(receipt["roles"]["source_test_open"])
            self.assertIn(
                "preserve_REF_callable_AN_and_TARGET_observed_mask",
                receipt["parameters"]["rare_factor_missingness_policy"],
            )
            self.assertEqual(
                receipt["inputs"]["panel_vcf"]["sha256"],
                BRIDGE.sha256_file(inputs["panel_vcf"]),
            )
            self.assertEqual(
                receipt["inputs"]["mosaic_vcf"]["sha256"],
                BRIDGE.sha256_file(inputs["mosaic_vcf"]),
            )

            expected_reference_order = tuple(
                sample for ancestry in BRIDGE.ANCESTRIES
                for sample in reference_input_order if f"_{ancestry}_" in sample
            )
            sample_map = (outdir / BRIDGE.OUTPUT_NAMES["sample_map"]).read_text().splitlines()
            self.assertEqual(tuple(line.split("\t")[0] for line in sample_map),
                             expected_reference_order)

    def test_flare_bgzf_keeps_only_complete_snvs_and_does_not_open_other_roles(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs, _, target_samples = fixture(root)
            outdir = root / "bridge"
            receipt = run_bridge(inputs, outdir)
            ref_lines = read_gzip_lines(outdir / BRIDGE.OUTPUT_NAMES["reference_vcf"])
            target_lines = read_gzip_lines(outdir / BRIDGE.OUTPUT_NAMES["target_vcf"])
            ref_header = next(line for line in ref_lines if line.startswith("#CHROM"))
            target_header = next(line for line in target_lines if line.startswith("#CHROM"))
            self.assertEqual(tuple(target_header.split("\t")[9:]), target_samples)
            self.assertNotIn("valid_0", ref_header.split("\t")[9:])
            self.assertEqual(
                [int(line.split("\t")[1]) for line in ref_lines if not line.startswith("#")],
                [300],
            )
            self.assertEqual(
                [int(line.split("\t")[1]) for line in target_lines if not line.startswith("#")],
                [300],
            )
            for key in ("reference_vcf", "target_vcf"):
                raw_vcf = (outdir / BRIDGE.OUTPUT_NAMES[key]).read_bytes()
                self.assertEqual(raw_vcf[:4], b"\x1f\x8b\x08\x04")
                self.assertEqual(raw_vcf[12:16], b"BC\x02\x00")
                self.assertTrue(raw_vcf.endswith(BRIDGE.BGZF_EOF))
            self.assertEqual(receipt["counts"]["non_snv_or_non_biallelic_records_skipped"], 2)
            self.assertEqual(receipt["counts"]["biallelic_snv_records_for_factor_evaluation"], 3)
            self.assertEqual(receipt["counts"]["complete_biallelic_snv_records_for_flare"], 1)
            self.assertEqual(receipt["counts"]["snv_records_excluded_from_flare_for_missing_gt"], 2)
            self.assertEqual(
                receipt["counts"]["complete_snv_records_excluded_from_flare_for_ref_monomorphic"],
                0,
            )
            self.assertFalse(receipt["roles"]["source_valid_panel_genotypes_opened"])
            self.assertFalse(receipt["roles"]["source_test_panel_genotypes_opened"])

    def test_complete_ref_monomorphic_snv_is_excluded_from_flare_only(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs, _, _ = fixture(root, complete_monomorphic=True)
            outdir = root / "bridge"
            receipt = run_bridge(inputs, outdir)
            reference_rows = [
                line for line in read_gzip_lines(
                    outdir / BRIDGE.OUTPUT_NAMES["reference_vcf"]
                ) if not line.startswith("#")
            ]
            target_rows = [
                line for line in read_gzip_lines(
                    outdir / BRIDGE.OUTPUT_NAMES["target_vcf"]
                ) if not line.startswith("#")
            ]
            self.assertEqual([int(line.split("\t")[1]) for line in reference_rows], [300])
            self.assertEqual([int(line.split("\t")[1]) for line in target_rows], [300])
            self.assertEqual(
                receipt["counts"]["complete_snv_records_excluded_from_flare_for_ref_monomorphic"],
                1,
            )
            self.assertEqual(receipt["counts"]["rare_loci_selected"], 2)

    def test_source_valid_and_source_test_mosaic_roles_are_explicit(self):
        for role in BRIDGE.MOSAIC_DONOR_ROLES:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                inputs, _, _ = fixture(root)
                outdir = root / "bridge"
                receipt = run_bridge(inputs, outdir, mosaic_donor_role=role)
                header = "\n".join(
                    line for line in read_gzip_lines(
                        outdir / BRIDGE.OUTPUT_NAMES["target_vcf"]
                    ) if line.startswith("##m34_")
                )
                self.assertIn(f"##m34_mosaic_donor_role_upstream={role}", header)
                self.assertIn(f"##m34_bridge_vcf_role=TARGET_{role}_MOSAICS", header)
                self.assertEqual(receipt["roles"]["mosaic_donor_role_upstream"], role)
                self.assertEqual(
                    receipt["roles"]["source_test_mosaic_donors_upstream"],
                    role == "SOURCE_TEST",
                )
                self.assertFalse(receipt["roles"]["source_valid_panel_genotypes_opened"])
                self.assertFalse(receipt["roles"]["source_test_panel_genotypes_opened"])

    def test_selected_locus_ancestry_af_audit_is_ref_train_only(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs, _, _ = fixture(root, ancestry_enriched=True)
            receipt = run_bridge(inputs, root / "bridge")
            counts = receipt["counts"]
            for suffix in ("0.05", "0.10", "0.20"):
                self.assertEqual(
                    counts[f"selected_loci_max_ancestry_af_ge_{suffix}"], 1
                )
                self.assertEqual(
                    counts[f"selected_loci_by_ancestry_af_ge_{suffix}"],
                    {"AFR": 0, "EUR": 0, "NAM": 1},
                )
            self.assertEqual(receipt["counts"]["rare_loci_selected"], 3)
            self.assertEqual(
                receipt["parameters"]["ancestry_af_audit"],
                {
                    "thresholds": [0.05, 0.10, 0.20],
                    "source": "REF_TRAIN_only",
                    "used_for_primary_selection": False,
                },
            )

    def test_bgzf_multiblock_roundtrip(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "large.vcf.gz"
            value = "".join(chr(33 + (index * 37) % 90) for index in range(150_000))
            with BRIDGE.deterministic_bgzf_text(path) as handle:
                handle.write(value)
            self.assertEqual(gzip.decompress(path.read_bytes()).decode("utf-8"), value)
            payload = path.read_bytes()
            offset = 0
            blocks = 0
            while offset < len(payload):
                self.assertEqual(payload[offset:offset + 4], b"\x1f\x8b\x08\x04")
                block_size = int.from_bytes(payload[offset + 16:offset + 18], "little") + 1
                self.assertLessEqual(block_size, 65_536)
                offset += block_size
                blocks += 1
            self.assertEqual(offset, len(payload))
            self.assertGreaterEqual(blocks, 4)  # three data blocks plus canonical EOF

    def test_axis_mismatch_fails_closed_without_output(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs, _, _ = fixture(root, axis_mismatch=True)
            outdir = root / "bridge"
            with self.assertRaisesRegex(ValueError, "axis differs"):
                run_bridge(inputs, outdir)
            self.assertFalse(outdir.exists())

    def test_unphased_reference_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs, _, _ = fixture(root, unphased_reference=True)
            with self.assertRaisesRegex(ValueError, "unphased/non-diploid"):
                run_bridge(inputs, root / "bridge")

    def test_population_and_atomic_unit_must_be_role_disjoint(self):
        for option, expected in (
            ({"crossed_population": True}, "canonical_population crosses"),
            ({"crossed_unit": True}, "atomic_unit_id crosses"),
        ):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                inputs, _, _ = fixture(root, **option)
                with self.assertRaisesRegex(ValueError, expected):
                    run_bridge(inputs, root / "bridge")

    def test_target_axis_must_be_disjoint_from_split(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs, _, _ = fixture(root, target_collision=True)
            with self.assertRaisesRegex(ValueError, "mosaic targets overlap split"):
                run_bridge(inputs, root / "bridge")

    def test_all_artifacts_and_receipt_are_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            inputs, _, _ = fixture(root)
            first = root / "first"
            second = root / "second"
            run_bridge(inputs, first)
            run_bridge(inputs, second)
            self.assertEqual(set(path.name for path in first.iterdir()),
                             set(BRIDGE.OUTPUT_NAMES.values()))
            for name in BRIDGE.OUTPUT_NAMES.values():
                with self.subTest(name=name):
                    self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            receipt = json.loads((first / BRIDGE.OUTPUT_NAMES["receipt"]).read_text())
            self.assertEqual(receipt["decision"],
                             "PASS_EXPLORATORY_PANEL_FACTORS_SOURCE_VALID_MOSAICS")
            self.assertTrue(receipt["scope"]["exploratory_only"])
            self.assertFalse(receipt["scope"]["confirmatory_validation"])

    def test_frozen_selection_parameters_reject_drift(self):
        for kwargs, expected in (
            ({"min_mac": 3}, "freezes --min-mac"),
            ({"max_maf_exclusive": 0.02}, "freezes --max-maf-exclusive"),
        ):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                inputs, _, _ = fixture(root)
                with self.assertRaisesRegex(ValueError, expected):
                    run_bridge(inputs, root / "bridge", **kwargs)


if __name__ == "__main__":
    unittest.main()
