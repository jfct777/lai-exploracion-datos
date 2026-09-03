#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m38_build_f_minus_s660 import (  # noqa: E402
    FMinusS660ContractError,
    build_f_minus_s660,
)
from m38_stratify_rare_loci import validate_exact_locus_partition  # noqa: E402


WORKFLOW = ROOT / "workflows/m38_f_minus_s660_baseline.nf"
CONFIG = ROOT / "conf/m38_f_minus_s660_baseline.config"
FILTER_MODULE = ROOT / "modules/38_F_MINUS_S660_FILTER.nf"
INDEX_MODULE = ROOT / "modules/38_F_MINUS_S660_BGZIP_TABIX.nf"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vcf_text(
    samples: list[str],
    rows: list[tuple[int, str, str, list[str]]],
    *,
    vcf_role: str,
    donor_role: str = "SOURCE_VALID",
) -> str:
    header = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=22,length=50818468>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        f"##m34_bridge_vcf_role={vcf_role}\n"
        "##m34_reference_and_frequency_role=REF_TRAIN\n"
        f"##m34_mosaic_donor_role_upstream={donor_role}\n"
        + "\t".join(
            ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *samples]
        )
        + "\n"
    )
    body = "".join(
        "\t".join(
            ["22", str(position), ".", ref, alt, ".", "PASS", ".", "GT", *genotypes]
        )
        + "\n"
        for position, ref, alt, genotypes in rows
    )
    return header + body


def write_selected(path: Path, keys: list[tuple[str, int, str, str]]) -> None:
    np.savez(
        path,
        alt=np.asarray([key[3].encode("ascii") for key in keys], dtype="S1"),
        cM=np.arange(len(keys), dtype=np.float64),
        chrom=np.asarray([int(key[0].removeprefix("chr")) for key in keys], dtype=np.uint8),
        locus_id=np.arange(100, 100 + len(keys), dtype=np.uint64),
        pos=np.asarray([key[1] for key in keys], dtype=np.int64),
        ref=np.asarray([key[2].encode("ascii") for key in keys], dtype="S1"),
    )


def make_fixture(root: Path) -> dict[str, object]:
    rows = [
        (10, "A", "G", ["0|0", "0|1"]),
        (20, "C", "T", ["0|1", "1|0"]),
        (30, "G", "A", ["1|0", "0|0"]),
        (40, "T", "C", ["1|1", "0|1"]),
    ]
    target_rows = [
        (position, ref, alt, [genotypes[0]])
        for position, ref, alt, genotypes in rows
    ]
    reference = root / "reference.vcf"
    target = root / "target.vcf"
    selected = root / "selected.npz"
    reference.write_text(
        vcf_text(["REF1", "REF2"], rows, vcf_role="REFERENCE_REF_TRAIN"),
        encoding="utf-8",
    )
    target.write_text(
        vcf_text(["TARGET1"], target_rows, vcf_role="TARGET_SOURCE_VALID_MOSAICS"),
        encoding="utf-8",
    )
    selected_keys = [("22", 20, "C", "T"), ("22", 40, "T", "C")]
    write_selected(selected, selected_keys)
    return {
        "reference": reference,
        "target": target,
        "selected": selected,
        "selected_keys": selected_keys,
    }


def run_fixture(inputs: dict[str, object], outdir: Path, **changes: object):
    arguments = {
        "split": "FIT",
        "reference_vcf": inputs["reference"],
        "target_vcf": inputs["target"],
        "selected_loci": inputs["selected"],
        "expected_reference_sha256": digest(inputs["reference"]),
        "expected_target_sha256": digest(inputs["target"]),
        "expected_selected_sha256": digest(inputs["selected"]),
        "expected_chromosome": "22",
        "expected_full_count": 4,
        "expected_selected_count": 2,
        "expected_reference_samples": 2,
        "expected_target_samples": 1,
        "outdir": outdir,
    }
    arguments.update(changes)
    return build_f_minus_s660(**arguments)


class FMinusS660FilterTests(unittest.TestCase):
    def test_exact_partition_and_text_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            receipt = run_fixture(inputs, root / "out")
            self.assertEqual(
                receipt["status"],
                "PASS_F_FULL_EQUALS_F_MINUS_S660_DISJOINT_UNION_S660",
            )
            self.assertEqual(receipt["counts"]["f_minus_s660_loci"], 2)
            self.assertEqual(receipt["counts"]["selected_rare_loci"], 2)
            self.assertEqual(receipt["roles"]["upstream_mosaic_donor_role"], "SOURCE_VALID")
            self.assertTrue(receipt["downstream_constraints"]["identical_scoring_grid_required"])
            self.assertTrue(receipt["partition"]
                            ["F_full_equals_disjoint_union_F_minus_S660_and_S660"])
            self.assertFalse(receipt["identity"]["locus_id_used"])
            self.assertEqual(receipt["identity"]["source_contig_label"], "22")
            self.assertFalse(receipt["scope"]["local_ancestry_labels_read"])
            reference_output = root / "out/m38_f_minus_s660_reference.chr22.vcf"
            target_output = root / "out/m38_f_minus_s660_target.chr22.vcf"
            reference_lines = reference_output.read_text(encoding="utf-8").splitlines()
            target_lines = target_output.read_text(encoding="utf-8").splitlines()
            header_rows = 7
            self.assertEqual(reference_lines[:header_rows],
                             Path(inputs["reference"]).read_text().splitlines()[:header_rows])
            self.assertEqual(target_lines[:header_rows],
                             Path(inputs["target"]).read_text().splitlines()[:header_rows])
            self.assertEqual([line.split("\t")[1] for line in reference_lines[header_rows:]],
                             ["10", "30"])
            self.assertEqual([line.split("\t")[1] for line in target_lines[header_rows:]],
                             ["10", "30"])
            self.assertEqual(reference_lines[header_rows].split("\t")[9:], ["0|0", "0|1"])

    def test_deterministic_receipt_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            first = root / "first"
            second = root / "second"
            run_fixture(inputs, first)
            run_fixture(inputs, second)
            for name in (
                "m38_f_minus_s660_reference.chr22.vcf",
                "m38_f_minus_s660_target.chr22.vcf",
                "m38_f_minus_s660_filter.receipt.json",
            ):
                self.assertEqual(digest(first / name), digest(second / name))
            with self.assertRaisesRegex(FMinusS660ContractError, "overwrite"):
                run_fixture(inputs, first)

    def test_wrong_hash_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            outdir = root / "out"
            with self.assertRaisesRegex(FMinusS660ContractError, "SHA-256 mismatch"):
                run_fixture(inputs, outdir, expected_reference_sha256="0" * 64)
            self.assertFalse(outdir.exists())

    def test_same_position_different_allele_is_not_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            write_selected(Path(inputs["selected"]), [
                ("22", 20, "C", "G"), ("22", 40, "T", "C")
            ])
            with self.assertRaisesRegex(FMinusS660ContractError, "absent from the full FLARE axis"):
                run_fixture(inputs, root / "out")

    def test_distinct_alleles_at_same_position_are_preserved_in_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            reference_rows = [
                (10, "A", "G", ["0|0", "0|1"]),
                (20, "C", "T", ["0|1", "1|0"]),
                (20, "C", "G", ["0|0", "0|1"]),
                (30, "G", "A", ["1|0", "0|0"]),
                (40, "T", "C", ["1|1", "0|1"]),
            ]
            target_rows = [
                (position, ref, alt, [genotypes[0]])
                for position, ref, alt, genotypes in reference_rows
            ]
            Path(inputs["reference"]).write_text(
                vcf_text(
                    ["REF1", "REF2"], reference_rows,
                    vcf_role="REFERENCE_REF_TRAIN",
                ),
                encoding="utf-8",
            )
            Path(inputs["target"]).write_text(
                vcf_text(
                    ["TARGET1"], target_rows,
                    vcf_role="TARGET_SOURCE_VALID_MOSAICS",
                ),
                encoding="utf-8",
            )
            receipt = run_fixture(inputs, root / "out", expected_full_count=5)
            self.assertEqual(receipt["counts"]["f_minus_s660_loci"], 3)
            retained = (root / "out/m38_f_minus_s660_reference.chr22.vcf").read_text(
                encoding="utf-8"
            )
            self.assertIn("22\t20\t.\tC\tG\t", retained)
            self.assertNotIn("22\t20\t.\tC\tT\t", retained)

    def test_absent_selected_variant_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            write_selected(Path(inputs["selected"]), [
                ("22", 20, "C", "T"), ("22", 50, "A", "C")
            ])
            with self.assertRaisesRegex(FMinusS660ContractError, "absent from the full FLARE axis"):
                run_fixture(inputs, root / "out")

    def test_reference_target_axis_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            target = Path(inputs["target"])
            target.write_text(target.read_text().replace("30\t.\tG\tA", "30\t.\tG\tT"),
                              encoding="utf-8")
            with self.assertRaisesRegex(FMinusS660ContractError, "axes differ"):
                run_fixture(inputs, root / "out")

    def test_sample_drift_and_sample_overlap_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            with self.assertRaisesRegex(FMinusS660ContractError, "target sample count drifted"):
                run_fixture(inputs, root / "count", expected_target_samples=2)
            target = Path(inputs["target"])
            target.write_text(target.read_text().replace("\tTARGET1\n", "\tREF1\n", 1),
                              encoding="utf-8")
            with self.assertRaisesRegex(FMinusS660ContractError, "sample axes overlap"):
                run_fixture(inputs, root / "overlap")

    def test_m34_roles_are_authenticated_for_each_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            target = Path(inputs["target"])
            reference = Path(inputs["reference"])
            target.write_text(
                target.read_text(encoding="utf-8")
                .replace("TARGET_SOURCE_VALID_MOSAICS", "TARGET_SOURCE_TEST_MOSAICS")
                .replace("SOURCE_VALID", "SOURCE_TEST"),
                encoding="utf-8",
            )
            reference.write_text(
                reference.read_text(encoding="utf-8").replace(
                    "SOURCE_VALID", "SOURCE_TEST"
                ),
                encoding="utf-8",
            )
            receipt = run_fixture(
                inputs,
                root / "valid",
                split="VALID",
                expected_reference_sha256=digest(reference),
                expected_target_sha256=digest(target),
            )
            self.assertEqual(receipt["roles"]["target_partition"], "VALID")
            self.assertEqual(
                receipt["roles"]["upstream_mosaic_donor_role"], "SOURCE_TEST"
            )

            reference.write_text(
                reference.read_text(encoding="utf-8").replace(
                    "REFERENCE_REF_TRAIN", "REFERENCE_SOURCE_VALID"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FMinusS660ContractError, "role header differs"):
                run_fixture(
                    inputs,
                    root / "wrong_role",
                    split="VALID",
                    expected_reference_sha256=digest(reference),
                    expected_target_sha256=digest(target),
                )

    def test_duplicate_selected_and_duplicate_vcf_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            write_selected(Path(inputs["selected"]), [
                ("22", 20, "C", "T"), ("22", 20, "C", "T")
            ])
            with self.assertRaisesRegex(FMinusS660ContractError, "duplicate variants"):
                run_fixture(inputs, root / "selected_duplicate")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            reference = Path(inputs["reference"])
            last = reference.read_text().splitlines(keepends=True)[-1]
            with reference.open("a", encoding="utf-8") as handle:
                handle.write(last)
            with self.assertRaisesRegex(FMinusS660ContractError, "duplicate variant"):
                run_fixture(inputs, root / "vcf_duplicate", expected_full_count=5)

    def test_partition_helper_rejects_overlap(self) -> None:
        full = [("22", 10, "A", "G"), ("22", 20, "C", "T"), ("22", 30, "G", "A")]
        with self.assertRaisesRegex(ValueError, "intersects selected"):
            validate_exact_locus_partition(full, full[:2], full[1:])


class FMinusS660NextflowTests(unittest.TestCase):
    def test_wiring_is_nextflow_first_pinned_and_personal(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        config = CONFIG.read_text(encoding="utf-8")
        filter_module = FILTER_MODULE.read_text(encoding="utf-8")
        index_module = INDEX_MODULE.read_text(encoding="utf-8")
        self.assertIn("M38_F_MINUS_S660_FILTER", workflow)
        self.assertIn("M38_F_MINUS_S660_BGZIP_TABIX", workflow)
        self.assertIn("gs://teams-usp/frank/lai-exploracion-datos", config)
        self.assertIn("resourceLabels = [team: 'frank'", config)
        self.assertIn("m38_fminus_splits = ['FIT']", config)
        self.assertIn("@sha256:", config)
        self.assertIn("--network none", filter_module)
        self.assertIn("--network none", index_module)
        self.assertIn("overwrite: false", filter_module)
        self.assertIn("overwrite: false", index_module)
        self.assertIn("bgzip --threads 1", index_module)
        self.assertIn("tabix -p vcf", index_module)
        self.assertIn('receipt["identity"]["source_contig_label"]', index_module)
        self.assertNotIn("grep -Fxq '${params.m38_fminus_chromosome}'", index_module)
        self.assertIn("saveAs:", filter_module)
        self.assertIn("name.endsWith('.receipt.json') ? name : null", filter_module)
        self.assertIn("split, referenceVcf, targetVcf, filterReceipt", workflow)
        combined = "\n".join((workflow, config, filter_module, index_module)).lower()
        self.assertNotIn("local_ancestry_truth", combined)
        self.assertNotIn("prediction", combined)
        self.assertNotIn("model_scores", combined)

    @unittest.skipUnless(shutil.which("nextflow"), "Nextflow is not installed")
    def test_configuration_parses(self) -> None:
        completed = subprocess.run(
            ["nextflow", "-C", str(CONFIG), "config", "-flat"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "NXF_OFFLINE": "true"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)


if __name__ == "__main__":
    unittest.main()
