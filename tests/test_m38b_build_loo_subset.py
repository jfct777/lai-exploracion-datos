#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m38b_build_loo_subset import (  # noqa: E402
    LooSubsetContractError,
    build_loo_subset,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_samples() -> tuple[list[str], list[dict[str, str]]]:
    sample_ids: list[str] = []
    rows: list[dict[str, str]] = []
    for ancestry in ("AFR", "EUR"):
        for number in range(10):
            sample_id = f"{ancestry}_{number:02d}"
            sample_ids.append(sample_id)
            rows.append({
                "sample_id": sample_id,
                "ancestry": ancestry,
                "atomic_unit_id": f"{ancestry}_UNIT",
                "role": "REF_TRAIN",
            })
    for unit_number in range(1, 5):
        for number in range(5):
            sample_id = f"NAM_U{unit_number}_{number:02d}"
            sample_ids.append(sample_id)
            rows.append({
                "sample_id": sample_id,
                "ancestry": "NAM",
                "atomic_unit_id": f"NAM_U{unit_number}",
                "role": "REF_TRAIN",
            })
    sample_ids.append("NOT_REF_TRAIN")
    rows.append({
        "sample_id": "NOT_REF_TRAIN",
        "ancestry": "NAM",
        "atomic_unit_id": "NAM_OUTSIDE",
        "role": "SOURCE_VALID",
    })
    return sample_ids, rows


def locus_genotypes(
    sample_ids: list[str], *, supporting_units: int,
) -> list[tuple[int, str, str, list[str]]]:
    loci: list[tuple[int, str, str, list[str]]] = []
    for locus, (position, ref, alt) in enumerate((
        (100, "A", "G"),
        (200, "C", "T"),
        (300, "G", "A"),
    )):
        genotypes: list[str] = []
        for sample_id in sample_ids:
            if sample_id == "NOT_REF_TRAIN":
                genotypes.append("1|1" if locus != 1 else "0|0")
                continue
            if sample_id.startswith(("AFR_", "EUR_")):
                genotypes.append("0|0" if locus != 1 else "1|1")
                continue
            unit = int(sample_id.split("_")[1][1:])
            person = int(sample_id.rsplit("_", 1)[1])
            if locus == 0:
                genotype = "0|1" if unit <= supporting_units and person < 2 else "0|0"
            elif locus == 1:
                genotype = "0|1" if unit <= supporting_units and person < 2 else "1|1"
            else:
                genotype = "0|1" if unit in (1, 2) and person == 0 else "0|0"
                if unit == 4 and person == 0:
                    genotype = ".|."
            genotypes.append(genotype)
        loci.append((position, ref, alt, genotypes))
    return loci


def make_fixture(root: Path, *, supporting_units: int = 4) -> dict[str, Path]:
    samples, split_rows = make_samples()
    split = root / "m27f_split.private.tsv"
    with split.open("w", encoding="utf-8", newline="") as handle:
        handle.write("sample_id\tancestry\tatomic_unit_id\trole\n")
        for row in split_rows:
            handle.write("\t".join(row[column] for column in (
                "sample_id", "ancestry", "atomic_unit_id", "role")) + "\n")

    records = locus_genotypes(samples, supporting_units=supporting_units)
    panel = root / "reference.chr22.vcf"
    with panel.open("w", encoding="utf-8", newline="") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##contig=<ID=22,length=50818468>\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype">\n')
        handle.write("\t".join((
            "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO",
            "FORMAT", *samples,
        )) + "\n")
        for position, ref, alt, genotypes in records:
            handle.write("\t".join((
                "22", str(position), ".", ref, alt, ".", "PASS", ".", "GT",
                *genotypes,
            )) + "\n")

    selected = root / "m34_selected_loci.npz"
    np.savez(
        selected,
        chrom=np.full(3, 22, dtype=np.uint8),
        pos=np.asarray([record[0] for record in records], dtype=np.int64),
        ref=np.asarray([record[1].encode("ascii") for record in records], dtype="S1"),
        alt=np.asarray([record[2].encode("ascii") for record in records], dtype="S1"),
        cM=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        locus_id=np.asarray([1001, 1002, 1003], dtype=np.uint64),
    )
    return {"panel": panel, "split": split, "selected": selected}


def run_fixture(
    inputs: dict[str, Path], output: Path, **changes: object,
) -> tuple[dict[str, object], dict[str, Path]]:
    outputs = {
        "tsv": output / "m38b_loo_subset.tsv",
        "npz": output / "m38b_loo_subset.npz",
        "receipt": output / "m38b_loo_subset.receipt.json",
    }
    arguments: dict[str, object] = {
        "panel_vcf": inputs["panel"],
        "split_tsv": inputs["split"],
        "selected_loci": inputs["selected"],
        "expected_panel_sha256": digest(inputs["panel"]),
        "expected_split_sha256": digest(inputs["split"]),
        "expected_selected_sha256": digest(inputs["selected"]),
        "expected_chromosome": "22",
        "expected_loci": 3,
        "expected_nam_units": 4,
        "beta_priors": (0.5, 1.0),
        "q_top_threshold": 0.8,
        "min_remaining_nam_units": 2,
        "posterior_draws": 4096,
        "seed": 3802103,
        "output_tsv": outputs["tsv"],
        "output_npz": outputs["npz"],
        "output_receipt": outputs["receipt"],
    }
    arguments.update(changes)
    return build_loo_subset(**arguments), outputs


class M38BLooSubsetTests(unittest.TestCase):
    def test_minor_orientation_missingness_and_each_omission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            receipt, outputs = run_fixture(inputs, root / "out")
            self.assertEqual(receipt["status"], "PASS_PRIMARY_LOO_SUBSET_FROZEN")
            self.assertEqual(receipt["counts"]["NAM_atomic_units"], 4)
            self.assertEqual(len(receipt["counts"]["per_omission"]), 4)
            self.assertEqual(receipt["orientation"]["minor_code_zero_loci"], 1)
            self.assertEqual(receipt["orientation"]["minor_code_one_loci"], 2)
            self.assertFalse(receipt["scope"]["target_genotypes_read"])
            self.assertFalse(receipt["scope"]["local_ancestry_truth_read"])
            self.assertFalse(receipt["scope"]["predictions_read"])
            self.assertFalse(receipt["scope"]["king_used"])
            with np.load(outputs["npz"], allow_pickle=False) as archive:
                self.assertEqual(archive["loo_minor_ac"].shape, (4, 3, 3))
                self.assertEqual(archive["minor_code"].tolist(), [1, 0, 1])
                self.assertEqual(archive["pooled_alt_ac"].tolist(), [8, 72, 2])
                self.assertEqual(archive["pooled_callable_an"].tolist(), [80, 80, 78])
                self.assertEqual(archive["primary_mask"].tolist(), [1, 1, 0])
                self.assertEqual(
                    archive["remaining_nam_carrier_units"][:, 2].tolist(),
                    [1, 1, 2, 2],
                )
                # U1 is a carrier unit at locus 3; U4 is a non-carrier unit
                # containing one fully missing genotype.
                self.assertEqual(int(archive["loo_callable_an"][0, 2, 2]), 28)
                self.assertEqual(int(archive["loo_callable_an"][3, 2, 2]), 30)
                for name in ("q_top_prior_0p5", "q_top_prior_1p0"):
                    self.assertTrue(np.allclose(archive[name].sum(axis=1), 1.0))
                    self.assertTrue(np.all(archive[name][:, 2, :2] >= 0.8))
            lines = outputs["tsv"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1 + 3 * 4)

    def test_zero_subset_is_valid_and_thresholds_are_not_relaxed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root, supporting_units=2)
            receipt, outputs = run_fixture(inputs, root / "out")
            self.assertEqual(
                receipt["status"], "PASS_ZERO_PRIMARY_LOO_SUBSET_NO_RELAXATION")
            self.assertEqual(receipt["counts"]["primary_loci"], 0)
            self.assertFalse(receipt["selection_contract"]["post_outcome_relaxation_allowed"])
            with np.load(outputs["npz"], allow_pickle=False) as archive:
                self.assertEqual(int(archive["primary_mask"].sum()), 0)
                self.assertEqual(archive["primary_locus_id"].shape, (0,))

    def test_outputs_are_byte_reproducible_and_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            _, first = run_fixture(inputs, root / "first")
            _, second = run_fixture(inputs, root / "second")
            for output_type in ("tsv", "npz", "receipt"):
                self.assertEqual(digest(first[output_type]), digest(second[output_type]))
            with self.assertRaisesRegex(LooSubsetContractError, "overwrite"):
                run_fixture(inputs, root / "first")

    def test_hash_mismatch_fails_before_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            output = root / "out"
            with self.assertRaisesRegex(LooSubsetContractError, "SHA-256 mismatch"):
                run_fixture(
                    inputs, output,
                    expected_selected_sha256="0" * 64,
                )
            self.assertFalse(output.exists())

    def test_non_ref_train_rows_cannot_affect_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            _, outputs = run_fixture(inputs, root / "out")
            with np.load(outputs["npz"], allow_pickle=False) as archive:
                # The outside sample is 1|1 at the first locus, but ALT AC is
                # exactly eight from the four REF_TRAIN NAM units.
                self.assertEqual(int(archive["pooled_alt_ac"][0]), 8)
            receipt = json.loads(outputs["receipt"].read_text(encoding="utf-8"))
            self.assertEqual(receipt["counts"]["REF_TRAIN_people"], 40)

    def test_absent_locus_and_unphased_call_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            panel = inputs["panel"]
            panel.write_text(
                panel.read_text(encoding="utf-8").replace("22\t300\t", "22\t301\t", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LooSubsetContractError, "absent from panel"):
                run_fixture(inputs, root / "absent")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = make_fixture(root)
            panel = inputs["panel"]
            panel.write_text(
                panel.read_text(encoding="utf-8").replace("0|1", "0/1", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LooSubsetContractError, "unphased called genotype"):
                run_fixture(inputs, root / "unphased")

    def test_interface_has_no_downstream_information(self) -> None:
        parameters = set(inspect.signature(build_loo_subset).parameters)
        self.assertEqual(parameters, {
            "panel_vcf", "split_tsv", "selected_loci",
            "expected_panel_sha256", "expected_split_sha256",
            "expected_selected_sha256", "expected_chromosome", "expected_loci",
            "expected_nam_units", "beta_priors", "q_top_threshold",
            "min_remaining_nam_units", "posterior_draws", "seed",
            "output_tsv", "output_npz", "output_receipt",
        })
        forbidden = {"target", "truth", "prediction", "score", "king"}
        self.assertFalse(parameters & forbidden)


if __name__ == "__main__":
    unittest.main()
