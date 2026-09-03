#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m33_safe_bridge_core import write_deterministic_npz  # noqa: E402
from m38b_collect_oof import (  # noqa: E402
    M38BOofCollectionError,
    STATE_NAMES,
    TCN_SEEDS,
    collect_oof,
)
from m38b_make_folds import build as build_folds  # noqa: E402
from m38b_subset_factors import (  # noqa: E402
    M38BFactorSubsetError,
    OUTPUT_NAMES,
    subset_factors,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_factor_fixture(root: Path, *, mask: tuple[int, ...] = (1, 0, 1, 0),
                        loo_pos_shift: int = 0,
                        reference_ac_shift: int = 0) -> dict[str, Path]:
    locus_id = np.asarray([10, 11, 12, 13], dtype="<u8")
    selected_arrays = {
        "locus_id": locus_id,
        "chrom": np.full(4, 22, dtype="|u1"),
        "pos": np.asarray([100, 200, 300, 400], dtype="<i8"),
        "ref": np.asarray([b"A", b"C", b"G", b"T"], dtype="|S1"),
        "alt": np.asarray([b"G", b"T", b"A", b"C"], dtype="|S1"),
        "cM": np.asarray([0.1, 0.2, 0.3, 0.4], dtype="<f8"),
    }
    target_arrays = {
        "sample_key_sha256": np.asarray([b"a" * 64, b"b" * 64, b"c" * 64], dtype="|S64"),
        "locus_id": locus_id,
        "minor_dosage": np.asarray([[0, 1, 2, 0], [1, 0, 1, 2], [0, 0, 1, 1]], dtype="|i1"),
        "observed_mask": np.asarray([[1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 1, 1]], dtype="|u1"),
    }
    # Missing values must have dosage zero.
    target_arrays["minor_dosage"][target_arrays["observed_mask"] == 0] = 0
    ac = np.asarray([
        [1, 2, 1, 2],
        [1, 1, 2, 1],
        [4, 3, 5, 2],
    ], dtype="<u2")
    if reference_ac_shift:
        ac[0, 0] = np.uint16(int(ac[0, 0]) + reference_ac_shift)
    an = np.asarray([
        [20, 20, 20, 20],
        [20, 20, 20, 20],
        [8, 8, 8, 8],
    ], dtype="<u2")
    reference_arrays = {
        "ancestry": np.asarray([b"AFR", b"EUR", b"NAM"], dtype="|S4"),
        "locus_id": locus_id,
        "minor_ac": ac,
        "callable_an": an,
        "minor_af": ac.astype("<f8") / an,
        "observed_mask": (an > 0).astype("|u1"),
        "no_support": ((an > 0) & (ac == 0)).astype("|u1"),
    }
    primary_mask = np.asarray(mask, dtype="|u1")
    primary = np.flatnonzero(primary_mask)
    minor_code = np.asarray([1, 0, 1, 0], dtype="|i1")
    pooled_minor = ac.astype(np.int64).sum(axis=0)
    pooled_an = an.astype(np.int64).sum(axis=0)
    pooled_alt = np.where(minor_code == 1, pooled_minor, pooled_an - pooled_minor)
    loo_pos = selected_arrays["pos"].copy()
    loo_pos[0] += loo_pos_shift
    loo_arrays = {
        "ancestry": np.asarray(["AFR", "EUR", "NAM"]),
        "omitted_nam_unit": np.asarray(["U1", "U2", "U3", "U4"]),
        "locus_id": locus_id,
        "chrom": selected_arrays["chrom"],
        "pos": loo_pos,
        "ref": selected_arrays["ref"],
        "alt": selected_arrays["alt"],
        "cM": selected_arrays["cM"],
        "minor_code": minor_code,
        "pooled_alt_ac": pooled_alt.astype("<i8"),
        "pooled_callable_an": pooled_an.astype("<i8"),
        "full_minor_ac": ac.astype("<i8") if not reference_ac_shift else np.asarray([
            [1, 2, 1, 2], [1, 1, 2, 1], [4, 3, 5, 2]
        ], dtype="<i8"),
        "full_callable_an": an.astype("<i8"),
        "loo_minor_ac": np.broadcast_to(ac, (4, 3, 4)).copy().astype("<i8"),
        "loo_callable_an": np.broadcast_to(an, (4, 3, 4)).copy().astype("<i8"),
        "loo_minor_af": np.broadcast_to(ac / an, (4, 3, 4)).copy().astype("<f8"),
        "remaining_nam_carrier_units": np.full((4, 4), 2, dtype="|u1"),
        "q_nam_min_all_priors_omissions": np.full(4, 0.9, dtype="<f8"),
        "remaining_nam_carrier_units_min": np.full(4, 2, dtype="|u1"),
        "primary_mask": primary_mask,
        "primary_locus_id": locus_id[primary],
        "primary_pos": loo_pos[primary],
        "primary_ref": selected_arrays["ref"][primary],
        "primary_alt": selected_arrays["alt"][primary],
        "primary_cM": selected_arrays["cM"][primary],
        "primary_minor_code": minor_code[primary],
    }
    paths = {
        "selected": root / "selected.npz",
        "target": root / "target.npz",
        "reference": root / "reference.npz",
        "loo": root / "loo.npz",
        "loo_receipt": root / "loo.receipt.json",
    }
    for name, arrays in (("selected", selected_arrays), ("target", target_arrays),
                         ("reference", reference_arrays), ("loo", loo_arrays)):
        write_deterministic_npz(paths[name], arrays)
    status = ("PASS_PRIMARY_LOO_SUBSET_FROZEN" if primary.size
              else "PASS_ZERO_PRIMARY_LOO_SUBSET_NO_RELAXATION")
    write_json(paths["loo_receipt"], {
        "stage": "M38B_REF_TRAIN_LEAVE_ONE_NAM_UNIT_OUT_SUBSET",
        "status": status,
        "scope": {
            "frequency_role": "REF_TRAIN_only", "target_genotypes_read": False,
            "local_ancestry_truth_read": False, "predictions_read": False,
            "scores_read": False, "king_used": False,
        },
        "selection_contract": {
            "beta_priors": [0.5, 1.0], "q_top_threshold": 0.8,
            "minimum_remaining_NAM_carrier_units": 2,
            "all_omissions_required": True, "all_priors_required": True,
            "post_outcome_relaxation_allowed": False,
        },
        "counts": {"S660_loci": 4, "primary_loci": int(primary.size)},
        "inputs": {"selected_loci_sha256": digest(paths["selected"])},
        "outputs": {"loo_subset_npz_sha256": digest(paths["loo"])},
    })
    return paths


def run_subset(paths: dict[str, Path], outdir: Path) -> dict[str, object]:
    return subset_factors(
        loo_subset=paths["loo"], loo_receipt=paths["loo_receipt"],
        selected=paths["selected"], target=paths["target"],
        reference=paths["reference"], expected_loo_sha256=digest(paths["loo"]),
        expected_loo_receipt_sha256=digest(paths["loo_receipt"]),
        expected_selected_sha256=digest(paths["selected"]),
        expected_target_sha256=digest(paths["target"]),
        expected_reference_sha256=digest(paths["reference"]),
        outdir=outdir, expected_loci=4,
    )


class M38BFactorSubsetTests(unittest.TestCase):
    def test_primary_mask_filters_all_three_factors_without_axis_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = make_factor_fixture(root)
            receipt = run_subset(paths, root / "out")
            self.assertEqual(receipt["decision"], "PASS_PRIMARY_FACTORS_FROZEN_FOR_MODEL")
            with np.load(root / "out" / OUTPUT_NAMES["selected"], allow_pickle=False) as selected, \
                    np.load(root / "out" / OUTPUT_NAMES["target"], allow_pickle=False) as target, \
                    np.load(root / "out" / OUTPUT_NAMES["reference"], allow_pickle=False) as reference:
                np.testing.assert_array_equal(selected["locus_id"], [10, 12])
                np.testing.assert_array_equal(target["locus_id"], selected["locus_id"])
                np.testing.assert_array_equal(reference["locus_id"], selected["locus_id"])
                self.assertEqual(target["minor_dosage"].shape, (3, 2))
                self.assertEqual(reference["minor_ac"].shape, (3, 2))

    def test_factor_outputs_are_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "a").mkdir()
            (root / "b").mkdir()
            a = make_factor_fixture(root / "a")
            b = make_factor_fixture(root / "b")
            run_subset(a, root / "out-a")
            run_subset(b, root / "out-b")
            for name in ("selected", "target", "reference"):
                self.assertEqual(
                    digest(root / "out-a" / OUTPUT_NAMES[name]),
                    digest(root / "out-b" / OUTPUT_NAMES[name]),
                )

    def test_zero_mask_stops_without_exposing_empty_model_npz(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = make_factor_fixture(root, mask=(0, 0, 0, 0))
            receipt = run_subset(paths, root / "out")
            self.assertEqual(receipt["decision"], "STOP_MODEL_NO_PRIMARY_LOCI")
            self.assertFalse(receipt["empty_factor_npz_written"])
            for name in ("selected", "target", "reference"):
                self.assertFalse((root / "out" / OUTPUT_NAMES[name]).exists())

    def test_axis_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = make_factor_fixture(root, loo_pos_shift=1)
            with self.assertRaisesRegex(M38BFactorSubsetError, "LOO/original pos"):
                run_subset(paths, root / "out")

    def test_reference_orientation_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = make_factor_fixture(root, reference_ac_shift=1)
            with self.assertRaisesRegex(M38BFactorSubsetError, "counts or minor orientation"):
                run_subset(paths, root / "out")

    def test_hash_mismatch_fails_before_selection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = make_factor_fixture(root)
            with self.assertRaisesRegex(M38BFactorSubsetError, "LOO subset SHA-256"):
                subset_factors(
                    loo_subset=paths["loo"], loo_receipt=paths["loo_receipt"],
                    selected=paths["selected"], target=paths["target"],
                    reference=paths["reference"], expected_loo_sha256="0" * 64,
                    expected_loo_receipt_sha256=digest(paths["loo_receipt"]),
                    expected_selected_sha256=digest(paths["selected"]),
                    expected_target_sha256=digest(paths["target"]),
                    expected_reference_sha256=digest(paths["reference"]),
                    outdir=root / "out", expected_loci=4,
                )

    def test_loo_receipt_hash_mismatch_fails_before_selection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = make_factor_fixture(root)
            with self.assertRaisesRegex(M38BFactorSubsetError, "LOO receipt SHA-256"):
                subset_factors(
                    loo_subset=paths["loo"], loo_receipt=paths["loo_receipt"],
                    selected=paths["selected"], target=paths["target"],
                    reference=paths["reference"], expected_loo_sha256=digest(paths["loo"]),
                    expected_loo_receipt_sha256="0" * 64,
                    expected_selected_sha256=digest(paths["selected"]),
                    expected_target_sha256=digest(paths["target"]),
                    expected_reference_sha256=digest(paths["reference"]),
                    outdir=root / "out", expected_loci=4,
                )


def make_oof_fixture(root: Path, *, family: str = "tcn", arm: str = "RE",
                     seed_values: tuple[int, ...] = TCN_SEEDS,
                     marker_drift: tuple[int, int] | None = None,
                     sample_reverse: tuple[int, int] | None = None,
                     add_truth: tuple[int, int] | None = None) -> dict[str, object]:
    source = root / "fold-source.npz"
    samples = np.asarray([f"{index:064d}".encode() for index in range(96)], dtype="|S64")
    write_deterministic_npz(source, {"sample_key_sha256": samples})
    folds = root / "folds.npz"
    build_folds(source, folds, outer_seed=7, inner_seed_start=100)
    with np.load(folds, allow_pickle=False) as archive:
        roles = np.asarray(archive["roles"])
    marker = np.asarray([100, 200, 300, 400, 500], dtype="<i8")
    marker_cm = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype="<f8")
    predictions: list[Path] = []
    receipts: list[Path] = []
    for fold in range(3):
        score_samples = samples[roles[fold] == "SCORE"].copy()
        for seed in seed_values:
            local_cm = marker_cm.copy()
            if marker_drift == (fold, seed):
                local_cm[-1] += 0.01
            if sample_reverse == (fold, seed):
                score_samples = score_samples[::-1].copy()
            probability = np.empty((32, len(marker), len(STATE_NAMES)), dtype="<f4")
            base = 0.1 + (seed % 17) / 100.0
            probability.fill((1.0 - base) / 5.0)
            probability[:, :, 0] = base
            path = root / f"prediction.f{fold}.s{seed}.npz"
            payload = {
                "probabilities": probability,
                "sample_key_sha256": score_samples,
                "marker_pos": marker,
                "marker_cM": local_cm,
                "marker_axis_sha256": np.asarray(["axis-v1"]),
                "fold": np.asarray([fold], dtype="|u1"),
                "family": np.asarray([family]),
                "arm": np.asarray([arm]),
                "seed": np.asarray([seed], dtype="<i8"),
            }
            if add_truth == (fold, seed):
                payload["state_labels"] = np.zeros((32, len(marker)), dtype="|u1")
            write_deterministic_npz(path, payload)
            receipt = path.with_suffix(".receipt.json")
            write_json(receipt, {
                "stage": "M38B_TRAIN_AND_PREDICT_OOF",
                "status": "PASS_SCORE_TRUTH_INACCESSIBLE",
                "fold": fold, "family": family, "arm": arm, "seed": seed,
                "score_people": 32, "score_truth_input": None,
                "model_contract_receipt_sha256": "c" * 64,
                "base_contract_sha256": "b" * 64,
                "amendment_sha256": "a" * 64,
                "amendment_2_sha256": "d" * 64,
                "output_sha256": digest(path),
            })
            predictions.append(path)
            receipts.append(receipt)
    return {
        "folds": folds, "folds_receipt": folds.with_suffix(".receipt.json"),
        "predictions": predictions, "receipts": receipts, "samples": samples,
    }


def run_collect(fixture: dict[str, object], root: Path, *, family: str = "tcn") -> dict[str, object]:
    return collect_oof(
        folds=fixture["folds"], folds_receipt=fixture["folds_receipt"],
        expected_folds_sha256=digest(fixture["folds"]),
        predictions=fixture["predictions"],
        prediction_receipts=fixture["receipts"], family=family, arm="RE",
        output=root / "oof.npz", receipt_output=root / "oof.receipt.json",
    )


class M38BOofCollectorTests(unittest.TestCase):
    def test_tcn_averages_all_three_seeds_and_restores_original_axis(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_oof_fixture(root)
            receipt = run_collect(fixture, root)
            self.assertEqual(receipt["person_coverage_min"], 1)
            self.assertEqual(receipt["person_coverage_max"], 1)
            self.assertEqual(receipt["seeds"], list(TCN_SEEDS))
            with np.load(root / "oof.npz", allow_pickle=False) as observed:
                np.testing.assert_array_equal(observed["sample_key_sha256"], fixture["samples"])
                np.testing.assert_array_equal(observed["state_names"], STATE_NAMES)
                self.assertEqual(observed["probabilities"].shape, (96, 5, 6))
                expected = np.mean([0.1 + (seed % 17) / 100.0 for seed in TCN_SEEDS])
                np.testing.assert_allclose(observed["probabilities"][:, :, 0], expected)
                self.assertEqual(np.unique(observed["fold_ids"], return_counts=True)[1].tolist(),
                                 [32, 32, 32])

    def test_missing_seed_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_oof_fixture(root)
            fixture["predictions"] = fixture["predictions"][:-1]
            fixture["receipts"] = fixture["receipts"][:-1]
            with self.assertRaisesRegex(M38BOofCollectionError, "count"):
                run_collect(fixture, root)

    def test_duplicate_fold_seed_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_oof_fixture(root)
            fixture["predictions"][-1] = fixture["predictions"][0]
            fixture["receipts"][-1] = fixture["receipts"][0]
            with self.assertRaisesRegex(M38BOofCollectionError, "duplicate fold/seed"):
                run_collect(fixture, root)

    def test_unexpected_seed_set_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_oof_fixture(root, seed_values=(1103, 2207, 3302))
            with self.assertRaisesRegex(M38BOofCollectionError, "incomplete or unexpected"):
                run_collect(fixture, root)

    def test_marker_axis_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_oof_fixture(root, marker_drift=(2, 3301))
            with self.assertRaisesRegex(M38BOofCollectionError, "marker axis"):
                run_collect(fixture, root)

    def test_score_sample_axis_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_oof_fixture(root, sample_reverse=(1, 2207))
            with self.assertRaisesRegex(M38BOofCollectionError, "sample axis"):
                run_collect(fixture, root)

    def test_truth_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_oof_fixture(root, add_truth=(0, 1103))
            with self.assertRaisesRegex(M38BOofCollectionError, "truth is forbidden"):
                run_collect(fixture, root)

    def test_analytic_requires_one_preregistered_seed_per_fold(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_oof_fixture(root, family="analytic", seed_values=(1103,))
            receipt = run_collect(fixture, root, family="analytic")
            self.assertEqual(receipt["probability_aggregation"], "SINGLE_PREREGISTERED_SEED")

    def test_rd_off_cannot_be_collected_from_fitted_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_oof_fixture(root, arm="RD")
            with self.assertRaisesRegex(M38BOofCollectionError, "packed from exact"):
                collect_oof(
                    folds=fixture["folds"], folds_receipt=fixture["folds_receipt"],
                    expected_folds_sha256=digest(fixture["folds"]),
                    predictions=fixture["predictions"],
                    prediction_receipts=fixture["receipts"], family="tcn", arm="RD",
                    output=root / "rd.oof.npz", receipt_output=root / "rd.oof.receipt.json",
                )


if __name__ == "__main__":
    unittest.main()
