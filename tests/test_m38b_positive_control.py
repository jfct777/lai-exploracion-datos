#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m33_safe_bridge_core import write_deterministic_npz  # noqa: E402
from m38b_make_folds import build as build_folds  # noqa: E402
from m38b_partition_fold import partition_features  # noqa: E402
from m38b_positive_control import (  # noqa: E402
    DELTA_GRID,
    EVENT_IDENTITY_MEMBERS,
    EVENT_MASK_MEMBERS,
    M38BPositiveControlError,
    _array_bundle_sha256,
    build_positive_control,
)
from m38b_train_fold import verify_partition_receipt  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialization_receipt(path: Path, arm: str) -> Path:
    receipt = path.with_suffix(".receipt.json")
    write_json(receipt, {
        "schema_version": "1.0.0",
        "stage": "M37_TRACE_MATERIALIZE",
        "arm": arm,
        "output_sha256": digest(path),
    })
    return receipt


def make_fixture(root: Path) -> dict[str, Path | np.ndarray]:
    people, markers = 96, 8
    samples = np.asarray([f"{index:064d}".encode() for index in range(people)], dtype="|S64")
    marker_pos = np.arange(100, 900, 100, dtype="<i8")
    marker_cm = np.arange(markers, dtype="<f8") / 10.0
    baseline = np.full((people, markers, 6), 1.0 / 6.0, dtype="<f4")
    event_sample = np.repeat(np.arange(people, dtype=np.uint32), 2)
    event_locus = np.tile(np.asarray([0, 1], dtype=np.uint32), people)
    event_marker_left = np.tile(np.asarray([1, 5], dtype=np.uint32), people)
    event_marker_right = event_marker_left.copy()
    event_cm = marker_cm[event_marker_left]
    event_count = len(event_sample)
    row_magnitude = 1.0 + event_sample.astype(np.float32) / 100.0
    event_loglik = np.zeros((event_count, 6), dtype="<f4")
    event_loglik[:, 0] = row_magnitude
    event_values = np.zeros((event_count, 23), dtype="<f4")
    event_values[:, 0] = 1.0
    event_values[:, 4:10] = event_loglik
    event_values[:, 16:18] = 1.0
    real = {
        "sample_key_sha256": samples,
        "marker_pos": marker_pos,
        "marker_cM": marker_cm,
        "marker_axis_sha256": np.asarray(["axis-test"]),
        "state_names": np.asarray(["AA", "AE", "AN", "EE", "EN", "NN"]),
        "baseline_states": baseline,
        "evidence_field": np.ones_like(baseline),
        "event_counts": np.ones((people, markers, 1), dtype="<f4"),
        "calendar_marker": np.asarray([1, 5], dtype=np.uint32),
        "schedule_sample": event_sample.copy(),
        "schedule_marker": event_marker_left.copy(),
        "event_sample": event_sample,
        "event_locus": event_locus,
        "event_genotype": np.ones(event_count, dtype=np.uint8),
        "event_target_callable": np.ones(event_count, dtype=np.uint8),
        "event_reference_callable": np.ones(event_count, dtype=np.uint8),
        "event_loglik": event_loglik,
        "event_pooled_loglik": np.zeros_like(event_loglik),
        "event_uncertainty": np.zeros((event_count, 3), dtype="<f4"),
        "event_support": np.ones((event_count, 3), dtype="<f4"),
        "event_context_7mer": np.zeros(event_count, dtype=np.uint16),
        "event_carrier_support": np.ones(event_count, dtype="<f4"),
        "event_origin_support": np.zeros(event_count, dtype="<f4"),
        "event_values": event_values,
        "event_cM": event_cm.astype("<f8"),
        "event_marker_left": event_marker_left,
        "event_marker_right": event_marker_right,
        "event_delta_left_cM": np.zeros(event_count, dtype="<f4"),
        "event_delta_right_cM": np.zeros(event_count, dtype="<f4"),
        "ancestry_names": np.asarray(["AFR", "EUR", "NAM"]),
    }
    rd = {name: np.ascontiguousarray(value).copy() for name, value in real.items()}
    rd["evidence_field"].fill(0)
    for name in tuple(rd):
        if name.startswith("event_"):
            rd[name] = rd[name][:0].copy()
    real_path, rd_path = root / "real.npz", root / "rd.npz"
    write_deterministic_npz(real_path, real)
    write_deterministic_npz(rd_path, rd)
    truth_path = root / "truth.npz"
    state_labels = (
        np.arange(people, dtype=np.uint8)[:, None]
        + np.arange(markers, dtype=np.uint8)[None, :]
    ) % 6
    write_deterministic_npz(truth_path, {
        "sample_key_sha256": samples,
        "marker_pos": marker_pos,
        "state_labels": state_labels,
    })
    truth_receipt = root / "truth.receipt.json"
    write_json(truth_receipt, {
        "stage": "M38B_ALIGN_FULL_AND_TRUTH_TO_F_MINUS_S660",
        "outputs": {truth_path.name: {"sha256": digest(truth_path)}},
    })
    fold_source = root / "fold-source.npz"
    write_deterministic_npz(fold_source, {"sample_key_sha256": samples})
    folds = root / "folds.npz"
    build_folds(fold_source, folds, outer_seed=7, inner_seed_start=100)
    return {
        "real": real_path,
        "real_receipt": materialization_receipt(real_path, "RE"),
        "rd": rd_path,
        "rd_receipt": materialization_receipt(rd_path, "RD"),
        "truth": truth_path,
        "truth_receipt": truth_receipt,
        "folds": folds,
        "folds_receipt": folds.with_suffix(".receipt.json"),
        "state_labels": state_labels,
    }


def run_control(
    fixture: dict[str, Path | np.ndarray], root: Path, delta: float, *, fold: int = 0,
) -> tuple[dict[str, object], Path]:
    output = root / f"positive.f{fold}.d{delta:g}.npz"
    receipt = root / f"positive.f{fold}.d{delta:g}.receipt.json"
    document = build_positive_control(
        real_features=fixture["real"], real_receipt=fixture["real_receipt"],
        rd_features=fixture["rd"], rd_receipt=fixture["rd_receipt"],
        truth=fixture["truth"], truth_receipt=fixture["truth_receipt"],
        folds=fixture["folds"], folds_receipt=fixture["folds_receipt"],
        fold=fold, delta=delta, output=output, receipt_output=receipt,
    )
    return document, output


class M38BPositiveControlTests(unittest.TestCase):
    def test_delta_zero_keeps_matched_events_but_has_no_model_signal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_fixture(root)
            document, output = run_control(fixture, root, 0.0)
            self.assertNotEqual(digest(output), digest(fixture["rd"]))
            self.assertTrue(document["invariants"]["delta_zero_has_matched_events_and_zero_model_signal"])
            self.assertEqual(
                document["invariants"]["positive_tcn_execution_at_every_delta"],
                "SAME_ARCHITECTURE_LOSS_OPTIMIZER_SELECTION_FOLDS_AND_SEEDS",
            )
            self.assertEqual(document["invariants"]["positive_primary_contrast"],
                             "POS_DELTA_MINUS_POS_ZERO")
            with np.load(fixture["real"], allow_pickle=False) as real, \
                    np.load(output, allow_pickle=False) as observed:
                self.assertEqual(len(observed["event_sample"]), len(real["event_sample"]))
                for name in EVENT_IDENTITY_MEMBERS + EVENT_MASK_MEMBERS:
                    np.testing.assert_array_equal(observed[name], real[name])
                self.assertEqual(np.count_nonzero(observed["evidence_field"]), 0)
                self.assertEqual(np.count_nonzero(observed["event_values"]), 0)
                self.assertEqual(np.count_nonzero(observed["event_context_7mer"]), 0)

    def test_positive_delta_preserves_axes_events_masks_and_aligns_truth(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_fixture(root)
            document, output = run_control(fixture, root, 1.0)
            with np.load(fixture["real"], allow_pickle=False) as real, \
                    np.load(output, allow_pickle=False) as observed:
                for name in EVENT_IDENTITY_MEMBERS + EVENT_MASK_MEMBERS:
                    np.testing.assert_array_equal(observed[name], real[name])
                self.assertEqual(
                    _array_bundle_sha256(observed, EVENT_IDENTITY_MEMBERS),
                    document["real_event_identity_sha256"],
                )
                anchors = observed["event_marker_left"]
                states = fixture["state_labels"][observed["event_sample"], anchors]
                self.assertTrue(np.all(np.argmax(observed["event_loglik"], axis=1) == states))
                self.assertGreater(np.count_nonzero(observed["evidence_field"]), 0)
                # The only model-visible event content is the six-dimensional
                # synthetic state evidence.  All real biological channels are
                # absent, including genotype and callability columns.
                self.assertEqual(np.count_nonzero(observed["event_values"][:, :4]), 0)
                self.assertEqual(np.count_nonzero(observed["event_values"][:, 10:]), 0)
                self.assertEqual(np.count_nonzero(observed["event_context_7mer"]), 0)
                self.assertEqual(np.count_nonzero(observed["event_genotype"]), 0)
                self.assertEqual(np.count_nonzero(observed["event_support"]), 0)
                self.assertEqual(np.count_nonzero(observed["event_uncertainty"]), 0)

    def test_delta_cells_differ_only_in_the_synthetic_six_state_signal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_fixture(root)
            zero_dir, half_dir, one_dir = root / "zero", root / "half", root / "one"
            zero_dir.mkdir(); half_dir.mkdir(); one_dir.mkdir()
            _zero_document, zero_path = run_control(fixture, zero_dir, 0.0)
            _half_document, half_path = run_control(fixture, half_dir, 0.5)
            _one_document, one_path = run_control(fixture, one_dir, 1.0)
            signal_members = {"event_loglik", "event_values", "evidence_field"}
            with np.load(zero_path, allow_pickle=False) as zero, \
                    np.load(half_path, allow_pickle=False) as half, \
                    np.load(one_path, allow_pickle=False) as one:
                self.assertEqual(set(zero.files), set(half.files))
                self.assertEqual(set(zero.files), set(one.files))
                for name in set(zero.files) - signal_members:
                    np.testing.assert_array_equal(zero[name], half[name], err_msg=name)
                    np.testing.assert_array_equal(zero[name], one[name], err_msg=name)
                np.testing.assert_allclose(
                    half["event_loglik"], one["event_loglik"] * 0.5,
                    rtol=0, atol=1e-7,
                )
                np.testing.assert_allclose(
                    half["evidence_field"], one["evidence_field"] * 0.5,
                    rtol=0, atol=1e-7,
                )
                np.testing.assert_allclose(
                    half["event_values"][:, 4:10],
                    one["event_values"][:, 4:10] * 0.5,
                    rtol=0, atol=1e-7,
                )

    def test_robust_scale_uses_train_events_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first_root, second_root = root / "first", root / "second"
            first_root.mkdir(); second_root.mkdir()
            first = make_fixture(first_root)
            second = make_fixture(second_root)
            with np.load(second["folds"], allow_pickle=False) as folds:
                nontrain = np.flatnonzero(folds["roles"][0] != "TRAIN")
            with np.load(second["real"], allow_pickle=False) as archive:
                changed = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
            selected = np.isin(changed["event_sample"], nontrain)
            changed["event_loglik"][selected, 5] = 10_000.0
            changed["event_values"][selected, 4:10] = changed["event_loglik"][selected]
            second["real"].unlink()
            write_deterministic_npz(second["real"], changed)
            second["real_receipt"].unlink()
            second["real_receipt"] = materialization_receipt(second["real"], "RE")
            first_document, first_output = run_control(first, first_root, 0.5)
            second_document, second_output = run_control(second, second_root, 0.5)
            self.assertEqual(
                first_document["scale"]["robust_magnitude"],
                second_document["scale"]["robust_magnitude"],
            )
            # Non-TRAIN likelihoods are overwritten by the same injected
            # signal, so the resulting diagnostic features are also exact.
            self.assertEqual(digest(first_output), digest(second_output))

    def test_truth_axis_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_fixture(root)
            with np.load(fixture["truth"], allow_pickle=False) as archive:
                payload = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
            payload["marker_pos"][-1] += 1
            fixture["truth"].unlink()
            write_deterministic_npz(fixture["truth"], payload)
            fixture["truth_receipt"].unlink()
            write_json(fixture["truth_receipt"], {
                "stage": "M38B_ALIGN_FULL_AND_TRUTH_TO_F_MINUS_S660",
                "outputs": {fixture["truth"].name: {"sha256": digest(fixture["truth"])}},
            })
            with self.assertRaisesRegex(M38BPositiveControlError, "truth/features axes"):
                run_control(fixture, root, 1.0)

    def test_exact_delta_grid_and_diagnostic_namespace_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_fixture(root)
            source_hashes = digest(fixture["real"]), digest(fixture["rd"])
            for index, delta in enumerate(DELTA_GRID):
                target = root / f"grid-{index}"
                target.mkdir()
                document, _output = run_control(fixture, target, delta)
                self.assertEqual(document["namespace"], "DIAGNOSTIC_ONLY_NEVER_REAL_CANDIDATE")
                self.assertFalse(document["invariants"]["production_inputs_modified"])
                self.assertFalse(document["truth_use"]["score_truth_used_to_select_real_candidate_or_checkpoint"])
            self.assertEqual(source_hashes, (digest(fixture["real"]), digest(fixture["rd"])))
            with self.assertRaisesRegex(M38BPositiveControlError, "exact preregistered grid"):
                run_control(fixture, root, 0.1)

    def test_tampered_event_mask_receipt_or_output_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_fixture(root)
            write_json(fixture["real_receipt"], {
                "stage": "M37_TRACE_MATERIALIZE", "arm": "RE",
                "output_sha256": "0" * 64,
            })
            with self.assertRaisesRegex(M38BPositiveControlError, "receipt hash"):
                run_control(fixture, root, 1.0)

    def test_tcn_off_has_exact_fminus_output_and_zero_parameter_gradient(self) -> None:
        import torch

        from m37_trace_core import TraceSpec, build_tcn
        from m37_trace_train import probability_nll

        torch.manual_seed(11)
        model = build_tcn(
            TraceSpec(hidden_dim=32, depth=2, kernel_size=3, dropout=0.0,
                      dilations=(1, 2)),
            23,
        ).train()
        baseline = torch.softmax(torch.randn(2, 5, 6), dim=-1)
        empty = (
            torch.zeros((0, 23)), torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long), torch.zeros(0),
        )
        prediction = model(*empty, baseline)
        self.assertTrue(torch.equal(prediction, baseline))
        labels = torch.zeros((2, 5), dtype=torch.long)
        probability_nll(prediction, labels).backward()
        nonzero_gradient = 0
        for parameter in model.parameters():
            if parameter.grad is not None:
                nonzero_gradient += int(torch.count_nonzero(parameter.grad).item())
        self.assertEqual(nonzero_gradient, 0)

    def test_weighted_tcn_loss_uses_cm_geometry(self) -> None:
        import torch

        from m37_trace_train import probability_nll

        prediction = torch.tensor([[[.9, .02, .02, .02, .02, .02],
                                    [.9, .02, .02, .02, .02, .02],
                                    [.1, .18, .18, .18, .18, .18]]], dtype=torch.float32)
        labels = torch.zeros((1, 3), dtype=torch.long)
        uniform = probability_nll(prediction, labels)
        weighted = probability_nll(prediction, labels, torch.tensor([.05, .50, .45]))
        self.assertGreater(float(weighted), float(uniform))

    def test_sparse_scheduler_visits_train_event_rows_and_anchors_the_carrier(self) -> None:
        from m37_trace_train import _event_batch, event_centered_schedule_with_carrier

        with tempfile.TemporaryDirectory() as raw:
            fixture = make_fixture(Path(raw))
            with np.load(fixture["real"], allow_pickle=False) as archive:
                features = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
            # Match the observed sparsity pattern qualitatively: 24/96 people
            # have no carrier event, including 12 people in this TRAIN fixture.
            event_count = len(features["event_sample"])
            keep = features["event_sample"] % 4 != 0
            for name in tuple(features):
                if np.asarray(features[name]).ndim > 0 and len(features[name]) == event_count:
                    features[name] = np.ascontiguousarray(features[name][keep])
            train_people = np.arange(48, dtype=np.int64)
            first = event_centered_schedule_with_carrier(
                features, 100, 71, 0.2, train_people,
            )
            second = event_centered_schedule_with_carrier(
                features, 100, 71, 0.2, train_people,
            )
            self.assertEqual(first, second)
            eligible_rows = set(np.flatnonzero(np.isin(features["event_sample"], train_people)))
            self.assertEqual(len(set(train_people) - set(features["event_sample"])), 12)
            self.assertEqual({row[3] for row in first[:len(eligible_rows)]}, eligible_rows)
            for update, (left, right, anchor, event_row) in enumerate(first):
                self.assertIn(anchor, train_people)
                self.assertEqual(anchor, int(features["event_sample"][event_row]))
                batch = train_people[
                    np.arange(update * 8, update * 8 + 8) % len(train_people)
                ].copy()
                if anchor not in batch:
                    batch[0] = anchor
                self.assertEqual(len(np.unique(batch)), 8)
                self.assertTrue(set(batch).issubset(set(train_people)))
                self.assertIn(anchor, batch)
                event = _event_batch(features, batch, left, right, 2.0, 0.2)
                self.assertGreater(event[5].numel(), 0)

    def test_diagnostic_namespace_survives_partition_and_cannot_enter_re(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_fixture(root)
            _document, positive = run_control(fixture, root, 1.0)
            args = Namespace(
                source=positive,
                source_receipt=positive.with_suffix(".receipt.json"),
                arm="POSITIVE",
                folds=fixture["folds"], fold=0,
                fit_output=root / "positive.fit.npz",
                score_output=root / "positive.score.npz",
            )
            partition = partition_features(args)
            partition_receipt = root / "positive.partition.receipt.json"
            write_json(partition_receipt, partition)
            observed = verify_partition_receipt(
                partition_receipt, args.fit_output, args.score_output, 0, "POSITIVE",
            )
            self.assertEqual(observed["arm"], "POSITIVE")
            self.assertTrue(observed["diagnostic_only"])
            with self.assertRaisesRegex(Exception, "feature partition receipt differs"):
                verify_partition_receipt(
                    partition_receipt, args.fit_output, args.score_output, 0, "RE",
                )
            forged = Namespace(**vars(args)); forged.arm = "RE"
            forged.fit_output = root / "forged.fit.npz"
            forged.score_output = root / "forged.score.npz"
            with self.assertRaisesRegex(Exception, "receipt, arm, or hash"):
                partition_features(forged)


if __name__ == "__main__":
    unittest.main()
