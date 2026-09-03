from __future__ import annotations

import sys
import tempfile
import hashlib
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

from m37_trace_core import (MISSING_GENOTYPE, baseline_to_states,
                            reference_state_log_likelihood, deposit_evidence,
                            hmm_posterior, hmm_posterior_reference, transition_matrix)
from m37_trace_score import ece, match_count, score
from m37_trace_materialize import materialize
from m37_trace_successive_halving import promote
from m37_bind_marker_axis import bind_marker_axis


def test_phase_free_likelihood_and_controls_keep_missing_unassigned() -> None:
    genotype = np.array([[1, 0, MISSING_GENOTYPE], [2, 1, 0]], dtype=np.uint8)
    ac = np.array([[[1, 0, 1], [0, 2, 1], [0, 1, 0]],
                   [[0, 1, 1], [1, 1, 0], [0, 0, 1]]], dtype=np.int16)
    an = np.full_like(ac, 8)
    likelihood, pooled, uncertainty, support = reference_state_log_likelihood(genotype, ac, an)
    assert likelihood.shape == (2, 3, 6)
    assert np.all(likelihood[0, 2] == 0.0)
    assert np.all(uncertainty >= 0) and np.all((support >= 0) & (support <= 1))
    field_re, count_re, events = deposit_evidence(likelihood, pooled, genotype, np.array([.1, .2, .3]), np.array([.1, .2, .3]), "RE")
    field_rd, count_rd, _ = deposit_evidence(likelihood, pooled, genotype, np.array([.1, .2, .3]), np.array([.1, .2, .3]), "RD")
    field_pooled, _, _ = deposit_evidence(likelihood, pooled, genotype, np.array([.1, .2, .3]), np.array([.1, .2, .3]), "POOLED")
    assert np.any(field_re != 0) and np.all(field_rd == 0) and np.array_equal(count_re, count_rd)
    assert np.allclose(field_pooled[:, :, 0], field_pooled[:, :, 5])
    assert {tuple(row) for row in events.tolist()} >= {(0, 0), (0, 2), (1, 0), (1, 1)}


def test_hmm_and_phase_free_metrics_are_simplex_and_distance_aware() -> None:
    haploid = np.full((1, 2, 4, 3), 1 / 3, dtype=np.float32)
    baseline = baseline_to_states(haploid)
    evidence = np.zeros_like(baseline)
    result = hmm_posterior(baseline, evidence, np.array([0.0, .05, .1, .2]))
    reference = hmm_posterior_reference(baseline, evidence, np.array([0.0, .05, .1, .2]))
    assert result.shape == (1, 4, 6)
    assert np.allclose(result.sum(axis=2), 1.0) and np.allclose(result, reference, atol=1e-6, rtol=0)
    metrics = score(result, np.array([[0, 0, 1, 1]]), np.array([0.0, .05, .1, .2]))
    assert set(metrics["f1_boundary"]) == {"0.05", "0.1", "0.2", "0.5"}
    assert metrics["log_loss"] > 0 and metrics["false_transitions_per_morgan"] >= 0
    transition = transition_matrix(.1, 12.0)
    assert transition[0, 1] > transition[0, 3]  # AA->AE needs one ancestry change; AA->EE needs two.


def test_event_tcn_splats_by_cm_and_preserves_f0_without_events() -> None:
    import torch
    from m37_trace_core import TraceSpec, build_tcn
    model = build_tcn(TraceSpec(hidden_dim=32, depth=2, kernel_size=3, dropout=0.0, dilations=(1, 2)), 20).eval()
    baseline = torch.softmax(torch.randn(2, 5, 6), dim=-1)
    # Include a structural zero: RD must not gain merely because the model
    # internally floors and renormalizes the FLARE posterior.
    baseline[0, 0, 5] = 0.0
    baseline[0, 0] /= baseline[0, 0].sum()
    empty = (torch.zeros((0, 20)), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
             torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros(0))
    with torch.inference_mode():
        output = model(*empty, baseline)
    assert torch.equal(output, baseline)
    assert torch.equal(output.sum(dim=-1), baseline.sum(dim=-1))


def test_tcn_shards_match_full_sequence_with_receptive_halo() -> None:
    import torch
    from m37_trace_core import TraceSpec, build_tcn
    from m37_trace_train import _predict_batched, receptive_halo
    torch.manual_seed(9)
    baseline = torch.softmax(torch.randn(2, 13, 6), dim=-1).numpy().astype(np.float32)
    features = {"baseline_states": baseline, "event_values": np.ones((3, 20), dtype=np.float32),
                "event_context_7mer": np.asarray([1, 2, 3], dtype=np.uint16),
                "event_sample": np.asarray([0, 0, 1], dtype=np.uint32),
                "event_cM": np.asarray([.025, .027, .095], dtype=np.float64),
                "marker_cM": np.arange(13, dtype=np.float64) / 100,
                # event 2/3 crosses the central shard boundary and its weights are deliberately asymmetric
                "event_marker_left": np.asarray([2, 2, 9], dtype=np.uint32),
                "event_marker_right": np.asarray([3, 3, 10], dtype=np.uint32),
                "event_delta_left_cM": np.asarray([.01, .071, .01], dtype=np.float32),
                "event_delta_right_cM": np.asarray([.04, .013, .01], dtype=np.float32),
                "schedule_sample": np.asarray([0, 0, 1], dtype=np.uint32),
                "schedule_marker": np.asarray([2, 2, 9], dtype=np.uint32)}
    spec = TraceSpec(hidden_dim=32, depth=2, kernel_size=3, dropout=0.0, dilations=(1, 2))
    model = build_tcn(spec, 20).eval()
    full = _predict_batched(model, features, 2, 99, 1.0, .04, halo=receptive_halo(spec))
    sharded = _predict_batched(model, features, 2, 3, 1.0, .04, halo=receptive_halo(spec))
    # CPU convolution accumulation order differs by at most low-µfloat noise;
    # the bound is intentionally far below a posterior-calibration tolerance.
    assert np.allclose(full, sharded, atol=5e-6, rtol=0)


def test_event_splat_is_continuous_inside_cm_radius_and_zero_outside() -> None:
    from m37_trace_train import _event_batch
    features = {
        "baseline_states": np.full((1, 5, 6), 1 / 6, dtype=np.float32),
        "marker_cM": np.asarray([0.0, .01, .03, .05, .07]),
        "event_values": np.ones((1, 20), dtype=np.float32),
        "event_context_7mer": np.asarray([1], dtype=np.uint16),
        "event_sample": np.asarray([0], dtype=np.uint32),
        "event_cM": np.asarray([.03]),
    }
    _, _, _, splat_event, splat_marker, splat_weight = _event_batch(
        features, np.asarray([0]), 0, 5, event_radius_cm=.04,
    )
    assert np.array_equal(splat_event.numpy(), np.zeros(4, dtype=np.int64))
    assert np.array_equal(splat_marker.numpy(), np.asarray([0, 1, 2, 3]))
    assert np.allclose(splat_weight.numpy(), np.asarray([.25, .5, 1., .5]), atol=1e-7, rtol=0)


def test_positive_event_gate_can_change_f0() -> None:
    import torch
    from m37_trace_core import TraceSpec, build_tcn
    model = build_tcn(TraceSpec(hidden_dim=32, depth=2, kernel_size=3, dropout=0.0, dilations=(1, 2)), 20).eval()
    baseline = torch.full((1, 4, 6), 1 / 6)
    with torch.no_grad():
        model.head.bias[0] = 2.0
        model.confidence.bias.fill_(20.0)
    one_event = (torch.ones((1, 20)), torch.ones(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
                 torch.zeros(3, dtype=torch.long), torch.tensor([0, 1, 2]), torch.tensor([.5, 1., .5]))
    with torch.inference_mode():
        output = model(*one_event, baseline)
    assert not torch.allclose(output[:, :3], baseline[:, :3])
    assert torch.allclose(output[:, 3], baseline[:, 3])


def test_supported_tcn_residual_is_simplex_and_can_revive_flare_zero() -> None:
    import torch
    from m37_trace_core import TraceSpec, build_tcn
    model = build_tcn(
        TraceSpec(hidden_dim=32, depth=2, kernel_size=3, dropout=0.0,
                  dilations=(1, 2)),
        20,
    ).eval()
    baseline = torch.full((1, 3, 6), 0.2)
    baseline[:, :, 5] = 0.0
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[5] = 40.0
        model.confidence.weight.zero_()
        model.confidence.bias.fill_(20.0)
    one_event = (
        torch.ones((1, 20)), torch.ones(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
        torch.tensor([1]), torch.ones(1),
    )
    with torch.inference_mode():
        output = model(*one_event, baseline)
    assert torch.allclose(output.sum(dim=-1), torch.ones((1, 3)), atol=1e-6, rtol=0)
    assert output[0, 1, 5] > 0.5
    assert torch.equal(output[:, 0], baseline[:, 0])
    assert torch.equal(output[:, 2], baseline[:, 2])


def test_structural_zero_keeps_a_finite_training_gradient_while_rd_is_exact() -> None:
    import torch
    from m37_trace_core import PROBABILITY_FLOOR, TraceSpec, build_tcn
    from m37_trace_train import probability_nll
    model = build_tcn(
        TraceSpec(hidden_dim=32, depth=2, kernel_size=3, dropout=0.0,
                  dilations=(1, 2)),
        20,
    ).train()
    baseline = torch.full((1, 3, 6), 0.2)
    baseline[:, :, 5] = 0.0
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.confidence.weight.zero_()
        model.confidence.bias.zero_()
    event = (
        torch.ones((1, 20)), torch.ones(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
        torch.tensor([1]), torch.ones(1),
    )
    prediction = model(*event, baseline)
    loss = probability_nll(prediction[:, 1:2], torch.tensor([[5]]))
    loss.backward()
    gradient = model.head.bias.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert abs(float(gradient[5])) > 1e-3
    assert PROBABILITY_FLOOR == 1e-12


def test_rd_removes_every_ragged_event_channel_while_geometry_keeps_only_locations() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        selected, target, reference, f0, marker_source, marker = [root / f"{name}.npz" for name in ("selected", "target", "reference", "f0", "marker_source", "marker")]
        np.savez(selected, locus_id=np.asarray([11, 12]), cM=np.asarray([.1, .2]),
                 context_7mer=np.asarray([5, 6]), carrier_support=np.asarray([.4, .8]), origin_support=np.asarray([.1, .3]))
        np.savez(target, sample_key_sha256=np.asarray([b"a", b"b"]), locus_id=np.asarray([11, 12]), minor_dosage=np.asarray([[1, 0], [2, 0]], dtype=np.int8),
                 observed_mask=np.ones((2, 2), dtype=np.uint8))
        np.savez(reference, ancestry=np.asarray(["AFR", "EUR", "NAM"]), locus_id=np.asarray([11, 12]),
                 minor_ac=np.ones((3, 2), dtype=np.int16), callable_an=np.full((3, 2), 8, dtype=np.int16))
        np.savez(f0, sample_key_sha256=np.asarray([b"a", b"b"]), marker_pos=np.asarray([100, 200]),
                 F0=np.full((2, 2, 2, 3), 1 / 3, dtype=np.float32))
        np.savez(marker_source, marker_pos=np.asarray([100, 200]), marker_cM=np.asarray([.1, .2]))
        bind_marker_axis(f0, marker_source, marker)
        marker_receipt = marker.with_suffix(".receipt.json")
        rd = materialize(selected, target, reference, f0, marker, marker_receipt, "RD", .5)
        geometry = materialize(selected, target, reference, f0, marker, marker_receipt, "GEOMETRY", .5)
        pooled = materialize(selected, target, reference, f0, marker, marker_receipt, "POOLED", .5)
    assert len(rd["event_sample"]) == 0 and len(rd["event_values"]) == 0 and len(rd["event_context_7mer"]) == 0
    assert len(geometry["event_sample"]) > 0 and np.all(geometry["event_values"] == 0) and np.all(geometry["event_context_7mer"] == 0)
    assert np.all(pooled["event_uncertainty"] == 0) and np.all(pooled["event_support"] == 0) and np.all(pooled["event_values"][:, 10:16] == 0)


def test_marker_cm_without_its_physical_axis_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        f0, marker, output = root / "f0.npz", root / "marker.npz", root / "bound.npz"
        np.savez(f0, marker_pos=np.asarray([100, 200]),
                 F0=np.full((1, 2, 2, 3), 1 / 3, dtype=np.float32))
        np.savez(marker, marker_cM=np.asarray([.1, .2]))
        try:
            bind_marker_axis(f0, marker, output)
        except ValueError as error:
            assert "joint F0/marker source receipt" in str(error)
        else:
            raise AssertionError("an order-only marker cM vector must not be authenticated")
        receipt = root / "m34_f0.receipt.json"
        receipt.write_text(json.dumps({
            "stage": "M34_PARSE_FLARE_F0", "marker_count": 2,
            "outputs": {
                f0.name: {"sha256": hashlib.sha256(f0.read_bytes()).hexdigest()},
                marker.name: {"sha256": hashlib.sha256(marker.read_bytes()).hexdigest()},
            },
        }), encoding="utf-8")
        observed = bind_marker_axis(f0, marker, output, receipt)
        assert observed["binding_evidence"] == "JOINT_UPSTREAM_F0_MARKER_RECEIPT"


def test_ece_includes_confidence_one_and_boundary_assignment_is_optimal() -> None:
    probability = np.zeros((1, 6), dtype=float)
    probability[0, 0] = 1.0
    assert ece(probability, np.asarray([1])) == 1.0
    # Greedy nearest maps truth 0 to 0.05 and leaves 0.18 unmatched; optimal
    # 1-to-1 pairing maps it to -0.15 and obtains both matches.
    assert match_count([(0.0, 0, 1), (.18, 1, 2)], [(-.15, 0, 1), (.05, 1, 2)], .15) == 2
    assert match_count([(0.0, 0, 1)], [(0.0, 0, 2)], .15) == 0


def test_successive_halving_requires_re_to_beat_every_paired_comparator() -> None:
    baseline = {"f1_boundary": {"0.2": .10}, "log_loss": .40}
    def metric(f1: float, loss: float) -> dict:
        return {"f1_boundary": {"0.2": f1}, "baseline": baseline, "log_loss": loss}
    rows = [{"candidate_id": "c", "root": "R0", "arm": arm, "metrics": metric(f1, loss)}
            for arm, f1, loss in (("RE", .40, .20), ("RD", .20, .30), ("POOLED", .18, .32),
                                  ("SHAM", .19, .31), ("GEOMETRY", .17, .33))]
    ranked, criteria = promote(rows, .5)
    assert ranked[0]["rung_mode"] == "triage_or_expansion" and ranked[0]["promote"]
    assert set(ranked[0]["root_gates"]["R0"]["RE_minus_comparator_F1"]) == {
        "F0", "RD", "POOLED", "SHAM", "GEOMETRY"
    }
    assert criteria["minimum_strict_F1_gain"] == 0.0

    rows[-1]["metrics"] = metric(.41, .19)
    ranked, _ = promote(rows, .5)
    assert not ranked[0]["promote"] and ranked[0]["reason"] == "FAIL_PAIRED_GATE"


def test_calendar_uses_train_people_only_and_is_arm_invariant() -> None:
    from m37_trace_train import event_centered_schedule
    common = {"marker_cM": np.arange(11, dtype=float) / 100,
              "schedule_sample": np.asarray([0, 1, 2], dtype=np.uint32),
              "schedule_marker": np.asarray([1, 5, 9], dtype=np.uint32)}
    train = np.asarray([0, 2], dtype=np.int64)
    observed = event_centered_schedule(common, 4, 17, .01, train)
    # searchsorted follows the exact floating cM axis; the right edge of the
    # 0.09-cM event falls immediately before marker 0.10 in this fixture.
    allowed = {(0, 3), (8, 10)}
    assert set(observed).issubset(allowed)
    rd = {**common, "event_sample": np.empty(0, dtype=np.uint32)}
    assert event_centered_schedule(rd, 4, 17, .01, train) == observed


def test_fit_valid_people_must_be_identical_or_disjoint() -> None:
    from m37_trace_train import authenticate_feature_pair
    axis = {"marker_pos": np.asarray([10, 20]), "marker_cM": np.asarray([.1, .2]),
            "marker_axis_sha256": np.asarray(["axis"])}
    fit = {**axis, "sample_key_sha256": np.asarray([b"a", b"b"])}
    assert authenticate_feature_pair(fit, fit) == "FIT_TUNE"
    valid = {**axis, "sample_key_sha256": np.asarray([b"c", b"d"])}
    assert authenticate_feature_pair(fit, valid) == "SEALED_VALID"
    overlap = {**axis, "sample_key_sha256": np.asarray([b"b", b"c"])}
    try:
        authenticate_feature_pair(fit, overlap)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("partial FIT/VALID overlap must fail closed")
