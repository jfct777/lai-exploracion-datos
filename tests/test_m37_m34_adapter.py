from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m37_m34_adapter as subject
from m37_trace_core import m34_labels_to_states


def test_m34_truth_is_marginalized_to_unordered_diploid_states() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        truth, f0, output = root / "truth.npz", root / "f0.npz", root / "trace.npz"
        samples = np.asarray([b"a", b"b"])
        marker = np.asarray([10, 20])
        labels = np.asarray([[[0, 1], [1, 2]], [[2, 0], [2, 1]]], dtype=np.int8)
        np.savez(truth, sample_key_sha256=samples, marker_pos=marker, labels=labels)
        np.savez(f0, sample_key_sha256=samples, marker_pos=marker, F0=np.full((2, 2, 2, 3), 1 / 3, dtype=np.float32))
        receipt = subject.adapt(truth, f0, output)
        with np.load(output, allow_pickle=False) as observed:
            np.testing.assert_array_equal(observed["state_labels"], np.asarray([[1, 4], [5, 1]], dtype=np.uint8))
        assert receipt["rare_phase_assignment"] == "forbidden"


def test_adapter_rejects_truth_f0_axis_drift() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        truth, f0 = root / "truth.npz", root / "f0.npz"
        np.savez(truth, sample_key_sha256=np.asarray([b"a"]), marker_pos=np.asarray([10]), labels=np.zeros((1, 2, 1), dtype=np.int8))
        np.savez(f0, sample_key_sha256=np.asarray([b"wrong"]), marker_pos=np.asarray([10]), F0=np.full((1, 2, 1, 3), 1 / 3, dtype=np.float32))
        try:
            subject.adapt(truth, f0, root / "out.npz")
        except ValueError as error:
            assert "axes" in str(error)
        else:
            raise AssertionError("axis drift must fail closed")


def test_m34_conversion_is_symmetric_and_requires_two_haplotype_axes() -> None:
    labels = np.asarray([[[0, 1, 2], [1, 0, 2]]], dtype=np.uint8)
    np.testing.assert_array_equal(m34_labels_to_states(labels), np.asarray([[1, 1, 5]], dtype=np.uint8))
    try:
        m34_labels_to_states(np.zeros((1, 3, 2), dtype=np.uint8))
    except ValueError as error:
        assert "[N,2,M]" in str(error)
    else:
        raise AssertionError("invalid M34 axes must fail closed")
