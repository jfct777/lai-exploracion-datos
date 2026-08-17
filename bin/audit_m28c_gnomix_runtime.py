#!/usr/bin/env python3
"""Known-answer audit of the exact Gnomix logistic-regression runtime path."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np

from src.Base.models import LogisticRegressionBase
from src.laidataset import BREAKPOINT_PROBABILITY_TOLERANCE, normalize_breakpoint_probability


def array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def array_sequence_sha256(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(array_sha256(value).encode("ascii"))
    return digest.hexdigest()


def known_answer_data() -> tuple[np.ndarray, np.ndarray]:
    patterns = np.asarray(
        [
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.int8,
    )
    labels = np.tile(np.arange(3, dtype=np.int8), 12)
    one_window = patterns[labels]
    # Gnomix deliberately uses a non-divisible marker/window pair. Mirror that
    # production path here: 13 markers, M=4, W=3, remainder=1.
    features = np.concatenate([one_window, one_window, one_window, one_window[:, :1]], axis=1)
    window_labels = np.repeat(labels[:, None], 3, axis=1)
    return features, window_labels


def fit_once(features: np.ndarray, labels: np.ndarray) -> LogisticRegressionBase:
    model = LogisticRegressionBase(
        chm_len=13,
        window_size=4,
        num_ancestry=3,
        context=0,
        n_jobs=1,
        seed=42,
        verbose=False,
    )
    model.train(features, labels, verbose=False)
    return model


def audit_breakpoint_guard() -> dict:
    valid = np.asarray([0.25, 0.25, 0.5], dtype=np.float64)
    valid_corrected = normalize_breakpoint_probability(valid)
    if not np.array_equal(valid, valid_corrected):
        raise ValueError("Breakpoint guard changed an exactly valid distribution")

    epsilon = np.finfo(np.float64).eps
    roundoff = np.asarray([0.5, -epsilon, 0.5], dtype=np.float64)
    roundoff_corrected = normalize_breakpoint_probability(roundoff)
    if np.any(roundoff_corrected < 0.0) or abs(float(roundoff_corrected.sum()) - 1.0) > 1e-15:
        raise ValueError("Breakpoint guard did not normalize floating-point roundoff")

    material_rejected = False
    try:
        normalize_breakpoint_probability(np.asarray([0.5, -1e-6, 0.5], dtype=np.float64))
    except ValueError:
        material_rejected = True
    if not material_rejected:
        raise ValueError("Breakpoint guard accepted a materially negative probability")

    patch_path = Path("/opt/gnomix/GNOMIX_PATCH_SHA256")
    if not patch_path.is_file():
        raise ValueError("Gnomix numerical patch receipt is missing")
    patch_sha256 = patch_path.read_text(encoding="ascii").strip()
    if len(patch_sha256) != 64:
        raise ValueError("Gnomix numerical patch receipt is malformed")
    return {
        "tolerance": BREAKPOINT_PROBABILITY_TOLERANCE,
        "valid_distribution_exact": True,
        "roundoff_negative_count": int((roundoff < 0.0).sum()),
        "roundoff_corrected_nonnegative": True,
        "roundoff_corrected_sum": float(roundoff_corrected.sum()),
        "material_negative_rejected": material_rejected,
        "patch_sha256": patch_sha256,
    }


def audit() -> dict:
    features, labels = known_answer_data()
    breakpoint_guard = audit_breakpoint_guard()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = fit_once(features, labels)
        second = fit_once(features, labels)
    probabilities_first = first.predict_proba(features)
    probabilities_second = second.predict_proba(features)
    predicted = probabilities_first.argmax(axis=2)
    expected_classes = [0, 1, 2]
    classes = [model.classes_.tolist() for model in first.models]
    if classes != [expected_classes] * 3:
        raise ValueError(f"Unexpected class order in Gnomix base models: {classes}")
    if probabilities_first.shape != (36, 3, 3):
        raise ValueError(f"Unexpected probability shape: {probabilities_first.shape}")
    if not np.isfinite(probabilities_first).all():
        raise ValueError("Nonfinite known-answer probabilities")
    max_sum_error = float(np.abs(probabilities_first.sum(axis=2) - 1.0).max())
    if max_sum_error > 1e-12:
        raise ValueError(f"Known-answer probabilities do not sum to one: {max_sum_error}")
    if not np.array_equal(predicted, labels):
        raise ValueError("Known-answer class predictions differ from the expected labels")
    if not np.array_equal(probabilities_first, probabilities_second):
        raise ValueError("Repeated Gnomix fits are not deterministic")
    reloaded = pickle.loads(pickle.dumps(first, protocol=pickle.HIGHEST_PROTOCOL))
    probabilities_reloaded = reloaded.predict_proba(features)
    if not np.array_equal(probabilities_first, probabilities_reloaded):
        raise ValueError("Gnomix probabilities changed after serialization and reload")
    coefficients = [model.coef_ for model in first.models]
    intercepts = [model.intercept_ for model in first.models]
    return {
        "stage": "M28C_GNOMIX_RUNTIME_KNOWN_ANSWER",
        "gnomix_base_class": type(first).__name__,
        "estimator_class": type(first.models[0]).__name__,
        "scikit_learn_version": version("scikit-learn"),
        "numpy_version": version("numpy"),
        "scipy_version": version("scipy"),
        "classes_per_window": classes,
        "probability_shape": list(probabilities_first.shape),
        "maximum_probability_sum_error": max_sum_error,
        "expected_predictions_exact": True,
        "repeated_fit_probabilities_exact": True,
        "reload_probabilities_exact": True,
        "probabilities_sha256": array_sha256(probabilities_first),
        "coefficient_shapes": [list(value.shape) for value in coefficients],
        "coefficients_sha256": array_sequence_sha256(coefficients),
        "intercepts_sha256": array_sequence_sha256(intercepts),
        "parent_process_warnings": sorted({str(item.message) for item in caught}),
        "breakpoint_probability_guard": breakpoint_guard,
        "expected_worker_warning": "scikit-learn 1.7.2 deprecates multiclass liblinear OVR and announces removal in 1.8; version 1.7.2 is intentionally pinned.",
        "decision": "PASS_GNOMIX_RUNTIME_WITH_NUMERIC_GUARD",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "stage": report["stage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
