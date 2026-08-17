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


def audit() -> dict:
    features, labels = known_answer_data()
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
        "expected_worker_warning": "scikit-learn 1.7.2 deprecates multiclass liblinear OVR and announces removal in 1.8; version 1.7.2 is intentionally pinned.",
        "decision": "PASS_GNOMIX_LIBLINEAR_OVR_RUNTIME",
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
