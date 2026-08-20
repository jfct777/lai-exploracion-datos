#!/usr/bin/env python3
"""Authenticated streaming runner for the two-root M31 ordered-linear DEV.

The runner keeps feature materialization bounded to one diploid individual.
It never accepts VALID/TEST inputs.  The full reciprocal ``run`` remains
blocked pending a separate POST.  The only real-data path exposed here is a
one-way feasibility pilot split across durable ``fit-predict`` and
``score-pilot`` processes; ``benchmark-sample`` is truth-blind and never fits.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import importlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_ordered_linear as core  # noqa: E402


OUTPUT_NAMES = {
    "summary": "m31_ordered_linear.summary.json",
    "metrics": "m31_ordered_linear.metrics.tsv",
    "individual": "m31_ordered_linear.individual.tsv.gz",
    "manifest": "m31_ordered_linear.manifest.json",
    "provenance": "m31_ordered_linear.provenance.json",
}
FITTED_ARMS = ("C", "L", "D", "H")
THREAD_LIMIT_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
PILOT_FEATURE_COUNTS = {"C": 59, "L": 99, "D": 171}
AUDIT_METRIC_NAMES = (
    "boundary_f1_0.2cM",
    "false_transitions_per_cM_0.2cM",
    "macro_ancestry_dose_mae",
    "haplotype_brier",
)
GUARD_FAILURE_NAMES = (
    "macro_ancestry_dose_mae_gt_f0",
    "false_transitions_per_cM_0.2cM_gt_f0",
)


class RunnerError(RuntimeError):
    """Raised when an execution or scientific invariant fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def stable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty table {path.name}")
    fields = list(rows[0])
    require(all(list(row) == fields for row in rows), f"inconsistent columns in {path.name}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if json_safe(value) is None else json_safe(value) for key, value in row.items()})


def _git_commit() -> str | None:
    head = Path(".git/HEAD")
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref:"):
        ref = Path(".git") / value.split(None, 1)[1]
        if ref.is_file():
            value = ref.read_text(encoding="utf-8").strip()
    return value if len(value) == 40 and all(char in "0123456789abcdef" for char in value) else None


@dataclass(frozen=True)
class RootPaths:
    sites: Path
    target: Path
    tree: Path
    pools: Path
    truth: Path
    flare_vcf: Path
    flare_audit: Path

    def as_dict(self) -> dict[str, Path]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True)
class FeaturePaths:
    sites: Path
    target: Path
    tree: Path
    pools: Path
    flare_vcf: Path
    flare_audit: Path

    def as_dict(self) -> dict[str, Path]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True)
class TrainingPaths:
    sites: Path
    target: Path
    tree: Path
    pools: Path
    truth: Path
    flare_vcf: Path
    flare_audit: Path

    def feature_paths(self) -> FeaturePaths:
        return FeaturePaths(self.sites, self.target, self.tree, self.pools, self.flare_vcf, self.flare_audit)

    def as_dict(self) -> dict[str, Path]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass
class FeatureDataset:
    name: str
    seed: int
    genetic_map: core.GeneticMap
    rare: core.RareInput
    flare: core.FlareInput
    marker_positions: np.ndarray
    marker_cm: np.ndarray
    marker_weights_cm: np.ndarray
    cell_left_bp: np.ndarray
    cell_right_bp: np.ndarray
    rare_cm: np.ndarray
    ref_dosage: np.ndarray
    ref_people: tuple[str, ...]
    ref_labels: tuple[str, ...]
    _support_cache: dict[tuple[str, int | None], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def samples(self) -> tuple[str, ...]:
        return self.flare.samples

    def support(self, arm: str, replicate: int | None) -> tuple[np.ndarray, np.ndarray]:
        key = (arm, replicate)
        if key in self._support_cache:
            return self._support_cache[key]
        labels = self.ref_labels
        if arm in {"DSHAM", "HSHAM"}:
            require(replicate is not None, "sham arm lacks replicate")
            labels = core.permute_diploid_labels(labels, self.seed, replicate)
        result = core.ancestry_support(self.ref_dosage, labels)
        self._support_cache[key] = result
        return result

    def features(self, sample_index: int, arm: str, replicate: int | None = None) -> np.ndarray:
        base_arm = {"DSHAM": "D", "HSHAM": "H"}.get(arm, arm)
        support, no_support = self.support(arm, replicate)
        materialized = core.materialize_sample_features(
            self.marker_cm,
            self.rare_cm,
            self.flare.probabilities[:, sample_index],
            np.zeros_like(self.flare.probabilities[:, sample_index]),
            self.rare.hap_presence[:, sample_index],
            support,
            no_support,
            requested_arms=(base_arm,),
        )
        result = np.asarray(
            materialized.arms[base_arm].reshape(-1, materialized.arms[base_arm].shape[-1]), dtype=np.float32,
        )
        require(np.all(np.isfinite(result)), f"{self.name}/{arm} produced nonfinite features")
        return result


def _physical_voronoi(positions: np.ndarray, genetic_map: core.GeneticMap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(positions.ndim == 1 and positions.size >= 2 and np.all(np.diff(positions) > 0), "FLARE marker positions are not strictly ordered")
    left = np.empty_like(positions)
    right = np.empty_like(positions)
    left[0] = positions[0]
    for index in range(1, len(positions)):
        left[index] = (int(positions[index - 1]) + int(positions[index])) // 2 + 1
    right[:-1] = left[1:]
    right[-1] = positions[-1] + 1
    map_right = np.minimum(right, genetic_map.positions[-1])
    weights = np.asarray(genetic_map.cm_at(map_right), dtype=float) - np.asarray(genetic_map.cm_at(left), dtype=float)
    require(np.all(weights >= 0) and weights.sum() > 0, "FLARE Voronoi cells have invalid genetic weights")
    return left, right, weights


def _truth_boundary_rows(
    truth: Mapping[str, tuple[list[core.TruthSegment], list[core.TruthSegment]]],
    samples: Sequence[str],
    marker_cm: np.ndarray,
    genetic_map: core.GeneticMap,
    tolerance: float,
) -> tuple[np.ndarray, ...]:
    output = []
    for sample in samples:
        mask = np.zeros((len(marker_cm), 2), dtype=bool)
        for hap in (0, 1):
            boundaries = [float(genetic_map.cm_at(after.start)) for before, after in zip(truth[sample][hap], truth[sample][hap][1:]) if before.ancestry != after.ancestry]
            if boundaries:
                distance = np.min(np.abs(marker_cm[:, None] - np.asarray(boundaries)[None, :]), axis=1)
                mask[:, hap] = distance <= tolerance + 1e-12
        output.append(mask.reshape(-1))
    return tuple(output)


@dataclass(frozen=True)
class TruthBundle:
    root_name: str
    segments: Mapping[str, tuple[list[core.TruthSegment], list[core.TruthSegment]]]
    markers: np.ndarray
    boundary_rows: tuple[np.ndarray, ...]


def load_feature_root(name: str, seed: int, paths: RootPaths, genetic_map: core.GeneticMap, marker_count: int) -> FeatureDataset:
    """Load only truth-blind inputs used to prepare model predictions."""
    core.validate_flare_audit(paths.flare_audit, name, paths.flare_vcf)
    rare = core.load_ordered_rare(paths.sites, paths.target, seed)
    flare = core.load_flare(paths.flare_vcf)
    require(len(flare.loci) == marker_count, f"{name} exact FLARE marker count drifted")
    marker_positions = np.asarray([locus[1] for locus in flare.loci], dtype=np.int64)
    ref_dosage, people, labels = core.load_ref_minor_dosage(paths.tree, paths.pools, rare, genetic_map)
    marker_cm = np.asarray(genetic_map.cm_at(marker_positions), dtype=np.float64)
    rare_cm = np.asarray(genetic_map.cm_at(rare.positions), dtype=np.float64)
    cell_left, cell_right, weights = _physical_voronoi(marker_positions, genetic_map)
    return FeatureDataset(name, seed, genetic_map, rare, flare, marker_positions,
                          marker_cm, weights, cell_left, cell_right, rare_cm, ref_dosage, people, labels)


def load_truth_bundle(paths: RootPaths, features: FeatureDataset) -> TruthBundle:
    """Mount truth only after truth-blind prediction preparation is complete."""
    truth = core.load_truth(
        paths.truth, features.samples, int(features.marker_positions[0]), int(features.marker_positions[-1]) + 1,
    )
    core.validate_phase_binding(features.rare, features.flare, truth)
    markers = core.truth_at_markers(truth, features.samples, features.marker_positions)
    boundaries = _truth_boundary_rows(
        truth, features.samples, features.marker_cm, features.genetic_map, 0.2,
    )
    return TruthBundle(features.name, truth, markers, boundaries)


@dataclass
class RawRidgeStats:
    weight_sum: float
    sum_x: np.ndarray
    sum_y: np.ndarray
    sum_xx: np.ndarray
    sum_xy: np.ndarray
    individuals: int

    @classmethod
    def zero(cls, features: int) -> "RawRidgeStats":
        return cls(0.0, np.zeros(features), np.zeros(3), np.zeros((features, features)), np.zeros((features, 3)), 0)

    def add(self, other: "RawRidgeStats", sign: float = 1.0) -> None:
        self.weight_sum += sign * other.weight_sum
        self.sum_x += sign * other.sum_x
        self.sum_y += sign * other.sum_y
        self.sum_xx += sign * other.sum_xx
        self.sum_xy += sign * other.sum_xy
        self.individuals += int(sign * other.individuals)

    def copy(self) -> "RawRidgeStats":
        return RawRidgeStats(self.weight_sum, self.sum_x.copy(), self.sum_y.copy(), self.sum_xx.copy(), self.sum_xy.copy(), self.individuals)


def sample_stats(x: np.ndarray, residual: np.ndarray, weights: np.ndarray) -> RawRidgeStats:
    require(x.ndim == 2 and residual.shape == (len(x), 3) and weights.shape == (len(x),), "sample statistic dimensions differ")
    require(np.all(np.isfinite(x)) and np.all(np.isfinite(residual)) and np.all(np.isfinite(weights)) and np.all(weights >= 0), "sample statistics are nonfinite")
    x64 = np.asarray(x, dtype=np.float64)
    weighted_x = weights[:, None] * x64
    return RawRidgeStats(float(weights.sum()), weighted_x.sum(axis=0), (weights[:, None] * residual).sum(axis=0),
                         x64.T @ weighted_x, x64.T @ (weights[:, None] * residual), 1)


def _raw_stats_sha256(stats: RawRidgeStats) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([stats.weight_sum], dtype="<f8").tobytes())
    digest.update(np.asarray([stats.individuals], dtype="<i8").tobytes())
    for array in (stats.sum_x, stats.sum_y, stats.sum_xx, stats.sum_xy):
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def fit_from_stats(stats: RawRidgeStats, alpha: float) -> core.WeightedStandardizedRidgeResidual:
    require(stats.weight_sum > 0 and stats.individuals > 0, "ridge sufficient statistics are empty")
    mean_x = stats.sum_x / stats.weight_sum
    mean_y = stats.sum_y / stats.weight_sum
    centered_xx = stats.sum_xx - stats.weight_sum * np.outer(mean_x, mean_x)
    centered_xy = stats.sum_xy - np.outer(stats.sum_x, mean_y)
    variance = np.maximum(np.diag(centered_xx) / stats.weight_sum, 0.0)
    scale = np.sqrt(variance)
    scale[scale <= np.finfo(float).eps] = 1.0
    factor = stats.individuals / stats.weight_sum
    gram = factor * centered_xx / scale[:, None] / scale[None, :]
    gram.flat[:: gram.shape[0] + 1] += alpha
    rhs = factor * centered_xy / scale[:, None]
    coefficients = np.linalg.solve(gram, rhs)
    for array in (mean_x, scale, mean_y, coefficients):
        require(np.all(np.isfinite(array)), "ridge fit from sufficient statistics is nonfinite")
        array.setflags(write=False)
    return core.WeightedStandardizedRidgeResidual(mean_x, scale, mean_y, coefficients, alpha, float(stats.individuals), stats.individuals)


@dataclass(frozen=True)
class FittedArm:
    arm: str
    replicate: int | None
    alpha: float
    boundary_weight: float
    cv_boundary_f1: float
    cv_false_transitions_per_cm: float
    cv_macro_ancestry_dose_mae: float
    cv_brier: float
    guarded: bool
    selection_status: str
    model: core.WeightedStandardizedRidgeResidual
    feature_count: int
    f0_cv_metrics: Mapping[str, float] = field(default_factory=dict)
    candidate_table: tuple[Mapping[str, Any], ...] = ()
    guard_failures: tuple[str, ...] = ()
    sufficient_stats_sha256: Mapping[str, str] = field(default_factory=dict)
    fold_stats_sha256: Mapping[str, str] = field(default_factory=dict)


def _validate_workers(workers: int) -> int:
    require(isinstance(workers, int) and not isinstance(workers, bool) and workers >= 1,
            "workers must be an integer >=1")
    if workers > 1:
        drift = {name: os.environ.get(name) for name in THREAD_LIMIT_ENV if os.environ.get(name) != "1"}
        require(not drift, f"workers>1 requires all BLAS/thread environment limits to equal 1: {drift}")
    return workers


def _threadpool_runtime(workers: int) -> dict[str, Any]:
    """Return a path-independent runtime inventory and verify effective BLAS limits."""
    require(workers >= 1, "threadpool runtime inspection requires workers >=1")
    try:
        threadpoolctl = importlib.import_module("threadpoolctl")
    except ImportError:
        require(workers == 1, "workers>1 requires importable threadpoolctl runtime verification")
        return {
            "controller": "threadpoolctl",
            "controller_version": None,
            "status": "NOT_REQUIRED_SINGLE_WORKER_CONTROLLER_UNAVAILABLE",
            "pools": [],
        }
    try:
        raw_pools = threadpoolctl.threadpool_info()
    except Exception as error:
        if workers > 1:
            raise RunnerError(
                f"workers>1 threadpoolctl inspection failed: {type(error).__name__}: {error}"
            ) from error
        return {
            "controller": "threadpoolctl",
            "controller_version": str(getattr(threadpoolctl, "__version__", "UNKNOWN")),
            "status": f"NOT_REQUIRED_SINGLE_WORKER_INSPECTION_FAILED_{type(error).__name__}",
            "pools": [],
        }
    require(isinstance(raw_pools, list), "threadpoolctl returned a non-list pool inventory")
    normalized = []
    for raw in raw_pools:
        require(isinstance(raw, Mapping), "threadpoolctl returned a non-mapping pool entry")
        num_threads = raw.get("num_threads")
        require(isinstance(num_threads, int) and not isinstance(num_threads, bool) and num_threads >= 1,
                "threadpoolctl pool has invalid num_threads")
        normalized.append({
            "user_api": None if raw.get("user_api") is None else str(raw.get("user_api")),
            "internal_api": None if raw.get("internal_api") is None else str(raw.get("internal_api")),
            "prefix": None if raw.get("prefix") is None else str(raw.get("prefix")),
            "version": None if raw.get("version") is None else str(raw.get("version")),
            "num_threads": num_threads,
            "threading_layer": None if raw.get("threading_layer") is None else str(raw.get("threading_layer")),
            "architecture": None if raw.get("architecture") is None else str(raw.get("architecture")),
        })
    normalized.sort(key=lambda pool: tuple("" if pool[key] is None else str(pool[key]) for key in (
        "user_api", "internal_api", "prefix", "version", "threading_layer", "architecture", "num_threads",
    )))
    if workers > 1:
        require(bool(normalized), "workers>1 requires at least one BLAS pool reported by threadpoolctl")
        offenders = [
            f"{pool['internal_api'] or pool['user_api'] or pool['prefix']}={pool['num_threads']}"
            for pool in normalized if pool["num_threads"] != 1
        ]
        require(not offenders, f"workers>1 requires every effective BLAS pool num_threads=1: {offenders}")
        status = "VERIFIED_ALL_POOLS_SINGLE_THREAD"
    else:
        status = "OBSERVED_SINGLE_WORKER" if normalized else "NOT_REQUIRED_SINGLE_WORKER_NO_POOLS"
    return {
        "controller": "threadpoolctl",
        "controller_version": str(getattr(threadpoolctl, "__version__", "UNKNOWN")),
        "status": status,
        "pools": normalized,
    }


def _validate_worker_runtime(workers: int) -> tuple[int, dict[str, Any]]:
    workers = _validate_workers(workers)
    return workers, _threadpool_runtime(workers)


def fit_arm_streaming(
    root: FeatureDataset,
    truth: TruthBundle,
    arm: str,
    replicate: int | None,
    alphas: Sequence[float],
    boundary_weights: Sequence[float],
    cv_seed: int,
    workers: int = 1,
) -> FittedArm:
    """Fit with individual-only concurrency and serial, index-ordered reductions."""
    workers, _runtime = _validate_worker_runtime(workers)
    fold_ids = core.grouped_three_fold_ids(root.samples, seed=cv_seed)
    # Populate the only mutable truth-blind feature cache before threads start.
    if hasattr(root, "support"):
        root.support(arm, replicate)
    base_weight = np.repeat(root.marker_weights_cm, 2)
    with tempfile.TemporaryDirectory(prefix=f"m31-{root.name}-{arm}-") as cache_text:
        cache_dir = Path(cache_text)

        def materialize_and_measure(sample_index: int) -> tuple[int, int, dict[float, RawRidgeStats]]:
            x = root.features(sample_index, arm, replicate)
            require(x.ndim == 2, "individual feature cache is not a matrix")
            # Each worker owns one unique temporary-cache path. Durable
            # checkpoints/manifests remain parent-only operations.
            np.save(cache_dir / f"sample-{sample_index}.npy", x.astype(np.float32), allow_pickle=False)
            residual = (truth.markers[:, sample_index] - root.flare.probabilities[:, sample_index]).reshape(-1, 3)
            stats = {
                float(boundary_weight): sample_stats(
                    x, residual,
                    base_weight * np.where(truth.boundary_rows[sample_index], float(boundary_weight), 1.0),
                )
                for boundary_weight in boundary_weights
            }
            return sample_index, x.shape[1], stats

        try:
            executor = (
                ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"m31-{arm}")
                if workers > 1 else None
            )
        except Exception as error:
            raise RunnerError(
                f"parallel individual executor failed for {root.name}/{arm}: "
                f"{type(error).__name__}: {error}"
            ) from error

        def ordered_map(function: Any, indexes: Iterable[int]) -> list[Any]:
            values = tuple(indexes)
            if executor is None:
                return [function(index) for index in values]
            # executor.map yields in input order regardless of completion order.
            try:
                return list(executor.map(function, values))
            except Exception as error:
                raise RunnerError(
                    f"parallel individual worker failed for {root.name}/{arm}: "
                    f"{type(error).__name__}: {error}"
                ) from error

        try:
            materialized = ordered_map(materialize_and_measure, range(len(root.samples)))
            require([item[0] for item in materialized] == list(range(len(root.samples))),
                    "parallel individual materialization changed sample order")
            feature_counts = {item[1] for item in materialized}
            require(len(feature_counts) == 1, "feature count changed across individuals")
            feature_count = feature_counts.pop()
            per_sample = [item[2] for item in materialized]

            fold_models: dict[tuple[int, float, float], core.WeightedStandardizedRidgeResidual] = {}
            fold_stats_hashes: dict[str, str] = {}
            for fold in range(3):
                for boundary_weight in boundary_weights:
                    combined = RawRidgeStats.zero(feature_count)
                    for sample_index, stats_by_weight in enumerate(per_sample):
                        if fold_ids[sample_index] != fold:
                            combined.add(stats_by_weight[float(boundary_weight)])
                    fold_stats_hashes[
                        f"fold{fold}:{format(float(boundary_weight), '.17g')}"
                    ] = _raw_stats_sha256(combined)
                    for alpha in alphas:
                        fold_models[(fold, float(boundary_weight), float(alpha))] = fit_from_stats(combined, float(alpha))

            def score_f0(sample_index: int) -> ScoreCounts:
                prediction = np.array(root.flare.probabilities[:, sample_index], copy=True)
                return score_sample(root, truth, sample_index, prediction)[1]

            f0_counts = ordered_map(score_f0, range(len(root.samples)))
            f0_metrics_all = summarize_counts(f0_counts)
            f0_metrics = {name: float(f0_metrics_all[name]) for name in AUDIT_METRIC_NAMES}
            candidates: list[tuple[tuple[float, ...], float, float, dict[str, float], tuple[str, ...]]] = []
            guarded_candidates: list[tuple[tuple[float, ...], float, float, dict[str, float], tuple[str, ...]]] = []
            candidate_rows: list[dict[str, Any]] = []
            for boundary_weight in boundary_weights:
                for alpha in alphas:
                    fold_models_for_candidate = fold_models

                    def predict_and_score(sample_index: int) -> ScoreCounts:
                        x = np.load(cache_dir / f"sample-{sample_index}.npy", mmap_mode="r")
                        baseline = root.flare.probabilities[:, sample_index]
                        fold = int(fold_ids[sample_index])
                        prediction = fold_models_for_candidate[(fold, float(boundary_weight), float(alpha))].predict(
                            x, baseline.reshape(-1, 3)
                        ).reshape(baseline.shape)
                        return score_sample(root, truth, sample_index, prediction)[1]

                    counts = ordered_map(predict_and_score, range(len(root.samples)))
                    metrics_all = summarize_counts(counts)
                    metrics = {name: float(metrics_all[name]) for name in AUDIT_METRIC_NAMES}
                    failures: list[str] = []
                    if metrics["macro_ancestry_dose_mae"] > f0_metrics["macro_ancestry_dose_mae"] + 1e-15:
                        failures.append("macro_ancestry_dose_mae_gt_f0")
                    if metrics["false_transitions_per_cM_0.2cM"] > f0_metrics["false_transitions_per_cM_0.2cM"] + 1e-15:
                        failures.append("false_transitions_per_cM_0.2cM_gt_f0")
                    guard_failures = tuple(failures)
                    key = (-metrics["boundary_f1_0.2cM"], metrics["false_transitions_per_cM_0.2cM"],
                           metrics["macro_ancestry_dose_mae"], metrics["haplotype_brier"],
                           float(boundary_weight), -float(alpha))
                    candidate = (key, float(boundary_weight), float(alpha), metrics, guard_failures)
                    candidates.append(candidate)
                    if not guard_failures:
                        guarded_candidates.append(candidate)
                    candidate_rows.append({
                        "boundary_weight": float(boundary_weight),
                        "alpha": float(alpha),
                        **metrics,
                        "guarded": not guard_failures,
                        "guard_failures": list(guard_failures),
                        "selected": False,
                    })
            guarded = bool(guarded_candidates)
            selection_status = "GUARDED_CONFIG" if guarded else "NO_GUARDED_CONFIG"
            selection_pool = guarded_candidates if guarded else candidates
            _key, selected_weight, selected_alpha, selected_metrics, selected_failures = min(
                selection_pool, key=lambda item: item[0]
            )
            selected_matches = 0
            for row in candidate_rows:
                if row["boundary_weight"] == selected_weight and row["alpha"] == selected_alpha:
                    row["selected"] = True
                    selected_matches += 1
            require(selected_matches == 1, "candidate audit table does not identify exactly one selected configuration")
            totals_by_weight: dict[float, RawRidgeStats] = {}
            for boundary_weight in boundary_weights:
                total_for_weight = RawRidgeStats.zero(feature_count)
                for stats_by_weight in per_sample:
                    total_for_weight.add(stats_by_weight[float(boundary_weight)])
                totals_by_weight[float(boundary_weight)] = total_for_weight
            stats_hashes = {
                format(weight, ".17g"): _raw_stats_sha256(total_for_weight)
                for weight, total_for_weight in totals_by_weight.items()
            }
            total = totals_by_weight[selected_weight]
            model = fit_from_stats(total, selected_alpha)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
    return FittedArm(
        arm, replicate, selected_alpha, selected_weight,
        selected_metrics["boundary_f1_0.2cM"],
        selected_metrics["false_transitions_per_cM_0.2cM"],
        selected_metrics["macro_ancestry_dose_mae"],
        selected_metrics["haplotype_brier"], guarded, selection_status, model, feature_count,
        f0_metrics, tuple(candidate_rows), selected_failures, stats_hashes, fold_stats_hashes,
    )


def _truth_boundaries_exact(root: FeatureDataset, truth: TruthBundle, sample: str, hap: int) -> list[core.Boundary]:
    output = []
    segments = truth.segments[sample][hap]
    for before, after in zip(segments, segments[1:]):
        if before.ancestry != after.ancestry:
            output.append(core.Boundary(float(root.genetic_map.cm_at(after.start)),
                                        core.ANCESTRIES.index(before.ancestry),
                                        core.ANCESTRIES.index(after.ancestry)))
    return output


def _predicted_boundaries(root: FeatureDataset, probabilities: np.ndarray, hap: int) -> list[core.Boundary]:
    labels = np.argmax(probabilities[:, hap], axis=1)
    return [
        core.Boundary(float(root.genetic_map.cm_at(int(root.cell_left_bp[index]))),
                      int(labels[index - 1]), int(labels[index]))
        for index in range(1, len(labels)) if labels[index] != labels[index - 1]
    ]


def _macro_f1_confusion(confusion: np.ndarray) -> float:
    values = []
    for index, _state in enumerate(core.DIPLOID_CLASSES):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        values.append(2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn else 0.0)
    return float(np.mean(values))


@dataclass(frozen=True)
class ScoreCounts:
    sample_id: str
    dose_mae_numerator: np.ndarray
    brier_numerator: float
    total_cm: float
    confusion: np.ndarray
    boundary: Mapping[float, tuple[int, int, int, tuple[float, ...]]]


def summarize_counts(counts: Sequence[ScoreCounts]) -> dict[str, Any]:
    require(bool(counts), "cannot summarize empty score counts")
    total_cm = float(sum(item.total_cm for item in counts))
    require(total_cm > 0.0, "global score has zero cM")
    dose = sum((item.dose_mae_numerator for item in counts), np.zeros(3)) / total_cm
    confusion = sum((item.confusion for item in counts), np.zeros((6, 6)))
    output: dict[str, Any] = {
        "macro_ancestry_dose_mae": float(np.mean(dose)),
        **{f"ancestry_dose_mae_{ancestry}": float(dose[index]) for index, ancestry in enumerate(core.ANCESTRIES)},
        "haplotype_brier": float(sum(item.brier_numerator for item in counts) / total_cm),
        "diploid_macro_f1_fixed_six": _macro_f1_confusion(confusion),
    }
    for tolerance in core.BOUNDARY_TOLERANCES_CM:
        truth_count = sum(item.boundary[tolerance][0] for item in counts)
        prediction_count = sum(item.boundary[tolerance][1] for item in counts)
        matched = sum(item.boundary[tolerance][2] for item in counts)
        distances = [distance for item in counts for distance in item.boundary[tolerance][3]]
        precision = matched / prediction_count if prediction_count else (1.0 if truth_count == 0 else 0.0)
        recall = matched / truth_count if truth_count else (1.0 if prediction_count == 0 else 0.0)
        suffix = f"{tolerance:.1f}cM"
        output[f"boundary_f1_{suffix}"] = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        output[f"false_transitions_per_cM_{suffix}"] = (prediction_count - matched) / (2.0 * total_cm)
        output[f"matched_boundary_median_{suffix}"] = float(np.median(distances)) if distances else None
        output[f"matched_boundary_p90_{suffix}"] = float(np.quantile(distances, 0.9)) if distances else None
    return output


def score_sample(root: FeatureDataset, truth: TruthBundle, sample_index: int, predicted: np.ndarray) -> tuple[dict[str, Any], ScoreCounts]:
    """M30-compatible exact-cell metrics with direct haplotype binding."""
    sample = root.samples[sample_index]
    require(predicted.shape == (len(root.marker_positions), 2, 3), "outer prediction shape differs from FLARE grid")
    mae_num = np.zeros(3)
    brier_num = 0.0
    total_cm = 0.0
    confusion = np.zeros((6, 6), dtype=np.float64)
    state_index = {state: index for index, state in enumerate(core.DIPLOID_CLASSES)}
    require(truth.root_name == root.name, "truth bundle belongs to another root")
    truth_pair = truth.segments[sample]
    truth_indexes = [0, 0]
    for marker_index, (left, right) in enumerate(zip(root.cell_left_bp, root.cell_right_bp)):
        cursor = int(left)
        while cursor < int(right):
            for hap in (0, 1):
                while truth_pair[hap][truth_indexes[hap]].end <= cursor:
                    truth_indexes[hap] += 1
            end = min(int(right), truth_pair[0][truth_indexes[0]].end, truth_pair[1][truth_indexes[1]].end)
            require(end > cursor, f"truth integration stalled for {sample}")
            weight = float(root.genetic_map.cm_at(min(end, int(root.genetic_map.positions[-1])))) - float(root.genetic_map.cm_at(cursor))
            require(weight >= -1e-12, "negative genetic integration weight")
            weight = max(0.0, weight)
            labels = tuple(core.ANCESTRIES.index(truth_pair[hap][truth_indexes[hap]].ancestry) for hap in (0, 1))
            truth_dose = np.bincount(labels, minlength=3).astype(float)
            predicted_dose = predicted[marker_index].sum(axis=0)
            mae_num += weight * np.abs(predicted_dose - truth_dose) / 2.0
            targets = np.eye(3)[list(labels)]
            brier_num += weight * float(np.square(predicted[marker_index] - targets).sum()) / 4.0
            observed_state = tuple(sorted(labels))
            predicted_state = tuple(sorted(np.argmax(predicted[marker_index], axis=1).astype(int).tolist()))
            confusion[state_index[observed_state], state_index[predicted_state]] += weight
            total_cm += weight
            cursor = end
    require(total_cm > 0, "truth integration has zero cM")
    truth_boundaries = [_truth_boundaries_exact(root, truth, sample, hap) for hap in (0, 1)]
    prediction_boundaries = [_predicted_boundaries(root, predicted, hap) for hap in (0, 1)]
    boundary: dict[float, tuple[int, int, int, tuple[float, ...]]] = {}
    for tolerance in core.BOUNDARY_TOLERANCES_CM:
        pairs = [pair for hap in (0, 1) for pair in core.ordered_boundary_pairs(truth_boundaries[hap], prediction_boundaries[hap], tolerance)]
        distances = tuple(float(pair[2]) for pair in pairs)
        truth_count = sum(map(len, truth_boundaries)); prediction_count = sum(map(len, prediction_boundaries)); matched = len(pairs)
        boundary[float(tolerance)] = (truth_count, prediction_count, matched, distances)
    counts = ScoreCounts(sample, mae_num, brier_num, total_cm, confusion, boundary)
    return {"sample_id": sample, **summarize_counts([counts])}, counts


def bootstrap_counts(counts: Sequence[ScoreCounts]) -> dict[str, Any]:
    require(bool(counts) and len({item.sample_id for item in counts}) == len(counts),
            "bootstrap requires unique complete diploid individuals")
    rng = np.random.default_rng(core.BOOTSTRAP_SEED)
    draws: dict[str, list[float]] = {}
    for _replicate in range(core.BOOTSTRAP_REPLICATES):
        selected = [counts[index] for index in rng.integers(0, len(counts), size=len(counts))]
        metrics = summarize_counts(selected)
        for key, value in metrics.items():
            if value is not None:
                draws.setdefault(key, []).append(float(value))
    return {
        "unit": "complete_diploid_individual",
        "aggregation": "resample_individual_sufficient_counts_then_reconstruct_global_metrics",
        "replicates": core.BOOTSTRAP_REPLICATES,
        "seed": core.BOOTSTRAP_SEED,
        "interval": "percentile_95",
        "metrics": {
            key: {"lower": float(np.quantile(values, 0.025)), "upper": float(np.quantile(values, 0.975))}
            for key, values in draws.items()
        },
    }


@dataclass(frozen=True)
class PredictionArtifact:
    root_name: str
    sample_ids: tuple[str, ...]
    arm: str
    replicate: int | None
    arrays: tuple[np.ndarray, ...]
    sha256: str


def _prediction_sha256(root_name: str, samples: Sequence[str], arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update((root_name + "\n" + "\n".join(samples) + "\n").encode("utf-8"))
    for array in arrays:
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def prepare_predictions(
    root: FeatureDataset, fitted: FittedArm | None, arm: str, replicate: int | None,
) -> PredictionArtifact:
    """Truth-blind prediction phase; its immutable digest is scorer input."""
    predictions = []
    for sample_index, _sample in enumerate(root.samples):
        baseline = root.flare.probabilities[:, sample_index]
        if fitted is None:
            predicted = np.array(baseline, copy=True)
        else:
            x = root.features(sample_index, arm, replicate)
            predicted = fitted.model.predict(x, baseline.reshape(-1, 3)).reshape(baseline.shape)
        predicted.setflags(write=False)
        predictions.append(predicted)
    arrays = tuple(predictions)
    return PredictionArtifact(
        root.name, root.samples, arm, replicate, arrays,
        _prediction_sha256(root.name, root.samples, arrays),
    )


def score_prediction_artifact(
    root: FeatureDataset, truth: TruthBundle, artifact: PredictionArtifact,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[ScoreCounts]]:
    """Truth-aware phase that verifies and scores the frozen prediction hash."""
    require(artifact.root_name == root.name and artifact.sample_ids == root.samples,
            "prediction artifact identity differs from scoring root")
    require(artifact.sha256 == _prediction_sha256(root.name, root.samples, artifact.arrays),
            "prediction artifact SHA-256 changed before scoring")
    scored = [score_sample(root, truth, index, prediction) for index, prediction in enumerate(artifact.arrays)]
    rows = [item[0] for item in scored]
    counts = [item[1] for item in scored]
    return summarize_counts(counts), rows, counts


def evaluate_arm(
    root: FeatureDataset, truth: TruthBundle, fitted: FittedArm | None, arm: str, replicate: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], PredictionArtifact]:
    artifact = prepare_predictions(root, fitted, arm, replicate)
    summary, rows, counts = score_prediction_artifact(root, truth, artifact)
    return summary, rows, bootstrap_counts(counts), artifact


def arm_specs(sham_replicates: int = core.SHAM_REPLICATES) -> list[tuple[str, int | None]]:
    return [(arm, None) for arm in FITTED_ARMS] + [
        (arm, replicate) for arm in ("DSHAM", "HSHAM") for replicate in range(sham_replicates)
    ]


def _metric_row(direction: str, arm: str, replicate: int | None, fitted: FittedArm | None,
                summary: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "direction": direction,
        "arm": arm,
        "sham_replicate": "" if replicate is None else replicate,
        "selected_alpha": "" if fitted is None else fitted.alpha,
        "selected_boundary_weight": "" if fitted is None else fitted.boundary_weight,
        "inner_cv_boundary_f1_0.2cM": "" if fitted is None else fitted.cv_boundary_f1,
        "inner_cv_false_transitions_per_cM_0.2cM": "" if fitted is None else fitted.cv_false_transitions_per_cm,
        "inner_cv_macro_ancestry_dose_mae": "" if fitted is None else fitted.cv_macro_ancestry_dose_mae,
        "inner_cv_brier": "" if fitted is None else fitted.cv_brier,
        "inner_cv_guarded": "" if fitted is None else fitted.guarded,
        "inner_cv_selection_status": "" if fitted is None else fitted.selection_status,
        "feature_count": 0 if fitted is None else fitted.feature_count,
        **summary,
        "bootstrap_json": json.dumps(json_safe(bootstrap), sort_keys=True, separators=(",", ":")),
    }


def decide(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(row["direction"], row["arm"], row["sham_replicate"]): row for row in metrics}
    directions = sorted({str(row["direction"]) for row in metrics})

    def real(direction: str, arm: str) -> Mapping[str, Any]:
        return by_key[(direction, arm, "")]

    d_direction_pass = {}
    h_direction_pass = {}
    l_direction_pass = {}
    guard_fail = False
    ancestry_worse_both = {}
    unguarded = [
        {"direction": row["direction"], "arm": row["arm"], "sham_replicate": row["sham_replicate"]}
        for row in metrics if row["arm"] != "F0" and row.get("inner_cv_guarded") is not True
    ]
    null_evidence: dict[str, dict[str, Any]] = {}
    for direction in directions:
        f0, c, l, d, h = (real(direction, arm) for arm in ("F0", "C", "L", "D", "H"))
        d_gain = d["boundary_f1_0.2cM"] - l["boundary_f1_0.2cM"]
        h_gain = h["boundary_f1_0.2cM"] - l["boundary_f1_0.2cM"]
        null_evidence[direction] = {}
        for real_arm, sham_arm, real_gain in (("D", "DSHAM", d_gain), ("H", "HSHAM", h_gain)):
            sham_rows = [row for row in metrics if row["direction"] == direction and row["arm"] == sham_arm]
            replicate_ids = [row["sham_replicate"] for row in sham_rows]
            require(len(replicate_ids) == core.SHAM_REPLICATES and set(replicate_ids) == set(range(core.SHAM_REPLICATES)),
                    f"{direction}/{sham_arm} must contain exactly unique replicate IDs 0..31")
            gains = [row["boundary_f1_0.2cM"] - l["boundary_f1_0.2cM"] for row in sham_rows]
            exceeds = bool(real_gain > max(gains))
            null_evidence[direction][real_arm] = {
                "replicates": core.SHAM_REPLICATES,
                "exceeds_all_shams": exceeds,
                "exploratory_p": 1.0 / (core.SHAM_REPLICATES + 1),
            }
        d_guard = d["macro_ancestry_dose_mae"] <= f0["macro_ancestry_dose_mae"] and d["false_transitions_per_cM_0.2cM"] <= f0["false_transitions_per_cM_0.2cM"]
        h_guard = h["macro_ancestry_dose_mae"] <= f0["macro_ancestry_dose_mae"] and h["false_transitions_per_cM_0.2cM"] <= f0["false_transitions_per_cM_0.2cM"]
        d_direction_pass[direction] = bool(d.get("inner_cv_guarded") is True and d["boundary_f1_0.2cM"] > max(f0["boundary_f1_0.2cM"], c["boundary_f1_0.2cM"], l["boundary_f1_0.2cM"]) and null_evidence[direction]["D"]["exceeds_all_shams"] and d_guard)
        h_direction_pass[direction] = bool(h.get("inner_cv_guarded") is True and h["boundary_f1_0.2cM"] > max(f0["boundary_f1_0.2cM"], c["boundary_f1_0.2cM"], l["boundary_f1_0.2cM"]) and null_evidence[direction]["H"]["exceeds_all_shams"] and h_guard)
        l_direction_pass[direction] = bool(l["boundary_f1_0.2cM"] > max(f0["boundary_f1_0.2cM"], c["boundary_f1_0.2cM"]))
        guard_fail |= (d["boundary_f1_0.2cM"] > l["boundary_f1_0.2cM"] and not d_guard) or (h["boundary_f1_0.2cM"] > l["boundary_f1_0.2cM"] and not h_guard)
    for ancestry in core.ANCESTRIES:
        key = f"ancestry_dose_mae_{ancestry}"
        ancestry_worse_both[ancestry] = all(real(direction, "D")[key] > real(direction, "F0")[key] for direction in directions)
    d_pass = all(d_direction_pass.values()) and not any(ancestry_worse_both.values())
    h_pass = all(h_direction_pass.values())
    l_pass = all(l_direction_pass.values())
    if unguarded:
        label = "NO_GUARDED_CONFIG"
        d_pass = False
        h_pass = False
    elif d_pass:
        label = "GO_NEW_ROOTS"
    elif h_pass and not d_pass:
        label = "PHASE_CEILING_ONLY"
    elif l_pass and not d_pass:
        label = "LOAD_ONLY"
    elif guard_fail:
        label = "TRADEOFF"
    else:
        label = "STOP_LINEAR_ORDERED_RARE"
    return {
        "label": label,
        "D_direction_pass": d_direction_pass,
        "H_direction_pass": h_direction_pass,
        "L_direction_pass": l_direction_pass,
        "D_ancestry_worse_in_both_directions": ancestry_worse_both,
        "unguarded_fitted_arms": unguarded,
        "null_evidence": null_evidence,
        "directions_are_not_independent_replicates": True,
        "claim_scope": "two reciprocal DEV roots only; no confirmatory validation",
    }


def run_two_root_dev(
    root17: FeatureDataset,
    truth17: TruthBundle,
    root18: FeatureDataset,
    truth18: TruthBundle,
    contract: core.OrderedLinearContract,
    *,
    sham_replicates: int = core.SHAM_REPLICATES,
    bootstrap: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    require(root17.samples == root18.samples, "root sample order differs")
    require(root17.seed != root18.seed, "reciprocal DEV requires distinct roots")
    directions = ((root17, truth17, root18, truth18), (root18, truth18, root17, truth17))
    metric_rows: list[dict[str, Any]] = []
    individual_rows: list[dict[str, Any]] = []
    prediction_hashes: dict[str, dict[str, str]] = {}
    for train, train_truth, evaluation, evaluation_truth in directions:
        direction = f"train_{train.name}_test_{evaluation.name}"
        prediction_hashes[direction] = {}
        f0_summary, f0_individual, f0_bootstrap, f0_artifact = evaluate_arm(
            evaluation, evaluation_truth, None, "F0", None,
        )
        prediction_hashes[direction]["F0"] = f0_artifact.sha256
        if not bootstrap:
            f0_bootstrap = {"status": "SKIPPED_KNOWN_ANSWER"}
        metric_rows.append(_metric_row(direction, "F0", None, None, f0_summary, f0_bootstrap))
        individual_rows.extend({"direction": direction, "arm": "F0", "sham_replicate": "", **row} for row in f0_individual)
        for arm, replicate in arm_specs(sham_replicates):
            fitted = fit_arm_streaming(
                train, train_truth, arm, replicate, contract.alphas, contract.boundary_weights, cv_seed=train.seed,
            )
            summary, individuals, boot, artifact = evaluate_arm(
                evaluation, evaluation_truth, fitted, arm, replicate,
            )
            prediction_hashes[direction][f"{arm}:{replicate if replicate is not None else 'real'}"] = artifact.sha256
            if not bootstrap:
                boot = {"status": "SKIPPED_KNOWN_ANSWER"}
            metric_rows.append(_metric_row(direction, arm, replicate, fitted, summary, boot))
            individual_rows.extend({"direction": direction, "arm": arm, "sham_replicate": "" if replicate is None else replicate, **row} for row in individuals)
    decision = decide(metric_rows) if sham_replicates == core.SHAM_REPLICATES else {
        "label": "KNOWN_ANSWER_NO_SCIENTIFIC_DECISION",
        "claim_scope": "synthetic orchestration only",
    }
    summary = {
        "schema_version": core.SCHEMA_VERSION,
        "experiment_id": core.EXPERIMENT_ID,
        "stage": "M31_TWO_ROOT_ORDERED_LINEAR_DEV",
        "decision": decision,
        "directions": [f"train_{train.name}_test_{evaluation.name}" for train, _tt, evaluation, _et in directions],
        "truth_blind_prediction_sha256": prediction_hashes,
        "arms": ["F0", *FITTED_ARMS, "DSHAM", "HSHAM"],
        "sham_replicates_per_null_arm": sham_replicates,
        "cross_root_preprocessing": {
            "policy": "root_local_truth_blind_feature_transform",
            "FREQ_site_universe": "root-local; never intersected or pooled across roots",
            "REF_LAI_support": "root-local ancestry-frequency support; evaluation root does not fit scaler/ridge",
            "shared_schema": "identical named C/L/D/H channels and frozen signed rings",
            "normalization_stop_rule": "feature scaler and ridge coefficients fit only on training-root individuals",
            "claim_limit": "reciprocal synthetic roots are DEV directions, not independent validation replicates",
        },
        "truth_phase_binding": "ANP1/AN1->h0 and ANP2/AN2->h1; no post-truth swap",
        "claims_excluded": ["confirmatory_validation", "DNABR_generalization", "Native_American_LAI", "Brazil_novel_variant_effect", "deep_learning_benefit"],
    }
    return summary, metric_rows, individual_rows


def _write_output_bundle(
    outdir: Path,
    summary: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    individuals: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if outdir.exists():
        require(outdir.is_dir() and not any(outdir.iterdir()), f"output directory is not empty: {outdir}")
    else:
        outdir.mkdir(parents=True)
    stable_json(outdir / OUTPUT_NAMES["summary"], json_safe(summary))
    _write_tsv(outdir / OUTPUT_NAMES["metrics"], metrics)
    _write_tsv(outdir / OUTPUT_NAMES["individual"], individuals)
    stable_json(outdir / OUTPUT_NAMES["provenance"], json_safe(provenance))
    files = {}
    for role in ("summary", "metrics", "individual", "provenance"):
        path = outdir / OUTPUT_NAMES[role]
        files[role] = {"path": path.name, "bytes": path.stat().st_size, "sha256": core.sha256_file(path)}
    manifest = {
        "schema_version": core.SCHEMA_VERSION,
        "experiment_id": core.EXPERIMENT_ID,
        "files": files,
        "manifest_excludes_self": True,
    }
    stable_json(outdir / OUTPUT_NAMES["manifest"], manifest)
    return manifest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    require(not temporary.exists(), f"stale temporary output exists: {temporary}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_npy_fsync(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    require(not temporary.exists(), f"stale temporary output exists: {temporary}")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _checkpoint_float(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"checkpoint {label} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"checkpoint {label} is not finite")
    return result


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_fit_checkpoint(arm: str, fit: Any) -> None:
    """Reject incomplete or internally inconsistent fit audit payloads."""
    if arm == "F0":
        require(fit is None, "checkpoint F0 fit must be null")
        return
    require(arm in PILOT_FEATURE_COUNTS, f"checkpoint fit validation does not support arm {arm}")
    require(isinstance(fit, Mapping), f"checkpoint fit missing for {arm}")
    required_fit_keys = {
        "alpha", "boundary_weight", "cv_boundary_f1_0.2cM",
        "cv_false_transitions_per_cM_0.2cM", "cv_macro_ancestry_dose_mae", "cv_brier",
        "guarded", "selection_status", "feature_count", "f0_cv_metrics", "candidate_table",
        "guard_failures", "sufficient_stats_sha256", "fold_stats_sha256",
    }
    require(set(fit) == required_fit_keys, f"checkpoint fit fields mismatch for {arm}")
    require(fit["feature_count"] == PILOT_FEATURE_COUNTS[arm]
            and isinstance(fit["feature_count"], int) and not isinstance(fit["feature_count"], bool),
            f"checkpoint feature_count mismatch for {arm}")

    f0 = fit["f0_cv_metrics"]
    require(isinstance(f0, Mapping) and set(f0) == set(AUDIT_METRIC_NAMES),
            f"checkpoint F0 OOF metrics mismatch for {arm}")
    f0_metrics = {name: _checkpoint_float(f0[name], f"{arm}.f0_cv_metrics.{name}")
                  for name in AUDIT_METRIC_NAMES}

    rows = fit["candidate_table"]
    require(isinstance(rows, list) and len(rows) == 18,
            f"checkpoint candidate table must contain exactly 18 rows for {arm}")
    expected_fields = {
        "alpha", "boundary_weight", *AUDIT_METRIC_NAMES,
        "guarded", "guard_failures", "selected",
    }
    expected_pairs = {
        (float(weight), float(alpha))
        for weight in core.EXPECTED_BOUNDARY_WEIGHTS
        for alpha in core.EXPECTED_ALPHAS
    }
    observed_pairs: set[tuple[float, float]] = set()
    normalized_rows = []
    for index, row in enumerate(rows):
        require(isinstance(row, Mapping) and set(row) == expected_fields,
                f"checkpoint candidate fields mismatch for {arm} row {index}")
        weight = _checkpoint_float(row["boundary_weight"], f"{arm}.candidate[{index}].boundary_weight")
        alpha = _checkpoint_float(row["alpha"], f"{arm}.candidate[{index}].alpha")
        require((weight, alpha) in expected_pairs, f"checkpoint candidate grid mismatch for {arm}")
        require((weight, alpha) not in observed_pairs, f"checkpoint duplicate candidate for {arm}")
        observed_pairs.add((weight, alpha))
        metrics = {
            name: _checkpoint_float(row[name], f"{arm}.candidate[{index}].{name}")
            for name in AUDIT_METRIC_NAMES
        }
        failures = []
        if metrics["macro_ancestry_dose_mae"] > f0_metrics["macro_ancestry_dose_mae"] + 1e-15:
            failures.append(GUARD_FAILURE_NAMES[0])
        if (
            metrics["false_transitions_per_cM_0.2cM"]
            > f0_metrics["false_transitions_per_cM_0.2cM"] + 1e-15
        ):
            failures.append(GUARD_FAILURE_NAMES[1])
        require(isinstance(row["guard_failures"], list) and row["guard_failures"] == failures,
                f"checkpoint candidate guard failures mismatch for {arm}")
        require(isinstance(row["guarded"], bool) and row["guarded"] is (not failures),
                f"checkpoint candidate guarded flag mismatch for {arm}")
        require(isinstance(row["selected"], bool), f"checkpoint candidate selected flag invalid for {arm}")
        key = (-metrics["boundary_f1_0.2cM"], metrics["false_transitions_per_cM_0.2cM"],
               metrics["macro_ancestry_dose_mae"], metrics["haplotype_brier"], weight, -alpha)
        normalized_rows.append((key, weight, alpha, metrics, tuple(failures), row))
    require(observed_pairs == expected_pairs, f"checkpoint candidate grid is incomplete for {arm}")
    selected_rows = [row for row in normalized_rows if row[-1]["selected"]]
    require(len(selected_rows) == 1, f"checkpoint must select exactly one candidate for {arm}")
    guarded_rows = [row for row in normalized_rows if not row[4]]
    expected_selected = min(guarded_rows if guarded_rows else normalized_rows, key=lambda row: row[0])
    selected = selected_rows[0]
    require(selected[1:3] == expected_selected[1:3], f"checkpoint selected candidate is incoherent for {arm}")

    fit_alpha = _checkpoint_float(fit["alpha"], f"{arm}.alpha")
    fit_weight = _checkpoint_float(fit["boundary_weight"], f"{arm}.boundary_weight")
    require((fit_weight, fit_alpha) == selected[1:3], f"checkpoint fit hyperparameters mismatch for {arm}")
    fit_metrics = {
        "boundary_f1_0.2cM": _checkpoint_float(fit["cv_boundary_f1_0.2cM"], f"{arm}.cv_boundary_f1"),
        "false_transitions_per_cM_0.2cM": _checkpoint_float(
            fit["cv_false_transitions_per_cM_0.2cM"], f"{arm}.cv_false_transitions"
        ),
        "macro_ancestry_dose_mae": _checkpoint_float(
            fit["cv_macro_ancestry_dose_mae"], f"{arm}.cv_macro_ancestry_dose_mae"
        ),
        "haplotype_brier": _checkpoint_float(fit["cv_brier"], f"{arm}.cv_brier"),
    }
    require(fit_metrics == selected[3], f"checkpoint selected metrics mismatch for {arm}")
    require(isinstance(fit["guard_failures"], list) and tuple(fit["guard_failures"]) == selected[4],
            f"checkpoint selected guard failures mismatch for {arm}")
    any_guarded = bool(guarded_rows)
    require(isinstance(fit["guarded"], bool) and fit["guarded"] is any_guarded,
            f"checkpoint guarded summary mismatch for {arm}")
    expected_status = "GUARDED_CONFIG" if any_guarded else "NO_GUARDED_CONFIG"
    require(fit["selection_status"] == expected_status,
            f"checkpoint selection status mismatch for {arm}")

    total_hashes = fit["sufficient_stats_sha256"]
    expected_total_keys = {format(float(weight), ".17g") for weight in core.EXPECTED_BOUNDARY_WEIGHTS}
    require(isinstance(total_hashes, Mapping) and set(total_hashes) == expected_total_keys,
            f"checkpoint total sufficient-stat hashes mismatch for {arm}")
    require(all(_valid_sha256(value) for value in total_hashes.values()),
            f"checkpoint total sufficient-stat SHA-256 invalid for {arm}")
    fold_hashes = fit["fold_stats_sha256"]
    expected_fold_keys = {
        f"fold{fold}:{format(float(weight), '.17g')}"
        for fold in range(3) for weight in core.EXPECTED_BOUNDARY_WEIGHTS
    }
    require(isinstance(fold_hashes, Mapping) and set(fold_hashes) == expected_fold_keys,
            f"checkpoint fold sufficient-stat hashes mismatch for {arm}")
    require(all(_valid_sha256(value) for value in fold_hashes.values()),
            f"checkpoint fold sufficient-stat SHA-256 invalid for {arm}")


def _write_prediction_checkpoint(
    outdir: Path,
    artifact: PredictionArtifact,
    fitted: FittedArm | None,
    context_sha256: str,
) -> dict[str, Any]:
    token = artifact.arm
    prediction_path = outdir / f"pilot.{token}.predictions.npy"
    checkpoint_path = outdir / f"pilot.{token}.checkpoint.json"
    require(not prediction_path.exists() and not checkpoint_path.exists(), f"refusing to overwrite checkpoint for {token}")
    if fitted is not None:
        require(fitted.arm == token, f"fitted/checkpoint arm mismatch for {token}")
    fit_payload = None if fitted is None else {
        "alpha": fitted.alpha,
        "boundary_weight": fitted.boundary_weight,
        "cv_boundary_f1_0.2cM": fitted.cv_boundary_f1,
        "cv_false_transitions_per_cM_0.2cM": fitted.cv_false_transitions_per_cm,
        "cv_macro_ancestry_dose_mae": fitted.cv_macro_ancestry_dose_mae,
        "cv_brier": fitted.cv_brier,
        "guarded": fitted.guarded,
        "selection_status": fitted.selection_status,
        "feature_count": fitted.feature_count,
        "f0_cv_metrics": dict(fitted.f0_cv_metrics),
        "candidate_table": list(fitted.candidate_table),
        "guard_failures": list(fitted.guard_failures),
        "sufficient_stats_sha256": dict(fitted.sufficient_stats_sha256),
        "fold_stats_sha256": dict(fitted.fold_stats_sha256),
    }
    _validate_fit_checkpoint(token, fit_payload)
    stacked = np.stack(artifact.arrays, axis=0).astype(np.float64, copy=False)
    _atomic_npy_fsync(prediction_path, stacked)
    checkpoint = {
        "schema_version": core.SCHEMA_VERSION,
        "status": "COMPLETE_FSYNC",
        "context_sha256": context_sha256,
        "root_name": artifact.root_name,
        "arm": artifact.arm,
        "replicate": artifact.replicate,
        "sample_ids": list(artifact.sample_ids),
        "shape": list(stacked.shape),
        "dtype": str(stacked.dtype),
        "prediction_semantic_sha256": artifact.sha256,
        "prediction_file": prediction_path.name,
        "prediction_file_sha256": core.sha256_file(prediction_path),
        "fit": fit_payload,
    }
    _atomic_json_fsync(checkpoint_path, checkpoint)
    checkpoint["checkpoint_file"] = checkpoint_path.name
    checkpoint["checkpoint_file_sha256"] = core.sha256_file(checkpoint_path)
    return checkpoint


def _load_prediction_checkpoint(outdir: Path, arm: str, context_sha256: str) -> tuple[PredictionArtifact, dict[str, Any]]:
    checkpoint_path = outdir / f"pilot.{arm}.checkpoint.json"
    require(checkpoint_path.is_file(), f"checkpoint missing for {arm}")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    require(checkpoint.get("status") == "COMPLETE_FSYNC" and checkpoint.get("context_sha256") == context_sha256,
            f"checkpoint context/status mismatch for {arm}")
    require(checkpoint.get("arm") == arm and checkpoint.get("replicate") is None,
            f"checkpoint identity mismatch for {arm}")
    require("fit" in checkpoint, f"checkpoint fit field missing for {arm}")
    _validate_fit_checkpoint(arm, checkpoint["fit"])
    prediction_path = outdir / str(checkpoint.get("prediction_file"))
    require(prediction_path.is_file() and core.sha256_file(prediction_path) == checkpoint.get("prediction_file_sha256"),
            f"prediction file SHA-256 mismatch for {arm}")
    stacked = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    require(list(stacked.shape) == checkpoint.get("shape") and str(stacked.dtype) == checkpoint.get("dtype"),
            f"prediction shape/dtype mismatch for {arm}")
    arrays = tuple(np.asarray(stacked[index]) for index in range(stacked.shape[0]))
    semantic = _prediction_sha256(str(checkpoint.get("root_name")), checkpoint.get("sample_ids", []), arrays)
    require(semantic == checkpoint.get("prediction_semantic_sha256"), f"prediction semantic SHA-256 mismatch for {arm}")
    artifact = PredictionArtifact(
        str(checkpoint["root_name"]), tuple(checkpoint["sample_ids"]), arm, None, arrays, semantic,
    )
    return artifact, checkpoint


def _package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "UNKNOWN"))


def _verify_git_provenance(expected_commit: str, relevant_paths: Sequence[Path]) -> str:
    require(len(expected_commit) == 40 and all(char in "0123456789abcdef" for char in expected_commit),
            "expected git commit must be a lowercase 40-character SHA-1")
    repo = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True,
    ).stdout.strip()).resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(head == expected_commit, f"HEAD {head} differs from explicitly expected commit {expected_commit}")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", expected_commit, head], check=False)
    require(ancestor.returncode == 0, "expected commit is not contained in HEAD")
    relative = []
    for path in relevant_paths:
        resolved = path.resolve()
        require(resolved == repo or repo in resolved.parents, f"relevant provenance path lies outside repository: {path}")
        relative.append(str(resolved.relative_to(repo)))
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *relative], check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    require(tracked.returncode == 0, "at least one relevant code/contract file is not tracked by HEAD")
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *relative], check=False)
    require(clean.returncode == 0, "relevant code/contract files are dirty relative to HEAD")
    return head


def _runtime_environment(container_digest: str | None) -> dict[str, Any]:
    safe_environment = {
        key: os.environ[key]
        for key in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER", "CONTAINER_IMAGE", "HOSTNAME")
        if key in os.environ
    }
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tskit": _package_version("tskit"),
        "container_digest": container_digest,
        "safe_environment": safe_environment,
    }


def _base_provenance(
    mode: str,
    contract_path: Path,
    *,
    verified_git_commit: str | None = None,
    command: Sequence[str] | None = None,
    container_digest: str | None = None,
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    core_path = Path(core.__file__).resolve()
    return {
        "schema_version": core.SCHEMA_VERSION,
        "experiment_id": core.EXPERIMENT_ID,
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": verified_git_commit,
        "git_commit_verification": "HEAD_EXACT_RELEVANT_FILES_CLEAN_AND_TRACKED" if verified_git_commit else "NOT_DECLARED_UNVERIFIED",
        "command": list(command if command is not None else sys.argv),
        "command_shell_rendering": shlex.join(command if command is not None else sys.argv),
        "runtime": _runtime_environment(container_digest),
        "code_sha256": {"runner": core.sha256_file(runner_path), "core": core.sha256_file(core_path)},
        "contract": {"path": str(contract_path), "sha256": core.sha256_file(contract_path)},
        "partition_policy": "DEV roots 20260817/20260818 only; VALID and TEST are forbidden",
        "prediction_protocol": "truth-blind FeatureDataset -> immutable PredictionArtifact SHA-256 -> TruthBundle scorer",
    }


def run_known_answer(contract_path: Path, outdir: Path) -> dict[str, Any]:
    contract = core.load_contract(contract_path)
    core_answers = core.run_known_answer_selftest()
    synthetic = core.run_synthetic_end_to_end()
    require(core_answers["synthetic_end_to_end"] == "PASS" and synthetic["status"] == "PASS",
            "known-answer selftest failed")
    probabilities = np.full((3, 2, 2, 3), 1.0 / 3.0, dtype=np.float64)
    prediction_root = type("KnownAnswerFeatureRoot", (), {
        "name": "synthetic_root", "samples": ("EVAL0", "EVAL1"),
        "flare": type("KnownAnswerFlare", (), {"probabilities": probabilities})(),
    })()
    prediction_artifact = prepare_predictions(prediction_root, None, "F0", None)
    require(
        prediction_artifact.sha256
        == _prediction_sha256(prediction_artifact.root_name, prediction_artifact.sample_ids, prediction_artifact.arrays),
        "known-answer prediction digest failed",
    )
    summary = {
        "schema_version": core.SCHEMA_VERSION,
        "experiment_id": contract.experiment_id,
        "status": "PASS",
        "mode": "KNOWN_ANSWER_NO_REAL_DATA",
        "scientific_decision": "NOT_EVALUATED",
        "known_answers": core_answers,
        "feature_dimensions": {"C": 59, "L": 99, "D": 171, "H": 243},
        "global_count_scoring": True,
        "prediction_hash_before_truth_scoring": True,
        "synthetic_prediction_sha256": prediction_artifact.sha256,
    }
    metrics = [{
        "direction": "synthetic_train_test",
        "arm": "synthetic_corrector",
        "boundary_f1_0.2cM": synthetic["boundary_f1_0.2cM"],
        "baseline_haplotype_brier": synthetic["baseline_haplotype_brier"],
        "corrected_haplotype_brier": synthetic["corrected_haplotype_brier"],
        "selected_alpha": synthetic["selected_alpha"],
        "selected_boundary_weight": synthetic["selected_boundary_weight"],
    }]
    individuals = [{"direction": "synthetic_train_test", "arm": "synthetic_corrector",
                    "sample_id": f"EVAL{index}", "status": "PASS"} for index in range(2)]
    provenance = _base_provenance("known-answer", contract_path)
    provenance["real_input_access"] = False
    return _write_output_bundle(outdir, summary, metrics, individuals, provenance)


def _reject_forbidden_partitions(paths: Iterable[Path]) -> None:
    forbidden = {"valid", "validation", "test"}
    for path in paths:
        tokens = {token.lower() for part in path.parts for token in part.replace("-", "_").replace(".", "_").split("_")}
        require(not tokens.intersection(forbidden), f"VALID/TEST path is forbidden: {path}")


def _paths_from_args(args: argparse.Namespace) -> dict[str, RootPaths]:
    roots = {}
    for root in core.ROOTS:
        roots[root] = RootPaths(**{
            key: getattr(args, f"{root}_{key}")
            for key in ("sites", "target", "tree", "pools", "truth", "flare_vcf", "flare_audit")
        })
    return roots


def _authenticate_execution(args: argparse.Namespace) -> tuple[core.OrderedLinearContract, dict[str, RootPaths], dict[str, str]]:
    contract = core.load_contract(args.contract)
    roots = _paths_from_args(args)
    all_paths = [args.contract, args.genetic_map, Path(__file__).resolve(), Path(core.__file__).resolve()]
    all_paths.extend(path for root in roots.values() for path in root.as_dict().values())
    _reject_forbidden_partitions(all_paths)
    require(core.sha256_file(args.contract) == args.expected_contract_sha256, "contract SHA-256 mismatch")
    require(core.sha256_file(Path(__file__).resolve()) == args.expected_runner_sha256, "runner SHA-256 mismatch")
    require(core.sha256_file(Path(core.__file__).resolve()) == args.expected_core_sha256, "core SHA-256 mismatch")
    hashes = core.authenticate_frozen_run_inputs(
        args.genetic_map, {name: root.as_dict() for name, root in roots.items()},
    )
    return contract, roots, hashes


def _authenticate_exact_subset(paths: Mapping[str, Path]) -> dict[str, str]:
    observed = {name: core.sha256_file(path) for name, path in paths.items()}
    expected = core.EXPECTED_RUN_INPUT_SHA256
    require(set(observed).issubset(expected), "authentication subset contains an unknown input role")
    mismatches = [name for name, digest in observed.items() if digest != expected[name]]
    require(not mismatches, f"frozen input SHA-256 mismatch: {mismatches}")
    return observed


def _verify_code_contract_and_commit(args: argparse.Namespace) -> tuple[core.OrderedLinearContract, str, dict[str, str]]:
    contract = core.load_contract(args.contract)
    code_hashes = {
        "contract": core.sha256_file(args.contract),
        "runner": core.sha256_file(Path(__file__).resolve()),
        "core": core.sha256_file(Path(core.__file__).resolve()),
    }
    require(code_hashes["contract"] == args.expected_contract_sha256, "contract SHA-256 mismatch")
    require(code_hashes["runner"] == args.expected_runner_sha256, "runner SHA-256 mismatch")
    require(code_hashes["core"] == args.expected_core_sha256, "core SHA-256 mismatch")
    commit = _verify_git_provenance(
        args.expected_git_commit, (Path(__file__).resolve(), Path(core.__file__).resolve(), args.contract),
    )
    require(bool(args.container_digest.strip()), "container digest/image identity is required")
    return contract, commit, code_hashes


def _pilot_paths_from_args(args: argparse.Namespace) -> tuple[TrainingPaths, FeaturePaths]:
    train = TrainingPaths(**{
        key: getattr(args, f"train_root17_{key}")
        for key in ("sites", "target", "tree", "pools", "truth", "flare_vcf", "flare_audit")
    })
    evaluation = FeaturePaths(**{
        key: getattr(args, f"eval_root18_{key}")
        for key in ("sites", "target", "tree", "pools", "flare_vcf", "flare_audit")
    })
    return train, evaluation


def _pilot_input_hashes(genetic_map: Path, train: TrainingPaths, evaluation: FeaturePaths) -> dict[str, str]:
    roles: dict[str, Path] = {"genetic_map": genetic_map}
    roles.update({f"root17.{key}": value for key, value in train.as_dict().items()})
    roles.update({f"root18.{key}": value for key, value in evaluation.as_dict().items()})
    _reject_forbidden_partitions(roles.values())
    return _authenticate_exact_subset(roles)


def _scan_flare_shape(path: Path) -> tuple[int, int]:
    markers = 0
    samples: int | None = None
    with core.open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = max(0, len(line.rstrip("\n").split("\t")) - 9)
            elif not line.startswith("#") and line.strip():
                markers += 1
    require(samples is not None and samples > 0, f"cannot identify FLARE samples in {path}")
    return markers, samples


def dry_run_estimate(args: argparse.Namespace) -> dict[str, Any]:
    contract, roots, hashes = _authenticate_execution(args)
    shapes = {name: _scan_flare_shape(paths.flare_vcf) for name, paths in roots.items()}
    for name, (markers, _samples) in shapes.items():
        require(markers == 79791, f"{name} exact FLARE marker count drifted")
    largest_features = 243
    max_markers = max(value[0] for value in shapes.values())
    max_samples = max(value[1] for value in shapes.values())
    one_h_float64 = max_markers * 2 * largest_features * 8
    cache_float32 = max_markers * 2 * largest_features * 4 * max_samples
    rng = np.random.default_rng(3101)
    benchmark_x = rng.normal(size=(2000, largest_features)).astype(np.float32)
    benchmark_y = rng.normal(scale=0.1, size=(2000, 3))
    benchmark_weights = rng.uniform(0.1, 1.0, size=2000)
    started = time.perf_counter()
    benchmark_stats = sample_stats(benchmark_x, benchmark_y, benchmark_weights)
    fit_from_stats(benchmark_stats, 0.1)
    benchmark_seconds = time.perf_counter() - started
    return {
        "schema_version": core.SCHEMA_VERSION,
        "experiment_id": contract.experiment_id,
        "status": "DRY_RUN_AUTHENTICATED_NO_FIT",
        "authenticated_hashes": len(hashes) + 3,
        "root_shapes": {name: {"markers": value[0], "individuals": value[1]} for name, value in shapes.items()},
        "feature_dimensions": {"C": 59, "L": 99, "D": 171, "H": 243},
        "resource_estimate": {
            "largest_one_individual_H_float64_GiB": one_h_float64 / 2**30,
            "peak_transient_float32_cache_GiB": cache_float32 / 2**30,
            "recommended_RAM_GiB": "6-12",
            "recommended_free_local_disk_GiB": math.ceil(cache_float32 / 2**30 * 1.5),
            "single_process_wall_time": "approximately 24-96 hours; benchmark on the target VM before authorization",
            "basis": "136 fitted arm/direction combinations, each with 18 guarded grouped-CV candidates; sequential arm cache",
            "local_synthetic_2000x243_stats_and_solve_seconds": benchmark_seconds,
            "benchmark_is_not_a_real_data_runtime_claim": True,
        },
        "real_execution_started": False,
    }


PILOT_FITTED_ARMS = tuple(PILOT_FEATURE_COUNTS)
PILOT_OUTPUT_ARMS = ("F0", *PILOT_FITTED_ARMS)


def fit_predict_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Train on root17 truth and durably predict root18 without accepting its truth."""
    workers, threadpool_runtime = _validate_worker_runtime(getattr(args, "workers", 1))
    contract, commit, code_hashes = _verify_code_contract_and_commit(args)
    train_paths, evaluation_paths = _pilot_paths_from_args(args)
    input_hashes = _pilot_input_hashes(args.genetic_map, train_paths, evaluation_paths)
    context = {
        "protocol": "M31_PILOT_ROOT17_TO_ROOT18_FIT_PREDICT_V1",
        "train_root": "root17",
        "evaluation_root": "root18",
        "fitted_arms": list(PILOT_FITTED_ARMS),
        "output_arms": list(PILOT_OUTPUT_ARMS),
        "input_sha256": input_hashes,
        "code_sha256": code_hashes,
        "git_commit": commit,
        "container_digest": args.container_digest,
        "workers": workers,
        "thread_limit_environment": {name: os.environ.get(name) for name in THREAD_LIMIT_ENV},
        "threadpool_runtime": threadpool_runtime,
    }
    context_sha256 = _payload_sha256(context)
    outdir = args.outdir
    expected_names = {
        *(f"pilot.{arm}.predictions.npy" for arm in PILOT_OUTPUT_ARMS),
        *(f"pilot.{arm}.checkpoint.json" for arm in PILOT_OUTPUT_ARMS),
        "pilot.fit_predict.provenance.json",
        "pilot.fit_predict.manifest.json",
    }
    if outdir.exists():
        require(outdir.is_dir(), f"pilot outdir is not a directory: {outdir}")
        if not args.resume:
            require(not any(outdir.iterdir()), f"pilot outdir is not empty without --resume: {outdir}")
        unexpected = {path.name for path in outdir.iterdir()} - expected_names
        require(not unexpected, f"unexpected files in pilot checkpoint directory: {sorted(unexpected)}")
    else:
        outdir.mkdir(parents=True)
        _fsync_directory(outdir.parent)
    manifest_path = outdir / "pilot.fit_predict.manifest.json"
    if manifest_path.exists():
        require(args.resume, "completed pilot manifest already exists; use --resume for verification")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("context_sha256") == context_sha256 and manifest.get("status") == "COMPLETE_FSYNC",
                "completed pilot manifest context/status mismatch")
        for arm in PILOT_OUTPUT_ARMS:
            _artifact, checkpoint = _load_prediction_checkpoint(outdir, arm, context_sha256)
            recorded = manifest.get("checkpoints", {}).get(arm, {})
            require(recorded.get("checkpoint_file_sha256") == core.sha256_file(outdir / f"pilot.{arm}.checkpoint.json"),
                    f"manifest/checkpoint SHA-256 mismatch for {arm}")
        return {"status": "RESUMED_COMPLETE_VERIFIED", "manifest": str(manifest_path),
                "manifest_sha256": core.sha256_file(manifest_path)}

    genetic_map = core.load_genetic_map(args.genetic_map)
    train_features = load_feature_root("root17", 20260817, train_paths.feature_paths(), genetic_map, 79791)
    train_truth = load_truth_bundle(train_paths, train_features)
    evaluation_features = load_feature_root("root18", 20260818, evaluation_paths, genetic_map, 79791)
    require(train_features.samples == evaluation_features.samples, "pilot root17/root18 sample order differs")
    checkpoints: dict[str, Any] = {}
    for arm in PILOT_OUTPUT_ARMS:
        checkpoint_path = outdir / f"pilot.{arm}.checkpoint.json"
        prediction_path = outdir / f"pilot.{arm}.predictions.npy"
        if checkpoint_path.exists():
            require(args.resume, f"checkpoint already exists for {arm} without --resume")
            _artifact, checkpoint = _load_prediction_checkpoint(outdir, arm, context_sha256)
            checkpoint["checkpoint_file"] = checkpoint_path.name
            checkpoint["checkpoint_file_sha256"] = core.sha256_file(checkpoint_path)
            checkpoints[arm] = checkpoint
            continue
        require(not prediction_path.exists(), f"orphan prediction without checkpoint for {arm}")
        fitted = None if arm == "F0" else fit_arm_streaming(
            train_features, train_truth, arm, None,
            contract.alphas, contract.boundary_weights, cv_seed=train_features.seed, workers=workers,
        )
        artifact = prepare_predictions(evaluation_features, fitted, arm, None)
        checkpoints[arm] = _write_prediction_checkpoint(outdir, artifact, fitted, context_sha256)

    provenance = _base_provenance(
        "pilot-fit-predict", args.contract, verified_git_commit=commit,
        command=getattr(args, "invocation", None), container_digest=args.container_digest,
    )
    provenance.update({
        "input_sha256": input_hashes,
        "context_sha256": context_sha256,
        "evaluation_truth_was_accepted_or_read": False,
        "scientific_decision": "NO_SCIENTIFIC_DECISION",
        "workers": workers,
        "thread_limit_environment": {name: os.environ.get(name) for name in THREAD_LIMIT_ENV},
        "threadpool_runtime": threadpool_runtime,
    })
    provenance_path = outdir / "pilot.fit_predict.provenance.json"
    _atomic_json_fsync(provenance_path, provenance)
    manifest = {
        "schema_version": core.SCHEMA_VERSION,
        "status": "COMPLETE_FSYNC",
        "stage": "M31_PILOT_ROOT17_TO_ROOT18_FIT_PREDICT",
        "label": "NO_SCIENTIFIC_DECISION",
        "context": context,
        "context_sha256": context_sha256,
        "checkpoints": checkpoints,
        "provenance_file": provenance_path.name,
        "provenance_sha256": core.sha256_file(provenance_path),
        "producer_process_boundary": "process must terminate before score-pilot may mount root18 truth",
    }
    _atomic_json_fsync(manifest_path, manifest)
    return {"status": "COMPLETE_FSYNC_PROCESS_EXIT_REQUIRED", "manifest": str(manifest_path),
            "manifest_sha256": core.sha256_file(manifest_path)}


def score_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Score a completed pilot manifest in a separate invocation that may mount root18 truth."""
    _contract, commit, code_hashes = _verify_code_contract_and_commit(args)
    require(core.sha256_file(args.prediction_manifest) == args.expected_prediction_manifest_sha256,
            "prediction manifest SHA-256 mismatch")
    producer = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    require(producer.get("status") == "COMPLETE_FSYNC" and producer.get("stage") == "M31_PILOT_ROOT17_TO_ROOT18_FIT_PREDICT",
            "prediction manifest is not a completed pilot producer")
    require(set(producer.get("checkpoints", {})) == set(PILOT_OUTPUT_ARMS),
            "pilot producer must contain exactly F0/C/L/D and no H/shams")
    require(producer.get("context_sha256") == _payload_sha256(producer.get("context", {})),
            "pilot producer context SHA-256 mismatch")
    require(producer.get("context", {}).get("code_sha256") == code_hashes,
            "producer/scorer code or contract hashes differ")
    require(producer.get("context", {}).get("git_commit") == commit,
            "producer/scorer git commits differ")
    evaluation_paths = FeaturePaths(**{
        key: getattr(args, f"eval_root18_{key}")
        for key in ("sites", "target", "tree", "pools", "flare_vcf", "flare_audit")
    })
    score_roles: dict[str, Path] = {"genetic_map": args.genetic_map, "root18.truth": args.eval_root18_truth}
    score_roles.update({f"root18.{key}": value for key, value in evaluation_paths.as_dict().items()})
    _reject_forbidden_partitions(score_roles.values())
    score_hashes = _authenticate_exact_subset(score_roles)
    producer_inputs = producer.get("context", {}).get("input_sha256", {})
    for role, digest in score_hashes.items():
        if role != "root18.truth":
            require(producer_inputs.get(role) == digest, f"producer/scorer input hash differs for {role}")
    # Authenticate every durable prediction/checkpoint before mounting truth.
    source_dir = args.prediction_manifest.parent
    authenticated_predictions: dict[str, tuple[PredictionArtifact, dict[str, Any]]] = {}
    for arm in PILOT_OUTPUT_ARMS:
        artifact, checkpoint = _load_prediction_checkpoint(source_dir, arm, str(producer["context_sha256"]))
        recorded = producer.get("checkpoints", {}).get(arm, {})
        checkpoint_path = source_dir / f"pilot.{arm}.checkpoint.json"
        require(recorded.get("checkpoint_file_sha256") == core.sha256_file(checkpoint_path),
                f"producer manifest/checkpoint SHA-256 mismatch for {arm}")
        authenticated_predictions[arm] = (artifact, checkpoint)
    genetic_map = core.load_genetic_map(args.genetic_map)
    features = load_feature_root("root18", 20260818, evaluation_paths, genetic_map, 79791)
    truth_paths = TrainingPaths(
        evaluation_paths.sites, evaluation_paths.target, evaluation_paths.tree, evaluation_paths.pools,
        args.eval_root18_truth, evaluation_paths.flare_vcf, evaluation_paths.flare_audit,
    )
    truth = load_truth_bundle(truth_paths, features)
    metric_rows: list[dict[str, Any]] = []
    individual_rows: list[dict[str, Any]] = []
    prediction_hashes = {}
    for arm in PILOT_OUTPUT_ARMS:
        artifact, checkpoint = authenticated_predictions[arm]
        summary, individuals, counts = score_prediction_artifact(features, truth, artifact)
        bootstrap = bootstrap_counts(counts)
        fit = checkpoint.get("fit") or {}
        row = {
            "direction": "train_root17_predict_root18_pilot",
            "arm": arm,
            "selected_alpha": fit.get("alpha", ""),
            "selected_boundary_weight": fit.get("boundary_weight", ""),
            "inner_cv_guarded": fit.get("guarded", ""),
            "inner_cv_selection_status": fit.get("selection_status", ""),
            "feature_count": fit.get("feature_count", 0),
            **summary,
            "bootstrap_json": json.dumps(json_safe(bootstrap), sort_keys=True, separators=(",", ":")),
        }
        metric_rows.append(row)
        individual_rows.extend({"direction": row["direction"], "arm": arm, **individual} for individual in individuals)
        prediction_hashes[arm] = artifact.sha256
    summary = {
        "schema_version": core.SCHEMA_VERSION,
        "stage": "M31_PILOT_ROOT17_TO_ROOT18_SCORE",
        "label": "NO_SCIENTIFIC_DECISION",
        "direction": "train_root17_predict_root18",
        "arms": list(PILOT_OUTPUT_ARMS),
        "shams": 0,
        "H_included": False,
        "prediction_manifest_sha256": args.expected_prediction_manifest_sha256,
        "prediction_semantic_sha256": prediction_hashes,
        "claim_scope": "runtime/feasibility pilot only; no scientific decision or validation claim",
    }
    provenance = _base_provenance(
        "score-pilot", args.contract, verified_git_commit=commit,
        command=getattr(args, "invocation", None), container_digest=args.container_digest,
    )
    provenance.update({"input_sha256": score_hashes, "producer_manifest_sha256": args.expected_prediction_manifest_sha256})
    manifest = _write_output_bundle(args.outdir, summary, metric_rows, individual_rows, provenance)
    return {"status": "COMPLETE_NO_SCIENTIFIC_DECISION", "manifest": manifest}


def benchmark_real_sample(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize C/L/D/H for one authenticated real sample; never load truth or fit."""
    require(not args.output.exists(), f"refusing to overwrite benchmark report: {args.output}")
    _contract, commit, code_hashes = _verify_code_contract_and_commit(args)
    feature_paths = FeaturePaths(**{
        key: getattr(args, f"benchmark_{key}")
        for key in ("sites", "target", "tree", "pools", "flare_vcf", "flare_audit")
    })
    roles = {"genetic_map": args.genetic_map}
    roles.update({f"{args.root}.{key}": value for key, value in feature_paths.as_dict().items()})
    _reject_forbidden_partitions(roles.values())
    hashes = _authenticate_exact_subset(roles)
    seed = 20260817 if args.root == "root17" else 20260818
    genetic_map = core.load_genetic_map(args.genetic_map)
    features = load_feature_root(args.root, seed, feature_paths, genetic_map, 79791)
    if args.sample_id is None:
        sample_index = 0
    else:
        require(args.sample_id in features.samples, f"benchmark sample is absent: {args.sample_id}")
        sample_index = features.samples.index(args.sample_id)
    rows = []
    import resource
    for arm in FITTED_ARMS:
        started = time.perf_counter()
        matrix = features.features(sample_index, arm, None)
        elapsed = time.perf_counter() - started
        rows.append({
            "arm": arm,
            "feature_count": matrix.shape[1],
            "rows": matrix.shape[0],
            "dtype": str(matrix.dtype),
            "wall_seconds": elapsed,
            "matrix_bytes": matrix.nbytes,
            "matrix_sha256": hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest(),
            "max_rss_kib_after": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        })
        del matrix
    report = {
        "schema_version": core.SCHEMA_VERSION,
        "status": "COMPLETE_TRUTH_BLIND_NO_FIT",
        "root": args.root,
        "sample_id": features.samples[sample_index],
        "input_sha256": hashes,
        "code_sha256": code_hashes,
        "git_commit": commit,
        "container_digest": args.container_digest,
        "arms": rows,
        "truth_accessed": False,
        "fit_performed": False,
    }
    _atomic_json_fsync(args.output, report)
    return report


def execute_real(_args: argparse.Namespace) -> dict[str, Any]:
    raise RunnerError("POST_REQUIRED_FULL_RUN_BLOCKED: use pilot/fit-predict, then a separate score-pilot process")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    known = subparsers.add_parser("known-answer", help="run synthetic known answers only")
    known.add_argument("--contract", type=Path, required=True)
    known.add_argument("--outdir", type=Path, required=True)
    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--contract", type=Path, required=True)
    dry.add_argument("--genetic-map", type=Path, required=True)
    dry.add_argument("--expected-contract-sha256", required=True)
    dry.add_argument("--expected-runner-sha256", required=True)
    dry.add_argument("--expected-core-sha256", required=True)
    for root in core.ROOTS:
        for key in ("sites", "target", "tree", "pools", "truth", "flare-vcf", "flare-audit"):
            dry.add_argument(f"--{root}-{key}", dest=f"{root}_{key.replace('-', '_')}", type=Path, required=True)

    def add_verified_runtime(item: argparse.ArgumentParser) -> None:
        item.add_argument("--contract", type=Path, required=True)
        item.add_argument("--genetic-map", type=Path, required=True)
        item.add_argument("--expected-contract-sha256", required=True)
        item.add_argument("--expected-runner-sha256", required=True)
        item.add_argument("--expected-core-sha256", required=True)
        item.add_argument("--expected-git-commit", required=True)
        item.add_argument("--container-digest", required=True)

    for command in ("pilot", "fit-predict"):
        item = subparsers.add_parser(command, help="root17 fit -> root18 truth-blind durable predictions")
        add_verified_runtime(item)
        item.add_argument("--outdir", type=Path, required=True)
        item.add_argument("--resume", action="store_true")
        item.add_argument("--workers", type=int, default=1,
                          help="individual-only worker threads; >1 requires BLAS/OMP/MKL/NUMEXPR thread limits=1")
        for key in ("sites", "target", "tree", "pools", "truth", "flare-vcf", "flare-audit"):
            item.add_argument(f"--train-root17-{key}", dest=f"train_root17_{key.replace('-', '_')}", type=Path, required=True)
        for key in ("sites", "target", "tree", "pools", "flare-vcf", "flare-audit"):
            item.add_argument(f"--eval-root18-{key}", dest=f"eval_root18_{key.replace('-', '_')}", type=Path, required=True)

    score = subparsers.add_parser("score-pilot", help="separate process that mounts root18 truth and scores pilot artifacts")
    add_verified_runtime(score)
    score.add_argument("--prediction-manifest", type=Path, required=True)
    score.add_argument("--expected-prediction-manifest-sha256", required=True)
    score.add_argument("--outdir", type=Path, required=True)
    for key in ("sites", "target", "tree", "pools", "flare-vcf", "flare-audit"):
        score.add_argument(f"--eval-root18-{key}", dest=f"eval_root18_{key.replace('-', '_')}", type=Path, required=True)
    score.add_argument("--eval-root18-truth", type=Path, required=True)

    benchmark = subparsers.add_parser("benchmark-sample", help="truth-blind C/L/D/H materialization benchmark; no fit")
    add_verified_runtime(benchmark)
    benchmark.add_argument("--root", choices=core.ROOTS, required=True)
    benchmark.add_argument("--sample-id")
    benchmark.add_argument("--output", type=Path, required=True)
    for key in ("sites", "target", "tree", "pools", "flare-vcf", "flare-audit"):
        benchmark.add_argument(f"--benchmark-{key}", dest=f"benchmark_{key.replace('-', '_')}", type=Path, required=True)

    subparsers.add_parser("run", help="blocked until a separate full-run POST authorizes it")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    invocation = list(argv) if argv is not None else list(sys.argv)
    args = build_parser().parse_args(argv)
    args.invocation = invocation
    if args.command == "known-answer":
        report = {"status": "PASS", "manifest": run_known_answer(args.contract, args.outdir)}
    elif args.command == "dry-run":
        report = dry_run_estimate(args)
    elif args.command in {"pilot", "fit-predict"}:
        report = fit_predict_pilot(args)
    elif args.command == "score-pilot":
        report = score_pilot(args)
    elif args.command == "benchmark-sample":
        report = benchmark_real_sample(args)
    else:
        report = execute_real(args)
    sys.stdout.write(json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


def fail_closed_main(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (RunnerError, core.ContractError, ValueError) as error:
        sys.stderr.write(f"M31_FAIL_CLOSED: {error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(fail_closed_main())
