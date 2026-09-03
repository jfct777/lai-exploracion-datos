#!/usr/bin/env python3
"""Train the M36 CORA-Set continuous external-IBD experiment.

This executable accepts only materialized event tables.  Real training is
fail-closed on a materialization receipt; ``--synthetic-smoke`` is the sole
route that generates examples internally.  It never derives an outcome from
M14/M16.5 or from the rare event channel.
"""

from __future__ import annotations

import hashlib
import json
import math
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:  # Kept lazy so schema/receipt checks run outside the pinned ML image.
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - local contract-only environment
    np = None

# Nextflow may execute the staged trainer from its task wrapper directory while
# the companion model module remains in the task working directory.  Keep both
# explicit, bounded import roots so local and staged execution resolve the same
# implementation.
for import_root in (Path(__file__).resolve().parent, Path.cwd()):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from m36_cora_models import CoraModelSpec, available_specs, build_pair_regressor
from m36_cora_set import (
    ContractError, REAL_EVENT_STATE_COLUMNS, as_float, component_folds, event_tokens, read_tsv, validate_inputs,
)


NUMERIC_TOKEN_FIELDS = (
    "genotype_dosage", "mac_scaled", "callability", "cm", "common_copying_context",
    "common_copying_context_available", "mutation_context_available", "is_ac2_het", "is_ac2_homalt", "is_mac3_10",
)
COVARIATE_FIELDS = ("rare_burden", "rare_callability", "Q_AFR", "Q_EUR", "Q_NAM", "Q_EAS")
CATEGORICAL_COVARIATE_FIELDS = ("cohort",)
ARMS = ("rare_enabled", "carrier_permuted", "geometry_only", "baseline_only")


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - pinned-runtime concern
        raise ContractError("M36 training requires a pinned PyTorch runtime") from error
    return torch


def require_numpy():
    if np is None:
        raise ContractError("M36 training requires the pinned NumPy/PyTorch runtime")
    return np


def validate_materialization_receipt(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ContractError("real M36 training requires a materialization receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "stage": "M36_CORA_MATERIALIZE",
        "synthetic": False,
        "feature_schema": "m36_factorized_sparse_v1",
        "external_target_schema": "m36_external_common_pairs_log1p_v3_pair_total",
    }
    if receipt.get("status") not in {"MATERIALIZED_PASS", "PUBLISHED_PASS"}:
        raise ContractError("materialization receipt is not a successful chainable or published materialization")
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(f"materialization receipt drift at {key}")
    descriptors = receipt.get("input_descriptors")
    expected_inputs = {"loci", "carriers", "missing", "covariates", "components", "targets"}
    if not isinstance(descriptors, dict) or set(descriptors) != expected_inputs:
        raise ContractError("materialization receipt must bind exactly events/covariates/components/targets")
    for name, descriptor in descriptors.items():
        if not isinstance(descriptor, dict) or not all(
            isinstance(descriptor.get(field), str) and descriptor[field]
            for field in ("uri", "generation", "sha256")
        ):
            raise ContractError(f"materialization receipt descriptor is incomplete: {name}")
    return receipt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def arm_events(events: list[dict[str, str]], arm: str, sample_ids: list[str], seed: int,
               missing_pairs: set[tuple[str, str]] | None = None, factorized: bool = False) -> list[dict[str, str]]:
    if arm == "rare_enabled" or arm == "baseline_only":
        return [dict(row) for row in events]
    require_numpy()
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in events:
        by_event[row["event_id"]].append(dict(row))
    rng = np.random.default_rng(seed)
    result: list[dict[str, str]] = []
    for event_id, rows in sorted(by_event.items()):
        if arm == "carrier_permuted":
            carrier_rows = [row for row in rows if row.get("genotype_state", "ALT_CARRIER") == "ALT_CARRIER"]
            eligible = ([sample for sample in sample_ids if (sample, event_id) not in (missing_pairs or set())]
                        if factorized else [row["sample_id"] for row in rows if row.get("genotype_state", "ALT_CARRIER") != "MISSING"])
            if len(carrier_rows) > len(eligible):
                raise ContractError(f"event {event_id} has more carriers than materialized samples")
            if not set(eligible).issubset(sample_ids):
                raise ContractError(f"event {event_id} contains a sample outside the materialized universe")
            if factorized:
                assigned = rng.choice(eligible, size=len(carrier_rows), replace=False)
                for donor, sample_id in zip(carrier_rows, assigned, strict=True):
                    moved = dict(donor)
                    moved["sample_id"] = str(sample_id)
                    result.append(moved)
                continue
            # Dense synthetic fixture only: retain observability lattice.  Real
            # tables use implicit zero-evaluable calls and follow the branch above.
            transformed = {row["sample_id"]: dict(row) for row in rows}
            for row in carrier_rows:
                target = transformed[row["sample_id"]]
                target["genotype"] = "0"
                target["genotype_state"] = "ZERO_EVALUABLE"
                target["evaluable_mask"] = "1"
            assigned = rng.choice(eligible, size=len(carrier_rows), replace=False)
            for donor, sample_id in zip(carrier_rows, assigned, strict=True):
                target = transformed[str(sample_id)]
                target["genotype"] = donor["genotype"]
                target["genotype_state"] = "ALT_CARRIER"
                target["evaluable_mask"] = "1"
            result.extend(transformed.values())
        elif arm == "geometry_only":
            # Preserve each carrier's event geometry but erase dosage/MAC/class evidence.
            for row in rows:
                row["_geometry_only"] = "true"
            result.extend(rows)
        else:
            raise ContractError(f"Unsupported M36 arm: {arm}")
    return result


def geometry_only_tokens(events: list[dict[str, str]], sample_ids: list[str]) -> list[dict[str, Any]]:
    """Give every individual the identical evaluable zero-genotype locus set.

    This blocks the earlier leak: a set with only carrier rows encodes carrier
    identity through row presence even when dosage is numerically zero.
    """
    loci: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in events:
        loci[row["event_id"]].append(row)
    tokens: list[dict[str, Any]] = []
    for event_id, rows in sorted(loci.items()):
        reference = rows[0]
        geometry_callability = float(np.mean([as_float(row["callability"], "callability") for row in rows]))
        for sample in sample_ids:
            tokens.append({
                "sample_id": sample, "event_id": event_id, "event_class": "GEOMETRY_ONLY",
                "genotype_dosage": 0.0, "mac_scaled": 0.0, "callability": geometry_callability,
                "cm": as_float(reference["cm"], "cm"),
                "common_copying_context": as_float(reference["common_copying_context"], "common_copying_context"),
                "common_copying_context_available": 1, "mutation_context_available": 1,
                "is_ac2_het": 0, "is_ac2_homalt": 0, "is_mac3_10": 0,
                "mutation_context": reference["mutation_context"], "evaluable_mask": 1,
            })
    return tokens


def build_tokens(events: list[dict[str, str]], arm: str, sample_ids: list[str] | None = None,
                 factorized: bool = False) -> list[dict[str, Any]]:
    if arm == "geometry_only":
        if sample_ids is None:
            raise ContractError("geometry-only control requires the complete evaluable sample universe")
        if factorized:
            # The global axis is the locus table itself.  Since it is identical
            # for every person, the geometry arm exposes no individual token and
            # avoids materializing samples × loci.
            return []
        return geometry_only_tokens(events, sample_ids)
    tokens = event_tokens(events)
    return tokens


def validate_real_event_lattice(events: list[dict[str, str]], sample_ids: set[str]) -> None:
    if not events or any(not REAL_EVENT_STATE_COLUMNS.issubset(row) for row in events):
        raise ContractError("real M36 requires genotype_state and evaluable_mask per sample/event locus")
    event_ids = {row["event_id"] for row in events}
    pairs = {(row["sample_id"], row["event_id"]) for row in events}
    if set(row["sample_id"] for row in events) != sample_ids or len(pairs) != len(events) or len(events) != len(sample_ids) * len(event_ids):
        raise ContractError("real M36 event lattice must distinguish every zero-evaluable and missing sample/locus")


def factorized_carrier_events(loci: list[dict[str, str]], carriers: list[dict[str, str]], missing: list[dict[str, str]],
                              sample_ids: set[str]) -> tuple[list[dict[str, str]], set[tuple[str, str]]]:
    """Join sparse carrier calls to a global locus axis without densification."""
    locus_required = {"event_id", "chrom", "position", "mac", "callability", "mutation_context", "cm", "common_copying_context"}
    carrier_required = {"sample_id", "event_id", "minor_dosage"}
    missing_required = {"sample_id", "event_id"}
    if not loci or not locus_required.issubset(loci[0]) or not carriers or not carrier_required.issubset(carriers[0]):
        raise ContractError("real M36 requires factorized loci and sparse carrier tables")
    if missing and not missing_required.issubset(missing[0]):
        raise ContractError("real M36 missing table must contain sample_id/event_id")
    by_event = {row["event_id"]: row for row in loci}
    if len(by_event) != len(loci):
        raise ContractError("factorized loci event_id must be unique")
    missing_pairs = {(row["sample_id"], row["event_id"]) for row in missing}
    if len(missing_pairs) != len(missing):
        raise ContractError("factorized missing sample/event pairs must be unique")
    events = []
    seen = set()
    for row in carriers:
        key = (row["sample_id"], row["event_id"])
        if key in seen or key in missing_pairs or row["sample_id"] not in sample_ids or row["event_id"] not in by_event:
            raise ContractError("invalid sparse carrier/missing sample-event relationship")
        dosage = int(row["minor_dosage"])
        if dosage not in (1, 2):
            raise ContractError("minor_dosage must be one or two")
        locus = by_event[row["event_id"]]
        events.append({"sample_id": row["sample_id"], "event_id": row["event_id"], "chrom": locus["chrom"],
                       "position": locus["position"], "mac": locus["mac"], "genotype": str(dosage),
                       "callability": locus["callability"], "mutation_context": locus["mutation_context"],
                       "mutation_context_available": locus.get("mutation_context_available", "0"), "cm": locus["cm"],
                       "common_copying_context": locus["common_copying_context"],
                       "common_copying_context_available": locus.get("common_copying_context_available", "0"),
                       "genotype_state": "ALT_CARRIER", "evaluable_mask": "1"})
        seen.add(key)
    for sample, event_id in missing_pairs:
        if sample not in sample_ids or event_id not in by_event:
            raise ContractError("sparse missing table references an unknown sample or locus")
    return events, missing_pairs


class FitPreprocessor:
    """FIT-only normalization and categorical vocabularies."""

    def fit(self, tokens: list[dict[str, Any]], covariates: dict[str, dict[str, str]], fit_samples: set[str]) -> None:
        require_numpy()
        fit_tokens = [row for row in tokens if row["sample_id"] in fit_samples]
        if not fit_tokens:
            self.means = {field: 0.0 for field in NUMERIC_TOKEN_FIELDS}
            self.stds = {field: 1.0 for field in NUMERIC_TOKEN_FIELDS}
            self.context_vocab = {"<UNK>": 0}
        else:
            self.means = {field: float(np.mean([float(row[field]) for row in fit_tokens])) for field in NUMERIC_TOKEN_FIELDS}
            self.stds = {
                field: max(float(np.std([float(row[field]) for row in fit_tokens])), 1e-6)
                for field in NUMERIC_TOKEN_FIELDS
            }
            contexts = sorted({row["mutation_context"] for row in fit_tokens})
            self.context_vocab = {"<UNK>": 0, **{context: index + 1 for index, context in enumerate(contexts)}}
        fit_covariates = [covariates[sample] for sample in sorted(fit_samples)]
        if not fit_covariates:
            raise ContractError("FIT preprocessing requires at least one sample")
        self.cov_means = {
            field: float(np.mean([as_float(row[field], field) for row in fit_covariates]))
            for field in COVARIATE_FIELDS
        }
        self.cov_stds = {
            field: max(float(np.std([as_float(row[field], field) for row in fit_covariates])), 1e-6)
            for field in COVARIATE_FIELDS
        }
        cohorts = sorted({row["cohort"].strip() for row in fit_covariates})
        if not cohorts or any(not value for value in cohorts):
            raise ContractError("cohort must be nonempty in every FIT partition")
        self.cohort_vocab = {"<UNK>": 0, **{cohort: index + 1 for index, cohort in enumerate(cohorts)}}
        self.covariate_dim = len(COVARIATE_FIELDS) + len(self.cohort_vocab)

    def encode_people(self, tokens: list[dict[str, Any]], covariates: dict[str, dict[str, str]], sample_ids: list[str], torch):
        require_numpy()
        by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in tokens:
            by_sample[row["sample_id"]].append(row)
        max_events = max(1, max((len(by_sample[sample]) for sample in sample_ids), default=0))
        values = np.zeros((len(sample_ids), max_events, len(NUMERIC_TOKEN_FIELDS)), dtype=np.float32)
        contexts = np.zeros((len(sample_ids), max_events), dtype=np.int64)
        mask = np.zeros((len(sample_ids), max_events), dtype=bool)
        covariate_values = np.zeros((len(sample_ids), self.covariate_dim), dtype=np.float32)
        for index, sample in enumerate(sample_ids):
            for token_index, row in enumerate(sorted(by_sample[sample], key=lambda value: (value["cm"], value["event_id"]))):
                values[index, token_index] = [
                    (float(row[field]) - self.means[field]) / self.stds[field]
                    for field in NUMERIC_TOKEN_FIELDS
                ]
                contexts[index, token_index] = self.context_vocab.get(row["mutation_context"], 0)
                mask[index, token_index] = True
            covariate_values[index, :len(COVARIATE_FIELDS)] = [
                (as_float(covariates[sample][field], field) - self.cov_means[field]) / self.cov_stds[field]
                for field in COVARIATE_FIELDS
            ]
            cohort = covariates[sample]["cohort"].strip()
            cohort_index = self.cohort_vocab.get(cohort, self.cohort_vocab["<UNK>"])
            covariate_values[index, len(COVARIATE_FIELDS) + cohort_index] = 1.0
        return (
            torch.tensor(values), torch.tensor(contexts), torch.tensor(mask), torch.tensor(covariate_values)
        )


def pair_rows_for_fold(targets: list[dict[str, str]], component_map: dict[str, str], assignment: dict[str, int], fold: int, partition: str):
    rows = []
    for row in targets:
        left = assignment[component_map[row["sample_i"]]]
        right = assignment[component_map[row["sample_j"]]]
        if partition == "fit" and left != fold and right != fold:
            rows.append(row)
        elif partition == "validation" and left == fold and right == fold:
            rows.append(row)
    return rows


def fold_preprocessing_audit(targets, covariates, component_map, assignment, n_folds: int) -> dict[str, Any]:
    """Record the exact FIT-only cohort vocabulary and target support per fold."""
    audit: dict[str, Any] = {}
    for fold in range(n_folds):
        fit_pairs = pair_rows_for_fold(targets, component_map, assignment, fold, "fit")
        validation_pairs = pair_rows_for_fold(targets, component_map, assignment, fold, "validation")
        fit_samples = {sample for row in fit_pairs for sample in (row["sample_i"], row["sample_j"])}
        validation_samples = {sample for row in validation_pairs for sample in (row["sample_i"], row["sample_j"])}
        fit_cohorts = sorted({covariates[sample]["cohort"].strip() for sample in fit_samples})
        validation_cohorts = sorted({covariates[sample]["cohort"].strip() for sample in validation_samples})
        if not fit_pairs or not validation_pairs or not fit_cohorts:
            raise ContractError(f"outer fold {fold} lacks FIT/validation pairs or FIT cohort support")

        def partition_counts(rows):
            counts: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "zero": 0})
            for row in rows:
                stratum = row.get("target_stratum", "unspecified")
                outcome = "positive" if float(row["target"]) > 0 else "zero"
                counts[stratum][outcome] += 1
            return {key: dict(value) for key, value in sorted(counts.items())}

        audit[str(fold)] = {
            "fit_sample_count": len(fit_samples),
            "validation_sample_count": len(validation_samples),
            "cohort_vocabulary": {"<UNK>": 0, **{value: index + 1 for index, value in enumerate(fit_cohorts)}},
            "validation_unseen_cohorts": sorted(set(validation_cohorts) - set(fit_cohorts)),
            "fit_pair_count": len(fit_pairs),
            "validation_pair_count": len(validation_pairs),
            "fit_target_counts": partition_counts(fit_pairs),
            "validation_target_counts": partition_counts(validation_pairs),
        }
    return audit


def target_partition_coverage(targets, component_map, assignment) -> dict[str, Any]:
    """Count target pairs that can and cannot enter component-disjoint validation."""
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"validation_covered": 0, "cross_fold_not_scored": 0}
    )
    for row in targets:
        left_fold = assignment[component_map[row["sample_i"]]]
        right_fold = assignment[component_map[row["sample_j"]]]
        stratum = row.get("target_stratum", "unspecified")
        key = "validation_covered" if left_fold == right_fold else "cross_fold_not_scored"
        counts[stratum][key] += 1
    total = len(targets)
    covered = sum(value["validation_covered"] for value in counts.values())
    return {
        "total_pairs": total,
        "validation_covered_pairs": covered,
        "cross_fold_not_scored_pairs": total - covered,
        "validation_coverage_fraction": covered / total if total else None,
        "by_target_stratum": {key: dict(value) for key, value in sorted(counts.items())},
    }


def tensor_pairs(model, people, sample_index: dict[str, int], pairs: list[dict[str, str]], torch, include_rare: bool):
    tokens, contexts, mask, covariates = people
    left = torch.tensor([sample_index[row["sample_i"]] for row in pairs], dtype=torch.long)
    right = torch.tensor([sample_index[row["sample_j"]] for row in pairs], dtype=torch.long)
    target = torch.tensor([as_float(row["target"], "target") for row in pairs], dtype=torch.float32)
    return model(
        tokens[left], contexts[left], mask[left], tokens[right], contexts[right], mask[right],
        covariates[left], covariates[right], include_rare=include_rare
    ), target


def fit_one_fold(spec: CoraModelSpec, tokens, covariates, targets, component_map, assignment, fold: int,
                 arm: str, epochs: int, seed: int) -> list[dict[str, Any]]:
    torch = require_torch()
    torch.manual_seed(seed)
    fit_pairs = pair_rows_for_fold(targets, component_map, assignment, fold, "fit")
    validation_pairs = pair_rows_for_fold(targets, component_map, assignment, fold, "validation")
    if not fit_pairs or not validation_pairs:
        return []
    fit_samples = {row["sample_i"] for row in fit_pairs} | {row["sample_j"] for row in fit_pairs}
    all_samples = sorted(covariates)
    prep = FitPreprocessor()
    prep.fit(tokens, covariates, fit_samples)
    people = prep.encode_people(tokens, covariates, all_samples, torch)
    sample_index = {sample: index for index, sample in enumerate(all_samples)}
    model = build_pair_regressor(spec, prep.covariate_dim, len(prep.context_vocab))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    include_rare = arm not in ("baseline_only", "geometry_only")
    model.train()
    for _ in range(epochs):
        (prediction, _), target = tensor_pairs(model, people, sample_index, fit_pairs, torch, include_rare)
        loss = loss_fn(prediction, target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    model.eval()
    with torch.no_grad():
        (prediction, baseline), target = tensor_pairs(model, people, sample_index, validation_pairs, torch, include_rare)
    rows = []
    for row, observed, predicted, baseline_value in zip(validation_pairs, target.tolist(), prediction.tolist(), baseline.tolist(), strict=True):
        component_key = "|".join(sorted({component_map[row["sample_i"]], component_map[row["sample_j"]]}))
        rows.append({
            "outer_fold": fold, "arm": arm, "target_chrom": row["target_chrom"],
            "component_key": component_key, "target": observed, "prediction": predicted,
            "baseline_prediction": baseline_value, "squared_error": (observed - predicted) ** 2,
            "absolute_error": abs(observed - predicted),
        })
    return rows


def summarize_predictions(rows: list[dict[str, Any]], seed: int, bootstrap_reps: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require_numpy()
    if not rows:
        raise ContractError("no component-disjoint validation predictions were produced")
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_component[row["component_key"]].append(row)
    unit_rows = []
    for component, values in sorted(by_component.items()):
        unit_rows.append({
            "component_key": component,
            "n_pairs": len(values),
            "MAE": float(np.mean([row["absolute_error"] for row in values])),
            "MSE": float(np.mean([row["squared_error"] for row in values])),
        })
    rng = np.random.default_rng(seed)
    components = sorted(by_component)
    boot = []
    for _ in range(bootstrap_reps):
        selected_components = rng.choice(components, len(components), replace=True)
        selected = [row for component in selected_components for row in by_component[component]]
        if selected:
            boot.append(float(np.mean([row["squared_error"] for row in selected])))
    target = np.array([row["target"] for row in rows], dtype=float)
    prediction = np.array([row["prediction"] for row in rows], dtype=float)
    mse = float(np.mean((target - prediction) ** 2))
    return {
        "n_pairs": len(rows), "n_components": len(components),
        "MAE": float(np.mean(np.abs(target - prediction))), "MSE": mse,
        "R2": None if np.var(target) == 0 else float(1 - np.sum((target - prediction) ** 2) / np.sum((target - target.mean()) ** 2)),
        "bootstrap_component_MSE_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if boot else None,
    }, unit_rows


def run_successive_halving(events, covariates, targets, component_map, n_folds: int, families, budgets, eta, seed,
                           bootstrap_reps, missing_pairs: set[tuple[str, str]] | None = None, factorized: bool = False):
    assignment = component_folds(component_map, n_folds)
    candidates = available_specs(families)
    history = []
    selected: list[CoraModelSpec] = []
    sample_ids = sorted(covariates)
    for stage, epochs in enumerate(budgets):
        scored = []
        for spec in candidates:
            arm = "rare_enabled"
            transformed = arm_events(events, arm, sample_ids, stable_seed(str(seed), str(stage), spec.family, str(spec.hidden_dim)), missing_pairs, factorized)
            tokens = build_tokens(transformed, arm, sample_ids, factorized)
            rows = [prediction for fold in range(n_folds) for prediction in fit_one_fold(
                spec, tokens, covariates, targets, component_map, assignment, fold, arm, epochs,
                stable_seed(str(seed), str(stage), spec.family, str(spec.hidden_dim), str(fold))
            )]
            metrics, _ = summarize_predictions(rows, seed, bootstrap_reps)
            scored.append((metrics["MSE"], spec, metrics))
        scored.sort(key=lambda item: (item[0], item[1].family, item[1].hidden_dim))
        for rank, (mse, spec, metrics) in enumerate(scored):
            history.append({"stage": stage, "epochs": epochs, "validation_rank": rank, "family": spec.family,
                            "hidden_dim": spec.hidden_dim, "MSE": mse, "n_pairs": metrics["n_pairs"]})
        candidates = [item[1] for item in scored[:max(1, math.ceil(len(scored) / eta))]]
        selected = candidates
    final_spec = selected[0]
    arm_predictions = {}
    arm_metrics = {}
    unit_metrics = {}
    for arm in ARMS:
        transformed = arm_events(events, arm, sample_ids, stable_seed(str(seed), "carrier_permutation"), missing_pairs, factorized)
        tokens = build_tokens(transformed, arm, sample_ids, factorized)
        rows = [prediction for fold in range(n_folds) for prediction in fit_one_fold(
            final_spec, tokens, covariates, targets, component_map, assignment, fold, arm, budgets[-1],
            stable_seed(str(seed), "final", final_spec.family, str(final_spec.hidden_dim), str(final_spec.depth), str(fold))
        )]
        arm_predictions[arm] = rows
        arm_metrics[arm], unit_metrics[arm] = summarize_predictions(rows, seed, bootstrap_reps)
    return final_spec, assignment, history, arm_predictions, arm_metrics, unit_metrics


def positive_gate(metrics: dict[str, dict[str, Any]], minimum_relative_mse_reduction: float) -> dict[str, Any]:
    rare_mse = metrics["rare_enabled"]["MSE"]
    comparators = ("baseline_only", "carrier_permuted", "geometry_only")
    reductions = {
        arm: 1.0 - rare_mse / metrics[arm]["MSE"] if metrics[arm]["MSE"] > 0 else None
        for arm in comparators
    }
    passed = all(value is not None and value >= minimum_relative_mse_reduction for value in reductions.values())
    return {
        "passed": passed,
        "minimum_relative_mse_reduction": minimum_relative_mse_reduction,
        "rare_enabled_relative_mse_reduction": reductions,
    }


def parse_train_seeds(value: str) -> tuple[int, ...]:
    """Parse the explicit replicate axis used only for real-data training."""
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ContractError("train seeds must be comma-separated integers") from error
    if not seeds or any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ContractError("train seeds must be unique nonnegative integers")
    return seeds


def synthetic_inputs(control: str) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    require_numpy()
    rng = np.random.default_rng(1701)
    samples = [f"S{index:02d}" for index in range(24)]
    covariates, components, events = [], [], []
    signal = defaultdict(float)
    for index, sample in enumerate(samples):
        q_nam = float(rng.uniform(0.1, 0.7))
        q_afr = float(rng.uniform(0.05, 0.25))
        q_eur = 1 - q_nam - q_afr
        covariates.append({"sample_id": sample, "rare_burden": str(2 + index % 3), "rare_callability": "0.98",
                           "cohort": f"C{index % 2}", "Q_AFR": str(q_afr), "Q_EUR": str(q_eur),
                           "Q_NAM": str(q_nam), "Q_EAS": "0"})
        components.append({"sample_id": sample, "pcrelate_component": f"PC_{index // 4}"})
    patterns = [("AC2_HET", 2, [1, 1]), ("AC2_HOMALT", 2, [2]), ("MAC3_10", 4, [1, 1, 2])]
    for event_index in range(72):
        kind, mac, dosages = patterns[event_index % len(patterns)]
        carriers = rng.choice(samples, len(dosages), replace=False)
        carrier_dosage = dict(zip(carriers, dosages, strict=True))
        copying = float((event_index % 5) / 4)
        for sample in samples:
            dosage = carrier_dosage.get(sample, 0)
            state = "ALT_CARRIER" if dosage else "ZERO_EVALUABLE"
            events.append({"sample_id": sample, "event_id": f"E{event_index}", "chrom": "chr22", "position": str(100 + event_index),
                           "mac": str(mac), "genotype": str(dosage), "callability": "0.98", "mutation_context": "CpG" if event_index % 2 else "NonCpG",
                           "cm": str(0.01 * event_index), "common_copying_context": str(copying),
                           "genotype_state": state, "evaluable_mask": "1"})
            if dosage:
                signal[sample] += dosage * (1 + copying if control == "interaction" else 1)
    targets = []
    for component in range(6):
        members = samples[component * 4:(component + 1) * 4]
        for left_index, left in enumerate(members):
            for right in members[left_index + 1:]:
                base = 0.05 + 0.02 * abs(float(covariates[samples.index(left)]["Q_NAM"]) - float(covariates[samples.index(right)]["Q_NAM"]))
                rare = 0.12 * (abs(signal[left] - signal[right]) if control == "additive" else signal[left] * signal[right] / 16.0)
                targets.append({"sample_i": left, "sample_j": right, "target_chrom": "chr21", "target_source": "common_wgs_ibd", "target": str(base + rare)})
    return events, covariates, components, targets


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--positive-control", choices=("additive", "interaction", "both"), default="both")
    parser.add_argument("--events", type=Path, help="legacy dense synthetic fixture only")
    parser.add_argument("--loci", type=Path)
    parser.add_argument("--carriers", type=Path)
    parser.add_argument("--missing", type=Path)
    parser.add_argument("--covariates", type=Path)
    parser.add_argument("--components", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--materialization-receipt", type=Path)
    parser.add_argument("--feature-chrom", default="chr22")
    parser.add_argument("--model-families", default="deep_sets")
    parser.add_argument("--halving-budgets", default="32,128")
    parser.add_argument("--halving-eta", type=int, default=2)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--train-seeds", default="1701", help="comma-separated independent seeds for real-data training")
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    parser.add_argument("--positive-min-relative-mse-reduction", type=float, default=0.10)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def load_real_inputs(args: argparse.Namespace):
    if not all((args.loci, args.carriers, args.missing, args.covariates, args.components, args.targets)):
        raise ContractError("real M36 training requires factorized loci/carriers/missing, covariates, components and targets")
    receipt = validate_materialization_receipt(args.materialization_receipt)
    loci, carriers, missing, covariates, components, targets = map(
        read_tsv, (args.loci, args.carriers, args.missing, args.covariates, args.components, args.targets)
    )
    for name, path in {"loci": args.loci, "carriers": args.carriers, "missing": args.missing,
                       "covariates": args.covariates, "components": args.components, "targets": args.targets}.items():
        if sha256_file(path) != receipt["input_descriptors"][name]["sha256"]:
            raise ContractError(f"materialized artifact hash differs from receipt: {name}")
    sample_ids = {row["sample_id"] for row in covariates}
    events, missing_pairs = factorized_carrier_events(loci, carriers, missing, sample_ids)
    covariate_map, component_map, _ = validate_inputs(events, covariates, components, targets, args.feature_chrom)
    return events, covariate_map, component_map, targets, missing_pairs


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "smoke" and not args.synthetic_smoke:
        raise ContractError("smoke mode must set --synthetic-smoke; materialized inputs are for explicit train mode")
    controls = ("additive", "interaction") if args.positive_control == "both" else (args.positive_control,)
    families = tuple(value.strip() for value in args.model_families.split(",") if value.strip())
    budgets = tuple(int(value) for value in args.halving_budgets.split(","))
    if not families or any(value <= 0 for value in budgets):
        raise ContractError("M36 requires finite model families and positive budgets")
    if not 0 < args.positive_min_relative_mse_reduction < 1:
        raise ContractError("positive control gate must be a relative MSE reduction in (0, 1)")
    runs: dict[str, Any] = {}
    if args.mode == "smoke":
        run_specs = [(control, stable_seed(str(args.seed), control), control) for control in controls]
        real_inputs = None
    else:
        run_specs = [(f"seed_{seed}", seed, None) for seed in parse_train_seeds(args.train_seeds)]
        # The authenticated chromosome tables are immutable across seeds and
        # can be large, so load them once rather than once per replicate.
        real_inputs = load_real_inputs(args)
    for run_label, run_seed, control in run_specs:
        factorized = args.mode == "train"
        missing_pairs: set[tuple[str, str]] | None = None
        if args.mode == "smoke":
            events, covariate_rows, component_rows, targets = synthetic_inputs(str(control))
            covariates = {row["sample_id"]: row for row in covariate_rows}
            component_map = {row["sample_id"]: row["pcrelate_component"] for row in component_rows}
        else:
            assert real_inputs is not None
            events, covariates, component_map, targets, missing_pairs = real_inputs
        result = run_successive_halving(
            events, covariates, targets, component_map, args.n_folds, families, budgets,
            args.halving_eta, run_seed, args.bootstrap_reps, missing_pairs, factorized,
        )
        spec, assignment, history, predictions, metrics, units = result
        preprocessing_audit = fold_preprocessing_audit(
            targets, covariates, component_map, assignment, args.n_folds
        )
        prefix = f"m36_cora_{run_label}"
        args.outdir.mkdir(parents=True, exist_ok=True)
        write_tsv(args.outdir / f"{prefix}_halving.tsv", history)
        for arm, rows in predictions.items():
            write_tsv(args.outdir / f"{prefix}_{arm}_predictions.tsv", rows)
            write_tsv(args.outdir / f"{prefix}_{arm}_component_metrics.tsv", units[arm])
        gate = positive_gate(metrics, args.positive_min_relative_mse_reduction)
        runs[run_label] = {
            "selected": {"family": spec.family, "hidden_dim": spec.hidden_dim, "depth": spec.depth},
            "metrics": metrics, "incremental_gate": gate, "fold_component_assignment": assignment,
            "fit_only_preprocessing_by_fold": preprocessing_audit,
            "target_partition_coverage": target_partition_coverage(targets, component_map, assignment),
        }
        if args.mode == "smoke":
            runs[run_label]["positive_control_gate"] = gate
        if args.mode == "smoke" and not gate["passed"]:
            reductions = gate["rare_enabled_relative_mse_reduction"]
            raise ContractError(
                f"synthetic {control} positive control was not recovered; relative MSE reductions={reductions}"
            )
    summary = {
        "stage": "M36_CORA_SET_TRAIN",
        "mode": args.mode,
        "synthetic_smoke": args.synthetic_smoke,
        "real_materialization_used": args.mode == "train",
        "run_axis": "synthetic_control" if args.mode == "smoke" else "independent_model_seed",
        "target": "continuous_external_common_ibd_or_chromopainter",
        "target_response": "log1p_asIBD_cM_for_factorized_real_inputs",
        "pair_head": "symmetric_abs_difference_and_product_residual_over_covariate_baseline",
        "fit_only_preprocessing": [
            "mutation_context_vocabulary", "token_normalization", "numeric_covariate_normalization",
            "cohort_one_hot_vocabulary",
        ],
        "categorical_covariates": {
            "cohort": {"encoding": "one_hot", "vocabulary_scope": "FIT_only", "unknown_level": "<UNK>"}
        },
        "controls": list(ARMS),
        "geometry_control": "global shared locus axis; real factorized arm exposes no individual rare tokens",
        "outer_cv": "both_nodes_outside_fit_fold_or_both_nodes_inside_validation_fold; cross-fold_pairs_excluded",
        "bootstrap": "PC-Relate component; target is one cross-chromosome total per pair",
        "runs": runs,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "m36_cora_train_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    try:
        run(parse_args())
    except ContractError as error:
        raise SystemExit(f"M36 training error: {error}") from error


if __name__ == "__main__":
    main()
