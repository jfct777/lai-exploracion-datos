#!/usr/bin/env python3
"""Nested, control-matched M36B screen for rare-variant genealogy signal.

M36B predicts a cross-chromosome common-IBD pair total.  It is an exploratory
test of recent genealogical structure, not a local-ancestry experiment and not
an estimate of biological ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - pinned runtime concern
    np = None

for import_root in (Path(__file__).resolve().parent, Path.cwd()):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from m36_cora_models import CoraModelSpec, available_specs
from m36_cora_set import ContractError, as_float, component_folds, event_tokens
from m36_cora_train import (
    COVARIATE_FIELDS,
    NUMERIC_TOKEN_FIELDS,
    FitPreprocessor,
    load_real_inputs,
    pair_rows_for_fold,
    parse_train_seeds,
    stable_seed,
    target_partition_coverage,
)
from m36b_cora_models import build_cached_pair_regressor


ARMS = ("rare_enabled", "carrier_permuted", "geometry_only", "baseline_only")
COMPARATORS = ("baseline_only", "carrier_permuted", "geometry_only")
POSITIVE_CONTROLS = ("additive", "interaction")
Q_FIELDS = ("Q_AFR", "Q_EUR", "Q_NAM", "Q_EAS")


def require_numpy():
    if np is None:
        raise ContractError("M36B requires the pinned NumPy/PyTorch runtime")
    return np


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - pinned runtime concern
        raise ContractError("M36B requires the pinned NumPy/PyTorch runtime") from error
    return torch


def permutation_stratum(row: dict[str, str]) -> tuple[str, str, int]:
    """Return a fixed cohort/Q stratum without fitting data-dependent cut-points.

    Q is represented by the ancestry with maximum global proportion and a
    fixed quartile of that maximum.  Fixed bins avoid learning a boundary from
    held-out outcomes or held-out target pairs.
    """
    q = {field: as_float(row[field], field) for field in Q_FIELDS}
    dominant = min(Q_FIELDS, key=lambda field: (-q[field], field))
    purity_quartile = min(3, max(0, int(q[dominant] * 4.0)))
    return row["cohort"].strip(), dominant.removeprefix("Q_"), purity_quartile


def _margin_signature(events: Iterable[dict[str, str]]) -> tuple[dict[str, tuple], dict[str, tuple]]:
    by_event: dict[str, list[int]] = defaultdict(list)
    by_sample: dict[str, list[int]] = defaultdict(list)
    for row in events:
        dosage = int(row["genotype"])
        by_event[row["event_id"]].append(dosage)
        by_sample[row["sample_id"]].append(dosage)
    event_signature = {
        event: (len(values), sum(values), tuple(sorted(values)))
        for event, values in by_event.items()
    }
    sample_signature = {
        sample: (len(values), sum(values), tuple(sorted(values)))
        for sample, values in by_sample.items()
    }
    return event_signature, sample_signature


def degree_preserving_permutation(
    events: list[dict[str, str]],
    covariates: dict[str, dict[str, str]],
    missing_pairs: set[tuple[str, str]],
    seed: int,
    swap_multiplier: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Shuffle carrier identities with exact bipartite double-edge swaps.

    Within each cohort/Q/dosage stratum, two carrier edges ``(i,a)`` and
    ``(j,b)`` are replaced by ``(i,b)`` and ``(j,a)`` only when both crossed
    cells are callable non-carriers.  Equal-dosage swaps preserve, exactly:

    * minor-allele count, carrier count and dosage multiset at every locus;
    * rare-event count, dosage burden and dosage multiset for every person;
    * the missing-call mask;
    * cohort and coarse global-ancestry composition of carrier assignments.

    The number of attempted swaps is explicit and the realised movement is
    reported.  No failed stratum silently falls back to a broader stratum.
    """
    require_numpy()
    if swap_multiplier <= 0:
        raise ContractError("permutation swap multiplier must be positive")
    original_event, original_sample = _margin_signature(events)
    templates: dict[str, dict[str, str]] = {}
    occupied: set[tuple[str, str]] = set()
    original_occupied: set[tuple[str, str]] = set()
    for row in events:
        sample, event = row["sample_id"], row["event_id"]
        key = (sample, event)
        if key in occupied:
            raise ContractError("carrier table contains a duplicate sample/event edge")
        if sample not in covariates or key in missing_pairs:
            raise ContractError("carrier permutation received an unknown or missing carrier edge")
        dosage = int(row["genotype"])
        if dosage not in (1, 2):
            raise ContractError("carrier permutation supports minor dosages one and two")
        templates.setdefault(event, dict(row))
        occupied.add(key)
        original_occupied.add(key)

    # Rebuild explicit groups without relying on a mutable default-key trick.
    grouped: dict[tuple[str, int], list[list[Any]]] = defaultdict(list)
    for row in events:
        stratum = "|".join(map(str, permutation_stratum(covariates[row["sample_id"]])))
        grouped[(stratum, int(row["genotype"]))].append(
            [row["sample_id"], row["event_id"], int(row["genotype"])]
        )

    rng = np.random.default_rng(seed)
    group_diagnostics: list[dict[str, Any]] = []
    accepted_total = 0
    attempts_total = 0
    for (stratum, dosage), group in sorted(grouped.items()):
        attempts = int(math.ceil(swap_multiplier * len(group))) if len(group) >= 2 else 0
        accepted = 0
        for _ in range(attempts):
            first, second = rng.choice(len(group), size=2, replace=False)
            sample_a, event_a, _ = group[int(first)]
            sample_b, event_b, _ = group[int(second)]
            if sample_a == sample_b or event_a == event_b:
                continue
            cross_a, cross_b = (sample_a, event_b), (sample_b, event_a)
            if cross_a in occupied or cross_b in occupied or cross_a in missing_pairs or cross_b in missing_pairs:
                continue
            occupied.remove((sample_a, event_a))
            occupied.remove((sample_b, event_b))
            occupied.add(cross_a)
            occupied.add(cross_b)
            group[int(first)] = [sample_a, event_b, dosage]
            group[int(second)] = [sample_b, event_a, dosage]
            accepted += 1
        attempts_total += attempts
        accepted_total += accepted
        group_diagnostics.append({
            "stratum": stratum,
            "dosage": dosage,
            "n_edges": len(group),
            "attempted_swaps": attempts,
            "accepted_swaps": accepted,
        })

    result: list[dict[str, str]] = []
    for group in grouped.values():
        for sample, event, dosage in group:
            row = dict(templates[event])
            row["sample_id"] = sample
            row["genotype"] = str(dosage)
            row["genotype_state"] = "ALT_CARRIER"
            row["evaluable_mask"] = "1"
            result.append(row)
    result.sort(key=lambda row: (row["event_id"], row["sample_id"]))
    permuted_event, permuted_sample = _margin_signature(result)
    if permuted_event != original_event or permuted_sample != original_sample:
        raise ContractError("carrier permutation changed an event or person margin")
    if any((row["sample_id"], row["event_id"]) in missing_pairs for row in result):
        raise ContractError("carrier permutation assigned an allele to a missing call")
    moved = len(original_occupied - occupied)
    diagnostics = {
        "algorithm": "equal-dosage bipartite double-edge swap within fixed cohort/dominant-Q/purity-quartile strata",
        "seed": seed,
        "swap_multiplier": swap_multiplier,
        "attempted_swaps": attempts_total,
        "accepted_swaps": accepted_total,
        "carrier_edges": len(events),
        "moved_carrier_edges": moved,
        "moved_carrier_fraction": moved / len(events) if events else 0.0,
        "exact_invariants": [
            "per_locus_minor_allele_count",
            "per_locus_carrier_count",
            "per_locus_dosage_multiset",
            "per_person_event_count",
            "per_person_minor_dosage_burden",
            "per_person_dosage_multiset",
            "missing_call_mask",
            "cohort_and_fixed_Q_stratum",
        ],
        "groups": group_diagnostics,
    }
    return result, diagnostics


def partitioned_degree_preserving_permutation(
    events: list[dict[str, str]],
    covariates: dict[str, dict[str, str]],
    missing_pairs: set[tuple[str, str]],
    partitions: dict[str, set[str]],
    seed: int,
    swap_multiplier: float,
    minimum_moved_carrier_fraction: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Permute FIT and validation separately so the null cannot cross the split.

    The moved-edge threshold is a technical null-validity guard: it prevents a
    nominally permuted arm from remaining too similar to the observed carrier
    graph.  It is not a biological cutoff and does not claim that the swap
    Markov chain has reached its uniform stationary distribution.
    """
    if not 0 < minimum_moved_carrier_fraction <= 1:
        raise ContractError("minimum moved-carrier fraction must be in (0,1]")
    labels = sorted(partitions)
    for index, left in enumerate(labels):
        for right in labels[index + 1:]:
            if partitions[left] & partitions[right]:
                raise ContractError("carrier-permutation partitions overlap")
    covered = set().union(*partitions.values()) if partitions else set()
    result = [dict(row) for row in events if row["sample_id"] not in covered]
    diagnostics = {}
    for label in labels:
        samples = partitions[label]
        subset = [row for row in events if row["sample_id"] in samples]
        if not subset:
            raise ContractError(f"carrier-permutation partition {label} has no carrier edges")
        permuted, audit = degree_preserving_permutation(
            subset, covariates, missing_pairs,
            stable_seed(str(seed), label), swap_multiplier,
        )
        audit["minimum_moved_carrier_fraction"] = minimum_moved_carrier_fraction
        audit["mixing_gate_passed"] = (
            audit["moved_carrier_fraction"] >= minimum_moved_carrier_fraction
        )
        if not audit["mixing_gate_passed"]:
            raise ContractError(
                f"carrier permutation did not move enough edges in {label}: "
                f"{audit['moved_carrier_fraction']:.6f} < "
                f"{minimum_moved_carrier_fraction:.6f}"
            )
        result.extend(permuted)
        diagnostics[label] = audit
    if _margin_signature(result) != _margin_signature(events):
        raise ContractError("partitioned carrier permutation changed a global margin")
    return result, {
        "split_isolation": "FIT and outer validation permuted independently",
        "mixing_gate_scope": "technical null validity; not a biological threshold or proof of uniform mixing",
        "minimum_moved_carrier_fraction": minimum_moved_carrier_fraction,
        "partitions": diagnostics,
    }


def geometry_tokens(events: list[dict[str, str]], anchor_sample: str) -> list[dict[str, Any]]:
    """Represent the shared locus geometry once, without person or dosage identity."""
    references: dict[str, dict[str, str]] = {}
    for row in events:
        references.setdefault(row["event_id"], row)
    tokens = []
    for event, row in sorted(references.items(), key=lambda item: (as_float(item[1]["cm"], "cm"), item[0])):
        tokens.append({
            "sample_id": anchor_sample,
            "event_id": event,
            "event_class": "GEOMETRY_ONLY",
            "genotype_dosage": 0.0,
            "mac_scaled": 0.0,
            "callability": as_float(row["callability"], "callability"),
            "cm": as_float(row["cm"], "cm"),
            "common_copying_context": as_float(row["common_copying_context"], "common_copying_context"),
            "common_copying_context_available": int(row.get("common_copying_context_available", "0")),
            "mutation_context_available": 0,
            "is_ac2_het": 0,
            "is_ac2_homalt": 0,
            "is_mac3_10": 0,
            "mutation_context": "<GEOMETRY>",
        })
    return tokens


def positive_control_inputs(
    control: str,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, str], list[dict[str, str]], set[tuple[str, str]]]:
    """Build a deterministic known-answer set for the complete M36B path.

    Every person has exactly four heterozygous carrier events, so rare burden
    cannot identify the outcome.  The known signal instead depends on which
    loci a person carries through ``common_copying_context``.  The additive
    target uses the absolute difference between two person-level signals; the
    interaction target uses their product.  All people share one fixed
    cohort/global-ancestry stratum, leaving enough swappable carrier edges in
    every component-disjoint FIT and validation partition.
    """
    require_numpy()
    if control not in POSITIVE_CONTROLS:
        raise ContractError(f"unsupported M36B positive control: {control}")
    samples = [f"PC{index:02d}" for index in range(48)]
    covariates = {
        sample: {
            "sample_id": sample,
            "cohort": "SYNTHETIC",
            "rare_burden": "4",
            "rare_callability": "1",
            "Q_AFR": "0.2",
            "Q_EUR": "0.6",
            "Q_NAM": "0.2",
            "Q_EAS": "0",
        }
        for sample in samples
    }
    component_map = {
        sample: f"SYNTHETIC_COMPONENT_{index // 4:02d}"
        for index, sample in enumerate(samples)
    }
    events: list[dict[str, str]] = []
    signal = {sample: 0.0 for sample in samples}
    for event_index in range(96):
        # These two modular schedules form a regular bipartite graph: each
        # person carries four events and every event has two distinct carriers.
        carrier_indices = (event_index % 48, (5 * event_index + 1) % 48)
        copying_context = ((37 * event_index) % 97) / 96.0
        event_signal = copying_context - 0.5
        for carrier_index in carrier_indices:
            sample = samples[carrier_index]
            signal[sample] += event_signal
            events.append({
                "sample_id": sample,
                "event_id": f"POS_{event_index:03d}",
                "chrom": "chr22",
                "position": str(100_000 + 10_000 * event_index),
                "mac": "2",
                "genotype": "1",
                "callability": "1",
                "mutation_context": "CpG" if event_index % 2 else "NonCpG",
                "mutation_context_available": "1",
                "cm": str(event_index / 100.0),
                "common_copying_context": str(copying_context),
                "common_copying_context_available": "1",
                "genotype_state": "ALT_CARRIER",
                "evaluable_mask": "1",
            })
    if any(sum(row["sample_id"] == sample for row in events) != 4 for sample in samples):
        raise ContractError("M36B positive-control carrier graph is not degree-regular")
    signal_values = np.asarray([signal[sample] for sample in samples], dtype=float)
    signal_values = (signal_values - signal_values.mean()) / signal_values.std()
    standardized = dict(zip(samples, signal_values.tolist(), strict=True))
    pairs: list[tuple[str, str, float]] = []
    for left_index, left in enumerate(samples):
        for right in samples[left_index + 1:]:
            if control == "additive":
                latent = abs(standardized[left] - standardized[right])
            else:
                latent = standardized[left] * standardized[right]
            pairs.append((left, right, latent))
    latent_values = np.asarray([row[2] for row in pairs], dtype=float)
    latent_min = float(latent_values.min())
    latent_range = float(latent_values.max() - latent_min)
    if latent_range <= 0:
        raise ContractError(f"M36B {control} positive control has no target variation")
    targets = [
        {
            "sample_i": left,
            "sample_j": right,
            "target_chrom": "chr1-chr21",
            "target_source": f"synthetic_known_{control}",
            "target_stratum": "synthetic_positive_control",
            "target": str(0.2 + 2.0 * (latent - latent_min) / latent_range),
        }
        for left, right, latent in pairs
    ]
    return events, covariates, component_map, targets, set()


def _pair_indices(pairs, sample_index, torch):
    left = torch.tensor([sample_index[row["sample_i"]] for row in pairs], dtype=torch.long)
    right = torch.tensor([sample_index[row["sample_j"]] for row in pairs], dtype=torch.long)
    target = torch.tensor([as_float(row["target"], "target") for row in pairs], dtype=torch.float32)
    return left, right, target


def _predict_in_batches(model, embeddings, covariates, left, right, include_rare, pair_batch_size):
    predictions, baselines = [], []
    for start in range(0, len(left), pair_batch_size):
        stop = min(start + pair_batch_size, len(left))
        prediction, baseline = model.predict_from_embeddings(
            embeddings, covariates, left[start:stop], right[start:stop], include_rare=include_rare
        )
        predictions.append(prediction)
        baselines.append(baseline)
    return predictions, baselines


def fit_cached_fold(
    spec: CoraModelSpec,
    tokens: list[dict[str, Any]],
    geometry: list[dict[str, Any]],
    covariates: dict[str, dict[str, str]],
    fit_pairs: list[dict[str, str]],
    validation_pairs: list[dict[str, str]],
    component_map: dict[str, str],
    fold: int,
    arm: str,
    epochs: int,
    seed: int,
    pair_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    huber_delta: float,
) -> list[dict[str, Any]]:
    """Fit one fold, encoding each unique person once per optimisation step."""
    torch = require_torch()
    if not fit_pairs or not validation_pairs:
        raise ContractError(f"fold {fold} lacks FIT or validation pairs")
    torch.manual_seed(seed)
    fit_samples = sorted({sample for row in fit_pairs for sample in (row["sample_i"], row["sample_j"])})
    validation_samples = sorted({sample for row in validation_pairs for sample in (row["sample_i"], row["sample_j"])})
    if set(fit_samples) & set(validation_samples):
        raise ContractError(f"fold {fold} leaks an individual between FIT and validation")
    prep = FitPreprocessor()
    if arm == "geometry_only":
        anchored_geometry = [dict(row, sample_id=fit_samples[0]) for row in geometry]
        prep.fit(anchored_geometry, covariates, set(fit_samples))
    elif arm == "baseline_only":
        prep.fit([], covariates, set(fit_samples))
    else:
        prep.fit(tokens, covariates, set(fit_samples))

    def encode_partition(samples: list[str]):
        active_tokens = [] if arm in ("geometry_only", "baseline_only") else tokens
        return prep.encode_people(active_tokens, covariates, samples, torch)

    fit_people = encode_partition(fit_samples)
    validation_people = encode_partition(validation_samples)
    fit_index = {sample: index for index, sample in enumerate(fit_samples)}
    validation_index = {sample: index for index, sample in enumerate(validation_samples)}
    fit_left, fit_right, fit_target = _pair_indices(fit_pairs, fit_index, torch)
    val_left, val_right, val_target = _pair_indices(validation_pairs, validation_index, torch)
    model = build_cached_pair_regressor(spec, prep.covariate_dim, len(prep.context_vocab))
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.HuberLoss(delta=huber_delta, reduction="sum")
    include_rare = arm != "baseline_only"
    geometry_base = None
    if arm == "geometry_only":
        anchor = fit_samples[0]
        geometry_people = prep.encode_people(
            [dict(row, sample_id=anchor) for row in geometry], covariates, [anchor], torch
        )
        # The geometry is identical for every person and therefore cannot
        # explain pair-to-pair variation.  Encode it once and detach it; a
        # repeated trainable pass would only relearn an intercept at enormous
        # cost, which the pair head already supplies explicitly.
        with torch.no_grad():
            geometry_base = torch.nn.functional.normalize(
                model.encode_population(*geometry_people[:3]), p=2, dim=-1
            ).detach()

    def embeddings_for(people, samples):
        values, contexts, mask, _ = people
        if arm == "baseline_only":
            return None
        if arm == "geometry_only":
            assert geometry_base is not None
            return geometry_base.expand(len(samples), -1)
        return model.encode_population(values, contexts, mask)

    model.train()
    for _ in range(epochs):
        optimiser.zero_grad()
        fit_embeddings = embeddings_for(fit_people, fit_samples)
        prediction_chunks, _ = _predict_in_batches(
            model, fit_embeddings, fit_people[3], fit_left, fit_right, include_rare, pair_batch_size
        )
        loss = sum(loss_fn(prediction, fit_target[start:start + len(prediction)])
                   for start, prediction in zip(range(0, len(fit_target), pair_batch_size), prediction_chunks))
        (loss / len(fit_target)).backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        val_embeddings = embeddings_for(validation_people, validation_samples)
        prediction_chunks, baseline_chunks = _predict_in_batches(
            model, val_embeddings, validation_people[3], val_left, val_right, include_rare, pair_batch_size
        )
        predictions = torch.cat(prediction_chunks).tolist()
        baselines = torch.cat(baseline_chunks).tolist()
    rows = []
    for index, (row, target, prediction, baseline) in enumerate(
        zip(validation_pairs, val_target.tolist(), predictions, baselines, strict=True)
    ):
        component_i = component_map[row["sample_i"]]
        component_j = component_map[row["sample_j"]]
        pair_digest = hashlib.sha256(
            "|".join(sorted((row["sample_i"], row["sample_j"]))).encode("utf-8")
        ).hexdigest()[:20]
        rows.append({
            "pair_id": pair_digest,
            "outer_fold": fold,
            "target_stratum": row.get("target_stratum", "unspecified"),
            "arm": arm,
            "component_i": component_i,
            "component_j": component_j,
            "target": target,
            "prediction": prediction,
            "baseline_prediction": baseline,
            "squared_error": (target - prediction) ** 2,
            "absolute_error": abs(target - prediction),
        })
    return rows


def _fold_macro_mse(rows: list[dict[str, Any]]) -> float:
    by_fold: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_fold[int(row["outer_fold"])].append(float(row["squared_error"]))
    if not by_fold:
        raise ContractError("architecture screen produced no validation predictions")
    return float(np.mean([np.mean(values) for values in by_fold.values()]))


def select_architecture_inside_outer_fit(
    specs: list[CoraModelSpec],
    budgets: tuple[int, ...],
    eta: int,
    tokens: list[dict[str, Any]],
    covariates: dict[str, dict[str, str]],
    outer_fit_pairs: list[dict[str, str]],
    component_map: dict[str, str],
    outer_assignment: dict[str, int],
    outer_fold: int,
    inner_folds: int,
    seed: int,
    train_kwargs: dict[str, Any],
) -> tuple[CoraModelSpec, list[dict[str, Any]]]:
    """Nested selection: no outer-validation pair participates in ranking."""
    outer_fit_samples = {
        sample for row in outer_fit_pairs for sample in (row["sample_i"], row["sample_j"])
    }
    if any(outer_assignment[component_map[sample]] == outer_fold for sample in outer_fit_samples):
        raise ContractError("outer validation component entered architecture selection")
    inner_map = {sample: component_map[sample] for sample in outer_fit_samples}
    inner_assignment = component_folds(inner_map, inner_folds)
    candidates = list(specs)
    history: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    for stage, epochs in enumerate(budgets):
        ranked = []
        for spec in candidates:
            candidate_rows = []
            for inner_fold in range(inner_folds):
                fit_pairs = pair_rows_for_fold(
                    outer_fit_pairs, component_map, inner_assignment, inner_fold, "fit"
                )
                validation_pairs = pair_rows_for_fold(
                    outer_fit_pairs, component_map, inner_assignment, inner_fold, "validation"
                )
                candidate_rows.extend(fit_cached_fold(
                    spec, tokens, geometry, covariates, fit_pairs, validation_pairs,
                    component_map, inner_fold, "rare_enabled", epochs,
                    stable_seed(str(seed), "select", str(outer_fold), str(stage), spec.family,
                                str(spec.hidden_dim), str(inner_fold)),
                    **train_kwargs,
                ))
            score = _fold_macro_mse(candidate_rows)
            ranked.append((score, spec, len(candidate_rows)))
        ranked.sort(key=lambda item: (item[0], item[1].family, item[1].hidden_dim))
        for rank, (score, spec, n_pairs) in enumerate(ranked):
            history.append({
                "outer_fold": outer_fold,
                "stage": stage,
                "epochs": epochs,
                "rank": rank,
                "family": spec.family,
                "hidden_dim": spec.hidden_dim,
                "depth": spec.depth,
                "inner_fold_macro_mse": score,
                "inner_validation_pairs": n_pairs,
                "selection_scope": "outer_FIT_only",
            })
        candidates = [item[1] for item in ranked[:max(1, math.ceil(len(ranked) / eta))]]
    return candidates[0], history


def _node_weighted_mean(rows, values, rng):
    components = sorted({row["component_i"] for row in rows} | {row["component_j"] for row in rows})
    node_weights = dict(zip(components, rng.exponential(1.0, len(components)), strict=True))
    weights = np.array([
        node_weights[row["component_i"]]
        if row["component_i"] == row["component_j"]
        else node_weights[row["component_i"]] * node_weights[row["component_j"]]
        for row in rows
    ], dtype=float)
    return float(np.average(np.asarray(values, dtype=float), weights=weights))


def paired_effects(
    predictions: dict[str, list[dict[str, Any]]], bootstrap_reps: int, seed: int
) -> list[dict[str, Any]]:
    """Report paired errors with an approximate component-node bootstrap.

    The Bayesian node bootstrap gives each PC-Relate component an independent
    exponential weight.  A between-component pair receives the product of its
    two node weights; a within-component pair receives that component's weight.
    This retains shared-node dependence, but remains an exploratory network
    bootstrap approximation rather than an exact pedigree likelihood.
    """
    require_numpy()
    rare = {row["pair_id"]: row for row in predictions["rare_enabled"]}
    if len(rare) != len(predictions["rare_enabled"]):
        raise ContractError("rare-enabled predictions contain a duplicate pair identifier")
    effects: list[dict[str, Any]] = []
    for comparator in COMPARATORS:
        control = {row["pair_id"]: row for row in predictions[comparator]}
        if len(control) != len(predictions[comparator]):
            raise ContractError(f"{comparator} predictions contain a duplicate pair identifier")
        if set(rare) != set(control):
            raise ContractError(f"paired comparison {comparator} does not use identical validation pairs")
        paired = []
        for pair_id in sorted(rare):
            row = rare[pair_id]
            other = control[pair_id]
            if row["outer_fold"] != other["outer_fold"] or row["target"] != other["target"]:
                raise ContractError("paired predictions disagree on fold or target")
            paired.append(dict(row, effect=float(other["squared_error"]) - float(row["squared_error"]),
                               comparator_error=float(other["squared_error"])))
        strata = ["ALL"] + sorted({row["target_stratum"] for row in paired})
        for stratum in strata:
            selected = paired if stratum == "ALL" else [row for row in paired if row["target_stratum"] == stratum]
            fold_values: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in selected:
                fold_values[int(row["outer_fold"])].append(row)
            for fold, rows in sorted(fold_values.items()):
                delta = float(np.mean([row["effect"] for row in rows]))
                control_mse = float(np.mean([row["comparator_error"] for row in rows]))
                rng = np.random.default_rng(stable_seed(str(seed), comparator, stratum, str(fold)))
                boot = [
                    _node_weighted_mean(rows, [row["effect"] for row in rows], rng)
                    for _ in range(bootstrap_reps)
                ]
                effects.append({
                    "comparator": comparator,
                    "scope": "fold",
                    "outer_fold": fold,
                    "target_stratum": stratum,
                    "n_pairs": len(rows),
                    "n_components": len({r["component_i"] for r in rows} | {r["component_j"] for r in rows}),
                    "n_folds": 1,
                    "delta_mse_control_minus_rare": delta,
                    "relative_mse_reduction": delta / control_mse if control_mse > 0 else None,
                    "ci95_lower": float(np.quantile(boot, 0.025)),
                    "ci95_upper": float(np.quantile(boot, 0.975)),
                    "bootstrap_method": "approximate_PC-Relate_component-node_Bayesian_bootstrap",
                })
            if not fold_values:
                continue
            fold_deltas = [np.mean([row["effect"] for row in rows]) for rows in fold_values.values()]
            fold_control = [np.mean([row["comparator_error"] for row in rows]) for rows in fold_values.values()]
            macro_delta = float(np.mean(fold_deltas))
            macro_control = float(np.mean(fold_control))
            rng_by_fold = {
                fold: np.random.default_rng(stable_seed(str(seed), comparator, stratum, "macro", str(fold)))
                for fold in fold_values
            }
            boot = []
            for _ in range(bootstrap_reps):
                boot.append(float(np.mean([
                    _node_weighted_mean(rows, [row["effect"] for row in rows], rng_by_fold[fold])
                    for fold, rows in fold_values.items()
                ])))
            effects.append({
                "comparator": comparator,
                "scope": "fold_macro",
                "outer_fold": "ALL",
                "target_stratum": stratum,
                "n_pairs": len(selected),
                "n_components": len({r["component_i"] for r in selected} | {r["component_j"] for r in selected}),
                "n_folds": len(fold_values),
                "delta_mse_control_minus_rare": macro_delta,
                "relative_mse_reduction": macro_delta / macro_control if macro_control > 0 else None,
                "ci95_lower": float(np.quantile(boot, 0.025)),
                "ci95_upper": float(np.quantile(boot, 0.975)),
                "bootstrap_method": "approximate_PC-Relate_component-node_Bayesian_bootstrap",
            })
    return effects


def promotion_gate(
    effects, minimum_relative_reduction: float, minimum_positive_folds: int, expected_folds: int
) -> dict[str, Any]:
    criteria = {}
    for comparator in COMPARATORS:
        macro = next(row for row in effects if row["comparator"] == comparator
                     and row["scope"] == "fold_macro" and row["target_stratum"] == "ALL")
        folds = [row for row in effects if row["comparator"] == comparator
                 and row["scope"] == "fold" and row["target_stratum"] == "ALL"]
        positive_folds = sum(float(row["delta_mse_control_minus_rare"]) > 0 for row in folds)
        criteria[comparator] = {
            "relative_reduction_at_least_threshold": (
                macro["relative_mse_reduction"] is not None
                and macro["relative_mse_reduction"] >= minimum_relative_reduction
            ),
            "component_bootstrap_lower_bound_above_zero": macro["ci95_lower"] > 0,
            "positive_fold_count": positive_folds,
            "minimum_positive_folds": minimum_positive_folds,
            "all_expected_folds_present": len(folds) == expected_folds,
            "passed": (
                macro["relative_mse_reduction"] is not None
                and macro["relative_mse_reduction"] >= minimum_relative_reduction
                and macro["ci95_lower"] > 0
                and positive_folds >= minimum_positive_folds
                and len(folds) == expected_folds
            ),
        }
    return {
        "passed": all(value["passed"] for value in criteria.values()),
        "minimum_relative_mse_reduction": minimum_relative_reduction,
        "minimum_positive_folds": minimum_positive_folds,
        "comparators": criteria,
        "scope": "exploratory promotion only; not LAI validation or biological truth",
    }


def run_nested_screen(
    *,
    events: list[dict[str, str]],
    covariates: dict[str, dict[str, str]],
    component_map: dict[str, str],
    targets: list[dict[str, str]],
    missing_pairs: set[tuple[str, str]],
    specs: list[CoraModelSpec],
    budgets: tuple[int, ...],
    halving_eta: int,
    outer_folds: int,
    inner_folds: int,
    run_seed: int,
    bootstrap_reps: int,
    permutation_swap_multiplier: float,
    minimum_moved_carrier_fraction: float,
    minimum_relative_mse_reduction: float,
    minimum_positive_folds: int,
    train_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run one seed through the shared nested-CV and matched-control path."""
    outer_assignment = component_folds(component_map, outer_folds)
    rare_tokens = event_tokens(events)
    shared_geometry = geometry_tokens(events, sorted(covariates)[0])
    predictions: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    selected_by_fold: dict[str, Any] = {}
    permutation_by_fold: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    for outer_fold in range(outer_folds):
        outer_fit = pair_rows_for_fold(
            targets, component_map, outer_assignment, outer_fold, "fit"
        )
        outer_validation = pair_rows_for_fold(
            targets, component_map, outer_assignment, outer_fold, "validation"
        )
        selected, fold_history = select_architecture_inside_outer_fit(
            specs, budgets, halving_eta, rare_tokens, covariates, outer_fit,
            component_map, outer_assignment, outer_fold, inner_folds, run_seed, train_kwargs,
        )
        history.extend(fold_history)
        selected_by_fold[str(outer_fold)] = {
            "family": selected.family,
            "hidden_dim": selected.hidden_dim,
            "depth": selected.depth,
            "selected_using": "nested_inner_CV_on_outer_FIT_only",
        }
        fit_samples = {
            sample for row in outer_fit for sample in (row["sample_i"], row["sample_j"])
        }
        validation_samples = {
            sample for row in outer_validation for sample in (row["sample_i"], row["sample_j"])
        }
        permuted_events, permutation_audit = partitioned_degree_preserving_permutation(
            events, covariates, missing_pairs,
            {"FIT": fit_samples, "VALIDATION": validation_samples},
            stable_seed(str(run_seed), "carrier_permutation", str(outer_fold)),
            permutation_swap_multiplier,
            minimum_moved_carrier_fraction,
        )
        permutation_by_fold[str(outer_fold)] = permutation_audit
        arm_tokens = {
            "rare_enabled": rare_tokens,
            "carrier_permuted": event_tokens(permuted_events),
            "geometry_only": [],
            "baseline_only": [],
        }
        for arm in ARMS:
            rows = fit_cached_fold(
                selected, arm_tokens[arm], shared_geometry, covariates,
                outer_fit, outer_validation, component_map, outer_fold, arm,
                budgets[-1], stable_seed(str(run_seed), "final", str(outer_fold),
                                         selected.family, str(selected.hidden_dim)),
                **train_kwargs,
            )
            predictions[arm].extend(dict(row, run_seed=run_seed) for row in rows)
    effects = paired_effects(predictions, bootstrap_reps, run_seed)
    gate = promotion_gate(
        effects, minimum_relative_mse_reduction, minimum_positive_folds, outer_folds,
    )
    return {
        "selected_architecture_by_outer_fold": selected_by_fold,
        "promotion_gate": gate,
        "predictions": predictions,
        "effects": effects,
        "architecture_history": history,
        "outer_assignment": outer_assignment,
        "control_diagnostics": {
            "carrier_permutation_by_outer_fold": permutation_by_fold,
            "geometry_only": {
                "algorithm": "one detached L2-normalized shared encoding over every observed locus; positions/callability/common context retained; person identity, dosage, MAC and rare class erased",
                "n_shared_locus_events": len(shared_geometry),
                "materialized_person_by_locus_matrix": False,
            },
        },
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loci", required=True, type=Path)
    parser.add_argument("--carriers", required=True, type=Path)
    parser.add_argument("--missing", required=True, type=Path)
    parser.add_argument("--covariates", required=True, type=Path)
    parser.add_argument("--components", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--materialization-receipt", required=True, type=Path)
    parser.add_argument("--feature-chrom", default="chr22")
    parser.add_argument("--model-families", default="deep_sets,set_transformer")
    parser.add_argument("--halving-budgets", default="32,128")
    parser.add_argument("--halving-eta", type=int, default=2)
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--train-seeds", default="1701,2701,3701")
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    parser.add_argument("--pair-batch-size", type=int, default=1024)
    parser.add_argument("--permutation-swap-multiplier", type=float, default=10.0)
    parser.add_argument("--minimum-moved-carrier-fraction", type=float, default=0.50)
    parser.add_argument("--positive-control-budgets", default="16,64")
    parser.add_argument("--positive-control-seed", type=int, default=1701)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--minimum-relative-mse-reduction", type=float, default=0.10)
    parser.add_argument("--minimum-positive-folds", type=int, default=2)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_numpy()
    families = tuple(value.strip() for value in args.model_families.split(",") if value.strip())
    budgets = tuple(int(value) for value in args.halving_budgets.split(",") if value.strip())
    positive_control_budgets = tuple(
        int(value) for value in args.positive_control_budgets.split(",") if value.strip()
    )
    if args.outer_folds < 3 or args.inner_folds < 2:
        raise ContractError("M36B requires at least three outer and two inner folds")
    if len(budgets) < 2 or any(left >= right for left, right in zip(budgets, budgets[1:])):
        raise ContractError("halving budgets must be strictly increasing")
    if args.halving_eta < 2 or args.pair_batch_size <= 0 or args.bootstrap_reps < 40:
        raise ContractError("invalid halving, mini-batch or bootstrap configuration")
    if not 0 < args.minimum_relative_mse_reduction < 1:
        raise ContractError("minimum relative MSE reduction must be in (0,1)")
    if not 1 <= args.minimum_positive_folds <= args.outer_folds:
        raise ContractError("minimum positive folds must be between one and outer-fold count")
    if positive_control_budgets != (16, 64):
        raise ContractError("M36B triage positive controls are frozen at budgets 16,64")
    if args.positive_control_seed < 0:
        raise ContractError("positive-control seed must be nonnegative")
    if not 0 < args.minimum_moved_carrier_fraction <= 1:
        raise ContractError("minimum moved-carrier fraction must be in (0,1]")

    specs = available_specs(families)
    train_kwargs = {
        "pair_batch_size": args.pair_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "huber_delta": args.huber_delta,
    }

    # Known-answer recovery is deliberately completed before reading any real
    # materialization.  A failure therefore aborts the real-data lane rather
    # than allowing an instrument with unknown detectability to produce a
    # scientific-looking null result.
    positive_controls: dict[str, Any] = {}
    for control in POSITIVE_CONTROLS:
        control_events, control_covariates, control_components, control_targets, control_missing = (
            positive_control_inputs(control)
        )
        control_result = run_nested_screen(
            events=control_events,
            covariates=control_covariates,
            component_map=control_components,
            targets=control_targets,
            missing_pairs=control_missing,
            specs=specs,
            budgets=positive_control_budgets,
            halving_eta=args.halving_eta,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            run_seed=stable_seed(str(args.positive_control_seed), control),
            bootstrap_reps=args.bootstrap_reps,
            permutation_swap_multiplier=args.permutation_swap_multiplier,
            minimum_moved_carrier_fraction=args.minimum_moved_carrier_fraction,
            minimum_relative_mse_reduction=args.minimum_relative_mse_reduction,
            minimum_positive_folds=args.minimum_positive_folds,
            train_kwargs=train_kwargs,
        )
        positive_controls[control] = {
            "known_signal": (
                "absolute difference of person-level locus-weighted carrier sums"
                if control == "additive"
                else "product of person-level locus-weighted carrier sums"
            ),
            "n_people": len(control_covariates),
            "n_carrier_edges": len(control_events),
            "n_pairs": len(control_targets),
            "budgets": positive_control_budgets,
            "seed": stable_seed(str(args.positive_control_seed), control),
            "selected_architecture_by_outer_fold": control_result["selected_architecture_by_outer_fold"],
            "promotion_gate": control_result["promotion_gate"],
            "carrier_permutation_by_outer_fold": control_result["control_diagnostics"]["carrier_permutation_by_outer_fold"],
        }
        if not control_result["promotion_gate"]["passed"]:
            raise ContractError(
                f"M36B {control} positive control failed the complete nested/control-matched gate"
            )

    events, covariates, component_map, targets, missing_pairs = load_real_inputs(args)
    outer_assignment = component_folds(component_map, args.outer_folds)
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_predictions: list[dict[str, Any]] = []
    all_effects: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    control_diagnostics: dict[str, Any] = {}

    for run_seed in parse_train_seeds(args.train_seeds):
        result = run_nested_screen(
            events=events,
            covariates=covariates,
            component_map=component_map,
            targets=targets,
            missing_pairs=missing_pairs,
            specs=specs,
            budgets=budgets,
            halving_eta=args.halving_eta,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            run_seed=run_seed,
            bootstrap_reps=args.bootstrap_reps,
            permutation_swap_multiplier=args.permutation_swap_multiplier,
            minimum_moved_carrier_fraction=args.minimum_moved_carrier_fraction,
            minimum_relative_mse_reduction=args.minimum_relative_mse_reduction,
            minimum_positive_folds=args.minimum_positive_folds,
            train_kwargs=train_kwargs,
        )
        predictions = result["predictions"]
        effects = result["effects"]
        gate = result["promotion_gate"]
        all_history.extend(dict(row, run_seed=run_seed) for row in result["architecture_history"])
        all_predictions.extend(row for rows in predictions.values() for row in rows)
        all_effects.extend(dict(row, run_seed=run_seed) for row in effects)
        control_diagnostics[str(run_seed)] = result["control_diagnostics"]
        runs[str(run_seed)] = {
            "selected_architecture_by_outer_fold": result["selected_architecture_by_outer_fold"],
            "promotion_gate": gate,
        }

    write_tsv(args.outdir / "m36b_architecture_screen.tsv", all_history)
    write_tsv(args.outdir / "m36b_predictions.tsv", all_predictions)
    write_tsv(args.outdir / "m36b_paired_effects.tsv", all_effects)
    (args.outdir / "m36b_control_diagnostics.json").write_text(
        json.dumps(control_diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    effective_parameters = {
        "model_families": families,
        "halving_budgets": budgets,
        "halving_eta": args.halving_eta,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "train_seeds": parse_train_seeds(args.train_seeds),
        "bootstrap_reps": args.bootstrap_reps,
        "pair_batch_size": args.pair_batch_size,
        "permutation_swap_multiplier": args.permutation_swap_multiplier,
        "minimum_moved_carrier_fraction": args.minimum_moved_carrier_fraction,
        "positive_control_budgets": positive_control_budgets,
        "positive_control_seed": args.positive_control_seed,
        "positive_controls": POSITIVE_CONTROLS,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "huber_delta": args.huber_delta,
        "minimum_relative_mse_reduction": args.minimum_relative_mse_reduction,
        "minimum_positive_folds": args.minimum_positive_folds,
    }
    summary = {
        "stage": "M36B_CORA_SET_TRAIN",
        "status": "TRAINED_EXPLORATORY",
        "feature_chrom": args.feature_chrom,
        "scientific_target": "cross-chromosome common-IBD pair total as exploratory genealogy/structure outcome",
        "scientific_direction": "chr22 rare-event features predict common-IBD totals from autosomes chr1-chr21; one direction only",
        "not_an_LAI_test": True,
        "pre_real_positive_controls": {
            "executed_before_real_data_read": True,
            "required_controls": POSITIVE_CONTROLS,
            "all_passed": all(
                result["promotion_gate"]["passed"] for result in positive_controls.values()
            ),
            "runs": positive_controls,
        },
        "architecture_selection": "nested inner component-disjoint CV within each outer FIT partition",
        "outer_evaluation": "component-disjoint outer validation; cross-fold pairs excluded",
        "primary_aggregation": "equal-weight macro average across outer folds",
        "uncertainty": "paired approximate PC-Relate component-node Bayesian bootstrap",
        "effective_parameters": effective_parameters,
        "target_partition_coverage": target_partition_coverage(targets, component_map, outer_assignment),
        "runs": runs,
        "cross_seed_promotion": {
            "eligible": set(parse_train_seeds(args.train_seeds)) == {1701, 2701, 3701},
            "passed": (
                set(parse_train_seeds(args.train_seeds)) == {1701, 2701, 3701}
                and all(run["promotion_gate"]["passed"] for run in runs.values())
            ),
            "required": "all fixed seeds 1701, 2701 and 3701 pass the same three-comparator gate; one seed is triage only",
            "passed_seeds": [seed for seed, run in runs.items() if run["promotion_gate"]["passed"]],
            "total_seeds": len(runs),
        },
    }
    (args.outdir / "m36b_train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    try:
        run(parse_args())
    except ContractError as error:
        raise SystemExit(f"M36B training error: {error}") from error


if __name__ == "__main__":
    main()
