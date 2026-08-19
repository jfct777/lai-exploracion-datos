#!/usr/bin/env python3
"""M29 DEV gate: test whether rare alleles add LAI information at identical loci.

The analysis is deliberately diploid and unphased.  Homologue order never enters
the feature matrix or the primary ancestry-dose loss.  Two independent M28-v2
roots are used in both leave-one-root-out directions.  B0-cal, BR and every
Bsham use the same estimator and feature dimension; the sham changes only the
ancestry labels of complete REF_LAI individuals.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m28d_b0_scorer import (  # noqa: E402
    ANCESTRIES,
    GeneticMap,
    TruthSegment,
    load_fb,
    load_genetic_map,
    load_msp,
    load_truth,
)


FEATURE_NAMES = (
    "b0_AFR",
    "b0_EUR",
    "b0_ASIA",
    "rare_load",
    "rare_support_AFR",
    "rare_support_EUR",
    "rare_support_ASIA",
)
ARMS = ("B0_CAL", "BR")


@dataclass(frozen=True)
class Window:
    left: int
    right: int
    cm_left: float
    cm_right: float
    baseline: np.ndarray  # target x ancestry; diploid mean posterior


@dataclass
class RootData:
    name: str
    seed: int
    samples: list[str]
    windows: list[Window]
    truth: np.ndarray  # row x ancestry proportions
    lengths_cm: np.ndarray
    sample_index: np.ndarray
    real_features: np.ndarray
    b0_features: np.ndarray
    sham_features: list[np.ndarray]


def minor_diploid_dosage(states: Sequence[int], minor_code: int) -> int:
    """Count copies of the selected minor allele, including minor_code=0."""
    if minor_code not in (0, 1) or len(states) != 2 or any(state not in (0, 1) for state in states):
        raise ValueError("minor dosage requires two binary states and minor_code 0/1")
    return sum(int(state == minor_code) for state in states)


def open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_git_commit(value: str | None) -> str | None:
    if value is not None and not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("git_commit must be an exact lowercase 40-character hexadecimal commit")
    return value


def authenticate_script(path: Path, expected_sha256: str | None) -> str:
    observed = sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"M29 script sha256 mismatch: {observed} != {expected_sha256}")
    return observed


def require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label}: missing file {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label}: sha256 mismatch {observed} != {expected}")


def stable_seed(base: int, root: int, replicate: int) -> int:
    payload = f"M29|{base}|{root}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def permute_diploid_labels(labels: Sequence[str], seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    rng = np.random.default_rng(seed)
    permuted = labels[rng.permutation(len(labels))]
    if sorted(permuted.tolist()) != sorted(labels.tolist()):
        raise AssertionError("permutation changed ancestry counts")
    return permuted


def _load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    stage = contract.get("stage")
    if stage == "M29_SAME_LOCUS_DEV":
        if contract.get("status") != "PRE_FROZEN_BEFORE_DEV":
            raise ValueError("M29 preregistration is not frozen for DEV")
        if contract["model"]["C_grid"] != [0.01, 0.1, 1.0, 10.0]:
            raise ValueError("C grid differs from preregistration")
    elif stage == "M29R_MINOR_ORIENTATION_ERRATUM":
        if contract.get("status") != "FROZEN_DIAGNOSTIC_ERRATUM_NO_NEW_VALIDATION":
            raise ValueError("M29R erratum is not frozen for diagnostic rerun")
        original = contract.get("original_preregistration", {})
        if original.get("stage") != "M29_SAME_LOCUS_DEV" or not isinstance(original.get("sha256"), str) or len(original["sha256"]) != 64:
            raise ValueError("M29R does not reference an exact original M29 preregistration")
        if contract["model"].get("C_grid") != [10.0] or contract["model"].get("fixed_C") != 10.0:
            raise ValueError("M29R permits only the historically selected fixed C=10.0")
        if contract["model"].get("C_selection") != "none_fixed_from_historical_M29_all_arms":
            raise ValueError("M29R must not reselect C bidirectionally")
    else:
        raise ValueError("contract is neither M29 nor the M29R orientation erratum")
    if contract.get("version") != 1:
        raise ValueError("unsupported M29/M29R contract version")
    if contract["sham"]["replicates"] != 32:
        raise ValueError("sham count differs from preregistration")
    return contract


def _load_pool_manifest(path: Path) -> tuple[dict[str, list[tuple[str, tuple[int, int]]]], dict[int, str]]:
    grouped: dict[str, dict[str, list[int]]] = {}
    node_ancestry: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"role", "ancestry", "individual_id", "node_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("pool manifest has unexpected header")
        for row in reader:
            role, ancestry = row["role"], row["ancestry"]
            if ancestry not in ANCESTRIES or role not in {"FREQ", "REF_LAI", "DONOR"}:
                raise ValueError("pool manifest has an unexpected role or ancestry")
            node = int(row["node_id"])
            if node in node_ancestry:
                raise ValueError("pool manifest duplicates a node")
            node_ancestry[node] = ancestry
            grouped.setdefault(role, {}).setdefault(row["individual_id"], []).append(node)
    output: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    owners: dict[str, str] = {}
    for role, individuals in grouped.items():
        output[role] = []
        for individual, nodes in sorted(individuals.items()):
            if len(nodes) != 2:
                raise ValueError(f"{individual} does not have exactly two homologues")
            if individual in owners:
                raise ValueError(f"{individual} crosses roles")
            owners[individual] = role
            ancestries = {node_ancestry[node] for node in nodes}
            if len(ancestries) != 1:
                raise ValueError(f"{individual} crosses ancestries")
            output[role].append((individual, tuple(sorted(nodes))))
    if set(output) != {"FREQ", "REF_LAI", "DONOR"}:
        raise ValueError("pool manifest lacks a required role")
    return output, node_ancestry


def _load_rare_catalog(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chrom", "position", "minor_code", "mac", "an", "maf"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("rare catalog has unexpected header")
        for row in reader:
            position = int(row["position"])
            if row["chrom"].removeprefix("chr") != "22" or position in rows:
                raise ValueError("rare catalog chromosome or position is invalid")
            rows[position] = row
    return rows


def _load_target_rare(path: Path, selected_minor_codes: dict[int, int]) -> tuple[list[str], np.ndarray, np.ndarray]:
    positions: list[int] = []
    dosage_rows: list[list[float]] = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("rare haplotypes lack a header")
        fixed = {"chrom", "position", "minor_code"}
        hap_columns = [column for column in reader.fieldnames if column not in fixed]
        samples = sorted({column.rsplit("_h", 1)[0] for column in hap_columns})
        expected = [f"{sample}_h{hap}" for sample in samples for hap in (0, 1)]
        if sorted(hap_columns) != sorted(expected):
            raise ValueError("rare haplotypes do not contain complete diploid targets")
        for row in reader:
            position = int(row["position"])
            if position not in selected_minor_codes:
                continue
            target_minor_code = int(row["minor_code"])
            if target_minor_code != selected_minor_codes[position]:
                raise ValueError(f"target/catalog minor_code mismatch at {position}")
            values = []
            for sample in samples:
                pair = [row[f"{sample}_h0"], row[f"{sample}_h1"]]
                if any(value in {"", ".", "NA"} for value in pair):
                    values.append(float("nan"))
                else:
                    states = [int(pair[0]), int(pair[1])]
                    try:
                        dosage = minor_diploid_dosage(states, target_minor_code)
                    except ValueError as exc:
                        raise ValueError(f"target haplotype state is not binary at {position}") from exc
                    values.append(float(dosage))
            positions.append(position)
            dosage_rows.append(values)
    if set(positions) != set(selected_minor_codes):
        raise ValueError("target rare table does not cover the selected FREQ universe")
    order = np.argsort(np.asarray(positions))
    return samples, np.asarray(positions, dtype=np.int64)[order], np.asarray(dosage_rows, dtype=float)[order]


def _derive_freq_universe_and_ref_support(tree_path: Path, pool_path: Path, catalog_path: Path, genomic_offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    import tskit

    pools, node_ancestry = _load_pool_manifest(pool_path)
    catalog = _load_rare_catalog(catalog_path)
    ts = tskit.load(str(tree_path))
    freq_people = pools["FREQ"]
    ref_people = pools["REF_LAI"]
    freq_nodes = np.asarray([node for _, pair in freq_people for node in pair], dtype=int)
    ref_nodes = np.asarray([node for _, pair in ref_people for node in pair], dtype=int)
    ref_labels = [node_ancestry[pair[0]] for _, pair in ref_people]
    sample_index = {int(node): index for index, node in enumerate(ts.samples())}
    ordered_nodes = np.concatenate([freq_nodes, ref_nodes])
    genotype_indexes = np.asarray([sample_index[int(node)] for node in ordered_nodes], dtype=int)
    positions: list[int] = []
    minor_codes: list[int] = []
    ref_genotypes: list[np.ndarray] = []
    for variant in ts.variants():
        position = genomic_offset + int(variant.site.position)
        row = catalog.get(position)
        if row is None:
            continue
        if len(variant.alleles) != 2:
            raise ValueError(f"catalog position {position} is not biallelic")
        minor = int(row["minor_code"])
        if minor not in (0, 1):
            raise ValueError("minor code must be 0 or 1")
        geno = np.asarray(variant.genotypes)[genotype_indexes]
        if np.any(geno < 0):
            raise ValueError("simulation tree contains missing genotypes")
        freq_minor = (geno[: len(freq_nodes)] == minor).astype(np.int8)
        mac, an = int(freq_minor.sum()), len(freq_minor)
        carriers = sum(int(freq_minor[2 * i] + freq_minor[2 * i + 1] > 0) for i in range(len(freq_people)))
        if mac != int(row["mac"]) or an != int(row["an"]) or abs(mac / an - float(row["maf"])) > 1e-12:
            raise ValueError(f"FREQ catalog mismatch at {position}")
        if carriers < 2:
            continue
        ref_minor = (geno[len(freq_nodes) :] == minor).astype(np.int8).reshape(len(ref_people), 2).sum(axis=1)
        # FREQ alone defines rarity and carrier eligibility. REF_LAI enters only
        # as an operational observability gate: an unsupported allele cannot
        # create a real or permuted ancestry-support feature.
        if int(ref_minor.sum()) == 0:
            continue
        positions.append(position)
        minor_codes.append(minor)
        ref_genotypes.append(ref_minor)
    if not positions:
        raise ValueError("FREQ rare universe is empty after the two-carrier rule")
    if set(positions) - set(catalog):
        raise AssertionError("selected a site outside the FREQ catalog")
    return np.asarray(positions, dtype=np.int64), np.asarray(minor_codes, dtype=np.int8), np.asarray(ref_genotypes, dtype=np.int8), ref_labels, np.asarray([pair for _, pair in ref_people], dtype=int)


def ancestry_support(ref_dosage: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    counts = np.column_stack([ref_dosage[:, labels == ancestry].sum(axis=1) for ancestry in ANCESTRIES]).astype(float)
    totals = counts.sum(axis=1)
    support = np.zeros_like(counts)
    nonzero = totals > 0
    support[nonzero] = counts[nonzero] / totals[nonzero, None]
    return support


def _prediction_windows(fb_path: Path, msp_path: Path, genetic_map: GeneticMap, domain: tuple[int, int]) -> tuple[list[str], list[Window]]:
    fb_positions, _, fb_samples, fb_probabilities = load_fb(fb_path)
    msp, msp_samples, _ = load_msp(msp_path)
    if fb_samples != msp_samples or len(fb_positions) != len(msp):
        raise ValueError("FB/MSP target or window mismatch")
    left_domain, right_domain = domain
    boundaries = [left_domain]
    for before, after in zip(msp, msp[1:]):
        midpoint = (int(before["epos"]) + int(after["spos"])) // 2 + 1
        boundaries.append(midpoint)
    boundaries.append(right_domain)
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("derived Gnomix windows are not positive and ordered")
    windows: list[Window] = []
    for index, probabilities in enumerate(fb_probabilities):
        baseline = np.asarray(
            [[(probabilities[sample][0][a] + probabilities[sample][1][a]) / 2 for a in range(3)] for sample in fb_samples],
            dtype=float,
        )
        left, right = boundaries[index], boundaries[index + 1]
        windows.append(Window(left, right, genetic_map.cm_at(left), genetic_map.cm_at(min(right, genetic_map.positions[-1])), baseline))
    return fb_samples, windows


def _half_open_cm_span(genetic_map: GeneticMap, left: int, right: int) -> float:
    """Return an additive genetic length for a half-open physical interval.

    Internal interval boundaries are evaluated at the shared boundary, matching
    M28D.  The chromosome domain ends one base beyond the final map coordinate,
    so only that terminal edge is clipped to the authenticated map endpoint.
    """
    if right <= left:
        return 0.0
    right_coordinate = min(right, genetic_map.positions[-1])
    if right_coordinate < left:
        return 0.0
    return max(0.0, genetic_map.cm_at(right_coordinate) - genetic_map.cm_at(left))


def _integrated_truth(truth: dict[str, tuple[list[TruthSegment], list[TruthSegment]]], samples: Sequence[str], windows: Sequence[Window], genetic_map: GeneticMap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    lengths: list[float] = []
    sample_indexes: list[int] = []
    for sample_i, sample in enumerate(samples):
        for window in windows:
            length = window.cm_right - window.cm_left
            if length <= 0:
                continue
            dosage = np.zeros(3, dtype=float)
            for hap in (0, 1):
                for segment in truth[sample][hap]:
                    left, right = max(window.left, segment.start), min(window.right, segment.end)
                    if left < right:
                        dosage[ANCESTRIES.index(segment.ancestry)] += _half_open_cm_span(genetic_map, left, right)
            proportions = dosage / (2 * length)
            if not np.isclose(proportions.sum(), 1.0, atol=2e-5):
                raise ValueError("integrated truth does not sum to one")
            rows.append(proportions)
            lengths.append(length)
            sample_indexes.append(sample_i)
    return np.asarray(rows), np.asarray(lengths), np.asarray(sample_indexes)


def build_features(samples: Sequence[str], windows: Sequence[Window], rare_positions: np.ndarray, target_dosage: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if target_dosage.shape != (len(rare_positions), len(samples)) or support.shape != (len(rare_positions), 3):
        raise ValueError("rare feature arrays have inconsistent dimensions")
    real_rows: list[np.ndarray] = []
    b0_rows: list[np.ndarray] = []
    for sample_i in range(len(samples)):
        for window in windows:
            mask = (rare_positions >= window.left) & (rare_positions < window.right)
            callable_mask = mask & np.isfinite(target_dosage[:, sample_i])
            n_callable = int(callable_mask.sum())
            rare = np.zeros(4, dtype=float)
            if n_callable:
                dosage = target_dosage[callable_mask, sample_i]
                rare[0] = dosage.sum() / (2 * n_callable)
                rare[1:] = (dosage[:, None] * support[callable_mask]).sum(axis=0) / (2 * n_callable)
            base = window.baseline[sample_i]
            real_rows.append(np.concatenate([base, rare]))
            b0_rows.append(np.concatenate([base, np.zeros(4)]))
    return np.asarray(real_rows), np.asarray(b0_rows)


def assert_same_locus_identity(real: np.ndarray, sham: np.ndarray) -> None:
    if real.shape != sham.shape or real.shape[1] != len(FEATURE_NAMES):
        raise ValueError("arm feature dimensions differ")
    if not np.array_equal(real[:, :4], sham[:, :4], equal_nan=True):
        raise ValueError("sham altered baseline predictions, loci, genotype load or missingness")
    if not np.allclose(real[:, 4:].sum(axis=1), sham[:, 4:].sum(axis=1), atol=1e-12):
        raise ValueError("sham altered total rare dosage support")


def _fit_soft_multinomial(X: np.ndarray, truth: np.ndarray, lengths: np.ndarray, C: float, max_iter: int, tol: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    expanded_x = np.repeat(scaled, 3, axis=0)
    labels = np.tile(np.arange(3), len(X))
    weights = (truth * lengths[:, None]).reshape(-1)
    keep = weights > 0
    model = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=max_iter, tol=tol, random_state=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(expanded_x[keep], labels[keep], sample_weight=weights[keep])
    if any(issubclass(item.category, ConvergenceWarning) for item in caught) or np.any(model.n_iter_ >= max_iter):
        raise RuntimeError(f"multinomial fit did not converge for C={C}")
    if list(model.classes_) != [0, 1, 2]:
        raise RuntimeError("multinomial fit lost an ancestry class")
    return scaler, model


def _predict(scaler: StandardScaler, model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(scaler.transform(X))
    if not np.all(np.isfinite(probabilities)) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("invalid multinomial probabilities")
    return probabilities


def individual_macro_mae(pred: np.ndarray, truth: np.ndarray, lengths: np.ndarray, sample_index: np.ndarray, n_samples: int) -> np.ndarray:
    errors = np.abs(pred - truth).mean(axis=1)
    output = np.empty(n_samples, dtype=float)
    for sample in range(n_samples):
        mask = sample_index == sample
        output[sample] = np.average(errors[mask], weights=lengths[mask])
    return output


def _project_unordered_state(values: np.ndarray) -> np.ndarray:
    states = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], dtype=float)
    return np.argmin(((values[:, None, :] - states[None, :, :]) ** 2).sum(axis=2), axis=1)


def _secondary_metrics(pred: np.ndarray, truth: np.ndarray, lengths: np.ndarray) -> dict:
    ancestry_mae = {ancestry: float(np.average(np.abs(pred[:, i] - truth[:, i]), weights=lengths)) for i, ancestry in enumerate(ANCESTRIES)}
    brier = float(np.average(((pred - truth) ** 2).sum(axis=1), weights=lengths))
    observed, predicted = _project_unordered_state(truth), _project_unordered_state(pred)
    accuracy = float(np.average(observed == predicted, weights=lengths))
    f1s = []
    for state in range(6):
        tp = float(lengths[(observed == state) & (predicted == state)].sum())
        fp = float(lengths[(observed != state) & (predicted == state)].sum())
        fn = float(lengths[(observed == state) & (predicted != state)].sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"ancestry_mae": ancestry_mae, "composition_brier": brier, "unordered_state_accuracy": accuracy, "unordered_state_macro_f1_fixed_six": float(np.mean(f1s))}


def fit_and_score(train_x: np.ndarray, test_x: np.ndarray, train: RootData, test: RootData, C: float, max_iter: int, tol: float) -> tuple[dict, np.ndarray]:
    scaler, model = _fit_soft_multinomial(train_x, train.truth, train.lengths_cm, C, max_iter, tol)
    pred = _predict(scaler, model, test_x)
    individual = individual_macro_mae(pred, test.truth, test.lengths_cm, test.sample_index, len(test.samples))
    return {"macro_mae": float(individual.mean()), **_secondary_metrics(pred, test.truth, test.lengths_cm)}, individual


def _root_inputs(args: argparse.Namespace, label: str) -> dict[str, Path]:
    return {key: Path(getattr(args, f"{label}_{key}")) for key in ("tree", "pools", "report", "manifest", "catalog", "haplotypes", "truth", "fb", "msp")}


def load_root(label: str, paths: dict[str, Path], spec: dict, binding_path: Path, contract: dict, genetic_map: GeneticMap) -> RootData:
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("stage") != "M29_AUTHENTICATED_B0_BINDING" or binding.get("root_seed") != spec["root_seed"]:
        raise ValueError(f"{label}: invalid B0 binding")
    baseline_hashes = binding.get("sha256", {})
    if set(baseline_hashes) != {"fb", "msp"} or any(not isinstance(value, str) or len(value) != 64 for value in baseline_hashes.values()):
        raise ValueError(f"{label}: B0 binding must contain exact FB/MSP SHA-256 values")
    expected_hashes = {**spec["sha256"], **baseline_hashes}
    for key, expected in expected_hashes.items():
        require_file_hash(paths[key], expected, f"{label}.{key}")
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    if report.get("stage") != "M28_LAI_SIMULATION_PREFLIGHT" or report.get("root_seed") != spec["root_seed"]:
        raise ValueError(f"{label}: M28-v2 report/root mismatch")
    genomic_offset = int(contract["chromosome_domain"]["start_bp"])
    positions, minor_codes, ref_dosage, ref_labels, _ = _derive_freq_universe_and_ref_support(paths["tree"], paths["pools"], paths["catalog"], genomic_offset)
    selected_minor_codes = {int(position): int(minor) for position, minor in zip(positions, minor_codes)}
    target_samples, target_positions, target_dosage = _load_target_rare(paths["haplotypes"], selected_minor_codes)
    if not np.array_equal(np.sort(positions), target_positions):
        raise ValueError(f"{label}: tree and target rare positions differ")
    order = np.argsort(positions)
    positions, minor_codes, ref_dosage = positions[order], minor_codes[order], ref_dosage[order]
    domain = (int(contract["chromosome_domain"]["start_bp"]), int(contract["chromosome_domain"]["end_bp_exclusive"]))
    fb_samples, windows = _prediction_windows(paths["fb"], paths["msp"], genetic_map, domain)
    if target_samples != fb_samples:
        raise ValueError(f"{label}: target order differs between rare haplotypes and Gnomix")
    truth_segments = load_truth(paths["truth"], fb_samples, "22", domain[0], domain[1])
    truth, lengths, sample_index = _integrated_truth(truth_segments, fb_samples, windows, genetic_map)
    real_support = ancestry_support(ref_dosage, ref_labels)
    real_features, b0_features = build_features(fb_samples, windows, positions, target_dosage, real_support)
    shams: list[np.ndarray] = []
    for replicate in range(contract["sham"]["replicates"]):
        permuted = permute_diploid_labels(ref_labels, stable_seed(contract["sham"]["base_seed"], spec["root_seed"], replicate))
        sham_support = ancestry_support(ref_dosage, permuted)
        sham_features, _ = build_features(fb_samples, windows, positions, target_dosage, sham_support)
        assert_same_locus_identity(real_features, sham_features)
        shams.append(sham_features)
    return RootData(label, spec["root_seed"], fb_samples, windows, truth, lengths, sample_index, real_features, b0_features, shams)


def run_gate(root_a: RootData, root_b: RootData, contract: dict) -> tuple[dict, list[dict], list[dict]]:
    Cs = contract["model"]["C_grid"]
    max_iter, tol = contract["model"]["max_iter"], contract["model"]["tol"]
    directions = [(root_a, root_b), (root_b, root_a)]
    metrics: list[dict] = []
    individual_rows: list[dict] = []
    for root in (root_a, root_b):
        matrices = [root.b0_features, root.real_features, *root.sham_features]
        if any(matrix.shape[1] != contract["model"]["feature_dimension"] for matrix in matrices):
            raise ValueError("B0-cal/BR/BSHAM do not have equal feature dimension")
    for train, test in directions:
        for C in Cs:
            for arm, train_x, test_x, sham in [("B0_CAL", train.b0_features, test.b0_features, None), ("BR", train.real_features, test.real_features, None)]:
                score, individuals = fit_and_score(train_x, test_x, train, test, C, max_iter, tol)
                metrics.append({"direction": f"{train.name}_to_{test.name}", "C": C, "arm": arm, "sham": "", **score, "ancestry_mae": json.dumps(score["ancestry_mae"], sort_keys=True)})
                for sample, value in zip(test.samples, individuals):
                    individual_rows.append({"direction": f"{train.name}_to_{test.name}", "C": C, "arm": arm, "sham": "", "sample": sample, "macro_mae": float(value)})
            for replicate in range(contract["sham"]["replicates"]):
                score, individuals = fit_and_score(train.sham_features[replicate], test.sham_features[replicate], train, test, C, max_iter, tol)
                metrics.append({"direction": f"{train.name}_to_{test.name}", "C": C, "arm": "BSHAM", "sham": replicate, **score, "ancestry_mae": json.dumps(score["ancestry_mae"], sort_keys=True)})
                for sample, value in zip(test.samples, individuals):
                    individual_rows.append({"direction": f"{train.name}_to_{test.name}", "C": C, "arm": "BSHAM", "sham": replicate, "sample": sample, "macro_mae": float(value)})
    def select_c(arm: str, sham: int | None = None) -> float:
        means = {C: np.mean([r["macro_mae"] for r in metrics if r["arm"] == arm and r["C"] == C and (sham is None or r["sham"] == sham)]) for C in Cs}
        return min(Cs, key=lambda value: (means[value], value))

    is_erratum = contract["stage"] == "M29R_MINOR_ORIENTATION_ERRATUM"
    if is_erratum:
        selected_b0 = selected_br = float(contract["model"]["fixed_C"])
        selected_shams = {replicate: selected_b0 for replicate in range(contract["sham"]["replicates"])}
    else:
        selected_b0 = select_c("B0_CAL")
        selected_br = select_c("BR")
        selected_shams = {replicate: select_c("BSHAM", replicate) for replicate in range(contract["sham"]["replicates"])}
    direction_results = []
    passed = True
    for train, test in directions:
        direction = f"{train.name}_to_{test.name}"
        b0 = next(r["macro_mae"] for r in metrics if r["direction"] == direction and r["C"] == selected_b0 and r["arm"] == "B0_CAL")
        br = next(r["macro_mae"] for r in metrics if r["direction"] == direction and r["C"] == selected_br and r["arm"] == "BR")
        sham_scores = [next(r["macro_mae"] for r in metrics if r["direction"] == direction and r["C"] == selected_shams[replicate] and r["arm"] == "BSHAM" and r["sham"] == replicate) for replicate in range(contract["sham"]["replicates"])]
        sham_improvements = [b0 - value for value in sham_scores]
        threshold = float(np.quantile(sham_improvements, 0.95, method="higher"))
        improvement = b0 - br
        direction_pass = improvement > 0 and improvement > threshold
        passed &= direction_pass
        direction_results.append({"direction": direction, "b0_macro_mae": b0, "br_macro_mae": br, "br_improvement": improvement, "sham_improvement_p95": threshold, "pass": bool(direction_pass)})
    if is_erratum:
        decision = "M29R_ERRATUM_SIGNAL_REPLICATED" if passed else "M29R_ERRATUM_NO_REPLICATED_INCREMENTAL_SIGNAL"
        scope = "DIAGNOSTIC_ERRATUM_DEV_ONLY_NO_NEW_VALIDATION"
    else:
        decision = "GO_FREEZE_DEV" if passed else "STOP_DEV_NO_REPLICATED_INCREMENTAL_SIGNAL"
        scope = "DEV_ONLY_NO_VALID_ACCESS"
    summary = {
        "stage": contract["stage"],
        "scope": scope,
        "selected_C": {"B0_CAL": selected_b0, "BR": selected_br, "BSHAM": {str(key): value for key, value in selected_shams.items()}},
        "directions": direction_results,
        "decision": decision,
        "C_policy": "fixed_historical_C_10_no_reselection" if is_erratum else "mean_bidirectional_DEV_selection",
        "primary_metric": "equal-individual mean of genetic-length-weighted macro ancestry-proportion MAE",
        "secondary_metrics": ["MAE_by_ancestry", "composition_Brier", "unordered_diploid_state_accuracy", "unordered_diploid_state_macro_F1_fixed_six"],
        "boundary_metric": "not_estimated_in_DEV: diploid window posteriors do not identify phased haplotype boundaries; no phase was imputed",
        "claims_excluded": contract.get(
            "claims_excluded",
            ["validated_LAI_improvement", "utility_in_DNABR", "Native_American_inference"],
        ),
    }
    return summary, metrics, individual_rows


def _write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty result table")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--genetic-map", required=True)
    parser.add_argument("--git-commit")
    parser.add_argument("--script-sha256")
    parser.add_argument("--outdir", required=True)
    for label in ("root_a", "root_b"):
        for key in ("tree", "pools", "report", "manifest", "catalog", "haplotypes", "truth", "fb", "msp"):
            parser.add_argument(f"--{label.replace('_', '-')}-{key.replace('_', '-')}", dest=f"{label}_{key}", required=True)
        parser.add_argument(f"--{label.replace('_', '-')}-binding", dest=f"{label}_binding", required=True)
    args = parser.parse_args()
    script_sha256 = authenticate_script(Path(__file__).resolve(), args.script_sha256)
    git_commit = validate_git_commit(args.git_commit)
    contract_path = Path(args.preregistration)
    contract = _load_contract(contract_path)
    genetic_map_path = Path(args.genetic_map)
    require_file_hash(genetic_map_path, contract["shared_inputs"]["genetic_map_sha256"], "genetic_map")
    genetic_map = load_genetic_map(genetic_map_path, "22")
    roots = contract["roots"]
    root_a = load_root("root_a", _root_inputs(args, "root_a"), roots["root_a"], Path(args.root_a_binding), contract, genetic_map)
    root_b = load_root("root_b", _root_inputs(args, "root_b"), roots["root_b"], Path(args.root_b_binding), contract, genetic_map)
    if root_a.seed == root_b.seed:
        raise ValueError("leave-one-root-out requires distinct roots")
    summary, metrics, individuals = run_gate(root_a, root_b, contract)
    summary["code_sha256"] = {"m29_same_locus_dev.py": script_sha256}
    summary["git_commit"] = git_commit
    summary["preregistration_sha256"] = sha256_file(contract_path)
    summary["input_sha256"] = {label: {**{key: sha256_file(path) for key, path in _root_inputs(args, label).items()}, "binding": sha256_file(Path(getattr(args, f"{label}_binding")))} for label in ("root_a", "root_b")}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "m29_dev_summary.public.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tsv(outdir / "m29_dev_metrics.tsv", metrics)
    _write_tsv(outdir / "m29_dev_individual_errors.tsv.gz", individuals)
    manifest = {
        "stage": contract["stage"],
        "git_commit": git_commit,
        "code_sha256": {"m29_same_locus_dev.py": script_sha256},
        "sha256": {path.name: sha256_file(path) for path in sorted(outdir.iterdir())},
    }
    (outdir / "m29_dev.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
