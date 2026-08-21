#!/usr/bin/env python3
"""Materialize and benchmark the truth-free M32 chr22 locus tensor.

This stage measures representation integrity and CPU/RAM behaviour only.  It
does not accept local-ancestry truth, boundary annotations, labels for model
training, or any argument capable of selecting a scientific radius.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import resource
import struct
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import tskit

from m31_ordered_linear import (
    ANCESTRIES,
    load_genetic_map,
    load_ordered_rare,
    load_ref_minor_dosage,
)
from m32_locus_contract import sha256_file, validate_git_commit


STAGE = "M32_PACKED_CHR22_BENCHMARK"
ROOTS = {"root17": 20260817, "root18": 20260818}
REQUIRED_SOURCES = {
    "bin/m32_source_auth.py",
    "bin/m32_packed_benchmark.py",
    "bin/m31_ordered_linear.py",
    "bin/m32_locus_contract.py",
    "bin/m32_locus_smoke.py",
    "conf/m32_packed_benchmark_preregistration.json",
    "conf/m32_packed_benchmark.config",
    "modules/32_PACKED_BENCHMARK.nf",
    "workflows/m32_packed_benchmark.nf",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def update_array_digest(digest: "hashlib._Hash", value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish one JSON atomically without ever replacing an existing path."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # fails with FileExistsError; no overwrite race
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write, reopen and publish a private NPZ without replacement."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as loaded:
            require(set(loaded.files) == set(arrays), "temporary tensor inventory drifted")
            for name, expected in arrays.items():
                observed = loaded[name]
                require(observed.shape == expected.shape and observed.dtype == expected.dtype,
                        f"temporary tensor metadata drifted for {name}")
                require(array_hash(observed) == array_hash(expected),
                        f"temporary tensor content drifted for {name}")
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_sources(values: Sequence[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        relative, separator, staged = value.partition("=")
        require(bool(separator) and bool(relative), "invalid source specification")
        require(not relative.startswith("/") and ".." not in Path(relative).parts, "unsafe source path")
        require(relative not in sources, "duplicate source specification")
        sources[relative] = Path(staged)
    require(set(sources) == REQUIRED_SOURCES, "source set does not cover the complete M32 implementation")
    return sources


def authenticate_staged_sources(manifest_path: Path, git_commit: str,
                                sources: dict[str, Path]) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("stage") == "M32_PACKED_SOURCE_AUTH", "source-auth stage drifted")
    require(manifest.get("status") == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES", "source-auth did not pass")
    require(manifest.get("git_commit") == git_commit, "source-auth commit drifted")
    expected = manifest.get("source_sha256", {})
    require(set(expected) == REQUIRED_SOURCES and set(sources) == REQUIRED_SOURCES,
            "source-auth inventory drifted")
    observed = {relative: sha256_file(path) for relative, path in sorted(sources.items())}
    require(observed == expected, "staged sources differ from the authenticated manifest")
    return observed


def load_contract(path: Path, root_label: str, root_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("stage") == STAGE and payload.get("version") == 1, "unsupported M32 contract")
    require(payload.get("status") == "TRUTH_FREE_TENSOR_IO_BENCHMARK_ONLY", "M32 contract status drifted")
    require(payload.get("radii_cm") == [0.05, 0.1, 0.2, 0.5], "radius screen drifted")
    require(payload.get("person_batches") == [1, 4, 8], "person-batch screen drifted")
    require(payload.get("token_budgets") == [65536, 262144, 1048576], "token-budget screen drifted")
    require(payload.get("backends") == ["contiguous_packed", "length_sorted_packed"], "backend screen drifted")
    require(payload.get("performance_marker_sample") == 512, "marker sample size drifted")
    require(payload.get("timing") == {"warmups": 1, "repetitions": 3, "cold_io_claimed": False},
            "timing contract drifted")
    require(payload.get("memory") == {
        "warning_fraction": 0.7, "stop_fraction": 0.8,
        "denominator": "process_cgroup_limit", "expected_limit_bytes": 8589934592,
    }, "memory contract drifted")
    require(payload.get("token_definition") == "one_valid_diploid_person_by_marker_by_rare_locus_cell",
            "token definition drifted")
    require(payload.get("interval_definition") == {
        "start": "bisect_left(rare_cm, marker_cm - radius_cm)",
        "stop": "bisect_right(rare_cm, marker_cm + radius_cm)",
        "bounds": "inclusive_in_cm_represented_as_half_open_index_interval",
        "independent_oracle": "streaming_two_pointer_all_markers",
    }, "interval contract drifted")
    require(payload.get("container_image_id") ==
            "sha256:2c30d018028636ac1b7a4890641e04b3e15be8c79d991dfade35b90db0e17bd1",
            "container image identity drifted")
    require(payload.get("axes") == {
        "flare": ["marker", "person", "haplotype", "ancestry"],
        "target_haplotype": ["locus", "person", "haplotype"],
        "target_diploid": ["locus", "person"],
        "reference": ["locus", "ancestry"],
        "ancestry_order": ["AFR", "EUR", "ASIA"],
        "haplotype_order": ["h0", "h1"],
    }, "axis contract drifted")
    require(payload.get("dtypes") == {
        "target_haplotype_presence": "int8_missing_minus_one",
        "target_minor_dosage": "int8_missing_minus_one",
        "target_observed_mask": "bool_not_biological_callability",
        "interval_start_stop": "int32",
        "ref_minor_count": "uint16_with_overflow_check",
        "ref_callable_alleles": "uint16_with_overflow_check",
        "ref_person_level_minor_dosage": "int8_missing_minus_one",
        "ref_person_level_observed_mask": "bool",
        "ref_person_level_ancestry_label": "int8_AFR0_EUR1_ASIA2",
        "ref_support": "float32",
        "ref_support_observed_mask": "bool_separate_from_minor_allele_presence",
        "ref_minor_supported": "bool_minor_count_greater_than_zero",
        "flare_probabilities_raw": "float32",
        "coordinates_cm": "float64",
    }, "dtype contract drifted")
    require(payload.get("simulation_limitations") == {
        "target_missing_expected": 0,
        "reference_missing_expected": 0,
        "missing_semantics_checked_only_by_separate_synthetic_fixture": True,
        "roots_are_not_independent_validation": True,
    }, "simulation limitation contract drifted")
    require(payload.get("reference_support_policy") == {
        "freq_role": "defines_the_minor_allele_in_the_M31_rare_locus_catalog",
        "ref_role": "measures_observed_minor_allele_support_in_parental_reference_people",
        "freq_and_ref_counts_are_not_required_to_match": True,
        "zero_ref_minor_count_with_nonzero_ref_callable_alleles": "observed_absence_not_missingness",
        "filter_freq_loci_by_ref_support": False,
        "retain_person_level_ref_dosage_labels_and_observed_mask": True,
    }, "reference support policy drifted")
    require(payload.get("stop_conditions") == [
        "input_or_source_sha256_mismatch", "truth_or_boundary_input_present",
        "sample_marker_locus_or_haplotype_axis_mismatch", "locus_loss_duplication_or_reordering",
        "interval_or_streaming_two_pointer_oracle_mismatch", "missing_converted_to_zero",
        "flare_haplotype_axis_changed", "reference_numerator_or_denominator_missing",
        "truncation_nonzero", "semantic_checksum_changes_across_operational_parameters",
        "single_context_exceeds_token_budget", "peak_rss_fraction_greater_than_or_equal_to_0_8",
        "output_exists", "source_not_in_git_commit",
    ], "stop-condition contract drifted")
    truth = payload.get("truth_policy", {})
    require(truth == {
        "truth_access": False,
        "boundary_access": False,
        "lai_performance_access": False,
        "selects_radius": False,
        "training_authorized": False,
        "scientific_run_authorized": False,
    }, "truth/training policy drifted")
    require(root_label in ROOTS and ROOTS[root_label] == root_seed, "root identity drifted")
    root = payload.get("roots", {}).get(root_label)
    require(root is not None and root.get("root_seed") == root_seed, "root is absent from contract")
    require(root.get("role") == "consumed_technical_only", "consumed root role drifted")
    return payload, root


def authenticate_inputs(paths: dict[str, Path], contract: dict[str, Any], root: dict[str, Any]) -> dict[str, str]:
    expected = dict(root["sha256"])
    expected["genetic_map"] = contract["genetic_map"]["sha256"]
    require(set(paths) == set(expected), "input set differs from the preregistration")
    observed = {name: sha256_file(path) for name, path in paths.items()}
    require(observed == expected, "one or more M32 input hashes drifted")
    return observed


def load_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    bp: list[int] = []
    cm: list[float] = []
    identifiers: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames == ["chrom", "bp", "cm", "locus_id"], "coordinate header drifted")
        for row in reader:
            require(row["chrom"] == "chr22", "coordinate chromosome drifted")
            bp.append(int(row["bp"]))
            cm.append(float(row["cm"]))
            identifiers.append(row["locus_id"])
    bp_array = np.asarray(bp, dtype=np.int64)
    cm_array = np.asarray(cm, dtype=np.float64)
    require(bp_array.size > 0 and np.all(np.diff(bp_array) > 0), "bp coordinates are not strictly ordered")
    require(np.all(np.isfinite(cm_array)) and np.all(np.diff(cm_array) >= 0), "cM coordinates are invalid")
    require(len(set(identifiers)) == len(identifiers), "coordinate identifiers are not unique")
    return bp_array, cm_array, tuple(identifiers)


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")


def load_flare_raw(path: Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, Any]]:
    """Read raw ANP1/ANP2 as marker x person x haplotype x ancestry."""
    ancestry_codes: dict[str, str] = {}
    samples: tuple[str, ...] | None = None
    positions: list[int] = []
    rows: list[np.ndarray] = []
    raw_sums: list[float] = []
    seen_formats: set[str] = set()
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##ANCESTRY=<"):
                for token in line.strip()[len("##ANCESTRY=<"):-1].split(","):
                    ancestry, code = token.split("=", 1)
                    ancestry_codes[code] = ancestry
                continue
            if line.startswith("##FORMAT=<ID="):
                seen_formats.add(line.split("ID=", 1)[1].split(",", 1)[0])
                continue
            if line.startswith("##") or not line.strip():
                continue
            if line.startswith("#CHROM"):
                samples = tuple(line.rstrip("\n").split("\t")[9:])
                require(bool(samples) and len(set(samples)) == len(samples), "FLARE samples are invalid")
                require({"AN1", "AN2", "ANP1", "ANP2"}.issubset(seen_formats), "FLARE haplotype FORMAT headers missing")
                continue
            if line.startswith("#"):
                continue
            require(samples is not None, "FLARE data precede header")
            require(ancestry_codes == {"0": "AFR", "1": "EUR", "2": "ASIA"}, "FLARE ancestry order drifted")
            fields = line.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(samples), f"malformed FLARE row {line_number}")
            require(fields[0].removeprefix("chr") == "22" and "," not in fields[4], "FLARE locus is not chr22 biallelic")
            position = int(fields[1])
            require(not positions or position > positions[-1], "FLARE positions are not strictly ordered")
            fmt = fields[8].split(":")
            indexes = {name: fmt.index(name) for name in ("AN1", "AN2", "ANP1", "ANP2") if name in fmt}
            require(len(indexes) == 4, "FLARE row lacks AN1/AN2/ANP1/ANP2")
            matrix = np.empty((len(samples), 2, 3), dtype=np.float32)
            for sample_index, raw_sample in enumerate(fields[9:]):
                values = raw_sample.split(":")
                for hap, (prob_name, hard_name) in enumerate((("ANP1", "AN1"), ("ANP2", "AN2"))):
                    probability = np.asarray([float(value) for value in values[indexes[prob_name]].split(",")], dtype=np.float32)
                    require(probability.shape == (3,) and np.all(np.isfinite(probability)), "FLARE probability shape/finiteness invalid")
                    require(np.all((probability >= 0.0) & (probability <= 1.0)), "FLARE probability outside [0,1]")
                    total = float(np.sum(probability, dtype=np.float64))
                    require(0.989999 <= total <= 1.010001, "FLARE rounded probability sum outside tolerance")
                    hard = int(values[indexes[hard_name]])
                    require(hard in (0, 1, 2) and float(probability[hard]) >= float(probability.max()) - 1e-7,
                            "FLARE hard ancestry is not a probability maximum")
                    matrix[sample_index, hap] = probability
                    raw_sums.append(total)
            positions.append(position)
            rows.append(matrix)
    require(samples is not None and rows, "FLARE VCF is empty")
    probabilities = np.asarray(rows, dtype=np.float32)
    sums = np.asarray(raw_sums, dtype=np.float64)
    return samples, np.asarray(positions, dtype=np.int64), probabilities, {
        "raw_probability_vectors": int(sums.size),
        "raw_sum_min": float(sums.min()),
        "raw_sum_max": float(sums.max()),
        "raw_sum_not_one_at_float_tolerance": int(np.count_nonzero(np.abs(sums - 1.0) > 1e-7)),
        "normalization_applied": False,
    }


def reference_channels(ref_dosage: np.ndarray, labels: Sequence[str]) -> dict[str, np.ndarray]:
    dosage = np.asarray(ref_dosage)
    require(dosage.ndim == 2 and np.all((dosage == -1) | ((dosage >= 0) & (dosage <= 2))),
            "REF dosage is invalid")
    label_array = np.asarray(labels, dtype=object)
    require(dosage.shape[1] == label_array.size, "REF label count differs from people")
    counts: list[np.ndarray] = []
    denominators: list[np.ndarray] = []
    label_codes = np.empty(label_array.size, dtype=np.int8)
    for ancestry_index, ancestry in enumerate(ANCESTRIES):
        mask = label_array == ancestry
        require(bool(np.any(mask)), f"REF lacks {ancestry}")
        label_codes[mask] = ancestry_index
        group = dosage[:, mask]
        group_observed = group >= 0
        count = np.where(group_observed, group, 0).sum(axis=1, dtype=np.uint64)
        denominator = 2 * group_observed.sum(axis=1, dtype=np.uint64)
        require(int(count.max(initial=0)) <= np.iinfo(np.uint16).max, "REF minor count overflows uint16")
        require(int(denominator.max(initial=0)) <= np.iinfo(np.uint16).max, "REF denominator overflows uint16")
        counts.append(count.astype(np.uint16))
        denominators.append(denominator.astype(np.uint16))
    minor_count = np.column_stack(counts)
    callable_alleles = np.column_stack(denominators)
    support_observed = callable_alleles > 0
    support = np.divide(minor_count, callable_alleles, dtype=np.float32,
                        out=np.zeros_like(minor_count, dtype=np.float32), where=callable_alleles > 0)
    return {
        "ref_dosage": dosage.astype(np.int8, copy=False),
        "ref_observed": dosage >= 0,
        "ref_label_codes": label_codes,
        "ref_minor_count": minor_count,
        "ref_callable_alleles": callable_alleles,
        "ref_support": support,
        "ref_support_observed_mask": support_observed,
        "ref_minor_supported": minor_count > 0,
    }


def target_channels(hap_presence: np.ndarray) -> dict[str, np.ndarray]:
    hap = np.asarray(hap_presence, dtype=np.float64)
    require(hap.ndim == 3 and hap.shape[2] == 2, "TARGET must expose locus x person x haplotype")
    finite = np.isfinite(hap)
    require(np.all((~finite) | (hap == 0.0) | (hap == 1.0)), "TARGET haplotype values are invalid")
    presence = np.where(finite, hap, -1).astype(np.int8)
    observed = np.all(finite, axis=2)
    dosage = np.where(observed, np.sum(np.where(finite, hap, 0.0), axis=2), -1).astype(np.int8)
    require(np.all((dosage == -1) | (dosage == 0) | (dosage == 1) | (dosage == 2)), "TARGET dosage invalid")
    return {"target_haplotype_presence": presence, "target_minor_dosage": dosage, "target_observed_mask": observed}


def missing_known_answer() -> dict[str, Any]:
    hap = np.asarray([[[0.0, 0.0], [np.nan, 0.0]], [[1.0, 0.0], [1.0, 1.0]]])
    channels = target_channels(hap)
    expected_dosage = [[0, -1], [1, 2]]
    expected_mask = [[True, False], [True, True]]
    require(channels["target_minor_dosage"].tolist() == expected_dosage, "missing fixture dosage failed")
    require(channels["target_observed_mask"].tolist() == expected_mask, "missing fixture mask failed")
    flare = np.asarray([[[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]]], dtype=np.float32)
    require(array_hash(flare[:, :, 0]) != array_hash(flare[:, :, 1]), "FLARE h0/h1 fixture is not discriminating")
    ref = reference_channels(
        np.asarray([[0, -1, 0, 0], [-1, -1, 2, 0]], dtype=np.int8),
        ["AFR", "AFR", "EUR", "ASIA"],
    )
    require(ref["ref_minor_count"].tolist() == [[0, 0, 0], [0, 2, 0]], "REF missing fixture count failed")
    require(ref["ref_callable_alleles"].tolist() == [[2, 2, 2], [0, 2, 2]], "REF missing fixture denominator failed")
    require(ref["ref_support_observed_mask"].tolist() == [[True, True, True], [False, True, True]],
            "REF support-observed fixture failed")
    require(ref["ref_support"].tolist() == [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "REF missing fixture support failed")
    return {
        "target_missing_is_not_hom_ref": True,
        "ref_zero_count_with_denominator_is_observed_absence": True,
        "ref_zero_denominator_is_unobserved": True,
        "flare_h0_h1_distinct": True,
    }


def interval_arrays(grid_cm: np.ndarray, rare_cm: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    start64 = np.searchsorted(rare_cm, grid_cm - radius, side="left")
    stop64 = np.searchsorted(rare_cm, grid_cm + radius, side="right")
    require(np.all((0 <= start64) & (start64 <= stop64) & (stop64 <= rare_cm.size)), "interval bounds invalid")
    require(int(stop64.max(initial=0)) <= np.iinfo(np.int32).max, "interval index overflows int32")
    starts, stops = start64.astype(np.int32), stop64.astype(np.int32)
    lower, upper = grid_cm - radius, grid_cm + radius
    left_ok = (starts == 0) | (rare_cm[np.maximum(starts - 1, 0)] < lower)
    start_ok = (starts == rare_cm.size) | (rare_cm[np.minimum(starts, rare_cm.size - 1)] >= lower)
    end_ok = (stops == 0) | (rare_cm[np.maximum(stops - 1, 0)] <= upper)
    right_ok = (stops == rare_cm.size) | (rare_cm[np.minimum(stops, rare_cm.size - 1)] > upper)
    require(bool(np.all(left_ok & start_ok & end_ok & right_ok)), "intervals differ from inclusive bisect oracle")
    return starts, stops


def streaming_interval_oracle(grid_cm: np.ndarray, rare_cm: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Independent O(M+L) two-pointer oracle for all half-open intervals."""
    starts = np.empty(grid_cm.size, dtype=np.int32)
    stops = np.empty(grid_cm.size, dtype=np.int32)
    left = right = 0
    for index, marker_cm in enumerate(grid_cm):
        lower, upper = marker_cm - radius, marker_cm + radius
        while left < rare_cm.size and rare_cm[left] < lower:
            left += 1
        if right < left:
            right = left
        while right < rare_cm.size and rare_cm[right] <= upper:
            right += 1
        starts[index], stops[index] = left, right
    return starts, stops


def explicit_csr_oracle(starts: np.ndarray, stops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = stops.astype(np.int64) - starts.astype(np.int64)
    indptr = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    indices = np.concatenate([np.arange(left, right, dtype=np.int32) for left, right in zip(starts, stops)])
    return indptr, indices


def padded_roundtrip_known_answer() -> dict[str, Any]:
    values = [np.asarray([3, 1], dtype=np.int8), np.asarray([2, 0, 1, 2], dtype=np.int8)]
    floats = [np.asarray([0.25, 0.75], dtype=np.float32), np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)]
    observed = [np.asarray([True, False]), np.asarray([True, True, False, True])]
    padded = np.full((2, 4), -9, dtype=np.int8)
    padded_float = np.full((2, 4), np.float32(12345.0), dtype=np.float32)
    padded_observed = np.ones((2, 4), dtype=bool)
    mask = np.zeros((2, 4), dtype=bool)
    for index, row in enumerate(values):
        padded[index, :row.size] = row
        padded_float[index, :row.size] = floats[index]
        padded_observed[index, :row.size] = observed[index]
        mask[index, :row.size] = True
    restored = [padded[index, mask[index]] for index in range(2)]
    restored_float = [padded_float[index, mask[index]] for index in range(2)]
    restored_observed = [padded_observed[index, mask[index]] for index in range(2)]
    require(all(np.array_equal(a, b) for a, b in zip(values, restored)), "padded known-answer roundtrip failed")
    require(all(np.allclose(a, b, rtol=1e-6, atol=1e-7) for a, b in zip(floats, restored_float)),
            "padded float known-answer roundtrip failed")
    require(all(np.array_equal(a, b) for a, b in zip(observed, restored_observed)),
            "padded bool known-answer roundtrip failed")
    require(not bool(np.any(padded[~mask] != -9)), "padding sentinel drifted")
    require(bool(np.all(padded_float[~mask] == np.float32(12345.0))), "adversarial float padding drifted")
    return {
        "packed_equals_padded_after_mask": True,
        "integer_boolean_float_channels_roundtrip": True,
        "padding_sentinel_cannot_enter_valid_values": True,
    }


def deterministic_marker_sample(lengths: np.ndarray, size: int) -> np.ndarray:
    require(size > 0, "performance marker sample must be positive")
    order = np.lexsort((np.arange(lengths.size), lengths))
    if size >= lengths.size:
        return np.sort(order)
    ranks = np.rint(np.linspace(0, lengths.size - 1, size)).astype(np.int64)
    selected = np.unique(order[ranks])
    require(selected.size == size, "marker rank sampling unexpectedly duplicated indexes")
    return np.sort(selected.astype(np.int32))


def backend_order(markers: np.ndarray, lengths: np.ndarray, backend: str) -> np.ndarray:
    if backend == "contiguous_packed":
        return np.sort(markers)
    require(backend == "length_sorted_packed", "unknown backend")
    local_lengths = lengths[markers]
    sorted_local = np.lexsort((markers, local_lengths))
    bucket = np.empty(markers.size, dtype=np.int8)
    bucket[sorted_local] = np.minimum(3, (np.arange(markers.size) * 4) // markers.size)
    return markers[np.lexsort((markers, local_lengths, bucket))]


def marker_chunks(order: np.ndarray, lengths: np.ndarray, people: int, budget: int) -> list[np.ndarray]:
    require(people > 0 and budget > 0, "batch/budget must be positive")
    chunks: list[np.ndarray] = []
    current: list[int] = []
    tokens = 0
    for marker in order.tolist():
        marker_tokens = people * int(lengths[marker])
        require(marker_tokens <= budget, "single context exceeds token budget")
        if current and tokens + marker_tokens > budget:
            chunks.append(np.asarray(current, dtype=np.int32))
            current, tokens = [], 0
        current.append(marker)
        tokens += marker_tokens
    if current:
        chunks.append(np.asarray(current, dtype=np.int32))
    require(sum(len(chunk) for chunk in chunks) == len(order), "chunking lost markers")
    return chunks


def memory_limit_bytes() -> int | None:
    candidates = (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    for path in candidates:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw != "max":
            value = int(raw)
            if 0 < value < (1 << 60):
                return value
    return None


def rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw * 1024 if os.uname().sysname != "Darwin" else raw)


def require_memory_gate(expected_limit: int) -> tuple[int, int, float]:
    limit = memory_limit_bytes()
    require(limit is not None, "finite cgroup memory limit is required")
    require(abs(limit - expected_limit) <= (1 << 20), "cgroup memory limit differs from the frozen task limit")
    peak = rss_bytes()
    fraction = peak / limit
    require(fraction < 0.8, "process reached the 80% cgroup memory stop")
    return limit, peak, fraction


def expected_assignment_hash(markers: np.ndarray, starts: np.ndarray, stops: np.ndarray, people: int) -> str:
    digest = hashlib.sha256()
    for marker in sorted(int(value) for value in markers):
        for person in range(people):
            digest.update(struct.pack("<IIII", marker, person, int(starts[marker]), int(stops[marker])))
    return digest.hexdigest()


def expected_content_hash(arrays: dict[str, np.ndarray], starts: np.ndarray, stops: np.ndarray,
                          markers: np.ndarray) -> str:
    marker_digests: dict[int, str] = {}
    people = arrays["target_minor_dosage"].shape[1]
    for raw_marker in markers:
        marker = int(raw_marker)
        left, right = int(starts[marker]), int(stops[marker])
        digest = hashlib.sha256()
        digest.update(struct.pack("<III", marker, left, right))
        locus = slice(left, right)
        distance = arrays["rare_cm"][locus].astype(np.float32) - np.float32(arrays["grid_cm"][marker])
        for value in (
            np.arange(left, right, dtype=np.int32), arrays["rare_bp"][locus], arrays["rare_cm"][locus],
            arrays["minor_codes"][locus], arrays["ref_dosage"][locus], arrays["ref_observed"][locus],
            arrays["ref_minor_count"][locus], arrays["ref_callable_alleles"][locus],
            arrays["ref_support"][locus], arrays["ref_support_observed_mask"][locus],
            arrays["ref_minor_supported"][locus], distance,
        ):
            update_array_digest(digest, value)
        for person in range(people):
            for value in (
                arrays["target_minor_dosage"][locus, person],
                arrays["target_observed_mask"][locus, person],
                arrays["target_haplotype_presence"][locus, person],
                arrays["flare_raw"][marker, person],
            ):
                update_array_digest(digest, value)
        marker_digests[marker] = digest.hexdigest()
    combined = hashlib.sha256()
    for marker in sorted(marker_digests):
        combined.update(struct.pack("<I", marker))
        combined.update(bytes.fromhex(marker_digests[marker]))
    return combined.hexdigest()


def consume_once(arrays: dict[str, np.ndarray], starts: np.ndarray, stops: np.ndarray,
                 markers: np.ndarray, person_batch: int, budget: int, backend: str,
                 verify_content: bool = True) -> dict[str, Any]:
    lengths = stops.astype(np.int64) - starts.astype(np.int64)
    order = backend_order(markers, lengths, backend)
    n_people = arrays["target_minor_dosage"].shape[1]
    totals = Counter(tokens=0, marker_person_contexts=0, chunks=0, dosage_sum=0, observed_count=0,
                     haplotype_presence_sum=0, haplotype_observed_count=0,
                     ref_minor_count_sum=0, ref_callable_alleles_sum=0,
                     ref_observed_count=0, ref_support_observed_count=0,
                     ref_minor_supported_count=0, ref_support_float32_bits_sum=0,
                     flare_float32_bits_sum=0, flare_h0_float32_bits_sum=0,
                     flare_h1_float32_bits_sum=0, payload_bytes=0)
    max_chunk_bytes = 0
    assignment_records: list[tuple[int, int, int, int]] = []
    marker_digests = {int(marker): hashlib.sha256() for marker in markers} if verify_content else {}
    for person_start in range(0, n_people, person_batch):
        person_stop = min(n_people, person_start + person_batch)
        people = person_stop - person_start
        for chunk in marker_chunks(order, lengths, people, budget):
            chunk_lengths = lengths[chunk]
            locus_index = np.concatenate([
                np.arange(starts[marker], stops[marker], dtype=np.int32) for marker in chunk
            ])
            marker_index = np.repeat(chunk.astype(np.int32), chunk_lengths)
            expected_locus_index = np.concatenate([
                np.arange(starts[marker], stops[marker], dtype=np.int32) for marker in chunk
            ])
            require(np.array_equal(locus_index, expected_locus_index), "packed locus identity/order drifted")
            require(np.array_equal(marker_index, np.repeat(chunk.astype(np.int32), chunk_lengths)),
                    "packed marker identity/order drifted")
            dosage = arrays["target_minor_dosage"][locus_index, person_start:person_stop]
            observed = arrays["target_observed_mask"][locus_index, person_start:person_stop]
            haplotype = arrays["target_haplotype_presence"][locus_index, person_start:person_stop]
            minor_codes = arrays["minor_codes"][locus_index]
            rare_bp = arrays["rare_bp"][locus_index]
            rare_cm = arrays["rare_cm"][locus_index]
            ref_dosage = arrays["ref_dosage"][locus_index]
            ref_count = arrays["ref_minor_count"][locus_index]
            ref_an = arrays["ref_callable_alleles"][locus_index]
            ref_observed = arrays["ref_observed"][locus_index]
            ref_support = arrays["ref_support"][locus_index]
            ref_support_observed = arrays["ref_support_observed_mask"][locus_index]
            ref_minor_supported = arrays["ref_minor_supported"][locus_index]
            distance = arrays["rare_cm"][locus_index].astype(np.float32) - arrays["grid_cm"][marker_index].astype(np.float32)
            flare = arrays["flare_raw"][chunk, person_start:person_stop]
            payload = sum(value.nbytes for value in (
                locus_index, marker_index, dosage, observed, haplotype, minor_codes, rare_bp, rare_cm,
                ref_dosage, ref_count, ref_an, ref_observed, ref_support, ref_support_observed,
                ref_minor_supported, distance, flare,
            ))
            max_chunk_bytes = max(max_chunk_bytes, payload)
            for marker in chunk.tolist():
                for person in range(person_start, person_stop):
                    assignment_records.append((int(marker), person, int(starts[marker]), int(stops[marker])))
            if verify_content:
                offset = 0
                for chunk_position, marker in enumerate(chunk.tolist()):
                    length = int(lengths[marker])
                    local = slice(offset, offset + length)
                    digest = marker_digests[int(marker)]
                    if person_start == 0:
                        digest.update(struct.pack("<III", int(marker), int(starts[marker]), int(stops[marker])))
                        for value in (
                            locus_index[local], rare_bp[local], rare_cm[local], minor_codes[local],
                            ref_dosage[local], ref_observed[local], ref_count[local], ref_an[local],
                            ref_support[local], ref_support_observed[local], ref_minor_supported[local],
                            distance[local],
                        ):
                            update_array_digest(digest, value)
                    for local_person in range(people):
                        for value in (
                            dosage[local, local_person], observed[local, local_person],
                            haplotype[local, local_person], flare[chunk_position, local_person],
                        ):
                            update_array_digest(digest, value)
                    offset += length
            totals["tokens"] += int(locus_index.size * people)
            totals["marker_person_contexts"] += int(chunk.size * people)
            totals["chunks"] += 1
            totals["dosage_sum"] += int(dosage[observed].sum(dtype=np.int64))
            totals["observed_count"] += int(observed.sum(dtype=np.int64))
            hap_observed = haplotype >= 0
            totals["haplotype_presence_sum"] += int(haplotype[hap_observed].sum(dtype=np.int64))
            totals["haplotype_observed_count"] += int(hap_observed.sum(dtype=np.int64))
            totals["ref_minor_count_sum"] += int(ref_count.sum(dtype=np.int64) * people)
            totals["ref_callable_alleles_sum"] += int(ref_an.sum(dtype=np.int64) * people)
            totals["ref_observed_count"] += int(ref_observed.sum(dtype=np.int64) * people)
            totals["ref_support_observed_count"] += int(ref_support_observed.sum(dtype=np.int64) * people)
            totals["ref_minor_supported_count"] += int(ref_minor_supported.sum(dtype=np.int64) * people)
            totals["ref_support_float32_bits_sum"] += int(ref_support.view(np.uint32).sum(dtype=np.uint64)) * people
            flare_bits = flare.view(np.uint32)
            totals["flare_float32_bits_sum"] += int(flare_bits.sum(dtype=np.uint64))
            totals["flare_h0_float32_bits_sum"] += int(flare[:, :, 0, :].view(np.uint32).sum(dtype=np.uint64))
            totals["flare_h1_float32_bits_sum"] += int(flare[:, :, 1, :].view(np.uint32).sum(dtype=np.uint64))
            totals["payload_bytes"] += int(payload)
    totals["max_chunk_payload_bytes"] = max_chunk_bytes
    expected_tokens = int(n_people * lengths[markers].sum(dtype=np.int64))
    require(totals["tokens"] <= expected_tokens, "packed representation duplicated tokens")
    totals["truncations"] = expected_tokens - totals["tokens"]
    require(totals["truncations"] == 0, "packed representation truncated tokens")
    require(len(assignment_records) == int(markers.size * n_people), "packed representation lost contexts")
    assignment_records.sort()
    digest = hashlib.sha256()
    for record in assignment_records:
        digest.update(struct.pack("<IIII", *record))
    result = dict(totals)
    result["assignment_sha256"] = digest.hexdigest()
    if verify_content:
        combined = hashlib.sha256()
        for marker in sorted(marker_digests):
            combined.update(struct.pack("<I", marker))
            combined.update(marker_digests[marker].digest())
        result["content_sha256"] = combined.hexdigest()
    return result


def benchmark_child(connection, arrays: dict[str, np.ndarray], starts: np.ndarray, stops: np.ndarray,
                    markers: np.ndarray, person_batch: int, budget: int, backend: str,
                    warmups: int, repetitions: int, memory_limit: int | None) -> None:
    try:
        base_rss = rss_bytes()
        integrity = consume_once(
            arrays, starts, stops, markers, person_batch, budget, backend, verify_content=True)
        for _ in range(warmups):
            consume_once(arrays, starts, stops, markers, person_batch, budget, backend, verify_content=False)
        durations: list[float] = []
        outputs: list[dict[str, int]] = []
        for _ in range(repetitions):
            started = time.perf_counter()
            outputs.append(consume_once(
                arrays, starts, stops, markers, person_batch, budget, backend, verify_content=False))
            durations.append(time.perf_counter() - started)
        require(all(output == outputs[0] for output in outputs), "benchmark repetitions are not deterministic")
        require(all(integrity[key] == outputs[0][key] for key in outputs[0]),
                "timed benchmark differs from the integrity pass")
        peak = rss_bytes()
        fraction = None if memory_limit is None else peak / memory_limit
        connection.send({
            "ok": True,
            "durations_seconds": durations,
            "median_seconds": statistics.median(durations),
            "range_seconds": [min(durations), max(durations)],
            "base_rss_bytes": base_rss,
            "peak_rss_bytes": peak,
            "delta_peak_rss_bytes": max(0, peak - base_rss),
            "memory_limit_bytes": memory_limit,
            "peak_rss_fraction": fraction,
            "totals": integrity,
        })
    except BaseException as exc:  # child must return a typed failure to the parent
        connection.send({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        connection.close()


def run_isolated_benchmark(arrays: dict[str, np.ndarray], starts: np.ndarray, stops: np.ndarray,
                           markers: np.ndarray, person_batch: int, budget: int, backend: str,
                           warmups: int, repetitions: int, memory_limit: int | None,
                           expected_content_sha256: str) -> dict[str, Any]:
    context = mp.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=benchmark_child, args=(child, arrays, starts, stops, markers,
                              person_batch, budget, backend, warmups, repetitions, memory_limit))
    process.start()
    child.close()
    if not parent.poll(3300):
        process.terminate()
        process.join(30)
        raise TimeoutError("isolated benchmark exceeded 55 minutes")
    result = parent.recv()
    process.join()
    require(process.exitcode == 0 and result.get("ok") is True, f"isolated benchmark failed: {result}")
    require(result["totals"]["assignment_sha256"] == expected_assignment_hash(
        markers, starts, stops, arrays["target_minor_dosage"].shape[1]),
        "packed assignment identity/order hash drifted")
    require(result["totals"]["content_sha256"] == expected_content_sha256,
            "packed content hash differs from the direct-array oracle")
    fraction = result.get("peak_rss_fraction")
    require(fraction is None or fraction < 0.8, "benchmark reached the 80% cgroup memory stop")
    result["memory_warning"] = bool(fraction is not None and fraction >= 0.7)
    totals = result["totals"]
    result["tokens_per_second"] = totals["tokens"] / result["median_seconds"]
    result["marker_person_contexts_per_second"] = totals["marker_person_contexts"] / result["median_seconds"]
    return result


def semantic_arrays_hash(arrays: dict[str, np.ndarray]) -> str:
    keys = (
        "grid_bp", "grid_cm", "rare_bp", "rare_cm", "target_haplotype_presence",
        "minor_codes", "target_minor_dosage", "target_observed_mask", "ref_dosage", "ref_observed",
        "ref_person_ids", "ref_label_codes", "ref_minor_count", "ref_callable_alleles", "ref_support",
        "ref_support_observed_mask", "ref_minor_supported", "flare_raw",
    )
    return json_hash({key: array_hash(arrays[key]) for key in keys})


def materialize(args: argparse.Namespace) -> int:
    contract, root = load_contract(args.preregistration, args.root_label, args.root_seed)
    require(args.container_image_id == contract["container_image_id"], "container image identity differs from contract")
    require(args.expected_memory_bytes == contract["memory"]["expected_limit_bytes"],
            "CLI memory expectation differs from contract")
    memory_limit, initial_peak_rss, initial_peak_fraction = require_memory_gate(args.expected_memory_bytes)
    input_paths = {name: getattr(args, name) for name in (
        "genetic_map", "grid_coordinates", "rare_coordinates", "sites", "target", "tree", "pools", "flare_vcf"
    )}
    input_hashes = authenticate_inputs(input_paths, contract, root)
    git_commit = validate_git_commit(args.git_commit)
    sources = parse_sources(args.source)
    source_hashes = authenticate_staged_sources(args.source_auth, git_commit, sources)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    grid_bp, grid_cm, grid_ids = load_coordinates(args.grid_coordinates)
    rare_bp, rare_cm, rare_ids = load_coordinates(args.rare_coordinates)
    timings["coordinate_parse_seconds"] = time.perf_counter() - started
    require_memory_gate(args.expected_memory_bytes)

    started = time.perf_counter()
    rare = load_ordered_rare(args.sites, args.target, args.root_seed)
    target = target_channels(rare.hap_presence)
    timings["target_parse_seconds"] = time.perf_counter() - started
    require_memory_gate(args.expected_memory_bytes)

    started = time.perf_counter()
    flare_samples, flare_positions, flare_raw, flare_audit = load_flare_raw(args.flare_vcf)
    timings["flare_parse_seconds"] = time.perf_counter() - started
    require_memory_gate(args.expected_memory_bytes)

    started = time.perf_counter()
    genetic_map = load_genetic_map(args.genetic_map)
    ref_dosage, ref_people, ref_labels = load_ref_minor_dosage(args.tree, args.pools, rare, genetic_map)
    reference = reference_channels(ref_dosage, ref_labels)
    timings["reference_parse_seconds"] = time.perf_counter() - started
    require_memory_gate(args.expected_memory_bytes)

    require(np.array_equal(rare.positions, rare_bp), "M31 sites and M32 rare coordinates differ")
    require(np.array_equal(flare_positions, grid_bp), "FLARE markers and M32 grid coordinates differ")
    require(tuple(rare.samples) == tuple(flare_samples), "TARGET and FLARE sample order differs")
    require(len(grid_ids) == grid_bp.size and len(rare_ids) == rare_bp.size, "coordinate identity loss")
    arrays: dict[str, np.ndarray] = {
        "grid_bp": grid_bp, "grid_cm": grid_cm, "rare_bp": rare_bp, "rare_cm": rare_cm,
        "minor_codes": rare.minor_codes.astype(np.int8, copy=False), "flare_raw": flare_raw,
        "ref_person_ids": np.asarray(ref_people, dtype=np.str_),
        **target, **reference,
    }
    require(int(np.count_nonzero(~target["target_observed_mask"])) == contract["simulation_limitations"]["target_missing_expected"],
            "observed TARGET missingness differs from the technical contract")
    require(int(np.count_nonzero(~reference["ref_observed"])) == contract["simulation_limitations"]["reference_missing_expected"],
            "observed REF missingness differs from the technical contract")
    known = {**missing_known_answer(), **padded_roundtrip_known_answer()}
    semantic_hash = semantic_arrays_hash(arrays)
    args.tensor_out.parent.mkdir(parents=True, exist_ok=True)
    require(not args.tensor_out.exists(), "private tensor output already exists")
    started = time.perf_counter()
    write_npz_atomic(args.tensor_out, arrays)
    timings["tensor_atomic_save_and_reopen_seconds"] = time.perf_counter() - started
    _limit, final_peak_rss, final_peak_fraction = require_memory_gate(args.expected_memory_bytes)
    tensor_hash = sha256_file(args.tensor_out)
    audit = {
        "stage": STAGE,
        "status": "PASS_MATERIALIZATION_TRUTH_FREE_NOT_SCIENTIFIC_EVIDENCE",
        "root_label": args.root_label,
        "root_seed": args.root_seed,
        "truth_accessed": False,
        "training_authorized": False,
        "axes": contract["axes"],
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "sample_order_sha256": json_hash(list(rare.samples)),
        "grid_identity_sha256": json_hash(list(grid_ids)),
        "rare_identity_sha256": json_hash(list(rare_ids)),
        "semantic_arrays_sha256": semantic_hash,
        "private_tensor_sha256": tensor_hash,
        "private_tensor_bytes": args.tensor_out.stat().st_size,
        "target_missing_diploid_cells": int(np.count_nonzero(~target["target_observed_mask"])),
        "ref_missing_diploid_cells": int(np.count_nonzero(~reference["ref_observed"])),
        "ref_people_by_ancestry": dict(Counter(ref_labels)),
        "ref_person_order_sha256": json_hash(list(ref_people)),
        "ref_loci_with_minor_support": {
            ancestry: int(np.count_nonzero(reference["ref_minor_count"][:, index] > 0))
            for index, ancestry in enumerate(ANCESTRIES)
        },
        "ref_loci_without_minor_support_any_ancestry": int(np.count_nonzero(np.all(reference["ref_minor_count"] == 0, axis=1))),
        "flare_raw_audit": flare_audit,
        "synthetic_known_answers": known,
        "timings": timings,
        "memory": {
            "cgroup_limit_bytes": memory_limit,
            "initial_peak_rss_bytes": initial_peak_rss,
            "initial_peak_rss_fraction": initial_peak_fraction,
            "final_peak_rss_bytes": final_peak_rss,
            "final_peak_rss_fraction": final_peak_fraction,
            "stop_fraction": 0.8,
        },
    }
    provenance = {
        "stage": STAGE,
        "root_label": args.root_label,
        "root_seed": args.root_seed,
        "truth_accessed": False,
        "git_commit": git_commit,
        "nextflow_version": args.nextflow_version,
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "source_auth_sha256": sha256_file(args.source_auth),
        "preregistration_sha256": sha256_file(args.preregistration),
        "private_tensor_sha256": tensor_hash,
        "container_image_id": args.container_image_id,
        "runtime_versions": {
            "python": platform.python_version(), "numpy": np.__version__, "tskit": tskit.__version__,
        },
    }
    write_json_atomic(args.materialization_audit, audit)
    write_json_atomic(args.materialization_provenance, provenance)
    return 0


def benchmark(args: argparse.Namespace) -> int:
    contract, _root = load_contract(args.preregistration, args.root_label, args.root_seed)
    require(args.expected_memory_bytes == contract["memory"]["expected_limit_bytes"],
            "CLI memory expectation differs from contract")
    require(args.container_image_id == contract["container_image_id"], "container image identity differs from contract")
    git_commit = validate_git_commit(args.git_commit)
    sources = parse_sources(args.source)
    source_hashes = authenticate_staged_sources(args.source_auth, git_commit, sources)
    audit = json.loads(args.materialization_audit.read_text(encoding="utf-8"))
    materialization_provenance = json.loads(args.materialization_provenance.read_text(encoding="utf-8"))
    for payload, name in ((audit, "materialization audit"),
                          (materialization_provenance, "materialization provenance")):
        require(payload.get("stage") == STAGE, f"{name} stage drifted")
        require(payload.get("root_label") == args.root_label, f"{name} root label drifted")
        require(payload.get("root_seed") == args.root_seed, f"{name} root seed drifted")
        require(payload.get("truth_accessed") is False, f"{name} accessed truth")
    require(materialization_provenance.get("git_commit") == git_commit,
            "materialization and benchmark Git commits differ")
    require(materialization_provenance.get("source_sha256") == source_hashes,
            "materialization and benchmark source hashes differ")
    require(materialization_provenance.get("source_auth_sha256") == sha256_file(args.source_auth),
            "materialization and benchmark source-auth manifests differ")
    require(materialization_provenance.get("preregistration_sha256") == sha256_file(args.preregistration),
            "materialization and benchmark contracts differ")
    require(materialization_provenance.get("container_image_id") == args.container_image_id,
            "materialization and benchmark container identities differ")
    memory_limit, initial_peak_rss, initial_peak_fraction = require_memory_gate(args.expected_memory_bytes)
    tensor_hash = sha256_file(args.tensor)
    require(tensor_hash == audit["private_tensor_sha256"],
            "private tensor hash differs from materialization")
    require(tensor_hash == materialization_provenance["private_tensor_sha256"],
            "private tensor hash differs from materialization provenance")
    started_load = time.perf_counter()
    with np.load(args.tensor, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    tensor_load_seconds = time.perf_counter() - started_load
    _limit, loaded_peak_rss, loaded_peak_fraction = require_memory_gate(args.expected_memory_bytes)
    require(set(arrays) == set(audit["shapes"]), "private tensor array inventory drifted")
    require(all(list(arrays[name].shape) == audit["shapes"][name] for name in arrays),
            "private tensor shape drifted")
    require(all(str(arrays[name].dtype) == audit["dtypes"][name] for name in arrays),
            "private tensor dtype drifted")
    require(semantic_arrays_hash(arrays) == audit["semantic_arrays_sha256"], "tensor semantic hash drifted")
    memory_limit = memory_limit_bytes()
    radii_reports: list[dict[str, Any]] = []
    all_configurations: list[dict[str, Any]] = []
    for radius in contract["radii_cm"]:
        starts, stops = interval_arrays(arrays["grid_cm"], arrays["rare_cm"], float(radius))
        oracle_starts, oracle_stops = streaming_interval_oracle(
            arrays["grid_cm"], arrays["rare_cm"], float(radius))
        oracle_passed = bool(np.array_equal(starts, oracle_starts) and np.array_equal(stops, oracle_stops))
        require(oracle_passed, "searchsorted intervals differ from the independent streaming oracle")
        lengths = stops.astype(np.int64) - starts.astype(np.int64)
        require(np.all(lengths > 0), "real M32 context unexpectedly empty")
        markers = deterministic_marker_sample(lengths, int(contract["performance_marker_sample"]))
        content_oracle_sha256 = expected_content_hash(arrays, starts, stops, markers)
        interval_hash = json_hash({"start": array_hash(starts), "stop": array_hash(stops)})
        radius_report = {
            "radius_cm": radius,
            "start_stop_sha256": interval_hash,
            "start_stop_bytes": int(starts.nbytes + stops.nbytes),
            "marker_count": int(lengths.size),
            "assignment_count": int(lengths.sum(dtype=np.int64)),
            "performance_sample_markers": int(markers.size),
            "performance_sample_sha256": array_hash(markers),
            "direct_content_oracle_sha256": content_oracle_sha256,
            "context_length": {
                "min": int(lengths.min()), "median": int(np.median(lengths)),
                "q95": int(np.quantile(lengths, 0.95, method="nearest")),
                "q99": int(np.quantile(lengths, 0.99, method="nearest")), "max": int(lengths.max()),
            },
            "streaming_two_pointer_oracle_all_markers": oracle_passed,
            "oracle_start_stop_sha256": json_hash({
                "start": array_hash(oracle_starts), "stop": array_hash(oracle_stops),
            }),
        }
        radii_reports.append(radius_report)
        for backend in contract["backends"]:
            for person_batch in contract["person_batches"]:
                for budget in contract["token_budgets"]:
                    result = run_isolated_benchmark(
                        arrays, starts, stops, markers, int(person_batch), int(budget), backend,
                        int(contract["timing"]["warmups"]), int(contract["timing"]["repetitions"]), memory_limit,
                        content_oracle_sha256,
                    )
                    totals = result["totals"]
                    integer_keys = (
                        "tokens", "marker_person_contexts", "dosage_sum", "observed_count",
                        "haplotype_presence_sum", "haplotype_observed_count",
                        "ref_minor_count_sum", "ref_callable_alleles_sum", "ref_observed_count",
                        "ref_support_observed_count", "ref_minor_supported_count",
                        "ref_support_float32_bits_sum", "flare_float32_bits_sum",
                        "flare_h0_float32_bits_sum", "flare_h1_float32_bits_sum", "truncations",
                    )
                    invariant = {key: int(totals[key]) for key in integer_keys}
                    invariant["assignment_sha256"] = totals["assignment_sha256"]
                    invariant["content_sha256"] = totals["content_sha256"]
                    all_configurations.append({
                        "radius_cm": radius, "backend": backend, "person_batch": person_batch,
                        "token_budget": budget, "semantic_totals": invariant, **{k: v for k, v in result.items() if k != "totals"},
                        "execution": totals,
                    })
        radius_configs = [row for row in all_configurations if row["radius_cm"] == radius]
        expected = radius_configs[0]["semantic_totals"]
        require(all(row["semantic_totals"] == expected for row in radius_configs),
                "semantic totals changed across batch/budget/backend")
    config_count = len(contract["radii_cm"]) * len(contract["backends"]) * len(contract["person_batches"]) * len(contract["token_budgets"])
    require(len(all_configurations) == config_count, "benchmark grid is incomplete")
    report = {
        "stage": STAGE,
        "status": "PASS_TRUTH_FREE_PACKED_BENCHMARK_NO_TRAINING",
        "root_label": args.root_label,
        "root_seed": args.root_seed,
        "truth_accessed": False,
        "training_authorized": False,
        "selects_radius": False,
        "observed_mask_scope": "simulated_genotype_observation_only_not_biological_DP_GQ_callability",
        "performance_scope": "deterministic_marker_microbenchmark_not_full_chromosome_training",
        "io_scope": "single_warm_NPZ_load_and_atomic_materialization_timings_not_cold_storage_benchmark",
        "memory_limit_bytes": memory_limit,
        "tensor_load_seconds": tensor_load_seconds,
        "initial_peak_rss_bytes": initial_peak_rss,
        "initial_peak_rss_fraction": initial_peak_fraction,
        "loaded_peak_rss_bytes": loaded_peak_rss,
        "loaded_peak_rss_fraction": loaded_peak_fraction,
        "radii": radii_reports,
        "configurations": all_configurations,
        "configuration_count": len(all_configurations),
        "semantic_arrays_sha256": audit["semantic_arrays_sha256"],
        "no_truncation": all(row["execution"]["truncations"] == 0 for row in all_configurations),
        "operational_invariance": all(
            len({json_hash(row["semantic_totals"]) for row in all_configurations if row["radius_cm"] == radius}) == 1
            for radius in contract["radii_cm"]
        ),
        "claims_excluded": ["LAI_improvement", "radius_selection", "model_selection", "DNABR_callability", "independent_validation"],
    }
    require(report["no_truncation"] and report["operational_invariance"], "benchmark capacity invariants failed")
    provenance = {
        "git_commit": git_commit,
        "nextflow_version": args.nextflow_version,
        "source_sha256": source_hashes,
        "source_auth_sha256": sha256_file(args.source_auth),
        "preregistration_sha256": sha256_file(args.preregistration),
        "materialization_audit_sha256": sha256_file(args.materialization_audit),
        "materialization_provenance_sha256": sha256_file(args.materialization_provenance),
        "private_tensor_sha256": sha256_file(args.tensor),
        "execution_interface": "workflows/m32_packed_benchmark.nf",
        "container_image_id": args.container_image_id,
        "runtime_versions": {
            "python": platform.python_version(), "numpy": np.__version__, "tskit": tskit.__version__,
        },
    }
    write_json_atomic(args.report, report)
    write_json_atomic(args.provenance, provenance)
    manifest = {"files": {args.report.name: sha256_file(args.report), args.provenance.name: sha256_file(args.provenance),
                           args.materialization_audit.name: sha256_file(args.materialization_audit),
                           args.materialization_provenance.name: sha256_file(args.materialization_provenance),
                           args.source_auth.name: sha256_file(args.source_auth)},
                "sources": source_hashes}
    write_json_atomic(args.manifest, manifest)
    receipt = {
        "stage": STAGE, "status": report["status"], "root_label": args.root_label,
        "git_commit": git_commit, "configuration_count": len(all_configurations),
        "manifest_sha256": sha256_file(args.manifest), "provenance_sha256": sha256_file(args.provenance),
        "no_truncation": report["no_truncation"],
        "operational_invariance": report["operational_invariance"],
        "training_authorized": False, "scientific_run_authorized": False,
    }
    write_json_atomic(args.receipt, receipt)
    return 0


def common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--root-label", choices=tuple(ROOTS), required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--nextflow-version", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--expected-memory-bytes", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize_parser = subparsers.add_parser("materialize")
    common_parser(materialize_parser)
    for name in ("genetic_map", "grid_coordinates", "rare_coordinates", "sites", "target", "tree", "pools", "flare_vcf"):
        materialize_parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    materialize_parser.add_argument("--tensor-out", type=Path, required=True)
    materialize_parser.add_argument("--materialization-audit", type=Path, required=True)
    materialize_parser.add_argument("--materialization-provenance", type=Path, required=True)

    benchmark_parser = subparsers.add_parser("benchmark")
    common_parser(benchmark_parser)
    benchmark_parser.add_argument("--tensor", type=Path, required=True)
    benchmark_parser.add_argument("--materialization-audit", type=Path, required=True)
    benchmark_parser.add_argument("--materialization-provenance", type=Path, required=True)
    benchmark_parser.add_argument("--report", type=Path, required=True)
    benchmark_parser.add_argument("--provenance", type=Path, required=True)
    benchmark_parser.add_argument("--manifest", type=Path, required=True)
    benchmark_parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return materialize(args) if args.command == "materialize" else benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
