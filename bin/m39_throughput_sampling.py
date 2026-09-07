"""Metadata-only sampling for a bounded ordered-model throughput profile.

The caller authenticates TRAIN membership and the store before using this
helper. Sampling does not inspect genotypes, rare dosage or ancestry labels.
The primary estimand is a steady-state pass with batches grouped by anchor;
the random replay is conditional on the identical sampled pair inventory.
"""
from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
import re

import numpy as np

SCHEMA = "m39-throughput-sampling-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _rng(seed, stream):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))


def build_sampling_manifest(store, seed: int) -> dict:
    """Select four distinct anchors per rank quartile, two queries per anchor.

    A separate query permutation supplies 32 distinct TRAIN indices. Query
    assignments are dependent without replacement but marginally uniform.
    Explicit streams keep anchor, query, execution and replay choices separate.
    """
    require(type(seed) is int and 0 <= seed < 2**32, "seed must be a uint32 integer")
    require(store.shape == (48, 660, 8), "frozen TRAIN48/660/K8 store required")
    keys = store.arrays["sample_key_sha256"]
    loci = store.arrays["locus_id"]
    require(keys.shape == (48,) and keys.dtype == np.dtype("S64")
            and len(set(keys.tolist())) == 48, "query key axis differs")
    decoded = [bytes(value).decode("ascii", errors="replace") for value in keys]
    require(all(_SHA.fullmatch(value) for value in decoded), "invalid anonymized query key")
    require(loci.shape == (660,) and loci.dtype.kind in "iu" and len(np.unique(loci)) == 660,
            "unique integer locus axis required")
    lengths = []
    for anchor in range(660):
        lo, hi = store.window_bounds(anchor)
        require(isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer))
                and 0 <= lo < hi, "nonempty complete anchor window required")
        lengths.append(int(hi - lo))
    order = np.lexsort((loci, np.asarray(lengths, dtype=np.int64)))
    strata, sampled = [], []
    anchor_rng = _rng(seed, 101)
    query_order = _rng(seed, 102).permutation(48).tolist()
    for h, indices in enumerate(np.array_split(order, 4)):
        strata.append({"stratum": h, "anchor_count": len(indices), "pair_count": 48 * len(indices),
                       "anchor_indices": indices.tolist(), "length_min": min(lengths[j] for j in indices),
                       "length_max": max(lengths[j] for j in indices),
                       "selected_anchor_count": 4, "anchor_inclusion_fraction": [4, len(indices)]})
        for anchor in anchor_rng.choice(indices, size=4, replace=False).tolist():
            slot = len(sampled)
            queries = query_order[2 * slot:2 * slot + 2]
            sampled.append({"stratum": h, "anchor_index": anchor, "anchor_length": lengths[anchor],
                            "pairs": [[query, anchor] for query in queries],
                            "query_key_sha256": [decoded[query] for query in queries]})
    execution_order = _rng(seed, 103).permutation(16).tolist()
    maximum = int(order[-1])
    manifest = {"schema_version": SCHEMA, "seed": seed,
                "rng": "numpy.PCG64/SeedSequence; streams=101,102,103,104",
                "store_shape": [48, 660, 8], "full_pair_count": 31680,
                "query_axis_sha256": hashlib.sha256(keys.tobytes()).hexdigest(),
                "locus_axis_sha256": hashlib.sha256(loci.tobytes()).hexdigest(),
                "length_axis_sha256": hashlib.sha256(np.asarray(lengths, dtype="<i8").tobytes()).hexdigest(),
                "anchor_lengths": lengths, "strata": strata, "anchor_samples": sampled,
                "execution_order": execution_order,
                "random_pair_order": _rng(seed, 104).permutation(32).tolist(),
                "warmup": {"stratum": None, "pairs": [[query, maximum] for query in query_order[32:34]],
                           "anchor_lengths": [lengths[maximum], lengths[maximum]]},
                "sampling_unit": "anchor then two query indices; 4 anchors per stratum",
                "primary_policy": "same-anchor batches; expected pass over all TRAIN query-anchor pairs",
                "random_policy": "same32pairs conditional replay; no full-pass estimator"}
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest):
    require(isinstance(manifest, dict) and manifest.get("schema_version") == SCHEMA,
            "sampling manifest schema differs")
    require(manifest.get("store_shape") == [48, 660, 8] and manifest.get("full_pair_count") == 31680,
            "sampling universe differs")
    lengths, strata, samples = (manifest.get(name) for name in ("anchor_lengths", "strata", "anchor_samples"))
    require(isinstance(lengths, list) and len(lengths) == 660
            and all(type(value) is int and value > 0 for value in lengths), "invalid full window lengths")
    require(hashlib.sha256(np.asarray(lengths, dtype="<i8").tobytes()).hexdigest()
            == manifest.get("length_axis_sha256"), "length axis digest differs")
    require(isinstance(strata, list) and len(strata) == 4, "four strata required")
    seen_anchors = []
    membership = {}
    for h, stratum in enumerate(strata):
        require(isinstance(stratum, dict) and stratum.get("stratum") == h
                and stratum.get("anchor_count") == 165 and stratum.get("pair_count") == 7920
                and stratum.get("selected_anchor_count") == 4
                and stratum.get("anchor_inclusion_fraction") == [4, 165], "stratum weights differ")
        indices = stratum.get("anchor_indices")
        require(isinstance(indices, list) and len(indices) == 165
                and all(type(j) is int and 0 <= j < 660 for j in indices), "stratum anchor axis differs")
        require(stratum.get("length_min") == min(lengths[j] for j in indices)
                and stratum.get("length_max") == max(lengths[j] for j in indices), "stratum length bounds differ")
        for j in indices:
            membership[j] = h
        seen_anchors.extend(indices)
    require(sorted(seen_anchors) == list(range(660)), "strata must partition all anchors exactly once")
    require(all(strata[h]["length_max"] <= strata[h + 1]["length_min"] for h in range(3)),
            "rank strata are not ordered by length")
    require(isinstance(samples, list) and len(samples) == 16, "sixteen selected anchors required")
    chosen, queries, query_hashes, counts = [], [], [], [0, 0, 0, 0]
    for sample in samples:
        require(isinstance(sample, dict), "sample descriptor required")
        h, j = sample.get("stratum"), sample.get("anchor_index")
        require(type(h) is int and type(j) is int and 0 <= h < 4 and membership.get(j) == h
                and sample.get("anchor_length") == lengths[j], "sample stratum/length differs")
        pairs, hashes = sample.get("pairs"), sample.get("query_key_sha256")
        require(isinstance(pairs, list) and len(pairs) == 2 and isinstance(hashes, list) and len(hashes) == 2,
                "two query pairs required per anchor")
        for pair, hashed in zip(pairs, hashes):
            require(isinstance(pair, list) and len(pair) == 2 and type(pair[0]) is int
                    and 0 <= pair[0] < 48 and type(pair[1]) is int and pair[1] == j,
                    "query pair is outside frozen axes")
            require(isinstance(hashed, str) and _SHA.fullmatch(hashed), "query key digest differs")
            queries.append(pair[0])
            query_hashes.append(hashed)
        chosen.append(j)
        counts[h] += 1
    require(len(set(chosen)) == 16 and len(set(queries)) == 32 and len(set(query_hashes)) == 32
            and counts == [4, 4, 4, 4],
            "selected anchors/queries are duplicated or stratum coverage differs")
    for name, n in (("execution_order", 16), ("random_pair_order", 32)):
        values = manifest.get(name)
        require(isinstance(values, list) and all(type(v) is int for v in values)
                and sorted(values) == list(range(n)), "execution/replay order is not a permutation")
    warmup = manifest.get("warmup")
    require(isinstance(warmup, dict) and warmup.get("stratum") is None
            and isinstance(warmup.get("pairs"), list) and len(warmup["pairs"]) == 2,
            "separate warmup required")
    for pair in warmup["pairs"]:
        require(isinstance(pair, list) and len(pair) == 2 and all(type(x) is int for x in pair)
                and 0 <= pair[0] < 48 and 0 <= pair[1] < 660 and lengths[pair[1]] == max(lengths),
                "warmup must use a complete maximum-length window")
    require(warmup["pairs"][0][0] != warmup["pairs"][1][0]
            and warmup["pairs"][0][1] == warmup["pairs"][1][1]
            and warmup.get("anchor_lengths") == [max(lengths), max(lengths)], "warmup axes differ")


def steps_for_case(manifest: dict, batch_size: int, policy: str) -> list[dict]:
    """Return measured rows only; warmup is never part of the sample estimator."""
    _validate_manifest(manifest)
    require(type(batch_size) is int and batch_size in (1, 2), "batch size must be 1 or 2")
    require(policy in ("grouped", "random") and (policy != "random" or batch_size == 2),
            "policy must be grouped, or random for batch2")
    ordered = [manifest["anchor_samples"][index] for index in manifest["execution_order"]]
    if policy == "random":
        all_pairs = [pair for sample in ordered for pair in sample["pairs"]]
        pairs = [all_pairs[index] for index in manifest["random_pair_order"]]
        return [{"stratum": None, "pairs": copy.deepcopy(pairs[start:start + 2]),
                 "anchor_lengths": [manifest["anchor_lengths"][j] for _, j in pairs[start:start + 2]]}
                for start in range(0, 32, 2)]
    rows = []
    for sample in ordered:
        for start in range(0, 2, batch_size):
            rows.append({"stratum": sample["stratum"],
                         "pairs": copy.deepcopy(sample["pairs"][start:start + batch_size]),
                         "anchor_lengths": [sample["anchor_length"]] * batch_size})
    return rows


def estimate_grouped_epoch(rows: list[dict], manifest: dict, batch_size: int) -> dict:
    """Weighted arithmetic total, conditional on warmed same-anchor batching.

    Each row must retain its sampled pairs/stratum and contain e2e_seconds.
    Ranges below describe the observed sample, not confidence or guaranteed
    bounds. B1 has eight timed rows but only four sampled anchors per stratum.
    """
    expected = steps_for_case(manifest, batch_size, "grouped")
    require(isinstance(rows, list) and len(rows) == len(expected), "measured grouped row count differs")
    expected_map = {tuple(map(tuple, row["pairs"])): row["stratum"] for row in expected}
    times, seen = [[] for _ in range(4)], set()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("pairs"), list), "timed row lacks sampled pairs")
        require(all(isinstance(pair, list) and len(pair) == 2
                    and all(type(value) is int for value in pair) for pair in row["pairs"]),
                "timed pairs require integer query/anchor indices")
        try:
            key = tuple(map(tuple, row["pairs"]))
        except TypeError as exc:
            raise ValueError("invalid timed pair inventory") from exc
        require(key in expected_map and key not in seen and type(row.get("stratum")) is int
                and row["stratum"] == expected_map[key], "timed inventory/strata differ; random is not estimable")
        elapsed = row.get("e2e_seconds")
        require(isinstance(elapsed, Real) and not isinstance(elapsed, (bool, np.bool_))
                and math.isfinite(elapsed) and elapsed > 0, "positive finite e2e_seconds required")
        seen.add(key)
        times[row["stratum"]].append(float(elapsed))
    strata, estimate, lower, upper = [], 0.0, 0.0, 0.0
    for h, values in enumerate(times):
        population_batches = manifest["strata"][h]["pair_count"] // batch_size
        mean = math.fsum(values) / len(values)
        estimate += population_batches * mean
        lower += population_batches * min(values)
        upper += population_batches * max(values)
        strata.append({"stratum": h, "sampled_anchors": 4, "measured_batches": len(values),
                       "population_pairs": 7920, "population_batches": population_batches,
                       "mean_e2e_seconds": mean, "minimum_e2e_seconds": min(values),
                       "maximum_e2e_seconds": max(values), "estimated_seconds": population_batches * mean})
    return {"estimated_grouped_pass_seconds": estimate, "estimated_pairs_per_second": 31680 / estimate,
            "full_pair_count": 31680, "batch_size": batch_size, "strata": strata,
            "descriptive_minmax_pass_seconds": [lower, upper],
            "uncertainty": "descriptive sample range, not CI or guaranteed bounds",
            "scope": "steady-state same-anchor batching; setup/warmup excluded; random replay not extrapolated"}
