#!/usr/bin/env python3
"""Bounded FIT window cost probe; never constructs a model or changes weights.

The probe authenticates one existing autosomal FIT request, reads one declared
person batch and tile, and counts the exact genetic-radius relationships used
by the current local-attention implementation.  It deliberately stops before
importing torch or constructing an attention graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

from r02_context_autosomal_dataset import open_autosomal


SCHEMA = "r02_structure_proportional_fit_probe_pre_v1"
CHROM22 = "ALLELE_POSITION_CHROM22_V1"
RARE_MEAN = "MEAN_WITH_EXPLICIT_BURDEN"
CODE_FILES = (
    "r02_structure_fit_probe.py",
    "r02_context_autosomal_dataset.py",
    "r02_event_mmap_reader.py",
    "r02_genomic_event_inputs.py",
    "r02_genomic_event_cache.py",
    "r02_genomic_pair_evidence.py",
    "r02_structure_increment.py",
    "r02_structure_input_plan.py",
    "r02_structure_materialize.py",
    "r02_structure_regions.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, expected_sha256):
    require(sha256(path) == expected_sha256, "probe plan SHA256 differs")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def validate_plan(plan):
    require(plan.get("schema") == SCHEMA and
            plan.get("status") == "ROOT_APPROVED_COUNT_ONLY_IMPLEMENTED_NOT_EXECUTED",
            "unreviewed probe plan")
    source, model, limits = plan.get("input", {}), plan.get("model", {}), plan.get("resources", {})
    require(source.get("stage") == "FIT", "probe accepts FIT only")
    require(type(source.get("chromosome")) is int and 1 <= source["chromosome"] <= 22 and
            type(source.get("person_batch_index")) is int and source["person_batch_index"] >= 0,
            "invalid chromosome/person batch")
    core = source.get("tile_core_bp")
    require(type(core) is list and len(core) == 2 and all(type(x) is int for x in core) and
            1 <= core[0] <= core[1], "invalid exact tile")
    require(model.get("family") == "structure_local_attention" and model.get("width") == 64 and
            model.get("depth") == 2 and model.get("heads") == 4 and model.get("dropout") == 0.0 and
            model.get("common_radius_cm") == model.get("rare_radius_cm") == 1.0 and
            model.get("arm") == "RARE" and model.get("seed") == 1701 and
            model.get("locus_descriptor") == CHROM22 and model.get("rare_pooling") == RARE_MEAN,
            "probe model differs from root-approved upper CW2 configuration")
    require(limits.get("cpu") == 1 and limits.get("memory_gib") == 3 and
            limits.get("wall_seconds") == 900 and limits.get("output_max_mib") == 10,
            "probe resource limits differ")
    for key in ("request_path", "request_sha256", "target_path", "target_sha256"):
        require(source.get(key), "missing authenticated FIT source " + key)
    return source, model, limits


def exact_neighbor_counts(sorted_queries, sorted_keys, radius):
    """Counts keys satisfying the literal abs(q-k)<=radius predicate.

    Two monotone cursors avoid q+/-radius cancellation and never materialize a
    query-by-key matrix. Inputs must be finite and sorted.
    """
    q = np.asarray(sorted_queries, dtype=np.float64)
    k = np.asarray(sorted_keys, dtype=np.float64)
    require(q.ndim == k.ndim == 1 and np.isfinite(q).all() and np.isfinite(k).all() and
            np.all(q[1:] >= q[:-1]) and np.all(k[1:] >= k[:-1]) and
            type(radius) in (int, float) and math.isfinite(radius) and radius > 0,
            "neighbor coordinates/radius differ")
    counts = np.zeros(len(q), dtype=np.int64)
    left = right = 0
    for index, value in enumerate(q):
        while left < len(k) and k[left] <= value and abs(value - k[left]) > radius:
            left += 1
        if right < left:
            right = left
        while right < len(k) and abs(value - k[right]) <= radius:
            right += 1
        counts[index] = right - left
    return counts


def shared_site_axis(cm, valid, label):
    cm, valid = np.asarray(cm), np.asarray(valid)
    require(cm.ndim == valid.ndim == 2 and cm.shape == valid.shape and valid.dtype == np.bool_ and
            cm.shape[0] <= 8 and cm.shape[1] > 0, label + " site axes differ")
    patterns = np.zeros(cm.shape[1], dtype=np.uint16)
    coordinates = np.full(cm.shape[1], np.nan, dtype=np.float64)
    for person in range(cm.shape[0]):
        mask = valid[person]
        require(np.isfinite(cm[person, mask]).all(), label + " valid coordinate is nonfinite")
        overlap = mask & np.isfinite(coordinates)
        require(np.array_equal(cm[person, overlap], coordinates[overlap]),
                label + " coordinate differs between people")
        coordinates[mask & ~np.isfinite(coordinates)] = cm[person, mask & ~np.isfinite(coordinates)]
        patterns |= mask.astype(np.uint16) << person
    require(np.array_equal(np.isfinite(coordinates), patterns != 0),
            label + " validity pattern/coordinate differs")
    return coordinates, patterns


def ideal_person_pairs(query_cm, query_valid, key_cm, key_valid, radius):
    values, maxima = [], []
    for person in range(query_valid.shape[0]):
        q = np.sort(query_cm[person, query_valid[person]].astype(np.float64))
        k = np.sort(key_cm[person, key_valid[person]].astype(np.float64))
        counts = exact_neighbor_counts(q, k, radius)
        values.append(int(counts.sum(dtype=np.int64)))
        maxima.append(int(counts.max(initial=0)))
    return values, maxima


def q1_candidate_key_counts(query_coordinates, query_patterns, key_coordinates, key_patterns, radius):
    """Exact key-union sizes made by candidate_keys for query blocks of one."""
    result = np.ones(len(query_coordinates), dtype=np.int64)  # forced numerical sentinel column zero
    observed_key_patterns = tuple(int(x) for x in np.unique(key_patterns) if x)
    key_groups = {pattern: np.sort(key_coordinates[key_patterns == pattern])
                  for pattern in observed_key_patterns}
    for query_pattern in (int(x) for x in np.unique(query_patterns) if x):
        indices = np.flatnonzero(query_patterns == query_pattern)
        order = np.argsort(query_coordinates[indices], kind="stable")
        sorted_indices = indices[order]
        queries = query_coordinates[sorted_indices]
        counts = np.zeros(len(queries), dtype=np.int64)
        for key_pattern, keys in key_groups.items():
            if query_pattern & key_pattern:
                counts += exact_neighbor_counts(queries, keys, radius)
        key0_is_neighbor = bool(key_patterns[0] & query_pattern) & (
            np.abs(queries - key_coordinates[0]) <= radius if key_patterns[0] else np.zeros(len(queries), dtype=bool))
        counts += (~key0_is_neighbor).astype(np.int64)
        result[sorted_indices] = counts
    return result


def assess_cost(event_q1_keys, common_q1_keys, people, model, limits, event_slots, common_slots):
    heads, width, depth = model["heads"], model["width"], model["depth"]
    head_width = width // heads
    attention_scores = int(people * heads * event_q1_keys.sum(dtype=np.int64))
    context_scores = int(people * common_q1_keys.sum(dtype=np.int64))
    # QK and attention-times-V each require at least one multiply and one add
    # per head coordinate. This excludes projections, softmax, MLP and backward.
    # A length-d dot product needs d multiplications and d-1 additions. QK and
    # A*V therefore need 2*(2d-1) operations per score; the context product
    # needs 2d-1. This avoids rounding the lower bound upward to 4d or 2d.
    attention_forward_flops = int(attention_scores * (4 * head_width - 2) * depth)
    context_forward_flops = int(context_scores * (2 * width - 1))
    forward_lower_bound = attention_forward_flops + context_forward_flops
    # Intel Broadwell AVX2 has a 32 FP32-FLOP/cycle FMA ceiling. Multiplying it
    # by four deliberately makes this admission bound generous to the run.
    hardware_genuine_upper_flops_second = int(2.2e9 * 32 * 4)
    possible_flops = hardware_genuine_upper_flops_second * limits["wall_seconds"]
    q1_peak_score_elements = int(people * heads * event_q1_keys.max(initial=0))
    q1_peak_context_elements = int(people * common_q1_keys.max(initial=0))
    lower_bound_bytes = int((people * (event_slots + common_slots) * width +
                             q1_peak_score_elements + q1_peak_context_elements) * 4)
    time_impossible = forward_lower_bound > possible_flops
    memory_lower_bound_exceeds = lower_bound_bytes > limits["memory_gib"] * 2**30
    allowed = not time_impossible and not memory_lower_bound_exceeds
    return {
        "q1_attention_score_elements_per_layer": attention_scores,
        "q1_context_matrix_elements": context_scores,
        "forward_flops_lower_bound": forward_lower_bound,
        "generous_one_cpu_flops_per_second_upper_bound": hardware_genuine_upper_flops_second,
        "generous_900_second_flop_capacity": possible_flops,
        "required_flops_per_second_for_forward_only": forward_lower_bound / limits["wall_seconds"],
        "q1_tensor_bytes_lower_bound": lower_bound_bytes,
        "attention_block_calls_forward_and_checkpoint_recompute": int(2 * depth * event_slots),
        "context_block_calls_forward_and_checkpoint_recompute": int(2 * event_slots),
        "decision": ("BLOCK_FORWARD_BACKWARD_TIME_LOWER_BOUND" if time_impossible else
                     "BLOCK_FORWARD_BACKWARD_MEMORY_LOWER_BOUND" if memory_lower_bound_exceeds else
                     "COUNTS_DO_NOT_PROVE_VIABILITY_ROOT_REVIEW_REQUIRED"),
        "forward_backward_allowed": False,
        "reason": ("Even forward-only mandatory attention arithmetic exceeds a fourfold-generous AVX2 hardware ceiling."
                   if time_impossible else
                   "A mandatory live-tensor lower bound exceeds 3 GiB." if memory_lower_bound_exceeds else
                   "Passing lower bounds is necessary but insufficient; no model execution is automatic."),
    }


def run(plan_path, plan_sha256, output):
    started = time.monotonic()
    plan = read_json(plan_path, plan_sha256)
    source, model, limits = validate_plan(plan)
    initial_sources = {"request": sha256(source["request_path"]), "target": sha256(source["target_path"])}
    require(initial_sources == {"request": source["request_sha256"], "target": source["target_sha256"]},
            "FIT request/target SHA256 differs")
    dataset = open_autosomal(source["request_path"], source["request_sha256"])
    request = dataset.request
    require(request["stage"] == "FIT" and request["common_radius_cm"] == model["common_radius_cm"] and
            request["rare_radius_cm"] == model["rare_radius_cm"] and
            request["attention_depth"] == model["depth"], "FIT request geometry differs")
    chromosome = str(source["chromosome"])
    require(source["tile_core_bp"] in request["tiles"][chromosome], "tile is absent from authenticated request")
    batch_index = source["person_batch_index"]
    require(batch_index < dataset.n_person_batches, "person batch is absent")
    ids, roles, components = dataset.person_batches[batch_index]
    reader = dataset._reader(chromosome)
    start_bp, end_bp = source["tile_core_bp"]
    view = reader.load_window(list(ids), start_bp, end_bp, request["block_id"], request["exclusion_scope"],
                              request["common_radius_cm"], request["rare_radius_cm"], request["attention_depth"])
    require(tuple(map(str, view["person_id"])) == ids and tuple(map(str, view["role"])) == roles and
            tuple(map(str, view["component_id"])) == components and set(roles) == {"FIT"},
            "FIT person identity/role/component differs")
    event_cm, event_valid = np.asarray(view["events"]["cm"]), np.asarray(view["events"]["valid"])
    common_cm, common_valid = np.asarray(view["common"]["cm"]), np.asarray(view["common"]["valid"])
    event_coordinates, event_patterns = shared_site_axis(event_cm, event_valid, "EVENT")
    common_coordinates, common_patterns = shared_site_axis(common_cm, common_valid, "COMMON")
    event_pairs, event_max = ideal_person_pairs(event_cm, event_valid, event_cm, event_valid,
                                                 model["rare_radius_cm"])
    context_pairs, context_max = ideal_person_pairs(event_cm, event_valid, common_cm, common_valid,
                                                     model["common_radius_cm"])
    event_q1 = q1_candidate_key_counts(event_coordinates, event_patterns,
                                       event_coordinates, event_patterns, model["rare_radius_cm"])
    common_q1 = q1_candidate_key_counts(event_coordinates, event_patterns,
                                        common_coordinates, common_patterns, model["common_radius_cm"])
    cost = assess_cost(event_q1, common_q1, len(ids), model, limits,
                       event_cm.shape[1], common_cm.shape[1])
    final_sources = {"request": sha256(source["request_path"]), "target": sha256(source["target_path"])}
    require(final_sources == initial_sources, "FIT source changed during probe")
    output = Path(output)
    require(not output.exists(), "probe output already exists")
    output.mkdir(parents=True)
    code_dir = Path(__file__).resolve().parent
    receipt = {
        "schema": "r02_structure_fit_neighbor_cost_receipt_v1",
        "status": "FIT_DENSE_WINDOW_COUNT_COMPLETE_NO_MODEL_NO_TRAINING",
        "plan": {"path": str(Path(plan_path).resolve()), "sha256": plan_sha256},
        "source_sha256": final_sources,
        "scope": {"stage": "FIT", "chromosome": int(chromosome), "tile_core_bp": [start_bp, end_bp],
                  "person_batch_index": batch_index, "people": len(ids), "persons_emitted": False,
                  "select_opened": False, "evaluate_opened": False},
        "model_configuration_for_cost": model,
        "counts": {
            "event_site_slots": int(event_cm.shape[1]), "common_site_slots": int(common_cm.shape[1]),
            "event_valid_total": int(event_valid.sum()), "common_valid_total": int(common_valid.sum()),
            "ideal_ordered_event_neighbor_pairs_by_person": event_pairs,
            "ideal_ordered_event_neighbor_pairs_total": int(sum(event_pairs)),
            "maximum_event_neighbors_by_person": event_max,
            "ideal_ordered_event_common_pairs_by_person": context_pairs,
            "ideal_ordered_event_common_pairs_total": int(sum(context_pairs)),
            "maximum_common_neighbors_by_event_by_person": context_max,
            "q1_candidate_keys_sum": int(event_q1.sum(dtype=np.int64)),
            "q1_candidate_keys_max": int(event_q1.max(initial=0)),
            "q1_context_keys_sum": int(common_q1.sum(dtype=np.int64)),
            "q1_context_keys_max": int(common_q1.max(initial=0)),
        },
        "cost": cost,
        "execution": {"model_created": False, "torch_imported": False, "attention_graph_created": False,
                      "optimizer_created": False, "optimizer_step": False, "weights_modified": False,
                      "elapsed_seconds": time.monotonic() - started,
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      "cpu_limit": limits["cpu"], "memory_limit_gib": limits["memory_gib"],
                      "wall_limit_seconds": limits["wall_seconds"]},
        "code_sha256": {name: sha256(code_dir / name) for name in CODE_FILES},
        "allowed_conclusion": "Operational cost of this exact real FIT window under the current blocked-attention equations.",
        "not_allowed": ["complete 22-autosome update duration", "optimization or convergence",
                        "SELECT/EVALUATE performance", "biological information"],
    }
    encoded = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    require(len(encoded.encode()) <= limits["output_max_mib"] * 2**20, "probe receipt exceeds output limit")
    (output / "receipt.json").write_text(encoded)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = run(args.plan, args.plan_sha256, args.output)
    print(json.dumps({"status": receipt["status"], "decision": receipt["cost"]["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
