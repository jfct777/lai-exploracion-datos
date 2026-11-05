#!/usr/bin/env python3
"""Inventory complete visible-flank EVENT pairs without reading genotypes.

The durable object is a compact recipe per cell and candidate radius.  Pair
indices are generated only for one requested cell; no chromosome-wide repeated
pair tensor is stored.  Inclusion depends only on authenticated locus geometry.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np


REQUEST_SCHEMA = "r02_masked_carrier_event_pair_inventory_request_v1"
MANIFEST_SCHEMA = "r02_masked_carrier_event_pair_manifest_v1"
RECIPE_SCHEMA = "r02_masked_carrier_event_pair_recipe_v1"
GENERATOR_VERSION = "visible_flanks_float64_v1"
SOURCE_MINOR = "source_minor_DNABR2723"
BASES = "ACGT"
PAIR_KINDS = ("left_left", "left_right", "right_right")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _hex64(value):
    return np.float64(value).item().hex()


def _hash_rows(rows):
    state = hashlib.sha256()
    for row in rows:
        state.update(canonical_bytes(row) + b"\n")
    return state.hexdigest()


def canonical_event_axis(chromosome, position_bp, position_cm, ref, alt):
    """Return a deterministic float64 geometry axis and its identity."""
    _require(type(chromosome) is int and 1 <= chromosome <= 22,
             "autosomal integer chromosome required")
    bp = np.asarray(position_bp, dtype=np.int64)
    cm = np.asarray(position_cm, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.uint8)
    alt = np.asarray(alt, dtype=np.uint8)
    _require(bp.ndim == cm.ndim == ref.ndim == alt.ndim == 1
             and len(bp) == len(cm) == len(ref) == len(alt) > 0,
             "nonempty equal one-dimensional EVENT geometry arrays required")
    _require(bool(np.isfinite(cm).all()) and bool((bp > 0).all())
             and bool((ref < 4).all()) and bool((alt < 4).all())
             and bool((ref != alt).all()), "invalid or non-finite EVENT geometry")
    ids = np.asarray([f"{chromosome}:{int(p)}:{BASES[int(r)]}:{BASES[int(a)]}"
                      for p, r, a in zip(bp, ref, alt)], dtype=object)
    _require(len(set(ids.tolist())) == len(ids), "duplicate EVENT locus identity")
    order = tuple(sorted(range(len(bp)),
        key=lambda i: (np.float64(cm[i]), int(bp[i]), str(ids[i]))))
    _require(order == tuple(range(len(bp))),
             "frozen EVENT axis is not ordered by cM, bp and locus identity")
    identity = _hash_rows((chromosome, int(p), _hex64(c), BASES[int(r)],
                           BASES[int(a)], str(identifier))
                          for p, c, r, a, identifier in
                          zip(bp, cm, ref, alt, ids))
    return {"chromosome": chromosome, "position_bp": bp, "position_cm": cm,
            "ref": ref, "alt": alt, "locus_ids": tuple(ids.tolist()),
            "sha256": identity, "count": len(bp)}


def canonical_cell_axis(edges):
    values = np.asarray(edges, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 2 and len(values) > 0
             and bool(np.isfinite(values).all())
             and bool((values[:, 1] > values[:, 0]).all()),
             "finite nonempty increasing cell intervals required")
    _require(len(values) == 1 or bool(np.equal(values[:-1, 1],
                                               values[1:, 0]).all()),
             "cell boundaries must be exactly shared in float64")
    return {"edges": values, "count": len(values),
            "sha256": _hash_rows((index, _hex64(left), _hex64(right))
                                  for index, (left, right) in enumerate(values))}


def _pair_counts(left_count, right_count):
    ll = left_count * (left_count - 1) // 2
    lr = left_count * right_count
    rr = right_count * (right_count - 1) // 2
    return ll, lr, rr, ll + lr + rr


def _summaries(rows, name):
    values = [int(row[name]) for row in rows]
    ordered = sorted(values)
    return {"minimum": ordered[0], "median": ordered[len(ordered) // 2],
            "maximum": ordered[-1], "sum_over_overlapping_cells": sum(values)}


def iter_cell_pairs(recipe, cell_index):
    """Yield complete global index pairs in LL, LR, RR lexicographic order."""
    _require(recipe.get("schema") == RECIPE_SCHEMA,
             "EVENT pair recipe schema differs")
    rows = recipe.get("cells")
    _require(type(cell_index) is int and isinstance(rows, list)
             and 0 <= cell_index < len(rows), "cell index outside recipe")
    row = rows[cell_index]
    left = range(row["left_start"], row["left_stop"])
    right = range(row["right_start"], row["right_stop"])
    for i in left:
        for j in range(i + 1, row["left_stop"]):
            yield i, j, 0
    for i in left:
        for j in right:
            yield i, j, 1
    for i in right:
        for j in range(i + 1, row["right_stop"]):
            yield i, j, 2


def _stream_pairs_hash(recipe, axis, limit):
    total = recipe["counts"]["pairs"]["sum_over_overlapping_cells"]
    if limit == 0 or total > limit:
        return {"status": "RECIPE_ONLY_NOT_ENUMERATED", "sha256": None,
                "pair_records": total, "limit": limit}
    state = hashlib.sha256()
    seen = 0
    for cell in range(len(recipe["cells"])):
        for left, right, kind in iter_cell_pairs(recipe, cell):
            state.update(canonical_bytes(
                [cell, axis["locus_ids"][left], axis["locus_ids"][right],
                 PAIR_KINDS[kind]]) + b"\n")
            seen += 1
    _require(seen == total, "streamed pair count differs from recipe")
    return {"status": "ENUMERATED_STREAM_HASH", "sha256": state.hexdigest(),
            "pair_records": seen, "limit": limit}


def build_recipe(axis, cells, radius_cm, *, stream_pair_hash_limit=0):
    """Build six compact index boundaries for every cell."""
    _require(axis.get("count") == len(axis.get("position_cm", ())),
             "canonical EVENT axis required")
    _require(cells.get("count") == len(cells.get("edges", ())),
             "canonical cell axis required")
    radius = np.float64(radius_cm)
    _require(np.isfinite(radius) and radius > 0,
             "finite positive float64 locus-to-cell radius required")
    cm, edges = axis["position_cm"], cells["edges"]
    domain_left, domain_right = edges[0, 0], edges[-1, 1]
    rows = []
    for index, (left_edge, right_edge) in enumerate(edges):
        left_boundary = np.maximum(domain_left, np.float64(left_edge - radius))
        right_boundary = np.minimum(domain_right, np.float64(right_edge + radius))
        left_start = int(np.searchsorted(cm, left_boundary, side="left"))
        left_stop = int(np.searchsorted(cm, left_edge, side="left"))
        hidden_start = left_stop
        hidden_stop = int(np.searchsorted(
            cm, right_edge, side="right" if right_edge == domain_right else "left"))
        right_start = int(np.searchsorted(cm, right_edge, side="left"))
        right_stop = int(np.searchsorted(
            cm, right_boundary,
            side="right" if right_boundary == domain_right else "left"))
        if right_edge == domain_right:
            right_start = right_stop
        left_count, hidden_count = left_stop-left_start, hidden_stop-hidden_start
        right_count = right_stop-right_start
        ll, lr, rr, pair_count = _pair_counts(left_count, right_count)
        rows.append({"cell_index": index, "left_start": left_start,
            "left_stop": left_stop, "hidden_start": hidden_start,
            "hidden_stop": hidden_stop, "right_start": right_start,
            "right_stop": right_stop, "left_loci": left_count,
            "hidden_loci": hidden_count, "right_loci": right_count,
            "left_left_pairs": ll, "left_right_pairs": lr,
            "right_right_pairs": rr, "pairs": pair_count,
            "cell_left_cm_float64_hex": _hex64(left_edge),
            "cell_right_cm_float64_hex": _hex64(right_edge),
            "left_boundary_cm_float64_hex": _hex64(left_boundary),
            "right_boundary_cm_float64_hex": _hex64(right_boundary)})
    compact_identity = digest({"schema": RECIPE_SCHEMA,
        "generator_version": GENERATOR_VERSION, "event_axis_sha256": axis["sha256"],
        "cell_axis_sha256": cells["sha256"], "radius_cm_float64_hex": _hex64(radius),
        "cells": rows})
    recipe = {"schema": RECIPE_SCHEMA, "generator_version": GENERATOR_VERSION,
        "identity_sha256": compact_identity, "event_axis_sha256": axis["sha256"],
        "cell_axis_sha256": cells["sha256"], "radius_cm": float(radius),
        "radius_cm_float64_hex": _hex64(radius),
        "radius_meaning": "MAXIMUM_LOCUS_TO_HIDDEN_CELL_DISTANCE_NOT_PAIR_DISTANCE",
        "territory": "[a-r,a)_UNION_[b,b+r)_CLIPPED_FINAL_DOMAIN_ENDPOINT_CLOSED",
        "ordering": "LEFT_LEFT_THEN_LEFT_RIGHT_THEN_RIGHT_RIGHT_LEXICOGRAPHIC",
        "cells": rows,
        "counts": {name: _summaries(rows, name) for name in
                   ("left_loci", "hidden_loci", "right_loci", "left_left_pairs",
                    "left_right_pairs", "right_right_pairs", "pairs")},
        "global_pair_tensor_materialized": False,
        "pair_array_bytes_per_cell": "16*P_s_FOR_INT64_PAIR_INDICES",
        "pair_geometry_bytes_per_cell":
            "ADDITIONAL_16*P_s_FOR_INT64_DELTA_BP_FLOAT64_DELTA_CM",
        "pair_segment_and_kind_bytes_per_cell":
            "ADDITIONAL_9*P_s_FOR_INT64_SEGMENT_UINT8_PAIR_KIND",
        "materialized_numpy_bytes_per_cell": "41*P_s_EXCLUDING_PYTHON_TEMPORARIES_AND_MODEL_ACTIVATIONS",
        "forward_activation_shape":
            "DOMINANT_B*Rb*K*(4+F)_PAIR_INPUT_PLUS_B*Rb*K*W_CONTRIBUTION_PLUS_B*Rb*S*W_OUTPUT",
        "autograd_memory":
            "UNBOUNDED_UNTIL_B_Rb_K_F_W_DTYPE_AND_CHECKPOINT_POLICY_ARE_FIXED"}
    recipe["streamed_pairs"] = _stream_pairs_hash(
        recipe, axis, stream_pair_hash_limit)
    return recipe


def materialize_cell_pairs(axis, recipe, cell_index, *, max_pairs):
    """Materialize one cell only and preserve raw float64 geometry."""
    _require(recipe.get("event_axis_sha256") == axis.get("sha256"),
             "EVENT axis differs from pair recipe")
    row = recipe["cells"][cell_index]
    _require(type(max_pairs) is int and max_pairs >= 0
             and row["pairs"] <= max_pairs,
             "cell pair count exceeds explicit materialization limit")
    records = list(iter_cell_pairs(recipe, cell_index))
    _require(len(records) == row["pairs"], "generated pair count differs")
    pairs = np.asarray([(a, b) for a, b, _ in records], dtype=np.int64).reshape(-1, 2)
    kinds = np.asarray([kind for _, _, kind in records], dtype=np.uint8)
    if len(pairs):
        delta_bp = np.abs(axis["position_bp"][pairs[:, 1]] -
                          axis["position_bp"][pairs[:, 0]])
        delta_cm = np.abs(axis["position_cm"][pairs[:, 1]] -
                          axis["position_cm"][pairs[:, 0]])
    else:
        delta_bp = np.empty(0, dtype=np.int64)
        delta_cm = np.empty(0, dtype=np.float64)
    _require(not len(pairs) or not np.isin(pairs, np.arange(
        row["hidden_start"], row["hidden_stop"])).any(),
        "hidden cell locus entered generated pairs")
    segments = np.zeros(len(pairs), dtype=np.int64)
    return {"pairs": pairs, "pair_kind": kinds, "delta_bp": delta_bp,
            "delta_cm": delta_cm,
            # A materialized one-cell fragment has one local segment.  The
            # chromosome-wide cell identity remains in the authenticated row.
            "pair_segment": segments,
            "has_pair_support": bool(len(pairs)),
            "materialized_bytes": sum(value.nbytes for value in
                                      (pairs, kinds, delta_bp, delta_cm, segments))}


def adapt_pairs_to_fragment(axis, materialized, fragment_locus_ids):
    """Map authenticated global locus identities to an exact fragment axis."""
    local = {value: index for index, value in enumerate(fragment_locus_ids)}
    _require(len(local) == len(fragment_locus_ids), "duplicate fragment locus identity")
    output = []
    for left, right in materialized["pairs"]:
        left_id, right_id = axis["locus_ids"][left], axis["locus_ids"][right]
        _require(left_id in local and right_id in local,
                 "fragment omits a required visible-flank EVENT locus")
        _require(local[left_id] < local[right_id],
                 "fragment EVENT axis order differs from authenticated axis")
        output.append((local[left_id], local[right_id]))
    return np.asarray(output, dtype=np.int64).reshape(-1, 2)


def inventory(axis, cells, radii, *, stream_pair_hash_limit=0,
              source_metadata=None, matched_information_primary=None):
    values = tuple(np.float64(value) for value in radii)
    _require(values and all(np.isfinite(value) and value > 0 for value in values)
             and len(set(value.hex() for value in values)) == len(values),
             "distinct finite positive candidate radii required")
    recipes = [build_recipe(axis, cells, value,
                            stream_pair_hash_limit=stream_pair_hash_limit)
               for value in values]
    result = {"schema": MANIFEST_SCHEMA,
        "status": "DESCRIPTIVE_GEOMETRY_ONLY_NOT_TRAINING_ADMISSION",
        "generator_version": GENERATOR_VERSION,
        "code_sha256": sha256(__file__), "chromosome": axis["chromosome"],
        "source_minor": SOURCE_MINOR, "event_locus_axis_sha256": axis["sha256"],
        "event_locus_count": axis["count"], "segment_axis_sha256": cells["sha256"],
        "cell_count": cells["count"], "candidate_recipes": recipes,
        "candidate_radius_cm": [float(value) for value in values],
        "candidate_radius_float64_hex": [value.hex() for value in values],
        "radius_adopted": None, "pair_feature_transform_adopted": None,
        "selection_forbidden": ["dosage", "carrier_status", "frequency", "person",
            "family", "loss", "score", "SELECT", "EVALUATE"],
        "human_training_admitted": False, "training_performed": False,
        "global_pair_tensor_materialized": False,
        "source_metadata": source_metadata or {}}
    if matched_information_primary is not None:
        primary = np.float64(matched_information_primary)
        _require(primary.hex() in {value.hex() for value in values},
                 "matched-information primary must be a candidate radius")
        result["matched_information_primary_radius_cm"] = float(primary)
        result["matched_information_primary_role"] = (
            "SAME_AUTHENTICATED_HISTORICAL_LOCAL_MODEL_TERRITORY_FOR_SUBSEQUENT_"
            "COMPARISON_NOT_OPTIMAL_GENOMIC_SCALE")
    result["manifest_sha256"] = digest(result)
    return result


def _load_authenticated_request(request):
    _require(request.get("schema") == REQUEST_SCHEMA
             and request.get("human_training_admitted") is False
             and request.get("select_opened") is False
             and request.get("evaluate_opened") is False
             and request.get("source_minor") == SOURCE_MINOR,
             "closed metadata-only inventory request required")
    limits = request.get("limits", {})
    radii = request.get("candidate_radius_cm", ())
    maximum = limits.get("max_candidate_radii")
    _require(type(maximum) is int and maximum > 0
             and isinstance(radii, (list, tuple)) and 0 < len(radii) <= maximum,
             "candidate radius count exceeds explicit limit")
    _require(all(type(value) in (int, float) and np.isfinite(value) and value > 0
                 for value in radii) and len(set(radii)) == len(radii),
             "candidate radii must be distinct positive finite numbers")
    source = request.get("event_source", {})
    declaration = request.get("source_declaration", {})
    declaration_path = Path(declaration.get("path", "")).resolve()
    _require(declaration_path.is_file()
             and sha256(declaration_path) == declaration.get("sha256"),
             "source declaration missing or changed")
    declared = json.loads(declaration_path.read_text()).get("source", {})
    _require(declared.get("store") == source.get("store")
             and declared.get("manifest_sha256") == source.get("manifest_sha256")
             and declared.get("rare_counted_allele") == SOURCE_MINOR,
             "authenticated source does not declare frozen source_minor")
    root = Path(source.get("store", "")).resolve()
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file() and sha256(manifest_path) == source.get("manifest_sha256"),
             "EVENT source manifest missing or changed")
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get("schema") == "r02_genomic_event_chromosome_v1"
             and int(manifest.get("chromosome")) == request.get("chromosome")
             and manifest.get("stage") == "COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING"
             and manifest.get("training_allowed") is False,
             "frozen raw EVENT source required")
    outputs = manifest.get("outputs_sha256", {})
    bound = {}
    for name in ("event_inputs.json", "source/materialize.json",
                 "source_verification.json"):
        path = root/name
        _require(path.is_file() and sha256(path) == outputs.get(name),
                 "EVENT source provenance file missing or changed")
        bound[name] = json.loads(path.read_text())
    _require(manifest.get("event_contract_sha256") == outputs["event_inputs.json"]
             and bound["event_inputs.json"].get("materialize_contract_sha256") ==
                 outputs["source/materialize.json"]
             and bound["source/materialize.json"].get("rare_allele_convention") ==
                 "SOURCE_MINOR_M02_1"
             and bound["source/materialize.json"].get(
                 "source_rare_rule", {}).get("ascertainment") ==
                 "FIXED_SOURCE_COHORT_NOT_FIT_RESELECTED"
             and bound["source_verification.json"].get("event_contract_sha256") ==
                 outputs["event_inputs.json"],
             "authenticated EVENT provenance does not bind fixed source-minor rare axis")
    chunks = manifest.get("channels", {}).get("rare")
    _require(isinstance(chunks, list) and chunks, "rare EVENT chunk inventory missing")
    limits = request.get("limits", {})
    max_loci, max_bytes = limits.get("max_event_loci"), limits.get("max_source_bytes")
    _require(type(max_loci) is int and max_loci > 0 and type(max_bytes) is int
             and max_bytes > 0 and sum(row["n_sites"] for row in chunks) <= max_loci
             and sum(row["bytes"] for row in chunks) <= max_bytes,
             "EVENT source exceeds explicit locus/byte limits")
    arrays = {name: [] for name in ("position_bp", "position_cm", "ref", "alt")}
    for descriptor in chunks:
        path = (root / descriptor["path"]).resolve()
        _require(root in path.parents and path.is_file()
                 and path.stat().st_size == descriptor["bytes"]
                 and sha256(path) == descriptor["sha256"],
                 "rare EVENT chunk missing or changed")
        with np.load(path, allow_pickle=False) as data:
            _require(set(arrays) <= set(data.files), "EVENT geometry arrays missing")
            for name in arrays:
                value = np.asarray(data[name])
                _require(value.ndim == 1 and len(value) == descriptor["n_sites"],
                         "EVENT geometry array shape/count differs")
                arrays[name].append(value.copy())
    axis = canonical_event_axis(request["chromosome"],
        *(np.concatenate(arrays[name]) for name in arrays))
    axis_dir = Path(request["cell_axis"]["directory"]).resolve()
    receipt_path, cells_path = axis_dir/"receipt.json", axis_dir/"cells.tsv"
    _require(receipt_path.is_file() and cells_path.is_file()
             and sha256(receipt_path) == request["cell_axis"]["receipt_sha256"]
             and sha256(cells_path) == request["cell_axis"]["cells_sha256"],
             "cell axis missing or changed")
    receipt = json.loads(receipt_path.read_text())
    _require(receipt.get("schema") == "r02_autosomal_cell_axis_receipt_v1"
             and receipt.get("status") == "COMPLETE"
             and receipt.get("cells_tsv_sha256") == request["cell_axis"]["cells_sha256"],
             "cell receipt differs")
    with cells_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    _require(len(rows) <= limits.get("max_cells", 0),
             "cell axis exceeds explicit limit")
    cells = canonical_cell_axis([[float(row["left_edge_cm"]),
                                  float(row["right_edge_cm"])] for row in rows])
    historical = request.get("historical_model_declaration", {})
    historical_path = Path(historical.get("path", "")).resolve()
    _require(historical_path.is_file()
             and sha256(historical_path) == historical.get("sha256"),
             "historical model declaration missing or changed")
    historical_value = json.loads(historical_path.read_text())
    primary = request.get("matched_information_primary_radius_cm")
    _require(historical_value.get("schema") == "r02_masked_chr22_fit_request_v1"
             and historical_value.get("condition") == "full"
             and np.float64(historical_value.get("model", {}).get("rare_radius_cm"))
                 == np.float64(primary),
             "matched-information radius differs from authenticated historical model")
    metadata = {"event_manifest_sha256": source["manifest_sha256"],
        "event_chunk_count": len(chunks), "event_chunk_bytes_authenticated":
        sum(row["bytes"] for row in chunks), "cell_receipt_sha256":
        request["cell_axis"]["receipt_sha256"], "cells_tsv_sha256":
        request["cell_axis"]["cells_sha256"],
        "npz_arrays_opened": list(arrays), "dosage_or_observed_opened": False,
        "frequency_or_counted_allele_opened": False}
    metadata["rare_orientation_evidence"] = {
        "materialize_contract_sha256": outputs["source/materialize.json"],
        "rare_allele_convention": "SOURCE_MINOR_M02_1",
        "ascertainment": "FIXED_SOURCE_COHORT_NOT_FIT_RESELECTED"}
    return axis, cells, metadata, primary


def run_request(request):
    started = time.monotonic()
    axis, cells, metadata, primary = _load_authenticated_request(request)
    result = inventory(axis, cells, request.get("candidate_radius_cm", ()),
        stream_pair_hash_limit=request["limits"].get("stream_pair_hash_limit", 0),
        source_metadata=metadata, matched_information_primary=primary)
    result["elapsed_seconds"] = time.monotonic() - started
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result["peak_rss_bytes"] = rss_bytes
    _require(result["elapsed_seconds"] <= request["limits"]["max_wall_seconds"]
             and rss_bytes <= request["limits"]["max_rss_bytes"],
             "inventory exceeded declared wall/RSS limit")
    _require(len(canonical_bytes(result)) + 1 <=
             request["limits"]["max_output_bytes"],
             "inventory exceeds declared output-byte limit")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    request_path, output = Path(args.request).resolve(), Path(args.output).resolve()
    _require(request_path.is_file() and not output.exists(),
             "request must exist and output must be fresh")
    result = run_request(json.loads(request_path.read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    _require(not temporary.exists(), "temporary output already exists")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_bytes(result) + b"\n")
            stream.flush(); os.fsync(stream.fileno())
        # Same-directory hard linking publishes without replacing a file that
        # appeared after the fresh-output check.
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise ValueError("output appeared concurrently; refusing to publish") from error
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.unlink(temporary)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
