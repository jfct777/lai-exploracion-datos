#!/usr/bin/env python3
"""Build admitted Deep Sets or local-attention autosomal person requests.

This is a successor to the frozen CN builder.  It retains the authenticated
readers, target checks, person partitioning and exact tile algorithm, while
budgeting the wider halos required by stacked local attention.
"""
import argparse
import bisect
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from r02_context_autosomal_dataset import (MMAP_READER_BACKEND, PERSON_BATCH_SCHEMA,
    PERSON_PARTITION_SCHEMA, READER_BACKEND_SCHEMA, partition_people, person_axis_sha256)
import r02_genomic_event_inputs as events
import r02_structure_person_batch_plan as structure_plan
from r02_structure_person_batch_plan import request_context

ROOT = Path(__file__).resolve().parents[1]
REAL_BUILDER = (ROOT / ".claude/runs/r02-corrections-20261006j/parallel-launch-20261008ac/"
                "person_batching/build_human_autosomal_person_batches.py")
EXACT_TILES = (ROOT / ".claude/runs/r02-corrections-20261006j/parallel-advancement-20261010ay/"
               "neural/frozen-v13d/budget_tiles_reader_exact.py")
LEGACY_BUILDER = (ROOT / ".claude/runs/r02-corrections-20261006j/neural-human-integration-20261007v/"
                  "code/build_human_autosomal_manifests.py")
FROZEN_PREDECESSORS = {
    REAL_BUILDER: "14df2fabb1aab632d0e4b8acee7905f5209a09d1839575a202ca6f5019c28f87",
    EXACT_TILES: "e377af3730101ab7e9cd2ef0d240738be53d71cb59ffb46e6e134680bd212250",
    LEGACY_BUILDER: "b2bc64f2149915ca89e5bdb4aafb2506ac75b18bb74466b34fca642079155f9a",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


for frozen_path, expected_sha256 in FROZEN_PREDECESSORS.items():
    if file_sha256(frozen_path) != expected_sha256:
        raise RuntimeError(f"frozen predecessor SHA256 differs: {frozen_path}")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load authenticated predecessor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


predecessor = _load("r02_person_batch_builder_predecessor", REAL_BUILDER)
exact_tiles = _load("r02_person_batch_exact_tiles", EXACT_TILES)
legacy = predecessor.legacy
require, sha, write_json = legacy.require, legacy.sha, legacy.write_json


def transitive_code_inventory():
    """Files that must be copied and reauthenticated in a deployable package."""
    paths = [Path(__file__).resolve(), Path(structure_plan.__file__).resolve(),
             Path(events.__file__).resolve(), ROOT / "bin/r02_context_autosomal_dataset.py",
             Path(structure_plan.CN_PLANNER),
             *FROZEN_PREDECESSORS]
    return [{"path": str(path), "sha256": file_sha256(path)} for path in paths]


def budget_tiles(reader, context, max_window_sites=None):
    """Reuse the exact binary search with the bounds read by load_window()."""
    event_halo = (context["rare_radius_cm"] or 0) * context["attention_depth"]
    common_halo = event_halo + context["common_radius_cm"]

    def actual_bounds(current_reader, start, end, halo):
        # exact_tiles asks with common_halo for COMMON and zero for RARE.
        actual_halo = event_halo if halo == 0 else halo
        return legacy.halo_bounds(current_reader, start, end, actual_halo)

    bounded_reader = reader
    if max_window_sites is not None:
        require(type(max_window_sites) is int and
                max_window_sites >= reader.r["max_window_sites"],
                "external preparation row limit differs")
        bounded_reader = copy.copy(reader)
        bounded_reader.r = {**reader.r, "max_window_sites": max_window_sites}
    channel_indices = {}
    for kind in ("common", "rare"):
        chunks = bounded_reader.manifest["channels"][kind]
        starts = [chunk["start_bp"] for chunk in chunks]
        ends = [chunk["end_bp"] for chunk in chunks]
        prefix = [0]
        for chunk in chunks:
            prefix.append(prefix[-1] + chunk["n_sites"])
        channel_indices[kind] = (starts, ends, prefix)

    def upper_sites(current_reader, kind, lower, upper):
        require(current_reader is bounded_reader, "reader identity changed during tile budgeting")
        starts, ends, prefix = channel_indices[kind]
        first = bisect.bisect_left(ends, lower)
        after = bisect.bisect_right(starts, upper)
        return prefix[after] - prefix[first]
    try:
        return exact_tiles.budget_tiles(bounded_reader, common_halo, actual_bounds,
                                        upper_sites, require)
    except ValueError as error:
        if "cannot fit one core" not in str(error) or max_window_sites is None:
            raise

    # The frozen candidate grid contains chunk edges.  After a preceding tile
    # ends, the next physical base can lie before the next edge; a point core is
    # safe under the measured limit but that next edge need not be.  Preserve
    # every frozen decision until this case, then binary-search physical bp.
    first, last = bounded_reader.points[0][0], bounded_reader.points[-1][0]
    candidates = {last}
    for kind in ("common", "rare"):
        for chunk in bounded_reader.manifest["channels"][kind]:
            candidates.update((max(first, chunk["start_bp"] - 1), min(last, chunk["end_bp"])))
    candidates = sorted(value for value in candidates if first <= value <= last)
    result, start, limit = [], first, bounded_reader.r["max_window_sites"]

    def admitted(end):
        common_bounds = actual_bounds(bounded_reader, start, end, common_halo)
        rare_bounds = actual_bounds(bounded_reader, start, end, 0)
        return (upper_sites(bounded_reader, "common", *common_bounds) <= limit and
                upper_sites(bounded_reader, "rare", *rare_bounds) <= limit)

    while start <= last:
        low_index = bisect.bisect_left(candidates, start)
        require(low_index < len(candidates) and admitted(start),
                "measured point-core row limit cannot fit one physical base")
        low, high = low_index, len(candidates) - 1
        if admitted(candidates[low]):
            while low < high:
                middle = (low + high + 1) // 2
                if admitted(candidates[middle]):
                    low = middle
                else:
                    high = middle - 1
            end = candidates[low]
        else:
            # No chunk-edge candidate fits. Find the largest exact bp before it.
            left, right = start, candidates[low] - 1
            while left < right:
                middle = (left + right + 1) // 2
                if admitted(middle):
                    left = middle
                else:
                    right = middle - 1
            end = left
        result.append([start, end]); start = end + 1
    require(result[-1][1] == last and
            all(a[1] + 1 == b[0] for a, b in zip(result, result[1:])),
            "corrected tiles do not cover mapped territory exactly")
    return result


def build(spec_path, spec_sha256, output):
    spec = legacy.read_json(spec_path, spec_sha256)
    require(spec.get("schema") == "r02_human_autosomal_person_batch_build_v1" and
            spec.get("status") == "PREPARATION_NOT_ADMISSION", "person-batch build spec differs")
    roster, stages = legacy.people(spec["persons"]["path"], spec["persons"]["sha256"])
    config = legacy.read_json(spec["resolved_config"]["path"], spec["resolved_config"]["sha256"])
    try:
        context = request_context(config)
    except ValueError as error:
        raise ValueError("resolved structure configuration differs: " + str(error)) from error
    partition = spec.get("person_partition", {})
    require(set(partition) == {"schema", "max_people", "method"} and
            partition["schema"] == PERSON_PARTITION_SCHEMA and
            partition["method"] == "CONTIGUOUS_FROZEN_ORDER", "explicit person partition policy required")
    backend = spec.get("reader_backend", {})
    require(set(backend) == {"schema", "name"} and backend["schema"] == READER_BACKEND_SCHEMA and
            backend["name"] == MMAP_READER_BACKEND, "explicit ZIP_STORED_MMAP_V1 backend required")
    stores = spec["stores"]
    require(isinstance(stores, list) and len(stores) == 22, "exactly 22 stores required")
    readers, lineage, max_verified_chunks = [], None, 0
    for index, descriptor in enumerate(stores, 1):
        require(set(descriptor) == {"chromosome", "path", "manifest_sha256"} and
                descriptor["chromosome"] == str(index) and Path(descriptor["path"]).is_absolute() and
                legacy.HEX.fullmatch(descriptor["manifest_sha256"]), "ordered store descriptor differs")
        reader = events.EventChromosome(descriptor["path"], descriptor["manifest_sha256"])
        readers.append(reader)
        current = (reader.manifest["event_contract_sha256"], reader.manifest["source_contract_sha256"])
        require(lineage is None or lineage == current, "store lineage differs")
        lineage = current
        max_verified_chunks += sum(len(reader.manifest["channels"][kind]) for kind in ("common", "rare"))
        require(partition["max_people"] <= reader.r["max_batch_people"],
                "person partition exceeds store max_batch_people")
        lookup = {person["sample_id"]: (person["role"], person["component"]) for person in reader.persons}
        require(all(lookup.get(row["sample_id"]) == (row["role"], row["component"]) for row in roster),
                "roster differs from store")
    task = spec["task"]
    require(set(task) == {"block_id", "exclusion_scope", "targets"} and
            task["exclusion_scope"] in ("BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME"),
            "task fields/exclusion differ")
    require(all(sum(block["block_id"] == task["block_id"] for block in reader.blocks) == 1 and
                task["exclusion_scope"] in {reader.source["exclusion_scope"],
                                             *reader.source.get("additional_exclusion_scopes", [])}
                for reader in readers), "block/exclusion not uniform across stores")
    tiles = {reader.chrom: budget_tiles(reader, context) for reader in readers}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    requests, target_checks, batch_counts = {}, {}, {}
    for stage in ("FIT", "SELECT"):
        descriptor = task["targets"][stage]
        require(set(descriptor) >= {"path", "sha256", "receipt_path", "receipt_sha256"},
                "target descriptor incomplete")
        target_checks[stage] = legacy.target(descriptor["path"], descriptor["sha256"], stage, stages[stage])
        receipt = legacy.read_json(descriptor["receipt_path"], descriptor["receipt_sha256"])
        require(receipt.get("arrays_sha256") == descriptor["sha256"] and
                receipt.get("target_contract_sha256") == spec["target_contract_sha256"] and
                receipt.get("block_id") == task["block_id"] and
                receipt.get("exclusion_scope") == task["exclusion_scope"],
                "target receipt/task lineage differs")
        ids = [row["sample_id"] for row in stages[stage]]
        roles = [stage] * len(ids)
        components = [row["component"] for row in stages[stage]]
        batches = partition_people(ids, roles, components, partition["max_people"])
        batch_counts[stage] = len(batches)
        request = {"schema": PERSON_BATCH_SCHEMA, "stage": stage, "person_ids": ids,
                   "roles": roles, "component_ids": components, "stores": stores,
                   "block_id": task["block_id"], "exclusion_scope": task["exclusion_scope"],
                   **context, "tiles": tiles,
                   "person_partition": {"schema": PERSON_PARTITION_SCHEMA,
                       "max_people": partition["max_people"], "method": "CONTIGUOUS_FROZEN_ORDER",
                       "complete_person_axis_sha256": person_axis_sha256(ids)},
                   "reader_backend": {"schema": READER_BACKEND_SCHEMA, "name": MMAP_READER_BACKEND,
                       "max_verified_chunks": max_verified_chunks}}
        path = out / (stage + ".autosomal-person-batches.json")
        write_json(path, request)
        requests[stage] = {"path": str(path.resolve()), "sha256": sha(path)}
    manifest = {"schema": "r02_human_autosomal_person_batch_manifest_v2",
        "status": "PREPARED_NOT_ADMITTED_FOR_TRAINING", "training_allowed": False,
        "dataset_mode": "AUTOSOMAL_PERSON_BATCHES_V1", "spec_sha256": spec_sha256,
        "source_builder_path": str(Path(__file__).resolve()), "source_builder_sha256": sha(__file__),
        "predecessor_builder_path": str(REAL_BUILDER), "predecessor_builder_sha256": sha(REAL_BUILDER),
        "transitive_code_inventory": transitive_code_inventory(),
        "resolved_config": spec["resolved_config"], "request_context": context,
        "target_contract_sha256": spec["target_contract_sha256"],
        "lineage": {"event_contract_sha256": lineage[0], "source_contract_sha256": lineage[1]},
        "counts": {"FIT": 1460, "SELECT": 476}, "person_batch_counts": batch_counts,
        "person_partition": partition, "reader_backend": backend,
        "max_verified_chunks": max_verified_chunks, "targets": task["targets"],
        "target_checks": target_checks, "requests": requests,
        "tile_counts": {chromosome: len(rows) for chromosome, rows in tiles.items()},
        "blocking_next_step": "APPROVED_DEVELOPMENT review must bind requests, targets and exact hashes"}
    write_json(out / "integration-manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.spec, args.spec_sha256, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
