#!/usr/bin/env python3
"""Prepare and measure the existing lazy 22-autosome structure input path.

The 196 GB chromosome stores and the 88 K03 FIT/SELECT targets remain shared.
``prepare`` computes the six exact tile inventories once and writes a compact
528-task index. ``materialize`` expands one selected task into the two request
files already consumed by ``r02_context_train.py``. ``measure`` traverses one
complete person batch without reading a target or running a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import time
import zipfile

import numpy as np

import r02_genomic_event_inputs as events
import r02_structure_person_batch_builder as existing
from r02_context_autosomal_dataset import (
    AUTOSOMES, MMAP_READER_BACKEND, PERSON_BATCH_SCHEMA,
    PERSON_PARTITION_SCHEMA, READER_BACKEND_LIMIT_SCHEMA,
    AutosomalDataset, person_axis_sha256,
)

HEX64 = re.compile(r"[0-9a-f]{64}")
INDEX_SCHEMA = "r02_structure_lazy_person_batch_index_v1"
AUTH_SCHEMA = "r02_structure_input_preparation_authorization_v1"
READER_LIMIT_SCHEMA = "r02_structure_reader_row_limits_v1"
RESTORE_SCHEMA = "r02_structure_restored_store_index_v1"
TARGET_STAGE_SCHEMA = "r02_structure_target_stage_receipt_v1"
V6_TILE_BUILDER_SHA256 = "a0f580a375982db7a685a2cbe009d63b399f88a7396476337cda916704694be6"
V6_PACKAGE_ARCHIVE_SHA256 = "7e639172b4bb0e8c7e328b4a6977c6832c2d0cdef9f661fc34460af593d96c99"
V6_RECOVERED_CONTEXT_SHA256 = {
    "2d74571328d62e617ae119b4d5fe26912e240538c4fd4b26c2c98b0ca1bdfcde":
        "3116333f12d10f48a8f0daca5d23f6683d1e2fa943c11a6f9cee3a4c2f662de0",
    "4fd642fb8d7136246b5ea2c4157cacb2bde5928eee42101c8f9f8ffa616704a1":
        "ad7491e2f059a41ee8a9b4182a740df6c60de1253ed1515622e047da90438f69",
    "a2030991d20f102d321266ead94d31d74cdba2899983113c0f82fbe274bc695f":
        "55e2f6f9b5a3c3a8d99d6209d5b3eb9200b4e8154459a5184af6d44c81a55449",
    "c531477a7aa2e813c990b49b2c4af1441fa3933986a55f7f024ad03abbd85443":
        "0c22d6d73d78b32a742bf7d1e47a2c77a402b375a366ec0ddd75f4c9a0cf73d9",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(path, expected):
    path = Path(path).resolve()
    require(path.is_file() and HEX64.fullmatch(expected or "") and sha256(path) == expected,
            "pinned JSON differs: " + str(path))
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key in " + str(path))
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ValueError("nonfinite JSON value: " + token)))


def descriptor(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")


def read_people(path, expected):
    require(sha256(path) == expected, "person roster differs")
    stages = {"FIT": [], "SELECT": []}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["sample_id", "component", "role"],
                "person roster columns differ")
        seen = set()
        for row in reader:
            require(row["sample_id"] not in seen and row["component"],
                    "duplicate person or missing component")
            seen.add(row["sample_id"])
            if row["role"] in stages:
                stages[row["role"]].append(row)
            else:
                require(row["role"] == "EVALUATE", "unknown person role")
    require({key: len(value) for key, value in stages.items()} == {"FIT": 1460, "SELECT": 476}
            and len(seen) == 2619, "person role counts differ")
    return stages


def normalized_request_context(context):
    """Validate the already-normalized context stored in the 528-task index.

    ``r02_structure_person_batch_plan.request_context`` accepts a resolved
    configuration with the key ``depth``.  STRUCTURE_INPUT_REQUESTS.json stores
    its output with the key ``attention_depth``; feeding that output back into
    the configuration function loses the depth and rejects every attention
    context.  This validator keeps the normalized values unchanged.
    """
    require(set(context) == {"family", "common_radius_cm", "rare_radius_cm",
                             "attention_depth"}, "normalized context fields differ")
    family = context["family"]
    common, rare, depth = (context["common_radius_cm"], context["rare_radius_cm"],
                           context["attention_depth"])
    require(type(common) in (int, float) and np.isfinite(common) and common > 0,
            "normalized common radius differs")
    if family == "structure_deep_sets":
        require(rare is None and type(depth) is int and depth == 0,
                "normalized Deep Sets context differs")
    elif family == "structure_local_attention":
        require(type(rare) in (int, float) and np.isfinite(rare) and rare > 0 and
                type(depth) is int and depth > 0,
                "normalized local-attention context differs")
    else:
        raise ValueError("normalized structure family differs")
    return {"common_radius_cm": common, "rare_radius_cm": rare,
            "attention_depth": depth}


def current_stores(restore, store_root):
    require(restore.get("schema") == RESTORE_SCHEMA and restore.get("chromosomes") == 22 and
            restore.get("status") ==
            "COMPLETE_22_RESTORED_AUTHENTICATED_STORE_PATHS_NO_ASSEMBLY_NO_TRAINING" and
            restore.get("channel_chunk_bytes_hashed") == 195593032185 and
            restore.get("training_started") is False and restore.get("evaluate_opened") is False,
            "restored store receipt differs")
    by_chrom = {str(row["chromosome"]): row for row in restore.get("stores", [])}
    require(tuple(sorted(by_chrom, key=int)) == AUTOSOMES, "restored autosome universe differs")
    root = Path(store_root).resolve()
    result = []
    for chrom in AUTOSOMES:
        store = root / ("chr" + chrom + ".neural-input-validation") / "store"
        manifest = store / "manifest.json"
        row = by_chrom[chrom]
        require(manifest.is_file() and sha256(manifest) == row["manifest_sha256"],
                "restored manifest differs for chromosome " + chrom)
        result.append({"chromosome": chrom, "path": str(store),
                       "manifest_sha256": row["manifest_sha256"]})
    return result


def target_descriptors(target_root, receipt_root, view_id, block_id, exclusion_scope):
    root = Path(target_root).resolve() / view_id
    stage_receipt_path = Path(receipt_root).resolve() / (view_id + ".stage.json")
    stage_receipt = json.loads(stage_receipt_path.read_text(encoding="utf-8"))
    require(stage_receipt.get("schema") == TARGET_STAGE_SCHEMA and
            stage_receipt.get("status") == "COMPLETE_AUTHENTICATED_FIT_SELECT_TARGET_STAGE" and
            stage_receipt.get("view_id") == view_id and stage_receipt.get("roles") == ["FIT", "SELECT"] and
            stage_receipt.get("evaluate_opened") is False and
            stage_receipt.get("training_allowed") is False,
            "target stage receipt differs for " + view_id)
    files = stage_receipt.get("files", {})
    require(set(files) == {"manifest.json", "FIT.target.private.npz", "FIT.target.receipt.json",
                           "SELECT.target.private.npz", "SELECT.target.receipt.json"},
            "target file inventory differs for " + view_id)
    manifest_path = root / "manifest.json"
    require(manifest_path.stat().st_size == files["manifest.json"]["bytes"] and
            sha256(manifest_path) == files["manifest.json"]["sha256"],
            "target manifest differs for " + view_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "r02_k03_neural_target_adapter_result_v1" and
            manifest.get("status") == "FIT_SELECT_TARGETS_COMPLETE_NOT_TRAINING_ADMISSION" and
            manifest.get("evaluate_targets_computed") is False and
            manifest.get("training_allowed") is False,
            "target manifest state differs for " + view_id)
    result = {}
    for stage in ("FIT", "SELECT"):
        array = root / (stage + ".target.private.npz")
        receipt = root / (stage + ".target.receipt.json")
        array_desc, receipt_desc = files[array.name], files[receipt.name]
        require(array.is_file() and array.stat().st_size == array_desc["bytes"] and
                receipt.is_file() and receipt.stat().st_size == receipt_desc["bytes"] and
                sha256(receipt) == receipt_desc["sha256"],
                "target file/receipt differs for " + view_id + "/" + stage)
        target_receipt = json.loads(receipt.read_text(encoding="utf-8"))
        require(target_receipt.get("arrays_sha256") == array_desc["sha256"] and
                target_receipt.get("role") == stage and
                target_receipt.get("block_id") == block_id and
                target_receipt.get("exclusion_scope") == exclusion_scope and
                target_receipt.get("evaluate_targets_computed") is False and
                target_receipt.get("training_allowed") is False,
                "target lineage differs for " + view_id + "/" + stage)
        item = manifest["targets"][stage]
        require(item["sha256"] == array_desc["sha256"] and
                item["receipt_sha256"] == receipt_desc["sha256"],
                "target stage/manifest hashes differ for " + view_id + "/" + stage)
        result[stage] = {"path": str(array), "sha256": array_desc["sha256"],
                         "bytes": array_desc["bytes"], "receipt_path": str(receipt),
                         "receipt_sha256": receipt_desc["sha256"],
                         "target_contract_sha256": target_receipt["target_contract_sha256"]}
    require(len({result[s]["target_contract_sha256"] for s in result}) == 1,
            "FIT/SELECT target contracts differ for " + view_id)
    return result


def prepare(requests_path, requests_sha256, restore_path, restore_sha256,
            authorization_path, authorization_sha256, store_root, target_root,
            receipt_root, people_path, people_sha256, reader_limits_path,
            reader_limits_sha256, output, reuse_contexts=None,
            preflight_only=False):
    requests = load_pinned(requests_path, requests_sha256)
    restore = load_pinned(restore_path, restore_sha256)
    authorization = load_pinned(authorization_path, authorization_sha256)
    reader_limits = load_pinned(reader_limits_path, reader_limits_sha256)
    require(authorization.get("schema") == AUTH_SCHEMA and
            authorization.get("status") == "AUTHORIZED_INPUT_PREPARATION_ONLY" and
            authorization.get("training_allowed") is False and
            authorization.get("evaluate_allowed") is False and
            authorization.get("maximum_cpu") == 1 and authorization.get("maximum_memory_gib") == 2,
            "input preparation authorization differs")
    require(requests.get("schema") == "r02_structure_search_input_requests_v2" and
            requests.get("request_count") == 528 and len(requests.get("requests", [])) == 528 and
            requests.get("view_count") == 88 and requests.get("context_count") == 6 and
            requests.get("evaluate_allowed") is False,
            "528-request catalogue differs")
    require(reader_limits.get("schema") == READER_LIMIT_SCHEMA and
            reader_limits.get("status") ==
            "COMPLETE_METADATA_BOUNDS_ALL_6_CONTEXTS_22_AUTOSOMES_BOTH_CHANNELS" and
            reader_limits.get("source_requests", {}).get("sha256") == requests_sha256 and
            reader_limits.get("channels") == ["common", "rare"] and
            reader_limits.get("autosomes") == 22 and
            reader_limits.get("genotype_chunks_opened") == 0 and
            reader_limits.get("individual_genotypes_read") is False and
            reader_limits.get("evaluate_opened") is False and
            reader_limits.get("training_started") is False,
            "external reader row-limit receipt differs")
    stages = read_people(people_path, people_sha256)
    stores = current_stores(restore, store_root)
    # Cheap complete preflight precedes the six costly tile inventories.  The
    # pinned catalogue is descriptive, so validate every identity and all 88
    # FIT/SELECT target receipts without opening a target array.
    context_definitions = {}
    for row in requests["contexts"]:
        key, context = row["context_sha256"], row["request_context"]
        require(HEX64.fullmatch(key or "") and key not in context_definitions and
                normalized_request_context(context) == {
            "common_radius_cm": context["common_radius_cm"],
            "rare_radius_cm": context["rare_radius_cm"],
            "attention_depth": context["attention_depth"]}, "context definition differs")
        context_definitions[key] = context
    require(len(context_definitions) == 6, "six normalized contexts required")
    limit_rows = {row["context_sha256"]: row for row in reader_limits.get("contexts", [])}
    require(set(limit_rows) == set(context_definitions) and all(
        row.get("request_context") == context_definitions[key] and
        type(row.get("required_max_window_sites")) is int and
        type(row.get("effective_max_window_sites")) is int and
        row["effective_max_window_sites"] >= max(100000, row["required_max_window_sites"]) and
        set(row.get("max_point_whole_chunk_upper", {})) == {"common", "rare"} and
        all(row["max_point_whole_chunk_upper"][kind]["rows"] <= row["effective_max_window_sites"]
            for kind in ("common", "rare")) for key, row in limit_rows.items()),
        "six per-context COMMON/RARE reader limits differ")
    preflight_tasks, targets = [], {}
    for request in requests["requests"]:
        identity = request["identity"]
        view_id, context_key = identity["view_id"], identity["context_sha256"]
        require(HEX64.fullmatch(request.get("request_id", "")) and
                set(identity) == {"view_id", "block_id", "exclusion_scope", "context_sha256"} and
                context_key in context_definitions and
                request["request_context"] == context_definitions[context_key] and
                request.get("dataset_mode") == "AUTOSOMAL_PERSON_BATCHES_V1" and
                request.get("status") == "DESCRIPTIVE_REQUEST_NOT_MATERIALIZED_NOT_ADMITTED" and
                request.get("training_allowed") is False and request.get("evaluate_allowed") is False and
                request.get("roles") == {"FIT": 1460, "SELECT": 476, "EVALUATE": "CLOSED_NOT_READ"} and
                request.get("person_partition", {}).get("max_people") == 8 and
                identity["exclusion_scope"] in ("BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME"),
                "request/context mismatch")
        if view_id not in targets:
            targets[view_id] = target_descriptors(target_root, receipt_root, view_id,
                                                  identity["block_id"], identity["exclusion_scope"])
        preflight_tasks.append({"request_id": request["request_id"], "view_id": view_id,
                      "block_id": identity["block_id"],
                      "exclusion_scope": identity["exclusion_scope"],
                      "context_sha256": context_key,
                      "family": request["request_context"]["family"],
                      "targets": targets[view_id]})
    require(len(preflight_tasks) == len({row["request_id"] for row in preflight_tasks}) == 528 and
            Counter(row["family"] for row in preflight_tasks) ==
            {"structure_deep_sets": 264, "structure_local_attention": 264} and len(targets) == 88,
            "preflight task universe differs")
    by_view = {}
    for row in preflight_tasks:
        by_view.setdefault(row["view_id"], set()).add(row["context_sha256"])
    require(len(by_view) == 88 and all(keys == set(context_definitions) for keys in by_view.values()),
            "each view must have all six contexts exactly once")
    if preflight_only:
        output = Path(output).resolve()
        require(not output.exists(), "fresh preflight output required")
        output.mkdir(parents=True)
        result = {"schema": "r02_structure_lazy_preflight_v1",
            "status": "PASS_22_STORES_88_TARGETS_528_IDENTITIES_6_CONTEXTS_NO_ARRAYS_READ",
            "counts": {"stores": 22, "targets": 88, "tasks": 528, "contexts": 6,
                       "fit_people": len(stages["FIT"]), "select_people": len(stages["SELECT"])},
            "source_requests": descriptor(requests_path),
            "authorization": descriptor(authorization_path),
            "restored_stores": descriptor(restore_path),
            "training_started": False, "evaluate_opened": False,
            "target_arrays_read": False, "tile_inventories_computed": 0}
        write_json(output / "PREFLIGHT.json", result)
        return result
    readers = {row["chromosome"]: events.EventChromosome(row["path"], row["manifest_sha256"])
               for row in stores}
    max_verified_chunks = sum(len(reader.manifest["channels"][kind])
                              for reader in readers.values() for kind in ("common", "rare"))
    output = Path(output).resolve()
    require(not output.exists(), "fresh preparation output required")
    (output / "contexts").mkdir(parents=True)
    recovered = {}
    if reuse_contexts:
        reuse_root = Path(reuse_contexts).resolve()
        recovery = json.loads((reuse_root / "RECOVERED_CONTEXTS.json").read_text())
        require(recovery.get("schema") == "r02_structure_recovered_contexts_v1" and
                recovery.get("source_requests_sha256") == requests_sha256 and
                recovery.get("tile_builder_sha256") in {
                    sha256(Path(existing.__file__).resolve()), V6_TILE_BUILDER_SHA256} and
                len(recovery.get("contexts", [])) <= 6,
                "recovered context receipt differs")
        recovered = {row["context_sha256"]: row for row in recovery["contexts"]}
        if recovery["tile_builder_sha256"] == V6_TILE_BUILDER_SHA256:
            require(recovery.get("source_package_sha256") == V6_PACKAGE_ARCHIVE_SHA256 and
                    recovery.get("status") ==
                    "FOUR_EXACT_V6_CONTEXTS_RECOVERED_FOR_INPUT_PREPARATION_ONLY" and
                    recovery.get("training_started") is False and
                    recovery.get("evaluate_opened") is False and
                    {key: row.get("sha256") for key, row in recovered.items()} ==
                    V6_RECOVERED_CONTEXT_SHA256,
                    "frozen v6 recovered context inventory differs")
    contexts = {}
    for key, context in context_definitions.items():
        tile_path = output / "contexts" / (key + ".tiles.json")
        if key in recovered:
            row = recovered[key]
            source = Path(reuse_contexts).resolve() / row["file"]
            shared = load_pinned(source, row["sha256"])
            require(shared.get("schema") == "r02_structure_shared_tiles_v1" and
                    shared.get("context_sha256") == key and
                    shared.get("request_context") == context and
                    tuple(sorted(shared.get("tiles", {}), key=int)) == AUTOSOMES,
                    "recovered tile context differs")
            shutil.copyfile(source, tile_path)
        else:
            row_limit = limit_rows[key]["effective_max_window_sites"]
            tiles = {chrom: existing.budget_tiles(readers[chrom], context, row_limit)
                     for chrom in AUTOSOMES}
            write_json(tile_path, {"schema": "r02_structure_shared_tiles_v1",
                                   "context_sha256": key, "request_context": context,
                                   "tiles": tiles})
        shared = json.loads(tile_path.read_text())
        tile_descriptor = descriptor(tile_path)
        tile_descriptor["path"] = str(tile_path.relative_to(output))
        contexts[key] = {"request_context": context, "tiles": tile_descriptor,
                         "tile_count": sum(map(len, shared["tiles"].values())),
                         "recovered": key in recovered,
                         "max_window_sites": limit_rows[key]["effective_max_window_sites"]}
    tasks = preflight_tasks
    index = {"schema": INDEX_SCHEMA,
             "status": "PREPARED_SHARED_LAZY_INPUTS_FOR_ALL_528_TASKS_NO_TRAINING",
             "authorization": descriptor(authorization_path),
             "source_requests": descriptor(requests_path),
             "restored_stores": descriptor(restore_path),
             "reader_limits": descriptor(reader_limits_path),
             "recovery": ({"receipt": descriptor(Path(reuse_contexts) / "RECOVERED_CONTEXTS.json"),
                 "policy": "REUSE_FOUR_EXACT_V6_HASH_PINNED_CONTEXTS_COMPUTE_ONLY_PENDING_TWO"}
                 if reuse_contexts else None),
             "store_authentication_reused": {"channel_chunk_bytes_hashed": 195593032185,
                 "policy": "DO_NOT_REHASH_RESTORED_CHUNKS_DURING_BOUNDED_MEASUREMENT"},
             "persons": descriptor(people_path), "person_counts": {"FIT": 1460, "SELECT": 476},
             "stores": stores, "max_verified_chunks": max_verified_chunks,
             "contexts": contexts, "tasks": tasks,
             "counts": {"tasks": 528, "views": 88, "contexts": 6, "stores": 22},
             "arms": ["NONE", "COUNTS", "RARE"],
             "training_allowed": False, "evaluate_allowed": False}
    write_json(output / "index.json", index)
    write_json(output / "PREPARED.json", {"schema": "r02_structure_lazy_preparation_receipt_v1",
        "status": index["status"], "index": descriptor(output / "index.json"),
        "physical_genotype_copies_created": 0, "physical_target_copies_created": 0,
        "shared_tile_inventories": 6, "task_references": 528,
        "training_started": False, "evaluate_opened": False})
    return index


def materialize_loaded(index, index_path, request_id, output, stages=None, shared_cache=None):
    """Write one trainer entry from an index already authenticated by its caller."""
    matches = [row for row in index["tasks"] if row["request_id"] == request_id]
    require(len(matches) == 1, "unknown or duplicate request ID")
    task = matches[0]
    context_row = index["contexts"][task["context_sha256"]]
    shared_cache = {} if shared_cache is None else shared_cache
    if task["context_sha256"] not in shared_cache:
        shared_path = Path(context_row["tiles"]["path"])
        if not shared_path.is_absolute():
            shared_path = Path(index_path).resolve().parent / shared_path
        shared_cache[task["context_sha256"]] = load_pinned(
            shared_path, context_row["tiles"]["sha256"])
    shared = shared_cache[task["context_sha256"]]
    context = shared["request_context"]
    if stages is None:
        stages = read_people(index["persons"]["path"], index["persons"]["sha256"])
    output = Path(output).resolve()
    require(not output.exists(), "fresh materialization output required")
    output.mkdir(parents=True)
    entries = {}
    for stage in ("FIT", "SELECT"):
        rows = stages[stage]
        ids = [row["sample_id"] for row in rows]
        request = {"schema": PERSON_BATCH_SCHEMA, "stage": stage, "person_ids": ids,
            "roles": [stage] * len(ids), "component_ids": [row["component"] for row in rows],
            "stores": index["stores"], "block_id": task["block_id"],
            "exclusion_scope": task["exclusion_scope"],
            "common_radius_cm": context["common_radius_cm"],
            "rare_radius_cm": context["rare_radius_cm"],
            "attention_depth": context["attention_depth"], "tiles": shared["tiles"],
            "person_partition": {"schema": PERSON_PARTITION_SCHEMA, "max_people": 8,
                "method": "CONTIGUOUS_FROZEN_ORDER",
                "complete_person_axis_sha256": person_axis_sha256(ids)},
            "reader_backend": {"schema": READER_BACKEND_LIMIT_SCHEMA, "name": MMAP_READER_BACKEND,
                "max_verified_chunks": index["max_verified_chunks"],
                "max_window_sites": context_row["max_window_sites"],
                "reader_limits_sha256": index["reader_limits"]["sha256"],
                "context_sha256": task["context_sha256"]}}
        request_path = output / (stage + ".autosomal-person-batches.json")
        write_json(request_path, request)
        target = task["targets"][stage]
        entries[stage] = {"path": target["path"], "sha256": target["sha256"],
            "receipt_path": target["receipt_path"], "receipt_sha256": target["receipt_sha256"],
            # FINALIZE_PREPARATION rewrites this relocatable path to the
            # published absolute path required by r02_context_train.py.
            "autosomal_request_path": request_path.name,
            "autosomal_request_sha256": sha256(request_path)}
    result = {"schema": "r02_structure_materialized_lazy_trainer_entries_v1",
        "status": "FIT_SELECT_ENTRIES_READY_FOR_ADMISSION_NO_TRAINING_STARTED",
        "request_id": request_id, "view_id": task["view_id"],
        "family": task["family"], "request_context": context,
        "target_contract_sha256": task["targets"]["FIT"]["target_contract_sha256"],
        "dataset_mode": "AUTOSOMAL_PERSON_BATCHES_V1", "entries": entries,
        "arms": ["NONE", "COUNTS", "RARE"], "training_allowed": False,
        "evaluate_allowed": False}
    write_json(output / "trainer-entries.json", result)
    return result


def materialize(index_path, index_sha256, request_id, output):
    index = load_pinned(index_path, index_sha256)
    require(index.get("schema") == INDEX_SCHEMA and index.get("training_allowed") is False and
            index.get("evaluate_allowed") is False, "lazy input index differs")
    return materialize_loaded(index, index_path, request_id, output)


def materialize_all(index_path, index_sha256, output, published_root=None):
    """Write all 528 small FIT/SELECT requests in one bounded process.

    The index, person roster, and six shared tile inventories are each checked
    once.  Genotype stores and target arrays are referenced and never copied.
    """
    index = load_pinned(index_path, index_sha256)
    require(index.get("schema") == INDEX_SCHEMA and index.get("training_allowed") is False and
            index.get("evaluate_allowed") is False and len(index.get("tasks", [])) == 528,
            "lazy input index differs")
    stages = read_people(index["persons"]["path"], index["persons"]["sha256"])
    output = Path(output).resolve()
    published_root = Path(published_root).resolve() if published_root else output
    require(not output.exists(), "fresh all-request materialization output required")
    output.mkdir(parents=True)
    shared_cache, rows = {}, []
    for task in index["tasks"]:
        request_id = task["request_id"]
        directory = output / request_id
        result = materialize_loaded(index, index_path, request_id, directory,
                                    stages=stages, shared_cache=shared_cache)
        # r02_context_train.py reads this file directly; rewrite its two request
        # paths to absolute paths in the published directory.
        for stage, entry in result["entries"].items():
            entry["autosomal_request_path"] = str(
                (published_root / request_id / (stage + ".autosomal-person-batches.json")).resolve())
        write_json(directory / "trainer-entries.json", result)
        rows.append({"request_id": request_id, "view_id": result["view_id"],
                     "family": result["family"],
                     "trainer_entries_path": str((published_root / request_id / "trainer-entries.json").resolve()),
                     "trainer_entries_sha256": sha256(directory / "trainer-entries.json")})
    require(len(rows) == len({row["request_id"] for row in rows}) == 528 and
            len(shared_cache) == 6, "all-request materialization universe differs")
    manifest = {"schema": "r02_structure_all_lazy_trainer_entries_v1",
        "status": "COMPLETE_528_FIT_SELECT_TRAINER_ENTRIES_NO_ARRAY_COPIES_NO_TRAINING",
        "index": descriptor(index_path), "request_count": 528,
        "stage_request_count": 1056, "shared_tile_inventories_read_once": 6,
        "physical_genotype_copies_created": 0, "physical_target_copies_created": 0,
        "entries": rows, "training_started": False, "evaluate_opened": False}
    write_json(output / "ALL_TRAINER_ENTRIES.json", manifest)
    return manifest


class ReceiptBackedChunkCache:
    """Use the completed restore receipt instead of hashing 195.6 GB again."""
    def __init__(self, original_cache, authenticated_bytes):
        self.original_cache = original_cache
        self.max_entries = original_cache.max_entries
        self.entries = {}
        self.bytes_hashed = 0
        self.authenticated_bytes = authenticated_bytes
        self.open_count = 0

    @contextmanager
    def open(self, path, chunk, n_people, max_array_mb):
        import r02_event_mmap_reader as mapped
        path = Path(path).resolve()
        before = mapped.signature(path)
        require(before[2] == chunk["bytes"], "restored chunk size changed")
        require(path.is_file(), "restored chunk missing")
        with zipfile.ZipFile(path) as archive:
            require(sum(row.file_size for row in archive.infolist()) * 4 <= max_array_mb * 2**20,
                    "chunk tensor bound differs")
        with mapped.mapped_arrays(path) as data:
            n = chunk["n_sites"]
            require(all(value.shape == ((n_people, n) if name in {"dosage", "observed"} else (n,))
                        for name, value in data.items()), "restored chunk array shape differs")
            yield data
            require(mapped.signature(path) == before, "restored chunk changed during read")
        self.open_count += 1


def array_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(array_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(array_bytes(item) for item in value)
    return 0


def measure(index_path, index_sha256, materialized_path, materialized_sha256,
            stage, person_batch_index, chromosome, core_bp, output):
    started = time.monotonic()
    index = load_pinned(index_path, index_sha256)
    materialized = load_pinned(materialized_path, materialized_sha256)
    require(index.get("schema") == INDEX_SCHEMA and
            materialized.get("schema") == "r02_structure_materialized_lazy_trainer_entries_v1" and
            stage in ("FIT", "SELECT") and materialized.get("evaluate_allowed") is False,
            "measurement inputs differ")
    entry = materialized["entries"][stage]
    request_path = Path(entry["autosomal_request_path"])
    if not request_path.is_absolute():
        request_path = Path(materialized_path).resolve().parent / request_path
    dataset = AutosomalDataset(request_path, entry["autosomal_request_sha256"])
    require(0 <= person_batch_index < dataset.n_person_batches, "person batch index out of range")
    require(chromosome in AUTOSOMES and type(core_bp) is int,
            "explicit real measurement chromosome/core required")
    receipt_cache = ReceiptBackedChunkCache(dataset.chunk_cache,
        index["store_authentication_reused"]["channel_chunk_bytes_hashed"])
    # MappedEventChromosome requires the exact base type. Create a small subclass
    # dynamically and copy the receipt-backed implementation onto it.
    import r02_event_mmap_reader as mapped
    class CompatibleReceiptCache(mapped.VerifiedChunkCache):
        def __init__(self, delegate):
            self.delegate = delegate
            self.max_entries = delegate.max_entries
            self.entries = {}
            self.bytes_hashed = 0
        @contextmanager
        def open(self, *args, **kwargs):
            with self.delegate.open(*args, **kwargs) as arrays:
                yield arrays
    dataset.chunk_cache = CompatibleReceiptCache(receipt_cache)
    totals = Counter()
    peaks = Counter()
    chromosomes = set()
    tiles = 0
    matching_tiles = [tile for tile in dataset.request["tiles"][chromosome]
                      if tile[0] <= core_bp <= tile[1]]
    require(len(matching_tiles) == 1, "measurement core must identify exactly one prepared tile")
    core_start, core_end = matching_tiles[0]
    ids, roles, components = dataset.person_batches[person_batch_index]
    reader = dataset._reader(chromosome)
    view = reader.load_window(list(ids), core_start, core_end, dataset.request["block_id"],
        dataset.request["exclusion_scope"], dataset.request["common_radius_cm"],
        dataset.request["rare_radius_cm"], dataset.request["attention_depth"])
    require(tuple(map(str, view["person_id"])) == ids and
            tuple(map(str, view["role"])) == roles and
            tuple(map(str, view["component_id"])) == components,
            "measured person identity/order differs")
    for view in (view,):
        common, events_view = view["common"], view["events"]
        b, common_sites = common["valid"].shape
        event_sites = events_view["valid"].shape[1]
        require(max(common_sites, event_sites) <= dataset.max_window_sites,
                "observed reader rows exceed authenticated external limit")
        numpy_bytes = array_bytes(view)
        totals.update(tiles=1, person_tile_rows=b, common_site_slots=common_sites,
                      event_site_slots=event_sites,
                      common_valid=int(common["valid"].sum()),
                      event_valid=int(events_view["valid"].sum()),
                      common_core=int(view["common_core"].sum()),
                      event_core=int(view["events_core"].sum()), numpy_bytes=numpy_bytes)
        peaks["numpy_bytes"] = max(peaks["numpy_bytes"], numpy_bytes)
        peaks["common_rows"] = max(peaks["common_rows"], common_sites)
        peaks["event_rows"] = max(peaks["event_rows"], event_sites)
        chromosomes.add(str(view["chromosome"])); tiles += 1
    require(chromosomes == {chromosome} and tiles == 1,
            "measurement did not traverse the one declared real interval")
    result = {"schema": "r02_structure_real_interval_reader_measurement_v1",
        "status": "COMPLETE_ONE_REAL_DENSE_INTERVAL_READER_MEASUREMENT_NO_MODEL_NO_TARGET",
        "request_id": materialized["request_id"], "view_id": materialized["view_id"],
        "family": materialized["family"], "request_context": materialized["request_context"],
        "stage": stage, "person_batch_index": person_batch_index,
        "person_batch_size": len(dataset.person_batches[person_batch_index][0]),
        "authenticated_max_window_sites": dataset.max_window_sites,
        "chromosome": chromosome, "requested_core_bp": core_bp,
        "measured_tile_core_bp": [core_start, core_end],
        "autosomes": 1, "totals": dict(totals), "peak_per_tile": dict(peaks),
        "restored_chunk_authentication": {"receipt_bytes_previously_hashed": receipt_cache.authenticated_bytes,
            "chunks_opened": receipt_cache.open_count, "bytes_rehashed": 0},
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "reader_channels_measured": ["common", "rare"],
        "target_arrays_read": False, "trainer_module_imported": False,
        "model_created": False, "training_started": False, "evaluate_opened": False}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("requests", "restore", "authorization", "reader-limits"):
        prep.add_argument("--" + name, required=True)
        prep.add_argument("--" + name + "-sha256", required=True)
    prep.add_argument("--store-root", required=True)
    prep.add_argument("--target-root", required=True)
    prep.add_argument("--receipt-root", required=True)
    prep.add_argument("--people", required=True)
    prep.add_argument("--people-sha256", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--reuse-contexts")
    prep.add_argument("--preflight-only", action="store_true")
    mat = sub.add_parser("materialize")
    mat.add_argument("--index", required=True); mat.add_argument("--index-sha256", required=True)
    mat.add_argument("--request-id", required=True); mat.add_argument("--output", required=True)
    all_mat = sub.add_parser("materialize-all")
    all_mat.add_argument("--index", required=True); all_mat.add_argument("--index-sha256", required=True)
    all_mat.add_argument("--output", required=True)
    all_mat.add_argument("--published-root")
    met = sub.add_parser("measure")
    met.add_argument("--index", required=True); met.add_argument("--index-sha256", required=True)
    met.add_argument("--materialized", required=True); met.add_argument("--materialized-sha256", required=True)
    met.add_argument("--stage", choices=("FIT", "SELECT"), required=True)
    met.add_argument("--chromosome", choices=AUTOSOMES, required=True)
    met.add_argument("--core-bp", type=int, required=True)
    met.add_argument("--person-batch-index", type=int, default=0); met.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.requests, args.requests_sha256, args.restore, args.restore_sha256,
                        args.authorization, args.authorization_sha256, args.store_root,
                        args.target_root, args.receipt_root, args.people, args.people_sha256,
                        args.reader_limits, args.reader_limits_sha256, args.output,
                        args.reuse_contexts, args.preflight_only)
    elif args.command == "materialize":
        value = materialize(args.index, args.index_sha256, args.request_id, args.output)
    elif args.command == "materialize-all":
        value = materialize_all(args.index, args.index_sha256, args.output, args.published_root)
    else:
        value = measure(args.index, args.index_sha256, args.materialized,
                        args.materialized_sha256, args.stage, args.person_batch_index,
                        args.chromosome, args.core_bp, args.output)
    print(json.dumps({"status": value["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
