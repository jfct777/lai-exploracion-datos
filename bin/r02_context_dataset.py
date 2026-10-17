#!/usr/bin/env python3
"""Connect an authenticated excluded individual view to an existing K02 target.

This is an explicitly bounded pilot adapter, NOT whole-autosome aggregation.
It neither splits people, chooses windows, refits marker selection nor trains.
The immutable source contract supplies roles, exclusions and target thresholds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

import r02_genomic_event_inputs as events
import r02_structure_materialize as material

require = material.require
sha, write_json = material.sha, material.write_json
SCHEMA = "r02_context_dataset_request_v1"
PILOT = "BOUNDED_CONTEXT_PILOT"


def read_json(path, expected):
    require(sha(path) == expected, "JSON SHA256 mismatch")
    return material.load_json(path)


def export_reserved(view, pairs, target, path, target_hash):
    """Separate EVALUATE serializer: never relabel reserved people as FIT.

    The array representation is identical to the producer's development export;
    the receipt schema is deliberately different and cannot train a model.
    """
    path = Path(path)
    require(path.suffix == ".npz" and not path.exists() and not path.with_suffix(".npz.json").exists(),
            "new .npz and receipt required; no overwriting")
    pairs, target = np.asarray(pairs), np.asarray(target)
    require(set(view["role"].tolist()) == {"EVALUATE"}, "only reserved people allowed")
    require((view["common"]["valid"] & view["common_core"]).any() or
            (view["events"]["valid"] & view["events_core"]).any(), "entirely unavailable reserved core")
    require(pairs.dtype.kind in "iu" and pairs.ndim == 2 and len(pairs) > 0 and pairs.shape[1] == 2 and
            np.all(pairs >= 0) and np.all(pairs < len(view["person_id"])) and
            len(set(map(tuple, pairs.tolist()))) == len(pairs) and np.all(pairs[:, 0] < pairs[:, 1]) and
            np.all(view["component_id"][pairs[:, 0]] != view["component_id"][pairs[:, 1]]),
            "reserved pairs must be between distinct components")
    require(target.shape == (len(pairs),) and np.isfinite(target).all(), "finite reserved targets required")
    arrays = {k: np.asarray(view[k]) for k in ("person_id", "component_id", "role")}
    for source, prefix in (("common", "common"), ("events", "event")):
        for field in ("cm", "chrom", "valid"):
            arrays[prefix + "_" + field] = view[source][field]
        for field in ("pos_bp", "core", "padding"):
            arrays[prefix + "_" + field] = view[source + "_" + field]
    arrays.update(common_features=view["common"]["features"],
                  event_metadata=view["events"]["features"][:, :, 1:],
                  event_dosage=view["events"]["features"][:, :, 0]*2,
                  pairs=pairs.astype(np.int64), target=target.astype(np.float64))
    np.savez(path, **arrays)
    receipt = dict(schema="r02_context_reserved_batch_v1", training_allowed=False,
        target_contract_sha256=target_hash, arrays_sha256=sha(path), n_people=len(view["person_id"]),
        n_pairs=len(pairs), missing_policy="VALID_MASK_NOT_CARRIER",
        common_features=list(events.COMMON_FEATURES), event_metadata_features=list(events.EVENT_FEATURES[1:]))
    for key in ("block_id", "exclusion_scope", "core_bp", "common_radius_cm", "rare_radius_cm",
                "attention_depth", "input_lineage", "excluded_bp"):
        receipt[key] = view[key]
    write_json(path.with_suffix(".npz.json"), receipt)
    return receipt


def bounded_archive(path, limit):
    require(type(limit) is int and limit > 0, "positive byte budget required")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and
                sum(x.file_size for x in archive.infolist()) <= limit,
                "duplicate members or uncompressed target allocation exceeds budget")


def build(request_path, request_sha256, output, admission_path=None, admission_sha256=None, verify_only=False,
          verified_path=None):
    request = read_json(request_path, request_sha256)
    require(request["schema"] == SCHEMA, "unsupported dataset request")
    verified = material.load_json(verified_path) if verified_path else None
    if verified is not None:
        require(verified["adapter_sha256"] == sha(__file__) and verified["request_sha256"] == request_sha256,
                "staged adapter/request differs from pre-cache verification")
    require(request["dataset_declared_scope"] == PILOT,
            "AUTOSOMAL_22 aggregation is not implemented; require explicit BOUNDED_CONTEXT_PILOT")
    require(request["purpose"] == "PREPARATION_OR_EXPLICIT_PILOT_NOT_PRIMARY_DELIVERY",
            "bounded views cannot substitute the principal autosomal objective")
    stage = request["stage"]
    require(stage in ("FIT", "SELECT", "EVALUATE"), "unknown frozen role")
    if stage == "EVALUATE":
        require(admission_path and admission_sha256, "reserved EVALUATE requires separate admission")
        admission = read_json(admission_path, admission_sha256)
        require(admission["status"] == "APPROVED_RESERVED_PILOT_PREPARATION" and
                admission["request_sha256"] == request_sha256, "reserved admission differs")
        selection = read_json(admission["frozen_selection"]["path"], admission["frozen_selection"]["sha256"])
        require(selection["final_evaluation"] == "NOT_ACCESSED" and selection["objective"] == "structure",
                "selection must be frozen before reserved preparation")
    descriptor = request["chromosome_store"]
    reader = events.EventChromosome(descriptor["path"], descriptor["manifest_sha256"])
    source_hash = reader.cfg["materialize_contract_sha256"]
    require(source_hash == request["source_contract_sha256"], "source contract differs")
    ids = request["sample_ids"]
    require(isinstance(ids, list) and ids and len(ids) == len(set(ids)) and
            all(p in reader.people and reader.persons[reader.people[p]]["role"] == stage for p in ids),
            "requested people do not have the frozen role")
    require(len(ids) <= reader.r["max_batch_people"], "batch people exceed producer budget before target construction")
    # Same conservative window-allocation terms as the underlying producer,
    # checked against this task's declared bound BEFORE it creates any tensor.
    r = reader.r
    window_bytes = len(ids)*r["max_window_sites"]*400+r["max_map_points"]*256+len(reader.persons)*r["chunk_sites"]*48
    require(type(request["max_batch_bytes"]) is int and window_bytes <= request["max_batch_bytes"],
            "conservative window allocation exceeds task budget before tensor creation")
    blocks = [b for b in reader.blocks if b["block_id"] == request["block_id"]]
    require(len(blocks) == 1, "target block is not frozen")
    block = blocks[0]
    # make_tasks only computes the already-defined target identity; no extraction.
    _, _, _, _, regions = material.load_context(reader.root / "source")
    tasks, membership = material.make_tasks(reader.source, regions)
    expected = tasks[membership[block["block_id"]]["target"]]
    target = request["target_partial"]
    target_root = Path(target["path"])
    require(sha(target_root / "manifest.json") == target["manifest_sha256"], "target manifest differs")
    bound = request["max_target_bytes"]
    require(type(bound) is int and 0 < bound <= reader.source["resources"]["max_array_mb"]*2**20,
            "target byte budget exceeds frozen source allocation")
    bounded_archive(target_root / "data.npz", bound)
    people = [p["sample_id"] for p in reader.persons]
    arrays, record = material.load_partial(target_root, expected, source_hash, people)
    require(record["source_intervals"] == [[expected["chromosome"], *v] for v in expected["intervals"]],
            "target intervals differ")
    markers = material.evaluator.table(target_root / "markers.tsv",
                                      ["variant_id", "chromosome", "position_bp", "counted_allele"])
    require(arrays["variant_ids"].tolist() == [m["variant_id"] for m in markers] and
            len({m["variant_id"] for m in markers}) == len(markers) and
            all(m["chromosome"] == str(block["chromosome"]) and m["counted_allele"] == "ALT" and
                int(block["start_bp"]) <= int(m["position_bp"]) <= int(block["end_bp"]) for m in markers),
            "target markers do not match the frozen target block")
    gt = arrays["genotypes"]
    require(gt.shape == (len(people), len(markers)) and
            np.isin(gt[np.isfinite(gt)], (0, 1, 2)).all() and not np.isinf(gt).any(), "invalid target genotypes")
    # Hashing/authenticating the source reads its bytes, including reserved rows;
    # target construction below selects FIT and this one development role only.
    indices = np.asarray([reader.people[p] for p in ids], dtype=np.int64)
    components = np.asarray([p["component"] for p in reader.persons])
    settings = reader.source["evaluator"]
    i, j, y, n, support = material.evaluator.target_pairs(
        gt, reader.roles["FIT"], indices, components, settings["min_fit_called"],
        settings["min_pair_markers"], settings["max_pairs"])
    local = {int(k): q for q, k in enumerate(indices)}
    pairs = np.asarray([[local[int(a)], local[int(b)]] for a, b in zip(i, j)], dtype=np.int64)
    view = reader.load_window(ids, *request["core_bp"], request["block_id"], request["exclusion_scope"],
                              request["common_radius_cm"], request["rare_radius_cm"], request["attention_depth"])
    if verified is not None:
        require(verified["input_lineage"] == view["input_lineage"] and
                verified["target_partial_manifest_sha256"] == target["manifest_sha256"],
                "staged source/target differs from pre-cache verification")
    output = Path(output)
    require(not output.exists(), "output already exists; no overwriting")
    if verify_only:
        result = dict(status="AUTHENTICATED_BOUNDED_VIEW_NOT_TRAINING_ADMISSION",
            request_sha256=request_sha256, adapter_sha256=sha(__file__), stage=stage,
            dataset_declared_scope=PILOT, input_lineage=view["input_lineage"],
            target_partial_manifest_sha256=target["manifest_sha256"],
            reserved_preparation_admission_sha256=admission_sha256 if stage == "EVALUATE" else None)
        write_json(output, result)
        return result
    output.mkdir(parents=True)
    path = output / "batch.npz"
    exporter = export_reserved if stage == "EVALUATE" else events.export_training_batch
    receipt = exporter(view, pairs, y, path, source_hash)
    bounded_archive(path, request["max_batch_bytes"])
    np.save(output / "joint_target_markers.npy", n, allow_pickle=False)
    descriptor = dict(path=str(path.resolve()), sha256=sha(path),
                      receipt_path=str(path.with_suffix(".npz.json").resolve()),
                      receipt_sha256=sha(path.with_suffix(".npz.json")))
    excluded = ([1, reader.source["sources"][str(block["chromosome"])]["length_bp"]]
                if request["exclusion_scope"] == "WHOLE_CHROMOSOME" else
                [int(block["excluded_start_bp"]), int(block["excluded_end_bp"])])
    result = dict(schema="r02_context_dataset_v1", status="PREPARED_NOT_ADMITTED_FOR_TRAINING",
        training_allowed=False, dataset_declared_scope=PILOT, stage=stage,
        request_sha256=request_sha256, adapter_sha256=sha(__file__), source_contract_sha256=source_hash,
        reserved_preparation_admission_sha256=admission_sha256 if stage == "EVALUATE" else None,
        target_contract_sha256=source_hash, target_partial_manifest_sha256=target["manifest_sha256"],
        target_formula="r02_structure_increment.target_pairs", target_support=support,
        target_joint_markers_sha256=sha(output / "joint_target_markers.npy"),
        target_value_roles_used=sorted({"FIT", stage}), source_archive_authenticated_all_rows=True,
        block_id=block["block_id"], exclusion_scope=request["exclusion_scope"],
        excluded_physical_interval=dict(chrom=int(block["chromosome"]), start_bp=excluded[0], end_bp=excluded[1]),
        persons={p:dict(component=str(components[reader.people[p]]),role=stage) for p in ids},
        batch=descriptor, common_features=len(events.COMMON_FEATURES),
        event_metadata_features=len(events.EVENT_FEATURES)-1,
        count_scope="owned core loci of this bounded view, not whole genome",
        evaluation_unit="component_macro_not_independent_pairs_or_windows",
        autosomal_aggregation="NOT_IMPLEMENTED_RHO_AND_HEAD_CURRENTLY_APPLIED_PER_BATCH",
        n_pairs=receipt["n_pairs"], n_people=receipt["n_people"],
        limitations=["No automatic task admission or selection of a biological target/window/person subset.",
                     "C_ij target formula is shared with K02; the existing neural component-macro loss is not K02 L_cov.",
                     "This product cannot substitute the principal 22-autosome representation."])
    write_json(output / "dataset.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reserved-admission")
    parser.add_argument("--reserved-admission-sha256")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verified")
    args = parser.parse_args()
    print(json.dumps(build(args.request, args.request_sha256, args.output,
                          args.reserved_admission, args.reserved_admission_sha256, args.verify_only,
                          args.verified), sort_keys=True))


if __name__ == "__main__":
    main()
