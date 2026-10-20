#!/usr/bin/env python3
"""Extend the frozen CN person-batch planner to admitted local attention.

The CN planner still owns admission, target, role, store and cache checks.  This
module replaces only its obsolete Deep-Sets-only numeric configuration check.
It prepares build specifications; it never grants development admission.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CN_PLANNER = (ROOT / ".claude/runs/r02-parallel-expansion-20261018cn/"
              "neural-structure/code/plan_admitted_person_batches.py")
CN_PLANNER_SHA256 = "1dcaf5da3c75943f4232b4bf1bbeeb20981c885ec51a470413826531285ede4e"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_cn_planner():
    if file_sha256(CN_PLANNER) != CN_PLANNER_SHA256:
        raise RuntimeError(f"frozen CN planner SHA256 differs: {CN_PLANNER}")
    spec = importlib.util.spec_from_file_location("r02_cn_admitted_person_batch_planner", CN_PLANNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen CN planner: {CN_PLANNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CN = _load_cn_planner()
req, descriptor, sha_obj = CN.req, CN.descriptor, CN.sha_obj


def request_context(config):
    """Return the exact radii/depth written into an autosomal request."""
    family = config.get("family")
    common = config.get("common_radius_cm")
    rare = config.get("rare_radius_cm")
    depth = config.get("depth")
    if type(common) not in (int, float) or not math.isfinite(common) or common <= 0:
        raise ValueError("resolved configuration requires a positive finite common radius")
    if family == "structure_deep_sets":
        if rare is not None:
            raise ValueError("structure_deep_sets requires rare_radius_cm=null")
        return {"common_radius_cm": common, "rare_radius_cm": None, "attention_depth": 0}
    if family == "structure_local_attention":
        if type(rare) not in (int, float) or not math.isfinite(rare) or rare <= 0:
            raise ValueError("structure_local_attention requires a positive finite rare radius")
        if type(depth) is not int or depth <= 0:
            raise ValueError("structure_local_attention requires a positive integer depth")
        return {"common_radius_cm": common, "rare_radius_cm": rare, "attention_depth": depth}
    raise ValueError("resolved structure family is unsupported")


def authenticated_json(value, label):
    descriptor(value, label)
    path = Path(value["path"])
    req(path.is_file() and file_sha256(path) == value["sha256"], label + " SHA256 differs")
    def unique(pairs):
        result = {}
        for key, item in pairs:
            req(key not in result, label + " has duplicate JSON key")
            result[key] = item
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ValueError(label + " has nonfinite JSON value: " + token)))


def plan(catalogue, admissions, stores, persons, cache, store_root):
    """Preserve CN admission checks and add explicit supported-family checks."""
    req(catalogue.get("schema") == "r02_common_territory_catalogue_v1", "catalogue schema differs")
    views = catalogue.get("views", [])
    claimed = catalogue.get("view_count")
    req(claimed == len(views) and len({row["view_id"] for row in views}) == claimed,
        "catalogue view universe differs")
    by_view = {row["view_id"]: row for row in views}
    req(stores.get("schema") == "r02_neural_store_request_index_v1" and
        len(stores.get("requests", [])) == 22, "22-store index differs")
    rows = sorted(stores["requests"], key=lambda row: int(row["chromosome"]))
    req([str(row["chromosome"]) for row in rows] == [str(index) for index in range(1, 23)],
        "store order differs")
    store_root = Path(store_root)
    req(store_root.is_absolute(), "store root must be absolute")
    store_refs = []
    for row in rows:
        descriptor(row["manifest"], "store manifest")
        store_refs.append({"chromosome": str(row["chromosome"]),
            "path": str(store_root / f"chr{row['chromosome']}.neural-input-validation" / "store"),
            "manifest_sha256": row["manifest"]["sha256"]})
    descriptor(persons, "persons")
    req(cache.get("embedded_manifest_sha256") and cache.get("packed_input", {}).get("generation") and
        isinstance(cache.get("bound_request_identity"), dict), "packed FULL22 identity differs")
    plans, signatures = [], set()
    for admission in admissions:
        req(admission.get("status") == "APPROVED_DEVELOPMENT" and
            admission.get("training_allowed") is True and admission.get("evaluate_allowed") is False,
            "configuration/view is not admitted")
        view = by_view.get(admission.get("view_id"))
        req(view is not None, "admitted view absent from catalogue")
        req(admission.get("block_id") == view.get("block_id") and
            admission.get("exclusion_scope") == view.get("exclusion_scope"), "view/exclusion differs")
        config_values = admission.get("resolved_config_values")
        request_context(config_values if isinstance(config_values, dict) else {})
        on_disk_config = authenticated_json(admission.get("resolved_config", {}), "resolved config")
        req(on_disk_config == config_values,
            "resolved config JSON differs from resolved_config_values")
        req(admission.get("roles") == {"FIT": 1460, "SELECT": 476, "EVALUATE": "CLOSED_NOT_READ"},
            "role policy differs")
        partition = admission.get("person_partition", {})
        req(set(partition) == {"schema", "max_people", "method"} and
            partition.get("schema") == "r02_context_person_partition_v1" and
            partition.get("method") == "CONTIGUOUS_FROZEN_ORDER" and
            type(partition.get("max_people")) is int and partition["max_people"] > 0,
            "documented technical person partition required")
        targets = admission.get("targets", {})
        req(set(targets) == {"FIT", "SELECT"}, "only FIT/SELECT targets allowed")
        for role in ("FIT", "SELECT"):
            descriptor(targets[role], role + " target")
            descriptor({"path": targets[role].get("receipt_path"),
                        "sha256": targets[role].get("receipt_sha256")}, role + " receipt")
        identity = {"view_id": admission["view_id"], "block_id": admission["block_id"],
            "exclusion_scope": admission["exclusion_scope"],
            "target_contract_sha256": admission["target_contract_sha256"],
            "resolved_config_sha256": admission["resolved_config"]["sha256"]}
        signature = sha_obj(identity)
        req(signature not in signatures, "duplicate target/config signature")
        signatures.add(signature)
        cache_identity = cache["bound_request_identity"]
        cache_matches = all(identity.get(key) == cache_identity.get(key) for key in
            ("view_id", "block_id", "exclusion_scope", "target_contract_sha256", "resolved_config_sha256"))
        old = admission.get("reuse_person_batch_request")
        if old is not None:
            req(cache_matches and old.get("signature") == signature and
                old.get("packed_cache_manifest_sha256") == cache["embedded_manifest_sha256"],
                "cached chr01/config request is incompatible")
        spec = {"schema": "r02_human_autosomal_person_batch_build_v1",
            "status": "PREPARATION_NOT_ADMISSION", "identity": {**identity, "signature": signature},
            "packed_cache_reuse": ({"allowed": True, "manifest_sha256": cache["embedded_manifest_sha256"]}
                if cache_matches else {"allowed": False,
                    "reason": "CACHE_BOUND_TO_DIFFERENT_TARGET_OR_NUMERIC_CONFIGURATION"}),
            "persons": persons, "resolved_config": admission["resolved_config"], "stores": store_refs,
            "task": {"block_id": admission["block_id"],
                     "exclusion_scope": admission["exclusion_scope"], "targets": targets},
            "target_contract_sha256": admission["target_contract_sha256"],
            "person_partition": partition,
            "reader_backend": {"schema": "r02_context_reader_backend_v1", "name": "ZIP_STORED_MMAP_V1"},
            "training_allowed": False, "evaluate_allowed": False}
        plans.append({"view_id": admission["view_id"], "signature": signature, "spec": spec,
                      "builder": "r02_structure_person_batch_builder.py"})
    return {"schema": "r02_admitted_structure_person_batch_plan_v2",
        "status": "READY_TO_BUILD_REQUESTS_FOR_ADMITTED_PAIRS_ONLY",
        "supported_families": ["structure_deep_sets", "structure_local_attention"],
        "catalogue_view_count": claimed, "planned_count": len(plans), "source_store_count": 22,
        "existing_request_bound_cache": cache["packed_input"], "plans": plans,
        "evaluate_allowed": False}


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    for name in ("catalogue", "admissions", "stores", "persons", "cache", "store-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = plan(load(args.catalogue), load(args.admissions), load(args.stores),
                  load(args.persons), load(args.cache), args.store_root)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
