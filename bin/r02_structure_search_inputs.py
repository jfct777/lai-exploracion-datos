#!/usr/bin/env python3
"""Describe the human input requests for the prospective Structure search.

This program reads metadata only.  It does not read genotype chunks, create
person batches, open EVALUATE responses, or grant scientific admission.  The
output enumerates every K03 target/exclusion view and every family/radius
context fixed by CW2 so that later admission records can be checked against a
complete, prospective universe.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA = "r02_structure_search_input_requests_v2"
K03_VIEW_SCHEMA = "r02_k03_fixed_source_all_views_v1"
NUMERIC_SCHEMA = "r02_structure_scientific_numeric_decision_v1"
STORE_SCHEMA = "r02_neural_store_request_index_v1"
TARGET_CATALOGUE_SCHEMA = "r02_structure_88_target_catalogue_v1"
DATASET_MODE = "AUTOSOMAL_PERSON_BATCHES_V1"
FAMILIES = ("structure_deep_sets", "structure_local_attention")
SCOPES = ("BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME")
ROLE_COUNTS = {"FIT": 1460, "SELECT": 476, "EVALUATE": 683}
HEX64 = re.compile(r"[0-9a-f]{64}")
VIEW = re.compile(
    r"(BLOCK_AND_LD_MARGIN|WHOLE_CHROMOSOME)__(chr(?:0[1-9]|1[0-9]|2[0-2])_part0[12])"
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


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_pinned(path, expected_sha256):
    # Nextflow stages authenticated inputs as symlinks. Resolve that transport
    # link and authenticate the target bytes rather than rejecting the workflow.
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink(), "missing input: " + str(path))
    require(HEX64.fullmatch(expected_sha256 or "") and sha256(path) == expected_sha256,
            "input SHA-256 differs: " + str(path))

    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    with path.open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=unique,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               ValueError("non-finite JSON value: " + value)))
    canonical(result)
    return result


def descriptor(path, expected_sha256):
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink(), "descriptor input missing: " + str(path))
    require(sha256(path) == expected_sha256, "descriptor SHA-256 differs: " + str(path))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": expected_sha256}


def validate_views(value):
    require(value.get("schema") == K03_VIEW_SCHEMA and
            value.get("scientific_models_admitted") is False,
            "K03 view plan schema/admission differs")
    rows = value.get("views")
    require(isinstance(rows, list) and len(rows) == 88, "exactly 88 K03 views required")
    expected = {f"{scope}__chr{chromosome:02d}_part{part:02d}"
                for scope in SCOPES for chromosome in range(1, 23) for part in (1, 2)}
    observed = {row.get("id") for row in rows}
    require(len(observed) == 88 and observed == expected, "K03 44-region/two-scope universe differs")
    result = []
    for row in rows:
        match = VIEW.fullmatch(row["id"])
        require(match is not None and HEX64.fullmatch(row.get("campaign_sha256", "")),
                "invalid K03 view record")
        scope, block = match.groups()
        result.append({"source_view_id": row["id"], "view_id": f"{block}__{scope}",
                       "block_id": block, "exclusion_scope": scope,
                       "campaign_sha256": row["campaign_sha256"]})
    return sorted(result, key=lambda row: (row["block_id"], SCOPES.index(row["exclusion_scope"])))


def validate_numeric(value):
    require(value.get("schema") == NUMERIC_SCHEMA and
            value.get("status") == "NUMERIC_SEARCH_DEFINED_UPDATE_BUDGET_AND_HUMAN_ADMISSION_PENDING" and
            value.get("training_allowed") is False and value.get("evaluate_allowed") is False,
            "CW2 numeric decision state differs")
    require(value.get("families") == list(FAMILIES), "CW2 family order/content differs")
    radii = value.get("search_space", {}).get("common_radius_cm")
    require(radii == [0.05, 0.2, 1.0] and all(math.isfinite(x) and x > 0 for x in radii),
            "CW2 common radii differ")
    policy = value.get("context_policy", {})
    require(value["search_space"].get("rare_radius_cm") == radii and
            policy.get("attention_rare_radius") == "match_common_radius" and
            policy.get("selection_design") == "single_frozen_fit_select" and
            policy.get("rare_effect_selection") == "report_only" and
            policy.get("fixed") == {"depth": 2, "dropout": 0.0, "heads": 4},
            "CW2 prospective context policy differs")
    contexts = []
    for family in FAMILIES:
        for radius in radii:
            request = {"family": family, "common_radius_cm": radius,
                       "rare_radius_cm": None if family == "structure_deep_sets" else radius,
                       "attention_depth": 0 if family == "structure_deep_sets" else 2}
            contexts.append({"context_sha256": digest(request), "request_context": request})
    require(len({row["context_sha256"] for row in contexts}) == 6,
            "six unique family/radius contexts required")
    return contexts


def validate_stores(index, linked):
    require(index.get("schema") == STORE_SCHEMA, "22-autosome store index schema differs")
    rows = index.get("requests")
    require(isinstance(rows, list) and len(rows) == 22, "22 autosomal store references required")
    require([str(row.get("chromosome")) for row in rows] == [str(i) for i in range(1, 23)],
            "autosomal store order differs")
    linked_stores = linked.get("stores", {})
    linked_rows = linked_stores.get("manifests", [])
    require(linked_stores.get("chromosomes") == 22 and linked_stores.get("chunks") == 72295 and
            linked_stores.get("people") == 2619 and len(linked_rows) == 22,
            "CN authenticated store inventory differs")
    linked_hashes = {row["chromosome"]: row["manifest"]["sha256"] for row in linked_rows}
    result = []
    for row in rows:
        chromosome = str(row["chromosome"])
        manifest = row.get("manifest", {})
        require(set(manifest) >= {"path", "sha256"} and
                HEX64.fullmatch(manifest.get("sha256", "")) and
                linked_hashes.get(chromosome) == manifest["sha256"],
                "store manifest differs from CN for chromosome " + chromosome)
        result.append({"chromosome": chromosome, "manifest": manifest})
    return result


def validate_people(linked):
    people = linked.get("people", {})
    source = people.get("descriptor", {})
    require(people.get("counts") == ROLE_COUNTS and people.get("cross_role_components") == 0 and
            set(source) >= {"path", "sha256", "bytes"}, "CN person roster receipt differs")
    path = Path(source["path"])
    descriptor(path, source["sha256"])
    counts = Counter()
    component_roles = defaultdict(set)
    sample_ids = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["sample_id", "component", "role"], "person roster columns differ")
        for row in reader:
            require(row["sample_id"] not in sample_ids and row["component"] and row["role"] in ROLE_COUNTS,
                    "person roster identity, component or role differs")
            sample_ids.add(row["sample_id"])
            counts[row["role"]] += 1
            component_roles[row["component"]].add(row["role"])
    require(dict(counts) == ROLE_COUNTS and len(sample_ids) == 2619 and
            all(len(roles) == 1 for roles in component_roles.values()),
            "person roster role/component separation differs")
    return {"descriptor": source, "role_counts": ROLE_COUNTS,
            "components": len(component_roles), "cross_role_components": 0}


def validate_targets(value, view_rows):
    require(value.get("schema") == TARGET_CATALOGUE_SCHEMA and
            value.get("status") == "AUTHENTICATED_88_FIT_SELECT_TARGETS_NO_TRAINING" and
            value.get("views") == 88 and value.get("new_direct_views") == 87 and
            value.get("reused_archive_views") == 1 and
            value.get("training_allowed") is False and
            value.get("evaluate_targets_present") is False,
            "88-view FIT/SELECT target catalogue state differs")
    completion = value.get("completion_source", {})
    require(set(completion) >= {"path", "bytes", "sha256"},
            "target completion source descriptor differs")
    descriptor(completion["path"], completion["sha256"])
    require(Path(completion["path"]).stat().st_size == completion["bytes"],
            "target completion source size differs")

    expected_names = {"manifest.json", "FIT.target.private.npz", "FIT.target.receipt.json",
                      "SELECT.target.private.npz", "SELECT.target.receipt.json"}
    expected_views = {row["view_id"] for row in view_rows}
    rows = value.get("targets")
    require(isinstance(rows, list) and len(rows) == 88, "exactly 88 target bundles required")
    require({row.get("view_id") for row in rows} == expected_views,
            "target catalogue and K03 view plan differ")

    def pinned_object(item, label):
        require(isinstance(item, dict) and set(item) >=
                {"uri", "generation", "bytes", "sha256", "md5_base64"} and
                str(item["uri"]).startswith("gs://projects-usp/dnaBr-lai/datalake/") and
                re.fullmatch(r"[1-9][0-9]*", str(item["generation"])) and
                isinstance(item["bytes"], int) and item["bytes"] > 0 and
                HEX64.fullmatch(item["sha256"]), "invalid pinned target object: " + label)

    result = {}
    direct_bytes = archive_bytes = 0
    sources = Counter()
    for row in rows:
        view_id = row["view_id"]
        require(row.get("roles") == ["FIT", "SELECT"] and row.get("evaluate") is False,
                "target roles/EVALUATE state differs for " + view_id)
        source = row.get("source")
        sources[source] += 1
        if source == "common88_direct_objects":
            objects = row.get("objects", {})
            require(set(objects) == expected_names, "direct target file inventory differs for " + view_id)
            for name, item in objects.items():
                pinned_object(item, view_id + "/" + name)
                require(item["uri"].endswith("/neural_" + view_id + "/" + name),
                        "target object URI/view differs for " + view_id)
                direct_bytes += item["bytes"]
        elif source == "verified_archive_member":
            require(view_id == "chr01_part01__WHOLE_CHROMOSOME",
                    "unexpected view supplied by verified archive")
            archive = row.get("archive", {})
            pinned_object(archive, view_id + "/archive")
            manifest = row.get("manifest", {})
            require(set(manifest) >= {"path", "bytes", "sha256"},
                    "archived target manifest descriptor differs")
            descriptor(manifest["path"], manifest["sha256"])
            require(Path(manifest["path"]).stat().st_size == manifest["bytes"],
                    "archived target manifest size differs")
            archive_bytes += archive["bytes"]
        else:
            raise ValueError("unknown target source for " + view_id)
        result[view_id] = {
            "catalogue_index": rows.index(row),
            "source": source,
            "target_descriptor_sha256": digest(row),
            "roles": ["FIT", "SELECT"],
            "evaluate": False,
        }
    require(sources == {"common88_direct_objects": 87, "verified_archive_member": 1},
            "87-direct/one-archive target provenance differs")
    return result, {
        "views": 88, "direct_views": 87, "archive_views": 1,
        "missing_views": [], "extra_views": [],
        "direct_published_object_bytes": direct_bytes,
        "archive_transport_bytes": archive_bytes,
        "published_transport_bytes": direct_bytes + archive_bytes,
        "completion_source": completion,
    }


def build(views, numeric, stores, linked, cache, targets, sources):
    view_rows = validate_views(views)
    contexts = validate_numeric(numeric)
    store_rows = validate_stores(stores, linked)
    people = validate_people(linked)
    target_rows, target_coverage = validate_targets(targets, view_rows)
    require(cache.get("schema") == "r02_full22_gcs_reuse_v1" and
            cache.get("status") == "PUBLISHED_VERIFIED_BY_EXISTING_RECEIPTS" and
            cache.get("scientific_scope", {}).get("training_admitted") is False,
            "CN FULL22 cache state differs")
    cache_identity = cache.get("bound_request_identity", {})
    require(cache_identity.get("view_id") == "chr01_part01__BLOCK_AND_LD_MARGIN" and
            HEX64.fullmatch(cache_identity.get("resolved_config_sha256", "")),
            "CN request-bound cache identity differs")

    shared_store_identity = digest([
        {"chromosome": row["chromosome"], "manifest_sha256": row["manifest"]["sha256"]}
        for row in store_rows
    ])
    blockers = [
        "APPROVED_DEVELOPMENT_REVIEW_REQUIRED_PER_VIEW_AND_CONTEXT",
        "RESOLVED_CONFIGURATION_HASH_REQUIRED_BEFORE_CN_BUILDER",
        "OPTIMIZER_UPDATE_AND_RESOURCE_BUDGET_REQUIRED_BEFORE_TRAINING",
    ]
    requests = []
    for view in view_rows:
        for context in contexts:
            identity = {"view_id": view["view_id"], "block_id": view["block_id"],
                        "exclusion_scope": view["exclusion_scope"],
                        "context_sha256": context["context_sha256"]}
            requests.append({
                "request_id": digest(identity),
                "status": "DESCRIPTIVE_REQUEST_NOT_MATERIALIZED_NOT_ADMITTED",
                "identity": identity,
                "source_view_id": view["source_view_id"],
                "view_campaign_sha256": view["campaign_sha256"],
                "request_context": context["request_context"],
                "dataset_mode": DATASET_MODE,
                "roles": {"FIT": 1460, "SELECT": 476, "EVALUATE": "CLOSED_NOT_READ"},
                "source_store_set_sha256": shared_store_identity,
                "fit_select_targets": target_rows[view["view_id"]],
                "person_partition": {"schema": "r02_context_person_partition_v1",
                                     "max_people": 8, "method": "CONTIGUOUS_FROZEN_ORDER",
                                     "classification": "CN_TECHNICAL_CHOICE_NOT_SCIENTIFIC_OPTIMUM"},
                "reader_backend": {"schema": "r02_context_reader_backend_v1",
                                   "name": "ZIP_STORED_MMAP_V1"},
                "cn_builder": "build_human_autosomal_person_batches_v13_reader_exact.py",
                "cn_admission_planner": "plan_admitted_person_batches.py",
                "cn_cache_reuse": {"allowed": False,
                    "reason": "FULL22_CACHE_IS_BOUND_TO_ONE_EXACT_TARGET_AND_RESOLVED_CONFIGURATION"},
                "required_before_materialization": blockers[:2],
                "training_allowed": False,
                "evaluate_allowed": False,
            })
    require(len(requests) == 528 and len({row["request_id"] for row in requests}) == 528,
            "88 views x six contexts must produce 528 unique requests")
    counts = Counter(row["request_context"]["family"] for row in requests)
    return {
        "schema": SCHEMA,
        "status": "ALL_528_REQUESTS_LINKED_TO_AUTHENTICATED_FIT_SELECT_TARGETS_MATERIALIZATION_BLOCKED",
        "question": ("For each of 88 K03 target/exclusion views, prepare the six CW2 "
                     "Deep Sets/local-attention radius contexts without opening EVALUATE."),
        "dataset_mode": DATASET_MODE,
        "view_count": 88,
        "biological_region_count": 44,
        "exclusion_scopes": list(SCOPES),
        "context_count": 6,
        "request_count": 528,
        "request_counts_by_family": dict(counts),
        "shared_inputs": {
            "sources": sources,
            "store_set_sha256": shared_store_identity,
            "store_count": 22,
            "chunk_count": 72295,
            "stores": store_rows,
            "people": people,
            "known_request_bound_cache": cache["packed_input"],
            "known_cache_bound_identity": cache_identity,
            "fit_select_target_coverage": target_coverage,
        },
        "contexts": contexts,
        "requests": requests,
        "blockers_before_materialization": blockers[:2],
        "blockers_before_training": blockers,
        "training_allowed": False,
        "evaluate_allowed": False,
        "does_not_certify": ["scientific admission", "current GCS object availability", "training budget",
                             "model performance", "biological validity"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("views-plan", "numeric-decision", "stores", "real-input-link", "cache",
                 "target-catalogue"):
        parser.add_argument("--" + name, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inputs = {}
    values = {}
    for name in ("views_plan", "numeric_decision", "stores", "real_input_link", "cache",
                 "target_catalogue"):
        path = getattr(args, name)
        expected = getattr(args, name + "_sha256")
        values[name] = load_pinned(path, expected)
        inputs[name] = descriptor(path, expected)
    result = build(values["views_plan"], values["numeric_decision"], values["stores"],
                   values["real_input_link"], values["cache"], values["target_catalogue"], inputs)
    output = Path(args.output)
    require(not output.exists() and output.parent.is_dir(), "fresh output path required")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("status", "view_count", "context_count", "request_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
