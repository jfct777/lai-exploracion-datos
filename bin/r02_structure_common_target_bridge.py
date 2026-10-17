#!/usr/bin/env python3
"""Produce authenticated common FIT/SELECT arrays without choosing a penalty.

This is the data bridge consumed by the finite common search and the neural
target adapter.  It binds one continuous common representation to one exact
prepared target shard and its exclusion scope.  It neither accepts nor emits
an alpha, a model-fit request, predictions, or EVALUATE targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

import r02_structure_increment as core
import r02_structure_k03_common_fit as legacy
import r02_structure_k03_prepare as prepare


require = core.require
CODE_FILES = tuple(dict.fromkeys((
    "r02_structure_common_target_bridge.py", *legacy.CODE_FILES,
)))


def implementation():
    return {name: core.sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def identity(records, names, roles=("FIT", "SELECT")):
    digest = hashlib.sha256()
    for role in roles:
        for name in names:
            value = np.asarray(records[role][name])
            digest.update(f"{role}:{name}:{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def write_target_product(output, persons, points, records, *, metadata,
                         target_audit, max_output_bytes, min_free_bytes):
    out = Path(output).resolve()
    require(not out.exists() and out.parent.is_dir(), "New common target output required")
    require(shutil.disk_usage(out.parent).free >= min_free_bytes + max_output_bytes,
            "Common target output disk reserve insufficient")
    out.mkdir(mode=0o700)
    prepare.features.discrete.materialize.input_plan.write_table(out / "persons.tsv", persons)
    np.savez(out / "arrays.private.npz", points=np.asarray(points), **{
        role + "_" + name: value for role, record in records.items()
        for name, value in record.items()
    })
    candidate_id = hashlib.sha256(
        ("COMMON_ONLY:" + metadata["block_id"] + ":" + metadata["exclusion_scope"]).encode()
    ).hexdigest()
    manifest = dict(
        schema="r02_k03_fit_select_arrays_v1",
        stage="FROZEN_FIT_SELECT_ARRAYS_NO_EVALUATE_TARGETS",
        arm="CONTINUOUS", candidate_id=candidate_id,
        target_identity=identity(records, ("i", "j", "y", "n", "w")),
        mask_control_identity=identity(records, ("i", "j", "n", "w", "extra")),
        select_target_identity=identity(records, ("i", "j", "y", "n", "w"), ("SELECT",)),
        target_audit=target_audit, common_only_profile=False,
        common_profile_purpose="FIT_SELECT_TARGET_SOURCE_NO_MODEL_OR_PENALTY",
        fit_weight_sum=float(np.asarray(records["FIT"]["w"]).sum()),
        alpha_selection_performed=False, penalty_present=False,
        model_fit_performed=False, evaluate_targets_computed=False,
        biological_validation=False, **metadata,
    )
    manifest["outputs_sha256"] = {
        "persons.tsv": core.sha(out / "persons.tsv"),
        "arrays.private.npz": core.sha(out / "arrays.private.npz"),
    }
    core.write_json(out / "manifest.json", manifest)
    require(sum(path.stat().st_size for path in out.iterdir()) <= max_output_bytes and
            shutil.disk_usage(out).free >= min_free_bytes,
            "Common target output byte/disk bound exceeded")
    return dict(
        schema="r02_k03_common_target_bridge_v1",
        stage="COMMON_FIT_SELECT_TARGET_SOURCE_COMPLETE_NO_FIT_REQUEST",
        block_id=metadata["block_id"], exclusion_scope=metadata["exclusion_scope"],
        common_axes=int(np.asarray(points).shape[1]),
        input_manifest_sha256=core.sha(out / "manifest.json"),
        fit_request_generated=False, penalty_present=False,
        model_fit_performed=False, evaluate_targets_computed=False,
        biological_validation=False,
    )


def run_request(request_path, request_sha256, admission_path, admission_sha256, output):
    os.umask(0o077)
    path = Path(request_path)
    require(path.stat().st_size <= 2**20 and core.sha(path) == request_sha256,
            "Common target bridge request changed")
    request = core.load_json(path)
    required = {"schema", "evidence_scope", "scientific_spec_sha256",
                "source_contract_sha256", "implementation_sha256", "block_id",
                "sources", "weighting", "resources"}
    require(set(request) == required and
            request["schema"] == "r02_k03_common_target_bridge_request_v1" and
            request["implementation_sha256"] == implementation() and
            request["evidence_scope"] in {"DNABR_R02", "SYNTHETIC_INSTRUMENT_ONLY"} and
            request["weighting"] == "COMPONENT_EQUAL",
            "Exact penalty-free common target request/code required")
    require(core.sha(admission_path) == admission_sha256,
            "Common target bridge admission changed")
    admission = core.load_json(admission_path)
    require(admission == dict(
        schema="r02_k03_common_target_bridge_admission_v1",
        status="COMMON_FIT_SELECT_TARGET_INPUT_ADMITTED",
        evidence_scope=request["evidence_scope"],
        scientific_spec_sha256=request["scientific_spec_sha256"],
        request_sha256=request_sha256, evaluate_allowed=False,
    ), "Exact external common target admission required")
    limits = request["resources"]
    require(set(limits) == {"max_metadata_bytes", "max_input_bytes", "max_array_bytes",
            "max_memory_bytes", "max_target_workspace_bytes", "max_output_bytes",
            "min_free_bytes"} and all(type(value) is int and value > 0
            for value in limits.values()), "Explicit positive common target limits required")
    sources = request["sources"]
    require(set(sources) == {"common", "prepared", "evaluation"},
            "Common target bridge accepts common and target sources only")
    auth = prepare.Authenticator(limits)
    prepared = sources["prepared"]
    require(set(prepared) == {"directory", "contract_sha256"},
            "Exact prepared target catalogue required")
    root = Path(prepared["directory"])
    contract = core.load_json(auth.pin(root / "contract.json",
                                       prepared["contract_sha256"], metadata=True))
    evaluation_desc = sources["evaluation"]
    require(set(evaluation_desc) == {"path", "sha256"},
            "Pinned scope-specific target configuration required")
    evaluation = core.load_json(auth.pin(evaluation_desc["path"],
                                         evaluation_desc["sha256"], metadata=True))
    require(evaluation["prepared_contract_sha256"] == prepared["contract_sha256"] and
            contract["source_contract_sha256"] == request["source_contract_sha256"],
            "Common target source or exclusion-specific configuration differs")
    store = core.BlockBundles(root, contract, evaluation)
    matches = [index for index, block in enumerate(store.blocks)
               if block["block_id"] == request["block_id"]]
    require(len(matches) == 1, "Exact common target block required")
    index = matches[0]
    shard = store.path(store.entries[index])
    receipt = core.load_json(shard / "manifest.json")
    for name, digest in receipt["outputs_sha256"].items():
        if name.endswith(".npz"):
            auth.array(shard / name, digest)
        else:
            auth.pin(shard / name, digest, metadata=True)
    people = store.persons
    allowed = legacy.validate_common_territory(
        store.blocks[index], contract["exclusion_scope"],
        legacy.common_manifest_allowed(sources["common"], auth))
    common, common_manifest = prepare.common_scores(
        sources["common"], auth, people,
        source_sha256=request["source_contract_sha256"], block_id=request["block_id"],
        scope=contract["exclusion_scope"], allowed=allowed,
        weighting=request["weighting"])
    values, block_contract = store.load(index)
    loaded_people, _, blocks, markers, genotypes, _, pairs = values
    records, target_audit = prepare.target_arrays(
        loaded_people, blocks[0], markers, genotypes, pairs[request["block_id"]],
        min_fit_called=block_contract["min_fit_called"],
        min_pair_markers=block_contract["min_pair_markers"],
        max_pairs=block_contract["max_pairs"],
        max_workspace_bytes=limits["max_target_workspace_bytes"])
    memory_bound = 6 * sum(auth.array_sizes.values()) + limits["max_target_workspace_bytes"]
    require(memory_bound <= limits["max_memory_bytes"],
            "Common target bridge workspace exceeds memory bound")
    metadata = dict(
        evidence_scope=request["evidence_scope"],
        scientific_spec_sha256=request["scientific_spec_sha256"],
        source_contract_sha256=request["source_contract_sha256"],
        block_id=request["block_id"], exclusion_scope=contract["exclusion_scope"],
        common_manifest_sha256=sources["common"]["manifest_sha256"],
        common_numerical_rank=common_manifest["numerical_rank"],
        weighting=request["weighting"], implementation_sha256=implementation(),
        input_request_sha256=request_sha256, memory_bound_bytes=memory_bound,
    )
    result = write_target_product(
        output, people, common, records, metadata=metadata,
        target_audit=target_audit, max_output_bytes=limits["max_output_bytes"],
        min_free_bytes=limits["min_free_bytes"])
    auth.verify(); store.verify()
    require(core.sha(path) == request_sha256 and
            implementation() == request["implementation_sha256"],
            "Common target inputs/code changed during run")
    core.write_json(Path(output) / "bridge-manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "request-sha256", "admission", "admission-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run_request(args.request, args.request_sha256, args.admission,
                         args.admission_sha256, args.output)
    print(json.dumps(result, sort_keys=True))
