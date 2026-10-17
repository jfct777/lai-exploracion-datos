#!/usr/bin/env python3
"""Materialize 88 penalty-free common-target bridge requests after staging."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import r02_structure_common_target_bridge as bridge


HEX = re.compile(r"[0-9a-f]{64}")
SCOPES = ("BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def pin(path, digest):
    path = Path(path).resolve()
    require(path.is_file() and HEX.fullmatch(digest or "") and sha(path) == digest,
            "Pinned request source differs: " + str(path))
    return path


def write(path, value):
    path = Path(path)
    require(not path.exists(), "Refuse overwrite: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return sha(path)


def build(*, catalogue_path, catalogue_sha, assembly_path, assembly_sha,
          legacy_request_path, legacy_request_sha, whole_evaluation_path,
          whole_evaluation_sha, ld_evaluation_path, ld_evaluation_sha,
          scientific_spec_path, scientific_spec_sha, source_contracts_path,
          source_contracts_sha, common_root, prepared_root, output, verify_staged=True):
    catalogue = load(pin(catalogue_path, catalogue_sha))
    assembly = load(pin(assembly_path, assembly_sha))
    legacy = load(pin(legacy_request_path, legacy_request_sha))
    pin(scientific_spec_path, scientific_spec_sha)
    source_contracts = load(pin(source_contracts_path, source_contracts_sha))
    require(source_contracts.get("schema") == "r02_common_prepared_scope_contracts_v1" and
            set(source_contracts.get("scopes", {})) == set(SCOPES),
            "Exact two-scope source contracts required")
    evaluations = {
        "WHOLE_CHROMOSOME": (pin(whole_evaluation_path, whole_evaluation_sha),
                             whole_evaluation_sha),
        "BLOCK_AND_LD_MARGIN": (pin(ld_evaluation_path, ld_evaluation_sha),
                                ld_evaluation_sha),
    }
    require(catalogue.get("schema") == "r02_common_territory_catalogue_v1" and
            catalogue.get("view_count") == 88 and
            assembly.get("schema") == "r02_common_territory_assembly_sources_v1" and
            assembly.get("view_count") == 88 and
            legacy.get("schema") == "r02_k03_common_fit_bridge_request_v1",
            "Exact catalogue, source map and frozen predecessor request required")
    common_root = Path(common_root).resolve(); prepared_root = Path(prepared_root).resolve()
    require(not verify_staged or (common_root.is_dir() and prepared_root.is_dir()),
            "Staged roots required")
    by_view = {row["view_id"]: row for row in assembly["views"]}
    require(len(by_view) == 88 and set(by_view) == {row["view_id"] for row in catalogue["views"]},
            "Exact 88-view source equality required")
    out = Path(output).resolve()
    require(out.parent.is_dir(), "Request output parent required")
    resume=out.exists();require(not resume or out.is_dir(),"Request output must be a directory")
    if not resume:out.mkdir(mode=0o700); (out / "requests").mkdir()
    else:require((out/"requests").is_dir(),"Resumable request directory required")
    def emit(path,value):
        if resume:
            require(path.is_file()and load(path)==value,"Existing request materialization differs: "+str(path))
            return sha(path)
        return write(path,value)
    rows = []
    for row in catalogue["views"]:
        view_id = row["view_id"]; block = row["block_id"]; scope = row["exclusion_scope"]
        source = by_view[view_id]["prepared"]
        common = common_root / ("target_" + view_id)
        prepared = prepared_root / scope
        common_manifest = common / "manifest.json"; prepared_contract = prepared / "contract.json"
        scope_contract = source_contracts["scopes"][scope]
        require(scope_contract["prepared_contract_sha256"] == source["contract_sha256"],
                "Prepared source-contract binding differs: " + view_id)
        if verify_staged:
            require(common_manifest.is_file() and sha(common_manifest) ==
                    row["common_representation_manifest_sha256"],
                    "Staged common representation differs: " + view_id)
            require(prepared_contract.is_file() and sha(prepared_contract) == source["contract_sha256"],
                    "Staged prepared contract differs: " + view_id)
            contract = load(prepared_contract)
            require(contract["source_contract_sha256"] == scope_contract["source_contract_sha256"] and
                    contract["scientific_protocol_sha256"] == scope_contract["scientific_protocol_sha256"],
                    "Staged source contract differs: " + view_id)
        evaluation_path, evaluation_sha = evaluations[scope]
        evaluation = load(evaluation_path)
        require(evaluation["prepared_contract_sha256"] == source["contract_sha256"] and
                evaluation["scientific_protocol_sha256"] == scope_contract["scientific_protocol_sha256"],
                "Scope-specific target configuration differs: " + view_id)
        request = {
            "schema": "r02_k03_common_target_bridge_request_v1",
            "evidence_scope": legacy["evidence_scope"],
            "scientific_spec_sha256": scientific_spec_sha,
            "source_contract_sha256": scope_contract["source_contract_sha256"],
            "implementation_sha256": bridge.implementation(),
            "block_id": block,
            "sources": {
                "common": {"directory": str(common),
                           "manifest_sha256": row["common_representation_manifest_sha256"]},
                "prepared": {"directory": str(prepared),
                             "contract_sha256": source["contract_sha256"]},
                "evaluation": {"path": str(evaluation_path), "sha256": evaluation_sha},
            },
            "weighting": legacy["weighting"],
            "resources": legacy["resources"],
        }
        request_path = out / "requests" / (view_id + ".json")
        request_sha = emit(request_path, request)
        rows.append({"id": view_id, "request": str(request_path),
                     "request_sha256": request_sha,
                     "admission_status": "NOT_GRANTED", "evaluate_allowed": False})
    batch = {
        "schema": "r02_k03_common_target_bridge_request_batch_v1",
        "status": "REQUESTS_COMPLETE_ADMISSIONS_REQUIRED",
        "scientific_spec_sha256": scientific_spec_sha,
        "implementation_sha256": bridge.implementation(),
        "view_count": 88, "views": rows,
        "penalty_present": False, "fit_requests_generated": False,
        "evaluate_allowed": False,
    }
    batch_sha = emit(out / "batch.json", batch)
    receipt = {
        "schema": "r02_k03_common_target_bridge_request_receipt_v1",
        "stage": "EIGHTY_EIGHT_PENALTY_FREE_REQUESTS_COMPLETE_ADMISSIONS_REQUIRED",
        "batch_sha256": batch_sha, "view_count": 88,
        "scope_counts": {scope: sum(row["id"].endswith("__" + scope) for row in rows)
                         for scope in SCOPES},
        "penalty_present": False, "model_fits_run": False,
        "evaluate_targets_computed": False,
    }
    emit(out / "manifest.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalogue", "catalogue-sha256", "assembly", "assembly-sha256",
                 "legacy-request", "legacy-request-sha256", "whole-evaluation",
                 "whole-evaluation-sha256", "ld-evaluation", "ld-evaluation-sha256",
                 "scientific-spec", "scientific-spec-sha256",
                 "source-contracts", "source-contracts-sha256",
                 "common-root", "prepared-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prepare-offline", action="store_true")
    args = parser.parse_args()
    result = build(catalogue_path=args.catalogue, catalogue_sha=args.catalogue_sha256,
        assembly_path=args.assembly, assembly_sha=args.assembly_sha256,
        legacy_request_path=args.legacy_request, legacy_request_sha=args.legacy_request_sha256,
        whole_evaluation_path=args.whole_evaluation,
        whole_evaluation_sha=args.whole_evaluation_sha256,
        ld_evaluation_path=args.ld_evaluation, ld_evaluation_sha=args.ld_evaluation_sha256,
        scientific_spec_path=args.scientific_spec,
        scientific_spec_sha=args.scientific_spec_sha256,
        source_contracts_path=args.source_contracts,
        source_contracts_sha=args.source_contracts_sha256,
        common_root=args.common_root, prepared_root=args.prepared_root, output=args.output,
        verify_staged=not args.prepare_offline)
    print(json.dumps(result, sort_keys=True))
