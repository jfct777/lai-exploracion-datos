#!/usr/bin/env python3
"""Authenticate the complete human Structure inputs without training.

This joins three already-produced facts: the final 22-store validation gate,
the 88 FIT/SELECT target views, and the existing autosomal trainer features.
It reads target JSON metadata only (plus the single reused local target pair),
so it does not restage the 195 GB event stores or download 87 target pairs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


HEX = re.compile(r"[0-9a-f]{64}")
VIEW = re.compile(r"(chr(?:0[1-9]|1[0-9]|2[0-2])_part(?:01|02))__(BLOCK_AND_LD_MARGIN|WHOLE_CHROMOSOME)")
ROLES = {"FIT": 1460, "SELECT": 476}
TARGET_FILES = {
    "FIT": ("FIT.target.private.npz", "FIT.target.receipt.json"),
    "SELECT": ("SELECT.target.private.npz", "SELECT.target.receipt.json"),
}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, expected_sha256: str | None = None) -> dict:
    require(path.is_file() and not path.is_symlink(), f"missing or linked JSON: {path}")
    if expected_sha256 is not None:
        require(HEX.fullmatch(expected_sha256) is not None, "invalid expected SHA-256")
        require(sha256(path) == expected_sha256, f"JSON SHA-256 differs: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def object_descriptor(value: dict, name: str, view_id: str) -> None:
    require(set(value) == {"bytes", "generation", "md5_base64", "sha256", "uri"},
            f"{view_id}: {name} object descriptor fields differ")
    require(type(value["bytes"]) is int and value["bytes"] > 0,
            f"{view_id}: {name} byte size is invalid")
    require(type(value["generation"]) is str and value["generation"].isdigit(),
            f"{view_id}: {name} generation is invalid")
    require(type(value["md5_base64"]) is str and value["md5_base64"],
            f"{view_id}: {name} MD5 is missing")
    require(HEX.fullmatch(value["sha256"]) is not None,
            f"{view_id}: {name} SHA-256 is invalid")
    require(value["uri"].endswith(f"/neural_{view_id}/{name}"),
            f"{view_id}: {name} URI does not identify the view")


def validate_receipt(receipt: dict, role: str, block_id: str, scope: str,
                     arrays_sha256: str, target_contract_sha256: str) -> dict:
    expected_fields = ["component_id", "joint_marker_count", "pair_weight", "pairs",
                       "person_id", "role", "target"]
    require(receipt.get("schema") == "r02_k03_neural_pair_target_v1"
            and receipt.get("status") == "TARGET_REPRESENTATION_COMPLETE_NOT_TRAINING_ADMISSION",
            f"{block_id}/{scope}/{role}: target receipt state differs")
    require(receipt.get("role") == role and receipt.get("n_people") == ROLES[role],
            f"{block_id}/{scope}/{role}: development role/person count differs")
    require(receipt.get("block_id") == block_id and receipt.get("exclusion_scope") == scope,
            f"{block_id}/{scope}/{role}: view identity differs")
    require(receipt.get("arrays_sha256") == arrays_sha256
            and receipt.get("target_contract_sha256") == target_contract_sha256
            and receipt.get("scientific_spec_sha256") == target_contract_sha256,
            f"{block_id}/{scope}/{role}: target lineage differs")
    require(receipt.get("fields") == expected_fields,
            f"{block_id}/{scope}/{role}: target fields differ")
    require(receipt.get("training_allowed") is False
            and receipt.get("evaluate_targets_computed") is False
            and receipt.get("biological_validation") is False,
            f"{block_id}/{scope}/{role}: forbidden state is open")
    semantics = receipt.get("neural_consumer_semantics")
    require(semantics == {
        "used_by_current_loss": ["pairs", "target", "pair_weight"],
        "authenticated_but_not_used_by_current_loss": ["joint_marker_count"],
    }, f"{block_id}/{scope}/{role}: trainer semantics differ")
    n_pairs = receipt.get("n_pairs")
    support = receipt.get("support", {})
    require(type(n_pairs) is int and n_pairs > 0
            and support.get("role_people") == ROLES[role]
            and support.get("emitted_between_component_pairs") == n_pairs
            and support.get("target_missing_values") == 0
            and type(support.get("minimum_joint_marker_count")) is int
            and support["minimum_joint_marker_count"] > 0
            and type(support.get("maximum_joint_marker_count")) is int
            and support["maximum_joint_marker_count"] >= support["minimum_joint_marker_count"]
            and type(support.get("pair_weight_sum")) in (int, float)
            and math.isfinite(support["pair_weight_sum"]) and support["pair_weight_sum"] > 0,
            f"{block_id}/{scope}/{role}: target support differs")
    return {
        "people": ROLES[role],
        "pairs": n_pairs,
        "pair_weight_sum": support["pair_weight_sum"],
        "minimum_joint_marker_count": support["minimum_joint_marker_count"],
        "maximum_joint_marker_count": support["maximum_joint_marker_count"],
        "arrays_sha256": arrays_sha256,
    }


def validate_manifest(manifest: dict, block_id: str, scope: str) -> None:
    require(manifest.get("schema") == "r02_k03_neural_target_adapter_result_v1"
            and manifest.get("status") == "FIT_SELECT_TARGETS_COMPLETE_NOT_TRAINING_ADMISSION",
            f"{block_id}/{scope}: manifest state differs")
    require(manifest.get("training_allowed") is False
            and manifest.get("evaluate_targets_computed") is False
            and manifest.get("biological_validation") is False,
            f"{block_id}/{scope}: forbidden manifest state is open")
    require(set(manifest.get("targets", {})) == set(ROLES),
            f"{block_id}/{scope}: FIT/SELECT targets are incomplete")
    require(HEX.fullmatch(str(manifest.get("target_contract_sha256", ""))) is not None,
            f"{block_id}/{scope}: target contract SHA-256 is invalid")


def validate_direct_target(item: dict, direct_root: Path) -> dict:
    view_id = item["view_id"]
    match = VIEW.fullmatch(view_id)
    require(match is not None, f"invalid view ID: {view_id}")
    block_id, scope = match.groups()
    require(item.get("source") == "common88_direct_objects"
            and item.get("roles") == ["FIT", "SELECT"]
            and item.get("evaluate") is False,
            f"{view_id}: catalogue state differs")
    objects = item.get("objects", {})
    require(set(objects) == {"manifest.json", *(x for pair in TARGET_FILES.values() for x in pair)},
            f"{view_id}: object inventory differs")
    folder = direct_root / f"neural_{view_id}"
    for name, descriptor in objects.items():
        object_descriptor(descriptor, name, view_id)
    manifest_descriptor = objects["manifest.json"]
    manifest_path = folder / "manifest.json"
    require(manifest_path.stat().st_size == manifest_descriptor["bytes"],
            f"{view_id}: manifest byte size differs")
    manifest = read_json(manifest_path, manifest_descriptor["sha256"])
    validate_manifest(manifest, block_id, scope)
    result = {}
    for role, (array_name, receipt_name) in TARGET_FILES.items():
        array_object, receipt_object = objects[array_name], objects[receipt_name]
        target = manifest["targets"][role]
        require(target == {"path": array_name, "receipt_path": receipt_name,
                           "receipt_sha256": receipt_object["sha256"],
                           "sha256": array_object["sha256"]},
                f"{view_id}/{role}: manifest/object link differs")
        receipt_path = folder / receipt_name
        require(receipt_path.stat().st_size == receipt_object["bytes"],
                f"{view_id}/{role}: receipt byte size differs")
        receipt = read_json(receipt_path, receipt_object["sha256"])
        result[role] = validate_receipt(receipt, role, block_id, scope,
                                        array_object["sha256"], manifest["target_contract_sha256"])
    return {"view_id": view_id, "source": item["source"], "block_id": block_id,
            "exclusion_scope": scope, "target_contract_sha256": manifest["target_contract_sha256"],
            "manifest_sha256": manifest_descriptor["sha256"], "roles": result}


def validate_reused_target(item: dict, reused_root: Path) -> dict:
    view_id = item["view_id"]
    match = VIEW.fullmatch(view_id)
    require(match is not None, f"invalid reused view ID: {view_id}")
    block_id, scope = match.groups()
    require(item.get("source") == "verified_archive_member"
            and item.get("roles") == ["FIT", "SELECT"]
            and item.get("evaluate") is False,
            f"{view_id}: reused catalogue state differs")
    archive = item.get("archive", {})
    require(set(archive) == {"bytes", "generation", "md5_base64", "sha256", "uri"}
            and type(archive["bytes"]) is int and archive["bytes"] > 0
            and type(archive["generation"]) is str and archive["generation"].isdigit()
            and HEX.fullmatch(archive["sha256"]) is not None
            and archive["uri"].startswith("gs://"), f"{view_id}: archive descriptor differs")
    manifest_descriptor = item.get("manifest", {})
    require(set(manifest_descriptor) == {"bytes", "path", "sha256"}
            and Path(manifest_descriptor["path"]).resolve() == (reused_root / "manifest.json").resolve(),
            f"{view_id}: reused manifest path differs")
    manifest_path = reused_root / "manifest.json"
    require(manifest_path.stat().st_size == manifest_descriptor["bytes"],
            f"{view_id}: reused manifest byte size differs")
    manifest = read_json(manifest_path, manifest_descriptor["sha256"])
    validate_manifest(manifest, block_id, scope)
    result = {}
    for role, (array_name, receipt_name) in TARGET_FILES.items():
        target = manifest["targets"][role]
        array_path, receipt_path = reused_root / array_name, reused_root / receipt_name
        require(target.get("path") == array_name and target.get("receipt_path") == receipt_name
                and sha256(array_path) == target.get("sha256")
                and sha256(receipt_path) == target.get("receipt_sha256"),
                f"{view_id}/{role}: reused archive member hash differs")
        receipt = read_json(receipt_path, target["receipt_sha256"])
        result[role] = validate_receipt(receipt, role, block_id, scope,
                                        target["sha256"], manifest["target_contract_sha256"])
    return {"view_id": view_id, "source": item["source"], "block_id": block_id,
            "exclusion_scope": scope, "target_contract_sha256": manifest["target_contract_sha256"],
            "manifest_sha256": manifest_descriptor["sha256"], "roles": result}


def validate_completion(path: Path, expected_sha256: str, publication_path: Path,
                        results_root: Path) -> dict:
    complete = read_json(path, expected_sha256)
    require(complete.get("schema") == "r02_neural_input_validation_complete_v1"
            and complete.get("status") == "COMPLETE_22_REMOTE_RECEIPTS_VERIFIED_NO_TRAINING"
            and complete.get("chromosomes") == 22
            and complete.get("training_started") is False
            and complete.get("evaluate_opened") is False,
            "22-store completion gate differs")
    publication = read_json(publication_path)
    require(publication.get("sha256") == expected_sha256
            and publication.get("bytes") == path.stat().st_size
            and publication.get("uri", "").endswith("/_COMPLETE.json")
            and str(publication.get("generation", "")).isdigit()
            and publication.get("md5_base64"), "completion publication differs")
    receipts = complete.get("receipts", [])
    require(len(receipts) == 22 and {str(x.get("chromosome")) for x in receipts} == {str(x) for x in range(1, 23)},
            "completion receipt chromosome inventory differs")
    bytes_hashed = file_count = 0
    for summary in receipts:
        chrom = str(summary["chromosome"])
        local_path = results_root / f"chr{chrom}.validation.local.json"
        local = read_json(local_path, summary["local_receipt_sha256"])
        require(local.get("schema") == "r02_neural_store_validation_receipt_v1"
                and local.get("status") == "COMPLETE_INPUT_VALIDATION_NO_TRAINING"
                and str(local.get("chromosome")) == chrom
                and local.get("derived_store_released") is True
                and local.get("training_started") is False
                and local.get("evaluate_opened") is False
                and local.get("request_sha256") == summary.get("request_sha256")
                and local.get("publication") == summary.get("publication")
                and local.get("file_count") == summary.get("file_count")
                and local.get("validation", {}).get("bytes_hashed") == summary.get("bytes_hashed")
                and local.get("validation", {}).get("n_people") == 2619
                and local.get("validation", {}).get("reader") == "bin/r02_event_mmap_reader.py:MappedEventChromosome"
                and local.get("validation", {}).get("reader_backend") == "ZIP_STORED_MMAP_V1",
                f"chr{chrom}: completion/local receipt differs")
        bytes_hashed += summary["bytes_hashed"]
        file_count += summary["file_count"]
    return {"status": complete["status"], "completed_utc": complete["completed_utc"],
            "sha256": expected_sha256, "publication": publication, "chromosomes": 22,
            "people_per_store": 2619, "bytes_hashed": bytes_hashed, "file_count": file_count,
            "derived_stores_released": 22, "currently_staged_stores": 0}


def source_symbols(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {x.name for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
    classes = {x.name: {y.name for y in x.body if isinstance(y, (ast.FunctionDef, ast.AsyncFunctionDef))}
               for x in tree.body if isinstance(x, ast.ClassDef)}
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return {"functions": functions, "classes": classes, "assignments": assignments}


def trainer_inventory(repo_root: Path) -> dict:
    paths = {
        "trainer": repo_root / "bin/r02_context_train.py",
        "checkpoint": repo_root / "bin/r02_context_checkpoint.py",
        "dataset": repo_root / "bin/r02_context_autosomal_dataset.py",
        "reader": repo_root / "bin/r02_event_mmap_reader.py",
        "target_consumer": repo_root / "bin/r02_k03_neural_target.py",
        "target_stage": repo_root / "bin/r02_neural_target_stage.py",
        "request_builder": repo_root / ".claude/runs/r02-corrections-20261006j/parallel-launch-20261008ac/person_batching/build_human_autosomal_person_batches.py",
        "workflow": repo_root / "workflows/r02_context_search.nf",
    }
    symbols = {name: source_symbols(path) for name, path in paths.items() if path.suffix == ".py"}
    require("TaskModel" in symbols["trainer"]["classes"]
            and {"forward_autosomal", "backward_autosomal"} <= symbols["trainer"]["classes"]["TaskModel"]
            and {"train", "checkpoint_identity"} <= symbols["trainer"]["functions"],
            "trainer autosomal/backward/checkpoint symbols differ")
    require("CheckpointStore" in symbols["checkpoint"]["classes"]
            and {"save", "restore"} <= symbols["checkpoint"]["classes"]["CheckpointStore"],
            "internal checkpoint symbols differ")
    require("AutosomalDataset" in symbols["dataset"]["classes"]
            and symbols["dataset"]["assignments"].get("PERSON_BATCH_SCHEMA") == "r02_context_autosomal_person_batches_v1",
            "person-batched dataset symbols/schema differ")
    require("weighted_pair_mse" in symbols["target_consumer"]["functions"]
            and symbols["target_consumer"]["assignments"].get("LOSS_POLICY") == "K03_COMPONENT_PAIR_WEIGHTED_MSE",
            "weighted target consumer symbols/policy differ")
    require({"run", "validate_bundle"} <= symbols["target_stage"]["functions"],
            "88-view target staging adapter symbols differ")
    require("build" in symbols["request_builder"]["functions"],
            "human autosomal request builder symbol differs")
    workflow = paths["workflow"].read_text(encoding="utf-8")
    require("--checkpoint-root" in workflow and "--checkpoint-every-steps" in workflow
            and "r02_context_checkpoint.py" in workflow, "Nextflow checkpoint wiring differs")
    evidence_paths = {
        "checkpoint": repo_root / ".claude/runs/r02-corrections-20261006j/neural-checkpoint-20261007t/RESULT.md",
        "bounded_memory_gradients": repo_root / ".claude/runs/r02-corrections-20261006j/neural-memory-completion-20261007v/RESULT.md",
        "person_batching": repo_root / ".claude/runs/r02-corrections-20261006j/parallel-launch-20261008ac/person_batching/RESULT.json",
        "real_target_load": repo_root / ".claude/runs/r02-corrections-20261006j/parallel-launch-20261008ac/structure_adapter/REAL_TARGET_LOAD.json",
    }
    person = read_json(evidence_paths["person_batching"])
    real = read_json(evidence_paths["real_target_load"])
    require(person.get("status") == "LOCAL_IMPLEMENTATION_TESTED_NO_HUMAN_TRAINING_RUN"
            and any(x.get("result") == "4/4 PASS" for x in person.get("verification", [])),
            "person-batching evidence differs")
    require(real.get("training_performed") is False and real.get("evaluate_targets_computed") is False
            and real.get("loss_policy") == "K03_COMPONENT_PAIR_WEIGHTED_MSE",
            "real target-load evidence differs")
    return {
        "consumer": "bin/r02_context_train.py",
        "dataset_mode": "AUTOSOMAL_PERSON_BATCHES_V1",
        "loss_policy": "K03_COMPONENT_PAIR_WEIGHTED_MSE",
        "capabilities": {
            "bounded_autosomal_recomputation_backward": True,
            "finite_gradient_guard": True,
            "atomic_internal_checkpoint_with_optimizer_and_rng": True,
            "lossless_contiguous_person_read_partitions": True,
            "single_global_weighted_loss": True,
            "catalogue_view_staging_adapter": True,
            "human_autosomal_request_builder": True,
        },
        "source_sha256": {str(path.relative_to(repo_root)): sha256(path) for path in paths.values()},
        "prior_verification_evidence": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in evidence_paths.items()
        },
        "verification_reused_not_rerun": True,
    }


def build(args: argparse.Namespace) -> dict:
    repo_root = Path(args.repo_root).resolve()
    catalogue_path = Path(args.catalogue).resolve()
    catalogue = read_json(catalogue_path, args.catalogue_sha256)
    require(catalogue.get("schema") == "r02_structure_88_target_catalogue_v1"
            and catalogue.get("status") == "AUTHENTICATED_88_FIT_SELECT_TARGETS_NO_TRAINING"
            and catalogue.get("views") == 88 and catalogue.get("new_direct_views") == 87
            and catalogue.get("reused_archive_views") == 1
            and catalogue.get("consumer") == "bin/r02_context_train.py"
            and catalogue.get("training_allowed") is False
            and catalogue.get("evaluate_targets_present") is False,
            "88-target catalogue state differs")
    items = catalogue.get("targets", [])
    require(len(items) == 88 and len({x.get("view_id") for x in items}) == 88,
            "88-target catalogue is incomplete or duplicated")
    views = []
    for item in items:
        if item.get("source") == "common88_direct_objects":
            views.append(validate_direct_target(item, Path(args.direct_root).resolve()))
        elif item.get("source") == "verified_archive_member":
            views.append(validate_reused_target(item, Path(args.reused_root).resolve()))
        else:
            raise ValueError(f"unknown target source: {item.get('source')}")
    expected = {(f"chr{chrom:02d}_part{part:02d}", scope)
                for chrom in range(1, 23) for part in (1, 2)
                for scope in ("BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME")}
    actual = {(x["block_id"], x["exclusion_scope"]) for x in views}
    require(actual == expected, "44 blocks x 2 exclusion scopes are not complete")
    contracts = Counter(x["target_contract_sha256"] for x in views)
    require(sorted(contracts.values()) == [1, 87], "expected direct/reused target contract lineages differ")
    complete = validate_completion(Path(args.completion).resolve(), args.completion_sha256,
                                   Path(args.completion_publication).resolve(),
                                   Path(args.completion_results).resolve())
    trainer = trainer_inventory(repo_root)
    result = {
        "schema": "r02_neural_human_integration_inventory_v1",
        "status": "FULL22_AND_88_TARGETS_AUTHENTICATED_PREPARED_NOT_ADMITTED",
        "question": "Are all human Structure inputs technically linked to the existing autosomal trainer without opening EVALUATE or starting training?",
        "catalogue": {"path": str(catalogue_path), "sha256": args.catalogue_sha256},
        "stores": complete,
        "targets": {
            "views": 88, "blocks": 44,
            "exclusion_scopes": ["BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME"],
            "roles_per_view": ["FIT", "SELECT"], "fit_people_per_view": 1460,
            "select_people_per_view": 476, "evaluate_views": 0,
            "direct_views": 87, "reused_views": 1,
            "target_contract_lineages": dict(sorted(contracts.items())), "inventory": views,
            "large_direct_npz_policy": "REUSE_CATALOGUE_GENERATION_MD5_SHA256_AND_JSON_RECEIPTS_NO_REDOWNLOAD",
        },
        "trainer": trainer,
        "integration": {
            "target_metadata_and_roles_ready": True,
            "full22_store_content_validated": True,
            "full22_stores_currently_materialized_together": False,
            "per_view_autosomal_person_batch_requests_materialized": 0,
            "training_allowed": False, "training_started": False, "evaluate_opened": False,
            "conclusion_allowed": "The 88 FIT/SELECT target views and all 22 event-store contents are authenticated and semantically compatible with the existing target consumer; a human training task is not yet admitted or materialized.",
        },
        "blocking_dependencies": [
            {
                "id": "RESTAGE_FULL22_FOR_ONE_ADMITTED_CONSUMER",
                "state": "MISSING",
                "detail": "The sequential validation released every derived store. One admitted training consumer must restage all 22 exact stores together; prior sizing requires a separate 250-300 GiB disk and must preserve a free-space reserve.",
            },
            {
                "id": "APPROVED_DEVELOPMENT_BINDING",
                "state": "MISSING",
                "detail": "No current APPROVED_DEVELOPMENT review binds one exact target view, its direct or reused target-contract lineage, FULL22 request hashes, 1460 FIT and 476 SELECT people, exclusion scope, and K03 component-pair weighted loss.",
            },
            {
                "id": "SCIENTIFIC_CONFIG_AND_SEARCH_BUDGET",
                "state": "MISSING",
                "detail": "No scientifically justified resolved model/search configuration, optimizer-step budget, thread count, or measured human checkpoint frequency was supplied; this inventory invents none.",
            },
            {
                "id": "AUTOSOMAL_PERSON_BATCH_REQUESTS",
                "state": "DEPENDENT",
                "detail": "The existing builder can emit AUTOSOMAL_PERSON_BATCHES_V1 only after the 22 local store paths and an admitted configuration fix radii, tiles, people and hashes. No placeholder requests were created.",
            },
            {
                "id": "EVALUATE",
                "state": "CLOSED",
                "detail": "EVALUATE targets remain absent and cannot be used for fitting or selection.",
            },
        ],
        "operations_performed": [
            "Revalidated the local final 22-store gate and all 22 local receipt hashes; no store was restaged.",
            "Read and hashed 87 manifests plus 174 FIT/SELECT receipts exposed by the published target mount; no direct target NPZ was downloaded.",
            "Hashed the single reused local FIT/SELECT target pair and its receipts.",
            "Parsed current trainer sources and authenticated prior checkpoint, bounded-memory gradient, person-batching and real-target-load evidence; tests were not repeated.",
        ],
        "biological_evidence_obtained": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--catalogue-sha256", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--completion-publication", required=True)
    parser.add_argument("--completion-results", required=True)
    parser.add_argument("--direct-root", required=True)
    parser.add_argument("--reused-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(not output.exists(), f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = build(args)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "sha256": sha256(output), "views": result["targets"]["views"]}, sort_keys=True))


if __name__ == "__main__":
    main()
