#!/usr/bin/env python3
"""Build the exact HUMAN structure instrumentation admission.

This module joins one authenticated K03 target view to the 22 staged event
stores.  It creates a real ``AUTOSOMAL_PERSON_BATCHES_V1`` build request and,
after that request has been materialized by the existing builder, emits the
development admission and training contract for the three matched information
arms.  EVALUATE is never accepted or emitted.

The admitted run is an engineering measurement: two optimizer updates over the
complete FIT axis, followed by SELECT scoring.  Its only purpose is to measure
memory, gradients, I/O and checkpoint restoration on human tensors.  It cannot
select a model or support a biological conclusion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile


HEX = re.compile(r"[0-9a-f]{64}")
ARMS = ("NONE", "COUNTS", "RARE")
VIEW = "chr01_part01__BLOCK_AND_LD_MARGIN"
BLOCK = "chr01_part01"
SCOPE = "BLOCK_AND_LD_MARGIN"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path, digest=None):
    path = Path(path).resolve()
    require(path.is_file(), f"missing input: {path}")
    if digest is not None:
        require(HEX.fullmatch(str(digest)) and sha(path) == digest,
                f"SHA256 differs: {path}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(
                          ValueError(f"nonfinite JSON: {x}")))


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")


def descriptor(item, label):
    require(set(item) == {"path", "sha256"} and Path(item["path"]).is_absolute(),
            f"exact absolute {label} descriptor required")
    value = read(item["path"], item["sha256"])
    return Path(item["path"]).resolve(), value


def people(path, digest):
    path = Path(path).resolve()
    require(sha(path) == digest, "persons.tsv SHA256 differs")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(rows and set(rows[0]) == {"sample_id", "component", "role"},
            "persons.tsv fields differ")
    counts = {role: sum(row["role"] == role for row in rows)
              for role in ("FIT", "SELECT", "EVALUATE")}
    require(counts == {"FIT": 1460, "SELECT": 476, "EVALUATE": 683}
            and len({row["sample_id"] for row in rows}) == 2619,
            "frozen human role/person counts differ")
    roles = {}
    for row in rows:
        roles.setdefault(row["component"], set()).add(row["role"])
    require(all(len(value) == 1 for value in roles.values()),
            "component crosses FIT/SELECT/EVALUATE")
    return rows, counts


def validate_spec(spec):
    require(spec.get("schema") == "r02_neural_structure_instrumentation_spec_v1"
            and spec.get("status") == "APPROVE_COMPLETE_HUMAN_ENGINEERING_MEASUREMENT_ONLY",
            "instrumentation spec/status differs")
    require(spec.get("view_id") == VIEW and spec.get("block_id") == BLOCK
            and spec.get("exclusion_scope") == SCOPE,
            "the bounded instrumentation view differs")
    require(spec.get("evaluate") == {"opened": False, "targets_present": False},
            "EVALUATE must remain closed")
    integration_path, integration = descriptor(spec["integration"], "AQ integration")
    require(integration.get("status") == "FULL22_AND_88_TARGETS_AUTHENTICATED_PREPARED_NOT_ADMITTED"
            and integration["stores"]["chromosomes"] == 22
            and integration["stores"]["people_per_store"] == 2619
            and integration["targets"]["views"] == 88
            and integration["targets"]["evaluate_views"] == 0,
            "AQ integration state differs")
    target_root = Path(spec["target_root"]).resolve()
    require(target_root.is_dir() and target_root.name == "neural_" + VIEW,
            "exact target view directory required")
    manifest = read(target_root / "manifest.json", spec["target_manifest_sha256"])
    require(manifest.get("schema") == "r02_k03_neural_target_adapter_result_v1"
            and manifest.get("training_allowed") is False
            and manifest.get("evaluate_targets_computed") is False
            and set(manifest.get("targets", {})) == {"FIT", "SELECT"},
            "target manifest scope differs")
    for role, count in (("FIT", 1460), ("SELECT", 476)):
        item = manifest["targets"][role]
        require(sha(target_root / item["path"]) == item["sha256"]
                and sha(target_root / item["receipt_path"]) == item["receipt_sha256"],
                f"{role} target bytes differ")
        receipt = read(target_root / item["receipt_path"])
        require(receipt.get("role") == role and receipt.get("n_people") == count
                and receipt.get("block_id") == BLOCK
                and receipt.get("exclusion_scope") == SCOPE
                and receipt.get("target_contract_sha256") ==
                manifest["target_contract_sha256"], f"{role} target lineage differs")
    persons_path = Path(spec["persons"]["path"]).resolve()
    roster, counts = people(persons_path, spec["persons"]["sha256"])
    block_path = Path(spec["blocks"]["path"]).resolve()
    require(sha(block_path) == spec["blocks"]["sha256"], "blocks.tsv SHA256 differs")
    with block_path.open(newline="") as handle:
        block_rows = list(csv.DictReader(handle, delimiter="\t"))
    matches = [row for row in block_rows if row["block_id"] == BLOCK]
    require(len(matches) == 1 and matches[0]["chromosome"] == "1",
            "target block is absent or duplicated")
    excluded = {"chrom": 1,
                "start_bp": int(matches[0]["excluded_start_bp"]),
                "end_bp": int(matches[0]["excluded_end_bp"])}
    require(1 <= excluded["start_bp"] <= excluded["end_bp"],
            "invalid frozen physical exclusion")
    store_index_path, store_index = descriptor(spec["store_request_index"],
                                                "22-store request index")
    require(store_index.get("schema") == "r02_neural_store_request_index_v1"
            and len(store_index.get("requests", [])) == 22,
            "exact 22-store request index required")
    rows = sorted(store_index["requests"], key=lambda row: int(row["chromosome"]))
    require([str(row["chromosome"]) for row in rows] == [str(x) for x in range(1, 23)],
            "store index must cover autosomes 1..22")
    for row in rows:
        for kind in ("manifest", "request"):
            item = row[kind]
            require(Path(item["path"]).is_absolute() and sha(item["path"]) == item["sha256"],
                    f"chr{row['chromosome']} {kind} differs")
    config = spec["instrumentation_config"]
    require(config == {"family": "structure_deep_sets", "common_radius_cm": 0.05,
                       "rare_radius_cm": None, "width": 32, "depth": 2,
                       "dropout": 0.0, "heads": None, "learning_rate": 0.001},
            "instrumentation anchor changed")
    require(spec.get("optimizer_updates") == 2 and spec.get("checkpoint_every_steps") == 1
            and spec.get("seed") == 1701 and spec.get("threads") == 1,
            "bounded recovery measurement policy changed")
    return dict(spec=spec, integration_path=integration_path, integration=integration,
                target_root=target_root, target_manifest=manifest,
                persons_path=persons_path, roster=roster, counts=counts,
                excluded=excluded, store_index_path=store_index_path,
                store_rows=rows)


def prepare(spec_path, spec_sha256, output):
    spec = read(spec_path, spec_sha256)
    state = validate_spec(spec)
    out = Path(output).resolve()
    require(not out.exists() and out.parent.is_dir(), "new packet output required")
    out.mkdir()
    config_path = out / "resolved.instrumentation.json"
    write(config_path, spec["instrumentation_config"])
    packet = {
        "schema": "r02_neural_structure_instrumentation_packet_v1",
        "status": "READY_TO_STAGE_FULL22_FOR_APPROVED_ENGINEERING_MEASUREMENT",
        "training_allowed": True,
        "scientific_model_selection_allowed": False,
        "biological_interpretation_allowed": False,
        "evaluate_allowed": False,
        "spec": {"path": str(Path(spec_path).resolve()), "sha256": spec_sha256},
        "integration": spec["integration"],
        "view": {"view_id": VIEW, "block_id": BLOCK, "exclusion_scope": SCOPE,
                 "target_manifest_sha256": spec["target_manifest_sha256"],
                 "target_contract_sha256": state["target_manifest"]["target_contract_sha256"],
                 "excluded_bp": state["excluded"]},
        "roles": {"FIT": 1460, "SELECT": 476, "EVALUATE": "CLOSED_683_NOT_READ"},
        "information_arms": {
            "NONE": "common context plus the identical rare-locus catalogue, metadata and observed-exposure mask; individual carrier/copy totals and per-locus dosage are zero",
            "COUNTS": "NONE plus genome-wide carrier count and counted-allele copy total; per-locus rare dosage remains zero",
            "RARE": "COUNTS plus person-by-locus counted-allele dosage, preserving rare identity/context",
        },
        "pair_objective": {
            "loss": "K03_COMPONENT_PAIR_WEIGHTED_MSE",
            "FIT_pairs": state["target_manifest"]["targets"]["FIT"]["sha256"],
            "SELECT_pairs": state["target_manifest"]["targets"]["SELECT"]["sha256"],
            "kinship_control": "within PC-Relate components excluded upstream; pair_weight equalizes represented component-pair strata",
            "independence_claim": False,
        },
        "instrumentation": {
            "resolved_config": {"path": str(config_path), "sha256": sha(config_path)},
            "optimizer_updates": 2, "checkpoint_every_steps": 1,
            "seed": 1701, "threads": 1,
            "required_recovery_observation": "terminate after confirmed step 1, restart exact identity, require restored_steps=1 and final_confirmed_step=2",
        },
        "search_policy_after_instrumentation": spec["search_policy"],
        "store_request_index": spec["store_request_index"],
        "target_root": str(state["target_root"]),
        "persons": spec["persons"], "blocks": spec["blocks"],
        "resources": spec["resources"],
        "next_gate": "PASS only if all 22 exact stores remain staged, all three arms finish two updates, RARE restores step 1, gradients are finite, RSS/I-O/checkpoint duration are measured, and SELECT is only descriptively scored; otherwise preserve disk and stop",
    }
    write(out / "PACKET.json", packet)
    write(out / "STORE_REQUESTS.json", {"schema": "r02_neural_structure_store_requests_v1",
        "requests": state["store_rows"]})
    return packet


def materialize_spec(packet_path, packet_sha256, staged_root, output):
    packet = read(packet_path, packet_sha256)
    require(packet.get("schema") == "r02_neural_structure_instrumentation_packet_v1"
            and packet.get("evaluate_allowed") is False, "packet scope differs")
    root = Path(staged_root).resolve()
    stores = []
    index = read(packet["store_request_index"]["path"],
                 packet["store_request_index"]["sha256"])
    for row in sorted(index["requests"], key=lambda x: int(x["chromosome"])):
        chrom = str(row["chromosome"])
        store = root / f"chr{chrom}.neural-input-validation" / "store"
        store_manifest = store / "manifest.json"
        require(store.is_dir() and store_manifest.is_file()
                and sha(store_manifest) == row["manifest"]["sha256"],
                f"staged chr{chrom} store differs")
        stores.append({"chromosome": chrom, "path": str(store),
                       "manifest_sha256": row["manifest"]["sha256"]})
    target_root = Path(packet["target_root"])
    target_manifest = read(target_root / "manifest.json", packet["view"]["target_manifest_sha256"])
    build = {
        "schema": "r02_human_autosomal_person_batch_build_v1",
        "status": "PREPARATION_NOT_ADMISSION",
        "persons": packet["persons"],
        "stores": stores,
        "resolved_config": packet["instrumentation"]["resolved_config"],
        "target_contract_sha256": packet["view"]["target_contract_sha256"],
        "task": {"block_id": BLOCK, "exclusion_scope": SCOPE,
                 "targets": {role: {**target_manifest["targets"][role],
                    "path": str((target_root / target_manifest["targets"][role]["path"]).resolve()),
                    "receipt_path": str((target_root / target_manifest["targets"][role]["receipt_path"]).resolve())}
                    for role in ("FIT", "SELECT")}},
        "person_partition": {"schema": "r02_context_person_partition_v1",
                             "max_people": 8, "method": "CONTIGUOUS_FROZEN_ORDER"},
        "reader_backend": {"schema": "r02_context_reader_backend_v1",
                           "name": "ZIP_STORED_MMAP_V1"},
    }
    write(output, build)
    return build


def finalize(packet_path, packet_sha256, build_manifest_path, output,
             synthetic_code_test=False, eligibility_manifest_path=None,
             eligibility_manifest_sha256=None):
    packet = read(packet_path, packet_sha256)
    require(packet.get("schema") == "r02_neural_structure_instrumentation_packet_v1"
            and packet.get("evaluate_allowed") is False, "packet scope differs")
    manifest = read(build_manifest_path)
    require(manifest.get("schema") == "r02_human_autosomal_person_batch_manifest_v1"
            and manifest.get("status") == "PREPARED_NOT_ADMITTED_FOR_TRAINING"
            and manifest.get("training_allowed") is False
            and set(manifest.get("requests", {})) == {"FIT", "SELECT"},
            "person-batch manifest differs")
    out = Path(output).resolve()
    require(not out.exists() and out.parent.is_dir(), "new final output required")
    roster, _ = people(packet["persons"]["path"], packet["persons"]["sha256"])
    persons = {row["sample_id"]: {"component": row["component"], "role": row["role"]}
               for row in roster if row["role"] in ("FIT", "SELECT")}
    eligibility = None
    if not synthetic_code_test:
        require(eligibility_manifest_path is not None and eligibility_manifest_sha256 is not None,
                "human structure training requires the joint eligibility manifest")
        eligibility_path = Path(eligibility_manifest_path).resolve()
        require(eligibility_path.is_file() and sha(eligibility_path) == eligibility_manifest_sha256,
                "joint eligibility manifest differs")
        eligibility = {"path": str(eligibility_path), "sha256": eligibility_manifest_sha256}
        from r02_neural_training_eligibility import require_contract_eligibility
        require_contract_eligibility({"objective": "structure", "persons": persons,
                                      "eligibility_manifest": eligibility})
    out.mkdir()
    target_root = Path(packet["target_root"])
    target_manifest = read(target_root / "manifest.json", packet["view"]["target_manifest_sha256"])
    batches = {}
    for role in ("FIT", "SELECT"):
        target = target_manifest["targets"][role]
        request = manifest["requests"][role]
        batches[role] = [{"path": str((target_root / target["path"]).resolve()),
                          "sha256": target["sha256"],
                          "receipt_path": str((target_root / target["receipt_path"]).resolve()),
                          "receipt_sha256": target["receipt_sha256"],
                          "autosomal_request_path": str(Path(request["path"]).resolve()),
                          "autosomal_request_sha256": request["sha256"]}]
    excluded = dict(packet["view"]["excluded_bp"])
    # Resolve genetic endpoints from the authenticated chr1 map.  The physical
    # interval remains the exact mask used by every input reader.
    fit_request = read(batches["FIT"][0]["autosomal_request_path"],
                       batches["FIT"][0]["autosomal_request_sha256"])
    import r02_genomic_event_inputs as events
    import r02_structure_regions as regions
    chr1 = fit_request["stores"][0]
    require(chr1["chromosome"] == "1", "autosomal request store order differs")
    reader = events.EventChromosome(chr1["path"], chr1["manifest_sha256"])
    excluded.update(start_cm=float(regions.interpolate_cm(reader.points, excluded["start_bp"])),
                    end_cm=float(regions.interpolate_cm(reader.points, excluded["end_bp"])))
    common_features = len(reader.manifest["common_features"])
    event_metadata_features = len(reader.manifest["event_features"]) - 1
    require(common_features > 0 and event_metadata_features > 0,
            "authenticated store feature inventory is empty")
    max_target_pairs = 0
    max_target_bytes = 0
    for role in ("FIT", "SELECT"):
        receipt = read(batches[role][0]["receipt_path"], batches[role][0]["receipt_sha256"])
        max_target_pairs = max(max_target_pairs, int(receipt["n_pairs"]))
        with zipfile.ZipFile(batches[role][0]["path"]) as archive:
            max_target_bytes = max(max_target_bytes,
                                   sum(item.file_size for item in archive.infolist()))
    target_definition = {
        "question": "predict held-out chr01_part01 common-genetic pair similarity from autosomal inputs outside the frozen block plus LD margin",
        "target": "authenticated K03 continuous pair response",
        "selection_role": "SELECT_DESCRIPTIVE_FOR_INSTRUMENTATION_ONLY",
    }
    count_scope = {
        "NONE": "rare carrier/copy totals zero; observed exposure retained",
        "COUNTS": "autosomal retained rare carrier/copy totals and observed exposure",
        "RARE": "COUNTS plus retained per-locus counted-allele dosage",
        "counted_allele": "source-minor allele fixed upstream; never inferred from REF/ALT here",
    }
    feature_definition = {
        "common": "fixed measured common-locus features outside the exclusion",
        "rare": "fixed rare catalogue, metadata, observation mask and arm-dependent dosage/counts outside the exclusion",
        "phase": "diploid dosage; no invented rare haplotype phase",
    }
    scope = "SYNTHETIC_CODE_TEST" if synthetic_code_test else "DEVELOPMENT"
    admission_status = "CODE_TEST_ONLY" if synthetic_code_test else "APPROVED_DEVELOPMENT"
    purpose = "SYNTHETIC_CONTAINER_RECOVERY_FIXTURE" if synthetic_code_test else "AUTOSOMAL_STRUCTURE_DEVELOPMENT"
    admission = {
        "schema": "r02_neural_structure_engineering_admission_v1",
        "status": admission_status, "scope": scope, "purpose": purpose,
        "dataset_mode": "AUTOSOMAL_PERSON_BATCHES_V1",
        "dataset_declared_scope": "AUTOSOMAL_22",
        "target_definition": target_definition, "feature_definition": feature_definition,
        "count_scope": count_scope, "persons": persons, "batches": batches,
        "target_contract_sha256": packet["view"]["target_contract_sha256"],
        "pair_loss_policy": "K03_COMPONENT_PAIR_WEIGHTED_MSE",
        **({"eligibility_manifest": eligibility} if eligibility is not None else {}),
        "excluded_intervals": [excluded], "evaluate_allowed": False,
        "optimizer_updates": 2,
        "scientific_model_selection_allowed": False,
        "allowed_conclusion": ("synthetic container wiring and exact-checkpoint recovery only"
                               if synthetic_code_test else
                               "human-scale memory, I/O, finite-gradient and exact-checkpoint recovery feasibility only"),
    }
    admission_path = out / "admission.json"; write(admission_path, admission)
    contract = {
        "schema": "r02_context_training_v1", "objective": "structure",
        "scope": scope, "purpose": purpose,
        "dataset_mode": "AUTOSOMAL_PERSON_BATCHES_V1",
        "dataset_declared_scope": "AUTOSOMAL_22",
        "admission": {"status": admission_status, "path": str(admission_path),
                      "sha256": sha(admission_path)},
        "target_definition": target_definition, "feature_definition": feature_definition,
        "count_scope": count_scope, "persons": persons, "batches": batches,
        "target_contract_sha256": packet["view"]["target_contract_sha256"],
        "pair_loss_policy": "K03_COMPONENT_PAIR_WEIGHTED_MSE",
        **({"eligibility_manifest": eligibility} if eligibility is not None else {}),
        "excluded_intervals": [excluded], "steps": 2, "threads": 1,
        "common_features": common_features,
        "event_metadata_features": event_metadata_features,
        "limits": {"max_common_sites": reader.r["max_window_sites"],
                   "max_event_sites": reader.r["max_window_sites"],
                   "max_targets": max_target_pairs,
                   "max_batch_bytes": max_target_bytes},
    }
    contract_path = out / "contract.json"; write(contract_path, contract)
    import r02_context_train as trainer
    trainer.validate_contract(contract)
    config = packet["instrumentation"]["resolved_config"]
    tasks = [{"view_id": VIEW, "arm": arm, "seed": packet["instrumentation"]["seed"],
              "contract_path": str(contract_path), "contract_sha256": sha(contract_path),
              "resolved_path": config["path"], "resolved_sha256": config["sha256"],
              "checkpoint_dir": str((out / "checkpoints" / arm).resolve())}
             for arm in ARMS]
    result = {"schema": "r02_neural_structure_instrumentation_final_v1",
              "status": ("READY_FOR_SYNTHETIC_CODE_TEST" if synthetic_code_test else
                         "READY_FOR_ROOT_LAUNCH_REVIEW"), "training_allowed": True,
              "scientific_model_selection_allowed": False, "evaluate_allowed": False,
              "contract": {"path": str(contract_path), "sha256": sha(contract_path)},
              "admission": {"path": str(admission_path), "sha256": sha(admission_path)},
              "resolved_config": config, "tasks": tasks,
              "checkpoint_protocol": packet["instrumentation"]["required_recovery_observation"],
              "next_gate": packet["next_gate"]}
    write(out / "FINAL.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("prepare"); pre.add_argument("--spec", required=True)
    pre.add_argument("--spec-sha256", required=True); pre.add_argument("--output", required=True)
    mat = sub.add_parser("materialize-spec"); mat.add_argument("--packet", required=True)
    mat.add_argument("--packet-sha256", required=True); mat.add_argument("--staged-root", required=True)
    mat.add_argument("--output", required=True)
    fin = sub.add_parser("finalize"); fin.add_argument("--packet", required=True)
    fin.add_argument("--packet-sha256", required=True); fin.add_argument("--build-manifest", required=True)
    fin.add_argument("--output", required=True)
    fin.add_argument("--synthetic-code-test", action="store_true")
    fin.add_argument("--eligibility-manifest")
    fin.add_argument("--eligibility-manifest-sha256")
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(args.spec, args.spec_sha256, args.output)
    elif args.command == "materialize-spec": result = materialize_spec(args.packet, args.packet_sha256, args.staged_root, args.output)
    else: result = finalize(args.packet, args.packet_sha256, args.build_manifest, args.output,
                            args.synthetic_code_test, args.eligibility_manifest,
                            args.eligibility_manifest_sha256)
    print(json.dumps({"status": result.get("status", "COMPLETE")}, sort_keys=True))


if __name__ == "__main__":
    main()
