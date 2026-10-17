#!/usr/bin/env python3
"""Audit and selectively retain a completed K02/K07 pair campaign.

The default ``audit`` mode is local and read-only.  ``publish`` is fail-closed:
it requires the exact 88-view plan, all 264 SCORE and two SELECT checkpoints,
their v2 receipts, content verification of every pinned GCS object, and a
complete selective runtime inventory.  It never starts/stops/deletes resources.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import r02_autosome_pipeline as pipeline
import r02_pair_publish_completed as checkpoints
import r02_retirement_gate as retirement
import r02_useful_products_publish as useful


SCHEMA = "r02_pair_selective_retention_v1"
SPEC_SCHEMA = "r02_pair_selective_retention_spec_v1"
AUTHORIZED_PREFIX = ("gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/"
                     "biologico/R02_20260930/r02-k02-k07-pair-select-20261009ap")
REQUIRED_LABELS = {
    "package_archive", "root_admission", "run_scope_amendment", "workflow",
    "nextflow_log", "nextflow_trace", "nextflow_timeline", "nextflow_report",
    "workflow_command", "exit_status", "resource_measurements",
}
ROLES = retirement.KEEP_ROLES | retirement.EXCLUDE_ROLES


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    return pipeline.sha(path)


def file_digests(path):
    path = Path(path)
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with path.open("rb") as handle:
        for raw in iter(lambda: handle.read(1 << 20), b""):
            size += len(raw); sha.update(raw); md5.update(raw)
    return {
        "bytes": size,
        "sha256": sha.hexdigest(),
        "md5_base64": base64.b64encode(md5.digest()).decode(),
    }


def campaign(plan_path, admission_path):
    plan_path, admission_path = Path(plan_path), Path(admission_path)
    tasks = checkpoints.validate_plan(plan_path)
    admission = checkpoints.load_json(admission_path)
    require(admission.get("schema") == "r02_k02_k07_pair_select_admission_v1" and
            admission.get("decision") == "ADMIT_FIT_SELECT" and
            admission.get("evaluate_allowed") is False and
            admission.get("view_count") == 88 and
            admission.get("score_checkpoint_count") == 264 and
            admission.get("global_selection_count") == 2 and
            admission.get("team_label") == "frank", "Exact admitted FIT/SELECT campaign required")
    require(admission.get("assembly_plan_sha256") == sha256(plan_path),
            "Admission belongs to another 88-view plan")
    expected_rosters = set()
    for task in tasks.values():
        root = task.get("prepared_root_objects", {})
        record = root.get("persons.tsv", {})
        digest = record.get("sha256")
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                "Plan task lacks an authenticated persons/roles roster")
        expected_rosters.add(digest)
    require(len(expected_rosters) == 1, "The 88-view plan uses incompatible persons/roles rosters")
    return {
        "tasks": tasks, "plan_sha256": sha256(plan_path),
        "admission_sha256": sha256(admission_path),
        "roster_sha256": expected_rosters.pop(), "admission": admission,
    }


def expected_checkpoints(state):
    expected = {}
    for view, task in state["tasks"].items():
        for arm in checkpoints.ARMS:
            relative = f"scores/{view}/{arm}/score-{view}-{arm}"
            expected[relative] = ("score", view, arm, task)
    for scope in checkpoints.SCOPES:
        relative = f"selection/{scope}/selection-{scope}"
        expected[relative] = ("selection", scope, None, None)
    require(len(expected) == 266, "Internal checkpoint plan is not 264 SCORE plus two SELECT")
    return expected


def _validate_record(record, uri):
    require(isinstance(record, dict) and record.get("uri") == uri and
            type(record.get("bytes")) is int and record["bytes"] >= 0 and
            re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))) and
            re.fullmatch(r"[1-9][0-9]*", str(record.get("generation", ""))),
            "Published object record differs: " + uri)


def _receipt_name(relative):
    return relative.replace("/", "__") + ".json"


def validate_checkpoints(output_root, receipts_dir, prefix, state):
    output_root, receipts_dir = Path(output_root).resolve(), Path(receipts_dir).resolve()
    require(output_root.is_dir() and receipts_dir.is_dir(), "Output and receipt directories must exist")
    expected = expected_checkpoints(state)
    actual = {p.name for p in receipts_dir.glob("*.json")}
    expected_names = {_receipt_name(relative) for relative in expected}
    require(actual == expected_names,
            f"Exact receipt set required; missing={len(expected_names-actual)} extra={len(actual-expected_names)}")
    objects, score_links, numerical_identity = [], {}, None
    roster_hashes, source_contracts, evaluations = set(), {}, {}
    receipt_rows = []
    for relative, (kind, first, second, task) in sorted(expected.items()):
        directory = output_root / relative
        require(directory.resolve() == directory and directory.is_dir() and not directory.is_symlink(),
                "Missing canonical checkpoint: " + relative)
        if kind == "score":
            files, manifest_sha = checkpoints.validate_score(directory, first, second, task, sha256)
            manifest = checkpoints.load_json(directory / "manifest.json")
            candidate_sha = sha256(directory / "candidate_scores.json")
            sources = manifest.get("sources_sha256", {})
            require(sources.get("persons.tsv") == state["roster_sha256"],
                    "SCORE persons/roles roster differs from the 88-view plan: " + relative)
            require(all(re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(field, ""))) for field in
                        ("source_contract_sha256", "evaluation_contract_sha256")),
                    "SCORE source/evaluation provenance is incomplete: " + relative)
            roster_hashes.add(sources["persons.tsv"])
            source_contracts.setdefault(task["exclusion_scope"], set()).add(
                manifest.get("source_contract_sha256"))
            evaluations.setdefault(task["exclusion_scope"], set()).add(
                manifest.get("evaluation_contract_sha256"))
            identity_fields = ("code_sha256", "versions", "numerical_backend", "operator_backend",
                "preprocessing_backend", "numerical_parameters", "solver", "numerical_config_sha256",
                "pair_operator_sha256", "pair_preconditioner_sha256", "run_scope_amendment_sha256")
            identity = {key: manifest.get(key) for key in identity_fields}
            require(identity["code_sha256"] == state["admission"].get("evaluator_sha256") and
                    identity["pair_operator_sha256"] == state["admission"].get("pair_operator_sha256") and
                    identity["pair_preconditioner_sha256"] == state["admission"].get("pair_preconditioner_sha256") and
                    identity["numerical_config_sha256"] == state["admission"].get("numerical_contract_sha256") and
                    identity["run_scope_amendment_sha256"] == state["admission"].get("run_scope_amendment_sha256"),
                    "SCORE code/configuration differs from root admission: " + relative)
            if numerical_identity is None:
                numerical_identity = identity
            require(identity == numerical_identity, "Numerical identity differs across SCORE checkpoints")
            score_links[(task["exclusion_scope"], task["block_id"], second)] = {
                "manifest_sha256": manifest_sha, "candidate_scores_sha256": candidate_sha,
                "source_contract_sha256": manifest["source_contract_sha256"],
                "evaluation_contract_sha256": manifest["evaluation_contract_sha256"],
            }
            identity = {"view_id": first, "block_id": task["block_id"], "arm": second,
                        "exclusion_scope": task["exclusion_scope"], "status": checkpoints.SCORE_STATUS}
        else:
            files, manifest_sha = checkpoints.validate_selection(directory, first, state["tasks"], sha256)
            manifest = checkpoints.load_json(directory / "manifest.json")
            require(manifest.get("assembly_plan_sha256") == state["plan_sha256"] and
                    manifest.get("numerical_identity") == numerical_identity,
                    "SELECT plan/numerical identity differs: " + relative)
            selection = checkpoints.load_json(directory / "frozen_selection.json")
            for key, link in selection["score_sources"].items():
                block, arm = key.rsplit(":", 1)
                require(link == score_links.get((first, block, arm)),
                        "SELECT does not bind the exact SCORE source: " + first + ":" + key)
            identity = {"exclusion_scope": first, "status": checkpoints.SELECT_STATUS}
        receipt_path = receipts_dir / _receipt_name(relative)
        receipt = checkpoints.load_json(receipt_path)
        ordered = sorted((p for p in files if p.name not in {"manifest.json", "complete.json"}),
                         key=lambda p: p.relative_to(directory).as_posix())
        ordered += [directory / "manifest.json", directory / "complete.json"]
        require(receipt.get("schema") == checkpoints.RECEIPT_SCHEMA and
                receipt.get("kind") == kind and receipt.get("relative") == relative and
                receipt.get("identity") == identity and receipt.get("manifest_sha256") == manifest_sha and
                receipt.get("complete_published_last") is True and
                len(receipt.get("files", [])) == len(ordered), "Checkpoint receipt identity differs: " + relative)
        for path, record in zip(ordered, receipt["files"]):
            uri = prefix.rstrip("/") + "/checkpoints/" + path.relative_to(output_root).as_posix()
            _validate_record(record, uri)
            digest = file_digests(path)
            require(record["bytes"] == digest["bytes"] and record["sha256"] == digest["sha256"],
                    "Local checkpoint differs from published receipt: " + str(path))
            objects.append((path, record))
        receipt_rows.append((receipt_path, receipt))
    require(len(score_links) == 264 and len(roster_hashes) == 1 and
            all(len(source_contracts[s]) == 1 and len(evaluations[s]) == 1 for s in checkpoints.SCOPES),
            "SCORE provenance/roster coverage is incompatible")
    return {"objects": objects, "receipts": receipt_rows, "score_count": len(score_links),
            "selection_count": 2, "roster_sha256": state["roster_sha256"],
            "output_root": str(output_root), "receipts_dir": str(receipts_dir),
            "source_contracts": {k: next(iter(v)) for k, v in source_contracts.items()},
            "evaluation_contracts": {k: next(iter(v)) for k, v in evaluations.items()},
            "numerical_identity": numerical_identity}


class GcloudContent:
    def __init__(self, project, account):
        self.base = ["gcloud", "--quiet", "--project=" + project, "--account=" + account]

    def _run(self, *args):
        result = subprocess.run([*self.base, *args], capture_output=True, text=True, timeout=7200)
        require(result.returncode == 0, "GCS verification failed: " + (result.stderr or "")[-1000:])
        return result

    def list_receipts(self, prefix):
        return set(self._run("storage", "ls", prefix.rstrip("/") + "/*.json").stdout.splitlines())

    def describe(self, uri):
        return json.loads(self._run("storage", "objects", "describe", uri, "--format=json").stdout)

    def download(self, uri, generation, destination):
        self._run("storage", "cp", uri + "#" + str(generation), str(destination))


def verify_content(records, cloud):
    enriched, total = [], 0
    seen = set()
    for record in records:
        key = (record["uri"], str(record["generation"]))
        if key in seen:
            continue
        seen.add(key)
        metadata = cloud.describe(record["uri"])
        require(str(metadata.get("generation")) == str(record["generation"]) and
                int(metadata.get("size", -1)) == record["bytes"], "Pinned GCS metadata differs")
        with tempfile.NamedTemporaryFile() as handle:
            cloud.download(record["uri"], record["generation"], handle.name)
            digest = file_digests(handle.name)
        require(digest["bytes"] == record["bytes"] and digest["sha256"] == record["sha256"],
                "Pinned GCS content differs: " + record["uri"])
        enriched.append({**record, **digest})
        total += record["bytes"]
    return enriched, total


def verify_remote_receipts(prefix, receipt_rows, cloud):
    remote_prefix = prefix.rstrip("/") + "/provenance/checkpoint-receipts"
    expected = {remote_prefix + "/" + path.name for path, _ in receipt_rows}
    require(cloud.list_receipts(remote_prefix) == expected, "Remote receipt namespace is not exactly 266 objects")
    records = []
    for path, _ in receipt_rows:
        uri = remote_prefix + "/" + path.name
        metadata = cloud.describe(uri)
        record = {"uri": uri, "generation": str(metadata.get("generation")), **file_digests(path)}
        require(int(metadata.get("size", -1)) == record["bytes"], "Remote receipt size differs")
        records.append(record)
    return verify_content(records, cloud)


def validate_spec(spec_path):
    spec = checkpoints.load_json(spec_path)
    require(spec.get("schema") == SPEC_SCHEMA and spec.get("authorization_reference") and
            spec.get("outside_roots_reviewed") is True, "Reviewed selective-retention specification required")
    retirement.labels.validate_plan(spec.get("resource_plan", {}))
    resources = spec["resource_plan"]["resources"]
    require(resources[0]["id"] == "3706342769844118382" and resources[1]["id"] == "8680383801666659206",
            "Retention specification targets different original resources")
    entries = spec.get("entries")
    require(isinstance(entries, list) and entries, "Explicit selective inventory entries required")
    labels, roots = set(), [Path(p) for p in spec.get("reviewed_roots", [])]
    require(roots and all(p.is_absolute() and p != Path("/") and p.resolve() == p for p in roots),
            "Safe canonical reviewed roots required")
    owned = []
    for entry in entries:
        path = Path(entry.get("path", ""))
        require(path.is_absolute() and path.resolve() == path and
                any(path == root or path.is_relative_to(root) for root in roots),
                "Inventory entry outside reviewed roots")
        require(entry.get("role") in ROLES and entry.get("reason", "").strip() and
                entry.get("disposition") in {"PRESERVE", "EXCLUDE"}, "Invalid inventory decision")
        require(not any(path == other or path.is_relative_to(other) or other.is_relative_to(path) for other in owned),
                "Inventory entries overlap")
        owned.append(path)
        if entry["disposition"] == "PRESERVE":
            require(entry["role"] in retirement.KEEP_ROLES and not path.is_symlink() and path.exists(),
                    "Preserved path must exist and cannot be a symlink")
            label = entry.get("label")
            require(isinstance(label, str) and re.fullmatch(r"[a-z][a-z0-9_-]*", label) and
                    label not in labels, "Unique safe retained labels required")
            labels.add(label)
        else:
            require(entry["role"] in retirement.EXCLUDE_ROLES, "Useful role cannot be excluded")
    require(REQUIRED_LABELS <= labels, "Missing required frozen code/Nextflow provenance labels")
    for root in roots:
        require(root.exists() and not root.is_symlink(), "Reviewed root is missing or a symlink")
        leaves = [p for p in root.rglob("*") if p.is_file() or p.is_symlink()]
        require(all(sum(p == entry or p.is_relative_to(entry) for entry in owned) == 1 for p in leaves),
                "Reviewed root contains unclassified or multiply classified paths")
    require(spec.get("resume", {}).get("mode") == "SCIENTIFIC_PRODUCTS" and
            spec["resume"].get("limitations", "").strip(), "Completed-campaign continuation limits required")
    return spec


def audit(plan, admission, output, receipts, prefix, spec=None):
    require(prefix.rstrip("/") == AUTHORIZED_PREFIX, "Exact authorized K02/K07 publication prefix required")
    state = campaign(plan, admission)
    reviewed_spec = validate_spec(spec) if spec is not None else None
    try:
        checked = validate_checkpoints(output, receipts, prefix, state)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        output_path, receipt_path = Path(output), Path(receipts)
        return {"schema": SCHEMA, "status": "HOLD_INCOMPLETE_OR_INVALID", "retirement_allowed": False,
            "reason": str(exc),
            "observed_score_markers": len(list(output_path.glob("scores/*/*/score-*/complete.json")))
                if output_path.is_dir() else 0,
            "observed_selection_markers": len(list(output_path.glob("selection/*/selection-*/complete.json")))
                if output_path.is_dir() else 0,
            "observed_local_receipts": len(list(receipt_path.glob("*.json"))) if receipt_path.is_dir() else 0,
            "required_score_checkpoints": 264, "required_selection_checkpoints": 2,
            "plan_sha256": state["plan_sha256"], "admission_sha256": state["admission_sha256"],
            "retention_spec_reviewed": reviewed_spec is not None}
    existing = []
    if reviewed_spec is not None:
        existing = [entry["object"] for entry in reviewed_spec.get("existing_objects", [])]
        for record in existing:
            _validate_record(record, record.get("uri"))
    object_records = [record for _, record in checked["objects"]]
    return {"schema": SCHEMA, "status": "READY_FOR_CONTENT_VERIFICATION" if reviewed_spec else "LOCAL_CHECKPOINTS_VERIFIED",
            "retirement_allowed": False, "score_checkpoints": checked["score_count"],
            "selection_checkpoints": checked["selection_count"], "evaluate_computed": False,
            "plan_sha256": state["plan_sha256"], "admission_sha256": state["admission_sha256"],
            "roster_sha256": checked["roster_sha256"], "source_contracts": checked["source_contracts"],
            "evaluation_contracts": checked["evaluation_contracts"],
            "checkpoint_objects": len(object_records),
            "checkpoint_bytes_to_reread": sum(r["bytes"] for r in object_records),
            "existing_objects_to_reread": len(existing), "checked": checked, "spec": reviewed_spec,
            "existing": existing}


def publish_final(audit_result, prefix, publication_receipts, cloud, publisher=useful.publish_file):
    require(audit_result["status"] == "READY_FOR_CONTENT_VERIFICATION", "Reviewed retention spec required")
    checked, spec = audit_result["checked"], audit_result["spec"]
    checkpoint_records, checkpoint_bytes = verify_content(
        [r for _, r in checked["objects"]], cloud)
    receipt_records, receipt_bytes = verify_remote_receipts(prefix, checked["receipts"], cloud)
    existing_records, existing_bytes = verify_content(audit_result["existing"], cloud)
    known = {(r["sha256"], r["bytes"]): r for r in checkpoint_records + receipt_records + existing_records}
    publication_receipts = Path(publication_receipts).resolve()
    publication_receipts.mkdir(parents=True, exist_ok=True, mode=0o700)
    inventory = []
    for entry in spec["entries"]:
        path = Path(entry["path"])
        if entry["disposition"] == "EXCLUDE":
            inventory.append({**entry})
            continue
        files = [path] if path.is_file() else [p for p in sorted(path.rglob("*")) if p.is_file()]
        require(files, "Preserved directory is empty: " + str(path))
        for source in files:
            require(not source.is_symlink(), "Symlink in preserved path")
            digest = file_digests(source)
            record = known.get((digest["sha256"], digest["bytes"]))
            if record is None:
                inside = source.name if path.is_file() else source.relative_to(path).as_posix()
                relative = "retention/objects/" + entry["label"] + "/" + inside
                record = publisher(source, relative, prefix.rstrip("/"), publication_receipts)
                record = {**record, **digest}
                known[(digest["sha256"], digest["bytes"])] = record
            inventory.append({"path": str(source), "disposition": "PRESERVE", "role": entry["role"],
                "reason": entry["reason"], "label": entry["label"], **digest, "object": record})
    for path, record in checked["objects"]:
        enriched = known[(record["sha256"], record["bytes"])]
        inventory.append({"path": str(path), "disposition": "PRESERVE", "role": "scientific_product",
            "reason": "Exact validated SCORE/SELECT checkpoint from the admitted 88-view plan.",
            **file_digests(path), "object": enriched})
    for (path, _), record in zip(checked["receipts"], receipt_records):
        inventory.append({"path": str(path), "disposition": "PRESERVE", "role": "provenance_log",
            "reason": "Checkpoint publication receipt re-read at its immutable generation.",
            **file_digests(path), "object": record})
    manifest = {"schema": retirement.SCHEMA, "authorization_reference": spec["authorization_reference"],
        "scientific_status": "COMPLETED", "pending_work": [], "resource_plan": spec["resource_plan"],
        "equivalent_read_only_sources": [], "inventory": {"roots": list(dict.fromkeys(
            spec["reviewed_roots"] + [checked["output_root"], checked["receipts_dir"]])),
            "entries": inventory, "outside_roots_reviewed": True, "unclassified_paths": [],
            "outside_roots_review": spec["outside_roots_review"]},
        "resume": spec["resume"], "completed_scope": "Exact 264 SCORE and two scope-wide SELECT checkpoints; EVALUATE remains closed."}
    retirement.validate_manifest(manifest)
    manifest_path = publication_receipts / "pair-retirement-manifest.json"
    require(not manifest_path.exists(), "Retention manifest already exists; inspect before retry")
    checkpoints.atomic_write_new(manifest_path, manifest)
    manifest_record = publisher(manifest_path, "retention/pair-retirement-manifest.json",
                                prefix.rstrip("/"), publication_receipts)
    complete = {"schema": SCHEMA, "status": "RETENTION_COMPLETE_AWAITING_SEPARATE_RETIREMENT_REVIEW",
        "retirement_allowed": False, "score_checkpoints": 264, "selection_checkpoints": 2,
        "evaluate_computed": False, "vm_id": "3706342769844118382", "disk_id": "8680383801666659206",
        "retirement_manifest": manifest_record, "remote_bytes_reread": checkpoint_bytes + receipt_bytes + existing_bytes,
        "no_resource_mutation": True, "no_gcs_deletion": True}
    complete_path = publication_receipts / "pair-retention-complete.json"
    checkpoints.atomic_write_new(complete_path, complete)
    complete_record = publisher(complete_path, "retention/complete.json", prefix.rstrip("/"), publication_receipts)
    return {**complete, "completion_record": complete_record}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "publish"), default="audit")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-receipts", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--retention-spec")
    parser.add_argument("--publication-receipts")
    args = parser.parse_args(argv)
    result = audit(args.plan, args.admission, args.output_root, args.checkpoint_receipts,
                   args.prefix, args.retention_spec)
    if args.mode == "publish":
        require(args.retention_spec and args.publication_receipts, "Publish needs reviewed spec and receipt directory")
        require(result.get("status") == "READY_FOR_CONTENT_VERIFICATION",
                "Incomplete or invalid campaign cannot publish final retention")
        spec = result["spec"]
        cloud = GcloudContent(spec["resource_plan"]["project"], spec["resource_plan"]["account"])
        result = publish_final(result, args.prefix, args.publication_receipts, cloud)
    else:
        result = {k: v for k, v in result.items() if k not in {"checked", "spec", "existing"}}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
