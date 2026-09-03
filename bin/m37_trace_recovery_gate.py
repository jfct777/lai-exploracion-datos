#!/usr/bin/env python3
"""Authenticate orphaned M37 bundles as non-consumable recovery evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Iterable


ARMS = ("RE", "RD", "POOLED", "SHAM", "GEOMETRY")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
AUDIT_BASENAME = "m37.recovered.nonconsumable.audit.json"
SUMMARY_BASENAME = "m37.recovered.nonconsumable.summary.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def _metadata_token() -> str:
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/"
        "service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)
    token = str(payload.get("access_token", ""))
    require(bool(token), "Google metadata service returned no access token")
    return token


def fetch_job(project: str, location: str, job_id: str) -> dict[str, Any]:
    """Read one live Batch descriptor through ADC available on the recovery VM."""
    url = (
        "https://batch.googleapis.com/v1/projects/"
        f"{project}/locations/{location}/jobs/{job_id}"
    )
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {_metadata_token()}"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    require(isinstance(payload, dict), f"Batch descriptor for {job_id} differs")
    return payload


def validate_job(job: dict[str, Any], expected: dict[str, Any], cloud: dict[str, Any]) -> dict[str, Any]:
    job_id = str(expected["job_id"])
    expected_name = f"projects/{cloud['project']}/locations/{cloud['location']}/jobs/{job_id}"
    require(job.get("name") == expected_name, f"Batch identity differs for {job_id}")
    status = job.get("status")
    require(isinstance(status, dict) and status.get("state") == "SUCCEEDED",
            f"Batch job {job_id} is not SUCCEEDED")
    counts = status.get("taskGroups", {}).get("group0", {}).get("counts", {})
    require(counts == {"SUCCEEDED": "1"}, f"Batch task counts differ for {job_id}")
    required_labels = cloud["required_labels"]
    require(job.get("labels") == required_labels,
            f"top-level Batch labels differ for {job_id}")
    allocation = job.get("allocationPolicy")
    require(isinstance(allocation, dict), f"Batch allocation policy missing for {job_id}")
    allocation_labels = allocation.get("labels", {})
    require(all(allocation_labels.get(key) == value for key, value in required_labels.items()),
            f"allocation Batch labels differ for {job_id}")
    groups = job.get("taskGroups")
    require(isinstance(groups, list) and len(groups) == 1,
            f"Batch task group count differs for {job_id}")
    runnables = groups[0].get("taskSpec", {}).get("runnables", [])
    require(isinstance(runnables, list) and len(runnables) == 1,
            f"Batch runnable count differs for {job_id}")
    container = runnables[0].get("container", {})
    require(container.get("imageUri") == cloud["container_digest"],
            f"Batch container digest differs for {job_id}")
    commands = container.get("commands")
    require(isinstance(commands, list) and commands,
            f"Batch runnable command missing for {job_id}")
    mounted_work_dir = expected["work_dir"].replace("gs://", "/mnt/disks/", 1)
    require(any(mounted_work_dir in str(command) for command in commands),
            f"Batch runnable does not bind expected work directory for {job_id}")
    return {
        "job_id": job_id,
        "state": "SUCCEEDED",
        "update_time": status.get("statusEvents", [])[-1].get("eventTime"),
        "descriptor_sha256": canonical_sha256(job),
        "work_dir": expected["work_dir"],
        "labels": required_labels,
        "container_digest": cloud["container_digest"],
    }


def validate_source_archive(path: Path, expected: dict[str, str]) -> dict[str, str]:
    require(expected and all(SHA256_RE.fullmatch(value) for value in expected.values()),
            "frozen downstream source contract differs")
    observed: dict[str, str] = {}
    with tarfile.open(path, mode="r:*") as archive:
        all_members = archive.getmembers()
        require(all((member.isfile() or member.isdir()) and
                    not member.issym() and not member.islnk() and
                    not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
                    for member in all_members),
                "frozen downstream source archive contains an unsafe member")
        members = [member for member in all_members if member.isfile()]
        names = [member.name.removeprefix("./") for member in members]
        require(len(names) == len(set(names)) and set(names) == set(expected),
                "frozen downstream source archive member set differs")
        for member, name in zip(members, names):
            handle = archive.extractfile(member)
            require(handle is not None, f"cannot read frozen source {name}")
            observed[name] = hashlib.sha256(handle.read()).hexdigest()
    require(observed == expected, "frozen downstream source archive hash set differs")
    return observed


def _index_by_family(paths: Iterable[Path], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        family = path.name.split(".", 1)[0]
        require(family in {"hmm", "tcn"} and family not in result,
                f"{label} family identity differs")
        result[family] = path
    require(set(result) == {"hmm", "tcn"}, f"{label} family set differs")
    return result


def validate_positive_control(payload_path: Path, receipt_path: Path,
                              contract: dict[str, Any]) -> dict[str, str]:
    expected = contract["positive_control"]
    require(sha256(payload_path) == expected["payload_sha256"] and
            sha256(receipt_path) == expected["receipt_sha256"],
            "positive-control frozen hashes differ")
    payload, receipt = load_json(payload_path), load_json(receipt_path)
    require(payload.get("stage") == "M37_TRACE_COMPACT_POSITIVE_CONTROL" and
            payload.get("run_id") == contract["origin_run_id"] and
            payload.get("container_digest") == contract["cloud"]["container_digest"],
            "positive-control identity differs")
    require(receipt.get("stage") == "M37_TRACE_COMPACT_POSITIVE_CONTROL" and
            receipt.get("run_id") == contract["origin_run_id"] and
            receipt.get("output_sha256") == sha256(payload_path),
            "positive-control receipt differs")
    return {payload_path.name: sha256(payload_path), receipt_path.name: sha256(receipt_path)}


def validate_family_bundle(
    family: str,
    metric_paths: list[Path],
    receipt_paths: list[Path],
    audit_path: Path,
    audit_receipt_path: Path,
    equivalence_path: Path,
    equivalence_receipt_path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    expected = contract["jobs"][family]
    run_id, root = contract["origin_run_id"], contract["root"]
    require(len(metric_paths) == len(receipt_paths) == int(expected["metric_count"]),
            f"{family} metric bundle count differs")
    require(len({path.name for path in metric_paths}) == len(metric_paths) and
            len({path.name for path in receipt_paths}) == len(receipt_paths),
            f"{family} metric bundle basenames are not unique")
    receipt_index: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for path in receipt_paths:
        receipt = load_json(path)
        key = (str(receipt.get("candidate_id", "")), str(receipt.get("arm", "")))
        require(receipt.get("stage") == "M37_TRACE_SCORE" and
                receipt.get("family") == family and receipt.get("root") == root and
                key[1] in ARMS and key not in receipt_index,
                f"{family} metric receipt identity differs")
        receipt_index[key] = (path, receipt)
    candidates: dict[str, set[str]] = {}
    metric_sha: dict[str, str] = {}
    metric_receipt_sha: dict[str, str] = {path.name: sha256(path) for path in receipt_paths}
    used: set[Path] = set()
    for path in metric_paths:
        metric = load_json(path)
        key = (str(metric.get("candidate_id", "")), str(metric.get("arm", "")))
        require(metric.get("family") == family and metric.get("root") == root and
                metric.get("run_id") == run_id and
                metric.get("evaluation_split") == contract["evaluation_split"] and
                key in receipt_index,
                f"{family} metric identity differs")
        receipt_path, receipt = receipt_index[key]
        require(receipt.get("output_sha256") == sha256(path),
                f"{family} metric hash differs from score receipt")
        candidates.setdefault(key[0], set()).add(key[1])
        metric_sha[path.name] = sha256(path)
        used.add(receipt_path)
    require(used == set(receipt_paths) and
            len(candidates) == int(expected["candidate_count"]) and
            all(arms == set(ARMS) for arms in candidates.values()),
            f"{family} paired candidate set differs")

    audit, audit_receipt = load_json(audit_path), load_json(audit_receipt_path)
    require(audit.get("stage") == "M37_TRACE_COMPACT_SWEEP" and
            audit.get("status") == "PASS_FIT_TUNE_ONLY" and
            audit.get("run_id") == run_id and audit.get("root") == root and
            audit.get("family") == family and
            audit.get("candidate_count") == int(expected["candidate_count"]) and
            audit.get("metric_count") == int(expected["metric_count"]) and
            audit.get("container_digest") == contract["cloud"]["container_digest"],
            f"{family} family audit identity differs")
    require(audit.get("metric_sha256") == metric_sha and
            audit.get("metric_receipt_sha256") == metric_receipt_sha,
            f"{family} family audit metric evidence differs")
    require(audit.get("authenticated_source_sha256") == contract["executed_source_sha256"],
            f"{family} executed source hash set differs")
    audit_field = {
        "candidate_manifest": "manifest_sha256",
        "parent_contract": "parent_contract_sha256",
        "contract_amendment": "contract_amendment_sha256",
        "canonical_metrics": "canonical_metrics_sha256",
        "canonical_metrics_receipt": "canonical_metrics_receipt_sha256",
        "truth": "truth_sha256",
        "run_overlay": None,
    }
    require(set(contract["frozen_input_sha256"]) == set(audit_field),
            f"{family} frozen input contract field set differs")
    for key, expected_sha in contract["frozen_input_sha256"].items():
        audit_key = audit_field[key]
        observed = audit.get(audit_key) if audit_key else audit.get("run_overlay", {}).get("sha256")
        require(observed == expected_sha, f"{family} frozen input {key} differs")
    require(audit.get("positive_control_sha256") ==
            contract["positive_control"]["payload_sha256"] and
            audit.get("positive_control_receipt_sha256") ==
            contract["positive_control"]["receipt_sha256"],
            f"{family} positive-control binding differs")
    require(audit_receipt.get("stage") == "M37_TRACE_COMPACT_SWEEP" and
            audit_receipt.get("run_id") == run_id and
            audit_receipt.get("root") == root and audit_receipt.get("family") == family and
            audit_receipt.get("output_sha256") == sha256(audit_path),
            f"{family} family audit receipt differs")

    equivalence, equivalence_receipt = load_json(equivalence_path), load_json(equivalence_receipt_path)
    require(equivalence.get("stage") == "M37_TRACE_COMPACT_EQUIVALENCE" and
            equivalence.get("status") == "PASS" and equivalence.get("run_id") == run_id and
            equivalence.get("root") == root and equivalence.get("family") == family,
            f"{family} equivalence identity differs")
    require(equivalence_receipt.get("stage") == "M37_TRACE_COMPACT_EQUIVALENCE" and
            equivalence_receipt.get("run_id") == run_id and
            equivalence_receipt.get("root") == root and
            equivalence_receipt.get("family") == family and
            equivalence_receipt.get("output_sha256") == sha256(equivalence_path),
            f"{family} equivalence receipt differs")
    require(audit.get("equivalence_sha256") == sha256(equivalence_path) and
            audit.get("equivalence_receipt_sha256") == sha256(equivalence_receipt_path),
            f"{family} family audit/equivalence binding differs")
    return {
        "candidate_count": len(candidates),
        "metric_count": len(metric_paths),
        "candidate_ids": sorted(candidates),
        "metric_sha256": dict(sorted(metric_sha.items())),
        "metric_receipt_sha256": dict(sorted(metric_receipt_sha.items())),
        "family_audit_sha256": sha256(audit_path),
        "family_audit_receipt_sha256": sha256(audit_receipt_path),
        "equivalence_sha256": sha256(equivalence_path),
        "equivalence_receipt_sha256": sha256(equivalence_receipt_path),
    }


def validate_contract(contract: dict[str, Any]) -> None:
    require(contract.get("schema_version") == "1.1.0" and
            contract.get("stage") == "M37_TRACE_ORPHAN_RECOVERY_CONTRACT" and
            contract.get("status") == "FROZEN" and contract.get("recovery_id") and
            contract.get("root") == "R0" and
            contract.get("evaluation_split") == "FIT_TUNE",
            "M37 recovery contract identity differs")
    policy = contract.get("recovery_policy", {})
    require(policy == {
        "rerun_family_jobs": False,
        "publish_raw_family_outputs": False,
        "publish_predictions_or_checkpoints": False,
        "open_valid_or_test": False,
        "consumable_for_candidate_selection_or_inference": False,
        "run_standard_decision_or_promotion": False,
        "publish_only_recovered_nonconsumable_audit_and_summary": True,
        "disposition": "RECOVER_DOWNSTREAM_AUDIT_ONLY_AFTER_SIGHUP",
    }, "M37 recovery policy differs")
    require(set(contract.get("jobs", {})) == {"hmm", "tcn"},
            "M37 recovery job family set differs")
    for family, job in contract["jobs"].items():
        require(job.get("work_dir", "").startswith(
                    "gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/"),
                f"{family} recovery work directory is outside the personal bucket")


def write_nonconsumable_outputs(
    output_dir: Path,
    contract_path: Path,
    source_archive: Path,
    source_evidence: dict[str, str],
    positive_evidence: dict[str, str],
    job_evidence: dict[str, dict[str, Any]],
    family_evidence: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    """Write an audit plus inventory summary that cannot act as a promotion input."""
    require(output_dir.is_dir(), "M37 recovery output directory does not exist")
    audit_path = output_dir / AUDIT_BASENAME
    summary_path = output_dir / SUMMARY_BASENAME
    audit_receipt_path = audit_path.with_suffix(".receipt.json")
    summary_receipt_path = summary_path.with_suffix(".receipt.json")
    outputs = (audit_path, audit_receipt_path, summary_path, summary_receipt_path)
    require(not any(path.exists() for path in outputs),
            "refusing to overwrite M37 non-consumable recovery evidence")

    common = {
        "recovery_id": contract["recovery_id"],
        "origin_run_id": contract["origin_run_id"],
        "root": contract["root"],
        "evaluation_split": contract["evaluation_split"],
        "consumable": False,
        "result_use": "DESCRIPTIVE_RECOVERY_AUDIT_ONLY",
        "candidate_selection": "FORBIDDEN",
        "model_promotion": "FORBIDDEN",
        "scientific_inference": "FORBIDDEN",
    }
    audit = {
        "schema_version": "1.1.0",
        "stage": "M37_TRACE_ORPHAN_RECOVERY_AUDIT",
        "status": "RECOVERED_NONCONSUMABLE",
        **common,
        "contract_sha256": sha256(contract_path),
        "source_commit": contract["source_commit"],
        "source_archive_sha256": sha256(source_archive),
        "downstream_source_sha256": source_evidence,
        "positive_control_sha256": positive_evidence,
        "jobs": job_evidence,
        "family_bundles": family_evidence,
        "recovery_policy": contract["recovery_policy"],
        "authenticated_source_sha256": {
            Path(__file__).name: sha256(Path(__file__)),
        },
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit_receipt = {
        "schema_version": "1.1.0",
        "stage": "M37_TRACE_ORPHAN_RECOVERY_AUDIT",
        "status": audit["status"],
        **common,
        "contract_sha256": sha256(contract_path),
        "output_sha256": sha256(audit_path),
    }
    audit_receipt_path.write_text(
        json.dumps(audit_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = {
        "schema_version": "1.1.0",
        "stage": "M37_TRACE_ORPHAN_RECOVERY_SUMMARY",
        "status": "RECOVERED_NONCONSUMABLE",
        **common,
        "standard_decision_executed": False,
        "published_payload": "AUTHENTICATED_INVENTORY_WITHOUT_MODEL_METRICS",
        "audit_sha256": sha256(audit_path),
        "audit_receipt_sha256": sha256(audit_receipt_path),
        "families": {
            family: {
                "job_state": job_evidence[family]["state"],
                "candidate_count": evidence["candidate_count"],
                "metric_count": evidence["metric_count"],
                "candidate_ids": evidence["candidate_ids"],
            }
            for family, evidence in sorted(family_evidence.items())
        },
        "total_candidate_count": sum(
            evidence["candidate_count"] for evidence in family_evidence.values()
        ),
        "total_metric_count": sum(
            evidence["metric_count"] for evidence in family_evidence.values()
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_receipt = {
        "schema_version": "1.1.0",
        "stage": "M37_TRACE_ORPHAN_RECOVERY_SUMMARY",
        "status": summary["status"],
        **common,
        "audit_sha256": sha256(audit_path),
        "audit_receipt_sha256": sha256(audit_receipt_path),
        "output_sha256": sha256(summary_path),
    }
    summary_receipt_path.write_text(
        json.dumps(summary_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit_path, audit_receipt_path, summary_path, summary_receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--positive-control", type=Path, required=True)
    parser.add_argument("--positive-control-receipt", type=Path, required=True)
    parser.add_argument("--metric", action="append", type=Path, required=True)
    parser.add_argument("--metric-receipt", action="append", type=Path, required=True)
    parser.add_argument("--family-audit", action="append", type=Path, required=True)
    parser.add_argument("--family-audit-receipt", action="append", type=Path, required=True)
    parser.add_argument("--equivalence", action="append", type=Path, required=True)
    parser.add_argument("--equivalence-receipt", action="append", type=Path, required=True)
    parser.add_argument(
        "--job-snapshot", action="append", type=Path,
        help="Offline test input; exactly two descriptors in hmm,tcn order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = load_json(args.contract)
    validate_contract(contract)
    source_evidence = validate_source_archive(
        args.source_archive, contract["downstream_source_sha256"]
    )
    positive_evidence = validate_positive_control(
        args.positive_control, args.positive_control_receipt, contract
    )
    if args.job_snapshot:
        require(len(args.job_snapshot) == 2,
                "offline recovery needs exactly two job snapshots")
        jobs = {family: load_json(path) for family, path in
                zip(("hmm", "tcn"), args.job_snapshot)}
    else:
        cloud = contract["cloud"]
        jobs = {
            family: fetch_job(cloud["project"], cloud["location"], spec["job_id"])
            for family, spec in contract["jobs"].items()
        }
    job_evidence = {
        family: validate_job(jobs[family], contract["jobs"][family], contract["cloud"])
        for family in ("hmm", "tcn")
    }
    audits = _index_by_family(args.family_audit, "family audit")
    audit_receipts = _index_by_family(args.family_audit_receipt, "family audit receipt")
    equivalences = _index_by_family(args.equivalence, "equivalence")
    equivalence_receipts = _index_by_family(args.equivalence_receipt, "equivalence receipt")
    family_evidence = {}
    for family in ("hmm", "tcn"):
        family_metrics = [path for path in args.metric if f".{family}." in path.name]
        family_receipts = [path for path in args.metric_receipt if f".{family}." in path.name]
        family_evidence[family] = validate_family_bundle(
            family, family_metrics, family_receipts,
            audits[family], audit_receipts[family],
            equivalences[family], equivalence_receipts[family], contract,
        )
    require(not (set(family_evidence["hmm"]["candidate_ids"]) &
                 set(family_evidence["tcn"]["candidate_ids"])),
            "M37 recovery candidate identifiers overlap between families")
    write_nonconsumable_outputs(
        args.output_dir,
        args.contract,
        args.source_archive,
        source_evidence,
        positive_evidence,
        job_evidence,
        family_evidence,
        contract,
    )
    print(json.dumps({
        "status": "RECOVERED_NONCONSUMABLE",
        "metrics": sum(row["metric_count"] for row in family_evidence.values()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
