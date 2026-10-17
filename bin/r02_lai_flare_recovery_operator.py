#!/usr/bin/env python3
"""Recover one retained FLARE archive, project six views, and publish them safely.

The operator is intentionally one-archive-at-a-time.  It downloads generation-
pinned immutable inputs, extracts only the three preserved inference VCFs, runs
the six predictor-arm/direction projections with Nextflow maxForks=1, publishes
useful products create-only, verifies remote size/MD5/generation, and only then
removes the local active directory.  A failure keeps every local byte for
diagnosis and ``--resume``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


GIB = 2**30
MINIMUM_FREE_AFTER_DOWNLOAD = 10 * GIB
PRODUCT_STATUS = "EXISTING_FLARE_POSTERIOR_PROJECTED_NO_TRAINING_NO_SELECTION"
ALLOWED_PREFIX = ("gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/"
                  "presentacion/biologico/R02_20260930/")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CAMPAIGN_DEADLINE_EPOCH = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_timeout(requested):
    require(requested > 0, "positive timeout required")
    if CAMPAIGN_DEADLINE_EPOCH is None:
        return requested
    remaining = CAMPAIGN_DEADLINE_EPOCH - time.time()
    require(remaining > 1, "48-hour campaign deadline exhausted")
    return min(requested, remaining)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_descriptor(path):
    path = Path(path)
    md5 = hashlib.md5(usedforsecurity=False)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(block); digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest(),
            "md5_base64": base64.b64encode(md5.digest()).decode()}


def run_checked(command, *, timeout, env=None, capture=True):
    result = subprocess.run(command, check=False, timeout=bounded_timeout(timeout), env=env,
                            text=True, capture_output=capture)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command,
                                            output=result.stdout if capture else None,
                                            stderr=result.stderr if capture else None)
    return result


def run_supervised(command, *, timeout, env, cwd):
    """Run a long tool in its own process group and stop the whole group on timeout."""
    process = subprocess.Popen(command, env=env, cwd=cwd, start_new_session=True)
    try:
        returncode = process.wait(timeout=bounded_timeout(timeout))
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def validate_cloud_record(record):
    require(set(("uri", "generation", "bytes", "sha256", "md5_base64")) <= set(record),
            "incomplete cloud record")
    require(record["uri"].startswith(ALLOWED_PREFIX) and "#" not in record["uri"],
            "source outside private R02 datalake")
    require(str(record["generation"]).isdigit() and int(record["bytes"]) > 0,
            "invalid generation or byte count")
    require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
            "invalid SHA-256")


def download_pinned(record, destination, runner=run_checked):
    validate_cloud_record(record); destination = Path(destination)
    if destination.exists():
        require(destination.is_file(), "download destination is not a file")
        require(destination.stat().st_size == int(record["bytes"]) and
                sha256(destination) == record["sha256"], "existing download differs")
        return "REUSED_VERIFIED_LOCAL"
    require(destination.parent.is_dir(), "download parent absent")
    runner(["gcloud", "storage", "cp", "--quiet",
            record["uri"] + "#" + str(record["generation"]), str(destination)], timeout=14400)
    require(destination.stat().st_size == int(record["bytes"]) and
            sha256(destination) == record["sha256"], "generation-pinned download differs")
    return "DOWNLOADED_GENERATION_PINNED"


def remote_metadata(uri, runner=run_checked):
    result = runner(["gcloud", "storage", "objects", "describe", uri, "--format=json"],
                    timeout=120)
    value = json.loads(result.stdout)
    require(isinstance(value, dict), "invalid remote metadata")
    return value


def publish_file(path, uri, runner=run_checked):
    path = Path(path); before = local_descriptor(path)
    require(uri.startswith(ALLOWED_PREFIX) and not any(x in uri for x in "#?\n\r"),
            "publication URI outside private R02 datalake")
    command = ["gcloud", "storage", "cp", "--quiet", "--if-generation-match=0",
               "--content-md5=" + before["md5_base64"], str(path), uri]
    result = subprocess.run(command, check=False, timeout=bounded_timeout(7200), text=True, capture_output=True,
                            env={**os.environ,
                                 "CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED": "false"})
    reused = result.returncode != 0
    if reused and "HTTPError 412:" not in (result.stderr or ""):
        raise subprocess.CalledProcessError(result.returncode, command,
                                            output=result.stdout, stderr=result.stderr)
    metadata = remote_metadata(uri, runner)
    remote_md5 = metadata.get("md5_hash", metadata.get("md5Hash"))
    require(int(metadata.get("size", -1)) == before["bytes"] and
            remote_md5 == before["md5_base64"] and str(metadata.get("generation", "")).isdigit(),
            "published object differs by size/MD5 or lacks generation")
    require(local_descriptor(path) == before, "local source changed during publication")
    return {"uri": uri, "generation": str(metadata["generation"]), **before,
            "operation": "REUSED_EQUIVALENT_HTTP_412" if reused else "CREATED_GENERATION_0"}


def verify_published_file(path, uri, expected_generation=None, runner=run_checked):
    local = local_descriptor(path); metadata = remote_metadata(uri, runner)
    remote_md5 = metadata.get("md5_hash", metadata.get("md5Hash"))
    generation = str(metadata.get("generation", ""))
    require(int(metadata.get("size", -1)) == local["bytes"] and
            remote_md5 == local["md5_base64"] and generation.isdigit(),
            "remote object differs from local verified file")
    if expected_generation is not None:
        require(generation == str(expected_generation), "remote generation changed")
    return {"uri":uri, "generation":generation, **local, "operation":"REUSED_VERIFIED_REMOTE"}


def select_source(manifest, source_id):
    require(manifest.get("schema") == "r02_lai_flare_recovery_manifest_v1" and
            manifest.get("status") == "READY_FOR_GENERATION_PINNED_READ_ONLY_RECOVERY",
            "recovery manifest not admitted")
    require(SAFE_ID.fullmatch(source_id) is not None, "unsafe source-id")
    rows = [row for row in manifest.get("tasks", []) if row.get("source_id") == source_id]
    require(len(rows) == 1, "source-id must resolve exactly once")
    source = rows[0]
    require(len(source.get("required_members", [])) == 3 and
            len(source.get("projection_targets", [])) == 2 and
            source.get("selection_status") == "PRESERVED_SENSITIVITY_NOT_SELECTED",
            "source does not preserve the three-arm/two-direction sensitivity")
    return source


def validate_extraction(folder, source):
    receipt = read_json(Path(folder) / "receipt.json")
    require(receipt.get("status") == "REQUESTED_MEMBERS_VERIFIED_AND_EXTRACTED",
            "extraction not verified")
    rows = {row["name"]: row for row in receipt.get("members", [])}
    require(set(rows) == {row["member"] for row in source["required_members"]},
            "extracted member set differs")
    for row in rows.values():
        path = Path(folder) / row["output"]
        require(path.is_file() and sha256(path) == row["sha256"], "extracted member differs")


def validate_batch(folder):
    manifest = read_json(Path(folder) / "REQUEST_MANIFEST.json")
    require(manifest.get("status") == "READY" and manifest.get("task_count") == 6 and
            manifest.get("predictor_source_count") == 3 and
            manifest.get("selection_performed") is False, "six-view request batch invalid")
    for task in manifest["tasks"]:
        require(SAFE_ID.fullmatch(task["task_id"]) is not None, "unsafe projection task-id")
        request = Path(task["request"]["path"])
        require(request.is_file() and sha256(request) == task["request"]["sha256"],
                "projection request differs")
    return manifest


def verified_products(output, batch):
    products = []
    for task in batch["tasks"]:
        folder = Path(output) / task["task_id"] / "projection"
        probability, receipt_path = folder / "baseline-probabilities.npy", folder / "receipt.json"
        receipt = read_json(receipt_path)
        require(receipt.get("status") == PRODUCT_STATUS and
                receipt.get("selection_performed") is False and
                receipt.get("truth_arrays_loaded") is False and
                receipt.get("truth_used_for_projection") is False,
                "projection receipt permits unsupported use")
        require(probability.is_file() and sha256(probability) == receipt["probabilities_sha256"],
                "projected probabilities differ from receipt")
        products.extend(((task["task_id"], probability), (task["task_id"], receipt_path)))
    return products


def cleanup_after_verified_publication(active, completion_proof):
    require(completion_proof.get("uri") and completion_proof.get("generation") and
            completion_proof.get("sha256"), "verified completion publication required for cleanup")
    shutil.rmtree(active)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--campaign-deadline-epoch", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    global CAMPAIGN_DEADLINE_EPOCH
    CAMPAIGN_DEADLINE_EPOCH = args.campaign_deadline_epoch
    require(60 < CAMPAIGN_DEADLINE_EPOCH-time.time() <= 48*3600+300,
            "campaign deadline must be active and at most 48 hours")
    require(args.output_prefix.startswith(ALLOWED_PREFIX) and args.output_prefix.startswith("gs://") and
            not any(x in args.output_prefix for x in "#?\n\r"), "private R02 output-prefix required")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", args.container) is not None or
            re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", args.container) is not None,
            "immutable container digest required")
    repo = args.repo_root.resolve(); scratch = args.scratch_root.resolve()
    require(repo.is_dir() and scratch.is_dir() and not scratch.is_symlink(), "existing canonical roots required")
    manifest_path = args.recovery_manifest.resolve(); recovery = read_json(manifest_path)
    source = select_source(recovery, args.source_id)
    active = scratch / "active" / args.source_id
    evidence = scratch / "receipts"; evidence.mkdir(parents=True, exist_ok=True, mode=0o700)
    prefix = args.output_prefix.rstrip("/") + "/sources/" + args.source_id
    completion_path = evidence / (args.source_id + ".completion.json")
    local_done = evidence / (args.source_id + ".publication.json")
    if local_done.exists():
        value = read_json(local_done); proof = value.get("completion", {})
        require(not active.exists(), "completed source still has active scratch")
        verify_published_file(completion_path, proof["uri"], proof["generation"])
        print(json.dumps(value, sort_keys=True)); return 0
    # Recover a crash after the remote completion marker but before local cleanup.
    if completion_path.exists():
        try:
            proof = verify_published_file(completion_path, prefix + "/completion.json")
        except subprocess.CalledProcessError:
            proof = None
        if proof is not None:
            if active.exists(): cleanup_after_verified_publication(active, proof)
            value = {"schema":"r02_lai_flare_recovery_publication_v1",
                     "status":"REMOTE_COMPLETION_VERIFIED", "completion":proof,
                     "active_scratch_removed":True}
            with local_done.open("x") as stream:
                json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")
            print(json.dumps(value, sort_keys=True)); return 0
    if active.exists(): require(args.resume, "active source exists; inspect it and use --resume")
    else: active.mkdir(parents=True, mode=0o700)
    archive_path, inventory_path = active / "models.tar", active / "models.inventory.json"
    required_free = int(source["archive"]["bytes"]) + int(source["archive_inventory"]["bytes"]) + MINIMUM_FREE_AFTER_DOWNLOAD
    require(shutil.disk_usage(active).free >= required_free, "less than archive plus 10 GiB reserve")
    download_pinned(source["archive"], archive_path)
    download_pinned(source["archive_inventory"], inventory_path)

    extracted = active / "recovered" / args.source_id
    if not extracted.exists():
        extracted.parent.mkdir(mode=0o700)
        command = [sys.executable, str(repo / "bin/r02_lai_archive_extract.py"),
                   "--archive", str(archive_path), "--archive-sha256", source["archive"]["sha256"],
                   "--inventory", str(inventory_path), "--inventory-sha256", source["archive_inventory"]["sha256"],
                   "--output-dir", str(extracted)]
        for member in source["required_members"]: command += ["--member", member["member"]]
        run_checked(command, timeout=14400, capture=False)
    validate_extraction(extracted, source)

    requests = active / "projection-requests"
    if not requests.exists():
        run_checked([sys.executable, str(manifest_path.parent / "PREPARE_PROJECTION_REQUESTS.py"),
                     "--recovered-root", str(active / "recovered"), "--source-id", args.source_id,
                     "--recovery-manifest", str(manifest_path),
                     "--output-dir", str(requests)], timeout=600, capture=False)
    batch = validate_batch(requests)
    output = active / "projection-output"; work = active / "nf-work"
    command = ["nextflow", "-log", str(active / "nextflow.log"),
               "-C", str(repo / "conf/r02_lai_flare_query_projection.config"),
               "run", str(repo / "workflows/r02_lai_flare_query_projection.nf"),
               "--request_manifest", str(requests / "REQUEST_MANIFEST.json"),
               "--output_dir", str(output), "--source_bin", str(repo / "bin"),
               "--container", args.container, "--max_forks", "1", "-work-dir", str(work)]
    if args.resume: command.append("-resume")
    nextflow_home = Path(os.environ.get("NXF_HOME", str(Path.home()/".nextflow"))).resolve()
    require((nextflow_home/"framework/26.04.6/nextflow-26.04.6-one.jar").is_file(),
            "verified local Nextflow 26.04.6 cache absent")
    env = {**os.environ, "NXF_HOME": str(nextflow_home), "NXF_VER":"26.04.6",
           "NXF_OFFLINE":"true", "NXF_DISABLE_CHECK_LATEST":"true",
           "R02_LAI_UID":str(os.getuid()), "R02_LAI_GID":str(os.getgid()),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    run_supervised(command, timeout=48 * 3600, env=env, cwd=active)
    products = verified_products(output, batch)

    records = []
    for task_id, path in products:
        records.append({k:v for k,v in publish_file(
            path, prefix + "/products/" + task_id + "/" + path.name).items() if k != "operation"})
    for task in batch["tasks"]:
        path = Path(task["request"]["path"])
        records.append({k:v for k,v in publish_file(
            path, prefix + "/requests/" + path.name).items() if k != "operation"})
    records.append({k:v for k,v in publish_file(
        requests / "REQUEST_MANIFEST.json", prefix + "/requests/REQUEST_MANIFEST.json").items()
        if k != "operation"})
    provenance_paths = [repo / "bin/r02_lai_archive_extract.py", repo / "bin/r02_lai_flare_query_projection.py",
                        repo / "bin/r02_lai_flare_recovery_operator.py",
                        repo / "workflows/r02_lai_flare_query_projection.nf",
                        repo / "conf/r02_lai_flare_query_projection.config",
                        manifest_path.parent / "PREPARE_PROJECTION_REQUESTS.py", manifest_path]
    for path in provenance_paths:
        records.append({k:v for k,v in publish_file(
            path, prefix + "/provenance/" + path.name).items() if k != "operation"})
    completion = {"schema": "r02_lai_flare_recovery_source_completion_v1",
        "status": "SIX_SENSITIVITY_PROJECTIONS_PUBLISHED_AND_VERIFIED",
        "source_id": args.source_id, "archive": source["archive"],
        "archive_inventory": source["archive_inventory"], "required_members": source["required_members"],
        "projection_count": 6, "predictor_arms": ["FILTERED.BASELINE", "ALL.BASELINE", "ALL.FULL"],
        "directions": ["FOLD_A_TO_FOLD_B", "FOLD_B_TO_FOLD_A"],
        "container": args.container, "max_forks": 1, "cpus_per_task": 1,
        "memory_bytes_per_task": 4 * GIB, "minimum_free_bytes_reserved": MINIMUM_FREE_AFTER_DOWNLOAD,
        "objects": sorted(records, key=lambda row: row["uri"]),
        "training_started": False, "selection_performed": False, "evaluate_opened": False,
        "source_gcs_deleted_or_overwritten": False,
        "scientific_limit": "Projection prepares truth-blind neural inputs; it does not evaluate or select FLARE."}
    if completion_path.exists(): require(read_json(completion_path) == completion, "existing completion differs")
    else:
        with completion_path.open("x") as stream: json.dump(completion, stream, indent=2, sort_keys=True); stream.write("\n")
    proof = publish_file(completion_path, prefix + "/completion.json")
    cleanup_after_verified_publication(active, proof)
    value = {"schema":"r02_lai_flare_recovery_publication_v1", "status":"REMOTE_COMPLETION_VERIFIED",
             "completion":proof, "active_scratch_removed":True}
    require(not local_done.exists(), "local publication proof already exists")
    with local_done.open("x") as stream: json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
