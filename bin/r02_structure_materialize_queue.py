#!/usr/bin/env python3
"""Run chromosome-local materialization with verified, recoverable publication.

This controller stages authenticated inputs outside scientific containers. It
does not choose markers, change roles, fit models, or interpret genetic results.
The scientific work and its cache belong to the materialization Nextflow entry.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import r02_autosome_pipeline as pipeline
import r02_publish_evidence as publication
import r02_region_support_queue as staging

require, sha, read, save = staging.require, staging.sha, staging.read, staging.save_once
SCHEMA = "r02_structure_materialize_queue_v1"
GIB = 2**30


def validate(root, launch_path, expected):
    require(root.is_absolute() and root.resolve() == root, "Canonical absolute root required")
    require(sha(launch_path) == expected, "Launch hash differs")
    job = read(launch_path)
    require(job["schema"] == SCHEMA and job["root"] == str(root), "Queue identity differs")
    require(re.fullmatch(r"[a-z0-9-]+", job["worker"]), "Invalid worker")
    require(job["output_prefix"].startswith(publication.PREFIX + "r02-structure-materialize-"), "Private output scope differs")
    require(job["training_allowed"] is False and job["phase"] == "extract", "Extraction only")
    require(0 < job["max_runtime_seconds"] <= 43200 and job["min_free_bytes"] >= 20*GIB, "Unsafe runtime/disk limits")
    require(job["intermediate_budget_bytes"] >= 40*GIB, "Insufficient intermediate admission")
    for key in ("task_cpus", "task_memory_gib", "max_forks", "vm_cpus", "vm_memory_gib", "memory_reserve_gib"):
        require(type(job[key]) is int and job[key] > 0, "Positive integer resource required: "+key)
    require(job["task_cpus"]*job["max_forks"] <= job["vm_cpus"], "Concurrent tasks exceed VM CPU allocation")
    require(job["memory_reserve_gib"] >= 2 and
            job["task_memory_gib"]*job["max_forks"] + job["memory_reserve_gib"] <= job["vm_memory_gib"],
            "Concurrent tasks leave insufficient VM/controller memory")
    chromosomes = job["chromosomes"]
    require(chromosomes and len(chromosomes) == len(set(chromosomes)) and
            set(chromosomes) <= {str(i) for i in range(1, 23)}, "Invalid chromosome ownership")
    for name, digest in job["files_sha256"].items():
        require(sha(root/publication.safe_relative(name)) == digest, "Frozen bundle changed: "+name)
    contract = read(root/"bundle/materialize.json")
    require(sha(root/"bundle/materialize.json") == job["contract_sha256"], "Data contract differs")
    require(contract["target_selection"] == "AUTHENTICATED_FIT_CACHE_REQUIRED", "Do not repeat FIT target selection")
    for chrom in chromosomes:
        for kind, record in job["sources"][chrom].items():
            source = contract["sources"][chrom][kind]
            require(source["sha256"] == record["sha256"] and
                    Path(source["path"]).parent == root/"sources"/f"chr{int(chrom):02d}", "Staging source differs")
    job["launch_sha256"] = expected
    return job


def fetch_receipt(uri, allow_missing=False):
    response = subprocess.run(["gcloud", "storage", "objects", "describe", uri, "--format=json"],
                              capture_output=True, text=True, timeout=120)
    if response.returncode:
        if allow_missing and staging.missing_gcs_object(response, uri):
            return None
        raise RuntimeError("Receipt unavailable, not confirmed absent: "+response.stderr[-800:])
    meta = json.loads(response.stdout)
    require(0 < int(meta["size"]) <= 8*2**20, "Receipt exceeds bound")
    payload = subprocess.check_output(["gcloud", "storage", "cat", uri+"#"+str(meta["generation"])], timeout=120)
    require(len(payload) == int(meta["size"]), "Receipt size differs")
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def fit_cache(root, job, chrom, folder, started):
    """Freeze the first complete upstream receipt; the producer checks science."""
    destination = folder/"fit"
    destination.mkdir(exist_ok=True, mode=0o700)
    descriptor = destination/"descriptor.json"
    if descriptor.exists():
        for record in read(descriptor).values():
            require(sha(record["path"]) == record["sha256"], "FIT cache changed")
        return descriptor
    origin = job["fit_inputs"][chrom]
    response = fetch_receipt(origin["uri"], allow_missing=True)
    if response is None:
        return None
    receipt, receipt_sha = response
    if origin.get("sha256"):
        require(receipt_sha == origin["sha256"], "Preserved FIT receipt changed")
    require(receipt["chromosome"] == chrom and receipt["launch_sha256"] == origin["launch_sha256"] and
            receipt["status"] == "FIT_SUPPORT_PUBLISHED_VERIFIED_NOT_MODEL_EVALUATION", "FIT producer differs")
    files = {record["name"]: record for record in receipt["files"]}
    require(len(files) == len(receipt["files"]), "Duplicate FIT artifact")
    save(destination/"upstream-receipt.json", receipt)
    result = {}
    for key, name in (("manifest", "manifest.json"), ("contract", "contract.json"), ("retained", "retained.tsv")):
        record = staging.normalize_record(files[name])
        require(record["uri"].startswith(origin["uri"].rsplit("/", 1)[0]+"/"), "FIT artifact escaped source prefix")
        target = destination/name
        staging.stage_one(record, target, started, job)
        result[key] = dict(path=str(target), sha256=record["sha256"])
    save(descriptor, result)
    return descriptor


def run_nextflow(root, job, chrom, folder, descriptor, started):
    params = dict(stage="extract", chromosome=chrom, bundle=str(root/"bundle"), code=str(root/"frozen/bin"),
        output_dir=str(folder/"published"), contract_sha256=job["contract_sha256"],
        fit_cache=str(descriptor), fit_cache_sha256=sha(descriptor), task_cpus=job["task_cpus"],
        memory_gb=job["task_memory_gib"], max_forks=job["max_forks"], task_time="7h")
    save(folder/"params.json", params)
    attempt = str(time.time_ns())
    trace, log = folder/("trace."+attempt+".tsv"), folder/("nextflow."+attempt+".log")
    command = [job["nextflow_path"], "-C", str(root/job["config_path"]), "-log", str(log), "run",
        str(root/"frozen/workflows/r02_structure_materialize.nf"), "-params-file", str(folder/"params.json"),
        "-work-dir", str(folder/"work"), "-with-trace", str(trace), "-ansi-log", "false"]
    if (folder/".nextflow").exists():
        command.append("-resume")
    save(folder/("command."+attempt+".json"), dict(command=command, launch_sha256=job["launch_sha256"]))
    env = {**os.environ, "NXF_OFFLINE": "true", "NXF_DISABLE_CHECK_LATEST": "true", "NXF_SYNTAX_PARSER": "v1",
           "NXF_OPTS": "-Xms64m -Xmx768m", "PYTHONDONTWRITEBYTECODE": "1"}
    with (folder/("console."+attempt+".log")).open("x") as console:
        process = subprocess.Popen(command, cwd=folder, env=env, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                staging.remaining(started, job)
                staging.check_disk(root, job)
                time.sleep(10)
            require(process.returncode == 0, "Nextflow failed; retain inputs and cache")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            raise
    with trace.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    require(len(rows) == job["expected_tasks_per_chromosome"] and all(
        row["status"] in {"COMPLETED", "CACHED"} and row["exit"] == "0" for row in rows), "Incomplete chromosome trace")


def publication_files(root, job, chrom, folder):
    published = folder/"published"
    partials = sorted(published.glob("partial_*"))
    require(len(partials) == job["expected_partials_per_chromosome"], "Incomplete chromosome partial set")
    task_ids = set()
    for partial in partials:
        manifest = read(partial/"manifest.json")
        require(manifest["schema"] == "r02_structure_materialize_partial_v1"
                and manifest["stage"] == "COMPLETE_SLICE" and manifest["training_allowed"] is False
                and manifest["contract_sha256"] == job["contract_sha256"]
                and manifest["request"]["chromosome"] == chrom, "Partial data contract/stage/chromosome differs")
        task_id = manifest["request"]["task_id"]
        require(task_id not in task_ids and partial.name == "partial_" + task_id, "Duplicate or misnamed partial task")
        task_ids.add(task_id)
        expected = {"data.npz", "selection.json", "common_markers.tsv"}
        if manifest["request"]["kind"] == "target":
            expected.add("markers.tsv")
        require(set(manifest["outputs_sha256"]) == expected, "Partial output inventory incomplete")
        for name, digest in manifest["outputs_sha256"].items():
            require(sha(partial/name) == digest, "Partial output changed before publication")
        for name, digest in manifest["code_sha256"].items():
            require(Path(name).name == name and sha(root/"frozen/bin"/name) == digest, "Partial producer code differs")
    selected = [p for p in published.rglob("*") if p.is_file()]
    selected += [p for p in folder.iterdir() if p.is_file() and
                 (p.suffix in {".log", ".tsv"} or p.name.startswith("command.") or p.name in {"params.json", "source-receipt.json"})]
    selected += [p for p in (folder/"fit").iterdir() if p.is_file()]
    records = []
    for path in sorted(selected):
        require(not path.is_symlink() and path.resolve().is_relative_to(folder), "Unsafe published file")
        name = path.relative_to(folder).as_posix()
        records.append(dict(name=name, path=str(path), uri=job["output_prefix"]+f"/chr{int(chrom):02d}/"+name,
                            **pipeline.publication_digests(path)))
    require(records and len({r["name"] for r in records}) == len(records), "Invalid publication inventory")
    return records


def publish_chromosome(root, job, chrom, folder):
    plan_path = folder/"publication-plan.json"
    if not plan_path.exists():
        save(plan_path, dict(chromosome=chrom, launch_sha256=job["launch_sha256"], files=publication_files(root, job, chrom, folder)))
    plan = read(plan_path)
    require(plan["launch_sha256"] == job["launch_sha256"] and plan["chromosome"] == chrom, "Publication plan differs")
    receipt_path = folder/"completed.publication.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        for record in read(folder/"completed.json")["files"]:
            publication.verify_remote(record, pinned=True)
        publication.verify_remote(receipt, pinned=True)
        return receipt
    for record in plan["files"]:
        require(sha(record["path"]) == record["sha256"], "Output changed before publication")
        publication.upload_one(record)
        publication.verify_remote(record, pinned=True)
    completion_path = folder/"completed.json"
    if completion_path.exists():
        # A failed upload of the completion receipt must not regenerate its
        # timestamp or rewrite the immutable scientific completion record.
        value = read(completion_path)
        require(value["status"] == "CHROMOSOME_PARTIALS_PUBLISHED_NOT_SCORED"
                and value["chromosome"] == chrom and value["contract_sha256"] == job["contract_sha256"]
                and value["launch_sha256"] == job["launch_sha256"] and value["training_allowed"] is False,
                "Previous completion belongs to another extraction")
        identity = lambda r: {k:r[k] for k in ("name", "uri", "bytes", "sha256", "md5_base64", "generation")}
        require([identity(r) for r in value["files"]] == [identity(r) for r in plan["files"]],
                "Published artifacts differ from the preserved completion")
    else:
        value = dict(schema=SCHEMA, status="CHROMOSOME_PARTIALS_PUBLISHED_NOT_SCORED", chromosome=chrom,
            contract_sha256=job["contract_sha256"], launch_sha256=job["launch_sha256"], files=plan["files"],
            training_allowed=False, generated_at=time.time())
        save(completion_path, value)
    receipt = dict(name="completed.json", path=str(folder/"completed.json"),
        uri=job["output_prefix"]+f"/chr{int(chrom):02d}/completed.json", **pipeline.publication_digests(folder/"completed.json"))
    publication.upload_one(receipt)
    publication.verify_remote(receipt, pinned=True)
    save(receipt_path, receipt)
    return receipt


def cleanup_recoverable(root, job, chrom, folder):
    """Remove only this worker's staging/cache, after verified publication."""
    require((folder/"completed.publication.json").is_file(), "Publication required before cleanup")
    contract = read(root/"bundle/materialize.json")
    for kind in ("common_vcf", "common_index", "rare_vcf", "rare_index"):
        record = staging.normalize_record(job["sources"][chrom][kind])
        staging.remote_check(record)
        path = Path(contract["sources"][chrom][kind]["path"])
        require(path.parent == root/"sources"/f"chr{int(chrom):02d}" and not path.is_symlink(), "Cleanup escaped assigned chromosome")
        if path.exists():
            path.unlink()
    work = folder/"work"
    require(folder == root/"chromosomes"/f"chr{int(chrom):02d}" and not work.is_symlink(), "Unsafe work cleanup")
    if work.exists():
        shutil.rmtree(work)
    save(folder/"cleanup.json", dict(removed="downloaded_source_copies_and_completed_nextflow_work_only",
         source_persistent=True, chromosome=chrom, publication_sha256=sha(folder/"completed.publication.json")))


def publish_worker_completion(root, job):
    """Publish the small worker receipt only after every chromosome is durable."""
    path = root/("complete-"+job["worker"]+".json")
    receipt_path = root/("complete-"+job["worker"]+".publication.json")
    if receipt_path.exists():
        record = read(receipt_path)
        require(sha(path) == record["sha256"], "Worker completion changed after publication")
        publication.verify_remote(record, pinned=True)
        return record
    record = dict(name=path.name, path=str(path), uri=job["output_prefix"]+"/"+path.name,
                  **pipeline.publication_digests(path))
    publication.upload_one(record)
    publication.verify_remote(record, pinned=True)
    save(receipt_path, record)
    return record


def verify_complete(root, job):
    """Fail closed unless the exact assigned chromosome products are durable.

    This check authorizes no deletion itself. The infrastructure owner must
    additionally match the VM/disk identities before retiring resources.
    """
    path = root/("complete-"+job["worker"]+".json")
    value = read(path)
    require(value["schema"] == SCHEMA and value["status"] == "ASSIGNED_CHROMOSOMES_PUBLISHED_NOT_SCORED"
            and value["worker"] == job["worker"] and value["launch_sha256"] == job["launch_sha256"]
            and value["contract_sha256"] == job["contract_sha256"] and value["training_allowed"] is False,
            "Worker completion scope differs")
    require(set(value["results"]) == set(job["chromosomes"]), "Worker chromosome coverage incomplete")
    final_receipt = read(root/("complete-"+job["worker"]+".publication.json"))
    require(final_receipt["uri"] == job["output_prefix"]+"/"+path.name
            and sha(path) == final_receipt["sha256"], "Worker final receipt differs")
    publication.verify_remote(final_receipt, pinned=True)
    files = 0
    for chrom in job["chromosomes"]:
        folder = root/"chromosomes"/f"chr{int(chrom):02d}"
        receipt = read(folder/"completed.publication.json")
        prefix = job["output_prefix"]+f"/chr{int(chrom):02d}/"
        require(value["results"][chrom] == receipt and receipt["uri"] == prefix+"completed.json"
                and sha(folder/"completed.json") == receipt["sha256"], "Chromosome completion linkage differs")
        publication.verify_remote(receipt, pinned=True)
        completed = read(folder/"completed.json")
        require(completed["schema"] == SCHEMA and completed["status"] == "CHROMOSOME_PARTIALS_PUBLISHED_NOT_SCORED"
                and completed["chromosome"] == chrom and completed["contract_sha256"] == job["contract_sha256"]
                and completed["launch_sha256"] == job["launch_sha256"] and completed["training_allowed"] is False,
                "Chromosome completion scope differs")
        require(completed["files"] and len({r["name"] for r in completed["files"]}) == len(completed["files"]),
                "Chromosome publication inventory is empty or duplicated")
        for record in completed["files"]:
            name = publication.safe_relative(record["name"])
            require(record["uri"] == prefix+name.as_posix(), "Published artifact escaped assigned chromosome")
            publication.verify_remote(record, pinned=True)
            files += 1
    return dict(status="SAFE_TO_RETIRE", worker=job["worker"], launch_sha256=job["launch_sha256"],
                verified_chromosomes=len(job["chromosomes"]), verified_artifacts=files,
                training_allowed=False, vm_and_disk_identity_check_still_required=True)


def execute(root, job, cache_preserver=None):
    started = time.monotonic()
    results = {}
    while len(results) < len(job["chromosomes"]):
        progressed = False
        for chrom in job["chromosomes"]:
            if chrom in results:
                continue
            staging.remaining(started, job)
            folder = root/"chromosomes"/f"chr{int(chrom):02d}"
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not (folder/"publication-plan.json").exists():
                descriptor = fit_cache(root, job, chrom, folder, started)
                if descriptor is None:
                    continue
                contract = read(root/"bundle/materialize.json")
                records = job["sources"][chrom]
                needed = sum(r["bytes"] for k, r in records.items() if not Path(contract["sources"][chrom][k]["path"]).exists())
                staging.check_disk(root, job, needed+job["intermediate_budget_bytes"])
                staged = {}
                for kind, record in records.items():
                    target = Path(contract["sources"][chrom][kind]["path"])
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    staged[kind] = staging.stage_one(staging.normalize_record(record), target, started, job)
                save(folder/"source-receipt.json", staged)
                run_nextflow(root, job, chrom, folder, descriptor, started)
            results[chrom] = publish_chromosome(root, job, chrom, folder)
            if cache_preserver is not None:
                # Operational continuations can preserve Nextflow state before
                # the original, verified source/cache cleanup. The scientific
                # extraction, assignment and published receipt stay unchanged.
                cache_preserver(root, job, chrom, folder)
            cleanup_recoverable(root, job, chrom, folder)
            progressed = True
        if not progressed:
            time.sleep(min(60, staging.remaining(started, job)))
    completion = dict(schema=SCHEMA, status="ASSIGNED_CHROMOSOMES_PUBLISHED_NOT_SCORED", worker=job["worker"],
        launch_sha256=job["launch_sha256"], contract_sha256=job["contract_sha256"], results=results, training_allowed=False)
    save(root/("complete-"+job["worker"]+".json"), completion)
    publish_worker_completion(root, job)
    verify_complete(root, job)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-complete", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    job = validate(args.root, args.launch, args.launch_sha256)
    if args.verify_only:
        print(json.dumps(dict(status="FROZEN_QUEUE_VERIFIED_NOT_EXECUTED", chromosomes=job["chromosomes"])))
        return
    if args.verify_complete:
        print(json.dumps(verify_complete(args.root, job)))
        return
    with (args.root/("lock-"+job["worker"])).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(execute(args.root, job)), flush=True)


if __name__ == "__main__":
    main()
