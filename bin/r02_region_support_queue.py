#!/usr/bin/env python3
"""Serial, resumable FIT-support queue with pinned inputs and verified publication.

This worker never fits a predictor, reads evaluation targets for model selection,
deletes a VM, or publishes private genotype intermediates. Missing upstream M02
receipts are pending work, not completed chromosomes.
"""
from __future__ import annotations

import argparse
import copy
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
from urllib.parse import unquote

import r02_autosome_pipeline as pipeline
import r02_publish_evidence as publication


require = publication.require
SCHEMA = "r02_region_support_queue_v1"
PREFIX = publication.PREFIX + "r02-region-support-20261004s"
RECOVERY_PREFIX = ("gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/checkpoints/"
                   "R02_20260930/r02-m02-recovery-20261004q/")
GIB = 2**30
PUBLIC_NAMES = {"fit_eligible.tsv", "retained.tsv", "target_support.tsv", "ld_diagnostic.tsv", "ld_tool.log"}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return publication.read_unique_json(Path(path))


def save_once(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, "Immutable local record changed: " + path.name)
    else:
        publication.write_receipt_new(path, value)


def remaining(started, manifest):
    value = manifest["max_runtime_seconds"] - (time.monotonic() - started)
    require(value > 0, "Queue deadline reached; unfinished work remains pending")
    return value


def check_disk(root, manifest, additional=0):
    require(shutil.disk_usage(root).free >= manifest["min_free_bytes"] + additional,
            "Local disk reserve or stage admission failed")


def normalize_record(record):
    result = dict(uri=record["uri"], generation=str(record["generation"]), bytes=int(record["bytes"]),
                  md5Hash=record.get("md5Hash", record.get("md5_base64")))
    require(result["uri"].startswith(pipeline.BUCKET) and "#" not in result["uri"], "Invalid source URI")
    require(re.fullmatch(r"[1-9][0-9]*", result["generation"]) and result["bytes"] > 0, "Invalid generation/size")
    require(isinstance(result["md5Hash"], str) and re.fullmatch(r"[A-Za-z0-9+/]{22}==", result["md5Hash"]), "Missing source MD5")
    if record.get("sha256"):
        result["sha256"] = publication.validate_sha(record["sha256"])
    return result


def remote_check(record):
    meta = publication.remote_metadata(record["uri"])
    require(str(meta["generation"]) == record["generation"] and int(meta["size"]) == record["bytes"] and
            meta.get("md5_hash", meta.get("md5Hash")) == record["md5Hash"], "Source generation/size/MD5 changed")


def missing_gcs_object(response, uri):
    """Recognize object absence, never an inaccessible or differently named object."""
    message = (response.stderr or "") + (response.stdout or "")
    if not response.returncode:
        return False
    # Preserve known older CLI renderings while rejecting permission/network
    # errors. Current GcsNotFoundError percent-encodes object-name slashes.
    if re.search(r"\b(?:401|403|429|50[0-9])\b", message):
        return False
    missing = re.search(
        r"\(gcloud\.storage\.objects\.describe\) (gs://[^\n]+) not found: 404\.\s*$", message)
    if missing:
        return unquote(missing[1]) == uri
    named_uris = {value.rstrip('.,:') for value in re.findall(r'''gs://[^\s'"<>()]+''', unquote(message))}
    return uri in named_uris and bool(re.search(
        r"HTTP(?:Error)?[^\n]*\b404\b|No URLs matched", message, flags=re.I))


def resolve_entry(entry, folder):
    """Persist the first authenticated resolution; never follow a changed receipt."""
    path = folder / "resolved-inputs.json"
    if path.exists():
        value = read(path)
        require(value["entry_sha256"] == hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest(), "Queue entry changed")
        return value
    origin = None
    if "recovery_receipt_uri" in entry:
        uri = entry["recovery_receipt_uri"]
        require(uri.startswith(RECOVERY_PREFIX) and uri.endswith("/chr" + entry["chromosome"].zfill(2) + "/completed.json"),
                "Recovery receipt outside chromosome assignment")
        response = subprocess.run(["gcloud", "storage", "objects", "describe", uri, "--format=json"],
                                  capture_output=True, text=True, timeout=120)
        if response.returncode:
            if missing_gcs_object(response, uri):
                return None
            raise subprocess.CalledProcessError(response.returncode, response.args, output=response.stdout, stderr=response.stderr)
        meta = json.loads(response.stdout)
        require(0 < int(meta["size"]) < 8 * 2**20, "Recovery receipt size is invalid")
        generation = str(meta["generation"])
        data = subprocess.check_output(["gcloud", "storage", "cat", uri + "#" + generation], timeout=120)
        require(len(data) == int(meta["size"]), "Recovery receipt size changed")
        receipt_path = folder / "upstream-recovery.json"
        if receipt_path.exists():
            require(receipt_path.read_bytes() == data, "Stored recovery receipt changed")
        else:
            with receipt_path.open("xb") as handle:
                handle.write(data)
        digest = pipeline.publication_digests(receipt_path)
        require(digest["md5_base64"] == meta.get("md5_hash", meta.get("md5Hash")), "Recovery receipt MD5 differs")
        receipt = read(receipt_path)
        require(receipt.get("status") == "M02_RECOVERED_PUBLISHED_VERIFIED_NOT_FIT_SELECTED" and
                str(receipt.get("chromosome")) == entry["chromosome"], "Recovery receipt scope/status differs")
        candidates = [r for r in receipt["files"] if r["uri"].endswith(".snv.bi.pass.vcf.gz")]
        require(len(candidates) == 1, "Recovery receipt must identify one M02 VCF")
        source = normalize_record(candidates[0])
        indexes = [r for r in receipt["files"] if r["uri"] == source["uri"] + ".tbi"]
        require(len(indexes) == 1, "Recovery receipt has no unique matching index")
        index = normalize_record(indexes[0])
        require(source["uri"].rsplit("/", 1)[0] == uri.rsplit("/", 1)[0], "Recovered source escaped receipt directory")
        origin = dict(uri=uri, generation=generation, **digest)
    else:
        source, index = normalize_record(entry["source"]), normalize_record(entry["index"])
    require(index["uri"] == source["uri"] + ".tbi", "Source/index mismatch")
    require(re.search(r"chr" + re.escape(entry["chromosome"]) + r"[._]", source["uri"].rsplit("/", 1)[-1]),
            "Source chromosome differs")
    for record in (source, index):
        remote_check(record)
    value = dict(chromosome=entry["chromosome"], source=source, index=index, recovery_receipt=origin,
                 entry_sha256=hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest())
    save_once(path, value)
    return value


def stage_one(record, target, started, manifest):
    remote_check(record)
    if not target.exists():
        part = target.with_name(target.name + ".partial")
        # A previous incomplete transfer is never accepted as a source. gcloud
        # may safely replace this exact worker-owned partial on explicit resume.
        require(not part.is_symlink(), "Unsafe partial transfer")
        subprocess.run(["gcloud", "storage", "cp", record["uri"] + "#" + record["generation"], str(part)],
                       check=True, timeout=min(7200, remaining(started, manifest)))
        digest = pipeline.publication_digests(part)
        require(digest["bytes"] == record["bytes"] and digest["md5_base64"] == record["md5Hash"] and
                ("sha256" not in record or digest["sha256"] == record["sha256"]), "Downloaded source checksum differs")
        os.link(part, target)
        part.unlink()
    require(not target.is_symlink() and target.resolve() == target, "Unsafe staged input")
    digest = pipeline.publication_digests(target)
    require(digest["bytes"] == record["bytes"] and digest["md5_base64"] == record["md5Hash"] and
            ("sha256" not in record or digest["sha256"] == record["sha256"]), "Staged source checksum differs")
    stat = target.stat()
    return dict(**record, path=str(target), local_sha256=digest["sha256"], mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def cleanup_bulk(root, folder):
    require(folder.parent == root / "chromosomes" and re.fullmatch(r"chr(?:0[1-9]|1[0-9]|2[0-2])", folder.name), "Unsafe cleanup scope")
    require(folder.resolve() == folder and not folder.is_symlink(), "Unsafe chromosome directory")
    for name in ("inputs", "work", "private_work"):
        target = folder / name
        require(target.resolve() == target and not target.is_symlink(), "Symlink in temporary cleanup")
        if target.exists():
            require(target.is_dir(), "Temporary path is not directory")
            shutil.rmtree(target)


def destination(manifest, chrom, name):
    publication.safe_relative(name)
    return manifest["output_prefix"] + "/chr" + chrom.zfill(2) + "/" + name


def publish_chromosome(root, manifest, folder, plan):
    chrom = plan["chromosome"]
    require(plan["launch_sha256"] == manifest["authenticated_launch_sha256"] and chrom in manifest["chromosomes"] and
            folder == root / "chromosomes" / ("chr" + chrom.zfill(2)), "Publication assignment differs")
    require(plan["files"] and len({r["name"] for r in plan["files"]}) == len(plan["files"]), "Empty/duplicate publication")
    for record in plan["files"]:
        require(record["uri"] == destination(manifest, chrom, record["name"]), "Publication escaped assigned prefix")
    completed = folder / "completed.json"
    if completed.exists():
        result = read(completed)
        require(result["publication_plan_sha256"] == sha(folder / "publication-plan.json") and
                result["launch_sha256"] == manifest["authenticated_launch_sha256"], "Completion receipt changed")
        for record in result["files"]:
            publication.verify_remote(record, pinned=True)
    else:
        for record in plan["files"]:
            publication.upload_one(record)
        for record in plan["files"]:
            publication.verify_remote(record, pinned=True)
        result = dict(schema="r02_region_support_completed_v1", status="FIT_SUPPORT_PUBLISHED_VERIFIED_NOT_MODEL_EVALUATION",
                      chromosome=chrom, launch_sha256=manifest["authenticated_launch_sha256"],
                      publication_plan_sha256=sha(folder / "publication-plan.json"), files=plan["files"])
        save_once(completed, result)
    receipt = dict(name="completed.json", path=str(completed), uri=destination(manifest, chrom, "completed.json"),
                   **pipeline.publication_digests(completed))
    receipt_path = folder / "completed.publication.json"
    if receipt_path.exists():
        previous = read(receipt_path)
        require(all(previous[k] == receipt[k] for k in ("uri", "bytes", "sha256", "md5_base64")), "Published receipt changed")
        receipt = previous
        publication.verify_remote(receipt, pinned=True)
    else:
        publication.upload_one(receipt)
        publication.verify_remote(receipt, pinned=True)
        save_once(receipt_path, receipt)
    cleanup_bulk(root, folder)
    return receipt


def execute_nextflow(root, manifest, folder, params, staged, started, attempt):
    save_once(folder / "params.json", params)
    names = {key: folder / f"{key}.{attempt}.{ext}" for key, ext in (("trace", "tsv"), ("nextflow", "log"), ("console", "log"))}
    require(not any(p.exists() for p in names.values()), "Attempt logs exist; use another explicit resumeN")
    args = [manifest["nextflow_path"], "-C", str(root / manifest["config_path"]), "-log", str(names["nextflow"]),
            "run", str(root / "frozen/workflows/r02_structure_region_support.nf"), "-params-file", str(folder / "params.json"),
            "-work-dir", str(folder / "work"), "-with-trace", str(names["trace"]), "-ansi-log", "false"]
    if (folder / ".nextflow").exists():
        args.append("-resume")
    env = {**os.environ, "NXF_OFFLINE": "true", "NXF_DISABLE_CHECK_LATEST": "true", "NXF_SYNTAX_PARSER": "v1",
           "NXF_OPTS": "-Xms64m -Xmx768m", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with names["console"].open("x") as console:
        process = subprocess.Popen(args, cwd=folder, env=env, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                remaining(started, manifest)
                check_disk(root, manifest)
                for record in staged.values():
                    stat = Path(record["path"]).stat()
                    require((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) ==
                            (record["bytes"], record["mtime_ns"], record["ctime_ns"]), "Staged source mutated during analysis")
                time.sleep(10)
            require(process.returncode == 0, "Support Nextflow failed; retained data and cache")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            # Nextflow termination should cancel its task. Also stop only this
            # campaign's explicitly labelled containers, if the launcher uses Docker.
            label = manifest.get("container_label")
            if label:
                require(re.fullmatch(r"dnabr\.run=[A-Za-z0-9._-]+", label), "Unsafe owned container label")
                ids = subprocess.check_output(["docker", "ps", "--filter", "label=" + label, "--format", "{{.ID}}"], text=True).split()
                require(all(re.fullmatch(r"[a-f0-9]{12,64}", item) for item in ids), "Unexpected owned container ID")
                for item in ids:
                    subprocess.run(["docker", "stop", "--time", "30", item], check=True, timeout=60)
            raise
    with names["trace"].open() as handle:
        trace = list(csv.DictReader(handle, delimiter="\t"))
    require(trace and all(r["status"] in {"COMPLETED", "CACHED"} and r["exit"] == "0" for r in trace), "Incomplete Nextflow trace")
    return args, names


def run_chromosome(root, manifest, entry, started, attempt):
    chrom = entry["chromosome"]
    folder = root / "chromosomes" / ("chr" + chrom.zfill(2))
    folder.mkdir(parents=True, mode=0o700, exist_ok=True)
    if (folder / "publication-plan.json").exists():
        return publish_chromosome(root, manifest, folder, read(folder / "publication-plan.json"))
    resolved = resolve_entry(entry, folder)
    if resolved is None:
        return None
    if "source_max_bytes" in manifest:
        require(resolved["source"]["bytes"] + resolved["index"]["bytes"] <= manifest["source_max_bytes"],
                "Resolved M02 source exceeds frozen disk admission ceiling")
    inputs = folder / "inputs"
    inputs.mkdir(mode=0o700, exist_ok=True)
    missing_bytes = sum(resolved[k]["bytes"] for k, name in (("source", "input.vcf.gz"), ("index", "input.vcf.gz.tbi")) if not (inputs/name).exists())
    check_disk(root, manifest, missing_bytes + manifest["intermediate_budget_bytes"])
    staged = {key: stage_one(resolved[key], inputs / name, started, manifest)
              for key, name in (("source", "input.vcf.gz"), ("index", "input.vcf.gz.tbi"))}
    save_once(folder / "source-receipt.json", staged)
    contract = copy.deepcopy(entry["contract_template"])
    for key, kind in (("common_vcf", "source"), ("common_index", "index")):
        contract["source"][key] = dict(path=staged[kind]["path"], sha256=staged[kind]["local_sha256"])
    require(str(contract["source"]["chromosome"]) == chrom, "Support template chromosome differs")
    save_once(folder / "contract.json", contract)
    params = dict(contract=str(folder / "contract.json"), contract_sha256=sha(folder / "contract.json"),
                  code=str(root / "frozen/bin"), output_dir=str(folder / "published"), task_cpus=2, memory_gb=8, task_time="4h")
    args, names = execute_nextflow(root, manifest, folder, params, staged, started, attempt)
    return finalize_chromosome(root, manifest, folder, chrom, args, names, attempt)


def finalize_chromosome(root, manifest, folder, chrom, args, names, attempt):
    """Publish verified task products; also used when adopting a finished task."""
    for name, digest in manifest["files_sha256"].items():
        require(sha(root / publication.safe_relative(name)) == digest, "Frozen bundle changed during support analysis")
    public = folder / "published/public"
    output_manifest = read(public / "manifest.json")
    require(output_manifest.get("schema") == "r02_structure_region_support_v1" and
            output_manifest.get("stage") == "FIT_SUPPORT_MEASURED_NO_BIOLOGICAL_VALIDATION" and
            str(output_manifest.get("chromosome")) == chrom, "Unexpected support output status/chromosome")
    declared = output_manifest["publish_files"]
    require(isinstance(declared, dict) and PUBLIC_NAMES - {"ld_tool.log"} <= set(declared) <= PUBLIC_NAMES,
            "Forbidden or missing public output")
    selected = {}
    for name, expected in declared.items():
        path = publication.safe_local(public / publication.safe_relative(name))
        require(sha(path) == publication.validate_sha(expected), "Public output digest differs")
        selected[name] = path
    selected["manifest.json"] = public / "manifest.json"
    provenance = dict(launch_sha256=manifest["authenticated_launch_sha256"], code_and_bundle_sha256=manifest["files_sha256"],
                      source_receipt_sha256=sha(folder / "source-receipt.json"), contract_sha256=sha(folder / "contract.json"),
                      nextflow_command=args, preparation_only=True, private_genotypes_published=False)
    if "ownership_receipt" in manifest:
        provenance["ownership_receipt_sha256"] = manifest["ownership_receipt"]["sha256"]
    if "execution_amendment_sha256" in manifest:
        provenance["execution_amendment_sha256"] = manifest["execution_amendment_sha256"]
    save_once(folder / "provenance.json", provenance)
    selected.update({"provenance.json": folder / "provenance.json", "source-receipt.json": folder / "source-receipt.json",
                     "contract.json": folder / "contract.json", "trace." + attempt + ".tsv": names["trace"]})
    files = [dict(name=name, path=str(path), uri=destination(manifest, chrom, name), **pipeline.publication_digests(path))
             for name, path in selected.items()]
    plan = dict(chromosome=chrom, launch_sha256=manifest["authenticated_launch_sha256"], files=files)
    save_once(folder / "publication-plan.json", plan)
    return publish_chromosome(root, manifest, folder, plan)


def verify_ownership(root, manifest):
    """A delegate may work only on chromosomes explicitly transferred to it."""
    descriptor = manifest.get("ownership_receipt")
    if descriptor is None:
        require("instance" not in manifest, "Named delegate requires a transfer receipt")
        return
    path = publication.safe_local(root / publication.safe_relative(descriptor["path"]))
    require(sha(path) == publication.validate_sha(descriptor["sha256"]) and
            manifest["files_sha256"].get(descriptor["path"]) == descriptor["sha256"], "Ownership receipt changed/unfrozen")
    transfer = read(path)
    version = transfer.get("schema")
    require(version in {"r02_region_support_transfer_v1", "r02_region_support_transfer_v2"} and
            transfer.get("original_instance") == "dnabr-r02-support-1004s", "No compatible queue transfer")
    publication.validate_sha(transfer["original_launch_sha256"])
    require(re.fullmatch(r"[1-9][0-9]+", transfer["source_instance_id"]), "Missing original VM identity")
    assignments = transfer["assignments"]
    owned = [a for a in assignments if a["instance"] == manifest.get("instance")]
    require(len(owned) == 1 and owned[0]["chromosomes"] == manifest["chromosomes"] and
            owned[0]["output_prefix"] == manifest["output_prefix"], "Delegate assignment differs")
    chromosomes = [c for a in assignments for c in a["chromosomes"]]
    require(chromosomes and len(chromosomes) == len(set(chromosomes)) and
            len({a["instance"] for a in assignments}) == len(assignments) and
            len({a["output_prefix"] for a in assignments}) == len(assignments), "Overlapping assignments")
    for assignment in assignments:
        require(re.fullmatch(r"dnabr-r02-fit-[a-z0-9-]+", assignment["instance"]) and
                assignment["output_prefix"] == PREFIX + "/worker-" + assignment["instance"].removeprefix("dnabr-r02-"),
                "Unsafe transferred destination")
    if version == "r02_region_support_transfer_v1":
        require(transfer.get("status") == "PENDING_SUPPORT_DELEGATED_SUPERVISOR_PAUSED" and
                set(chromosomes) == {"13", "14", "15", "16"}, "Unexpected v1 transfer")
        return
    require(transfer.get("status") == "PENDING_SUPPORT_DELEGATED_ORIGINAL_STOPPED" and
            transfer["source_instance_id"] == "5195112201394809158", "Original queue not fenced")
    require(manifest.get("original_launch_sha256") == transfer["original_launch_sha256"], "Original launch binding differs")
    original = root / "inputs/original-launch.json"
    require(manifest["files_sha256"].get("inputs/original-launch.json") == transfer["original_launch_sha256"] and
            sha(original) == transfer["original_launch_sha256"], "Original launch not frozen")
    fence_descriptor = transfer["source_fence"]
    fence_path = root / publication.safe_relative(fence_descriptor["path"])
    require(manifest["files_sha256"].get(fence_descriptor["path"]) == fence_descriptor["sha256"] and
            sha(fence_path) == publication.validate_sha(fence_descriptor["sha256"]), "Source fence changed/unfrozen")
    fence = read(fence_path)
    require(fence.get("name") == transfer["original_instance"] and str(fence.get("id")) == transfer["source_instance_id"] and
            fence.get("status") == "TERMINATED" and fence.get("scientific_autostart_disabled") is True,
            "Source fence does not prove stopped diagnostic-only machine")
    preserved = transfer["preserved_results"]
    kept = [r["chromosome"] for r in preserved]
    require(len(kept) == len(set(kept)) and set(kept) == {"13", "14", "15", "16", "17", "18", "19", "20", "22"} and
            not set(chromosomes) & set(kept) and set(chromosomes) | set(kept) == {str(i) for i in range(1, 23)},
            "Transferred/preserved chromosomes must partition the original 22 exactly")
    for item in preserved:
        record = item["receipt"]
        require(record["uri"].startswith(PREFIX + "/") and record["uri"].endswith("/chr" + item["chromosome"].zfill(2) + "/completed.json"),
                "Preserved result escaped original campaign")
        publication.validate_sha(record["sha256"])
        require(re.fullmatch(r"[1-9][0-9]*", str(record["generation"])), "Unpinned preserved result")


def verify_bundle(root, expected):
    require(root.resolve() == root and sha(root / "launch.json") == publication.validate_sha(expected), "Launch checksum/path differs")
    manifest = read(root / "launch.json")
    require(manifest["schema"] == SCHEMA and manifest["root"] == str(root), "Wrong queue/root")
    require(manifest["output_prefix"] == PREFIX or re.fullmatch(re.escape(PREFIX) + r"/worker-[a-z0-9-]+", manifest["output_prefix"]), "Wrong output scope")
    require(type(manifest["max_runtime_seconds"]) is int and 0 < manifest["max_runtime_seconds"] <= 72000 and manifest["min_free_bytes"] >= 20 * GIB and
            manifest["intermediate_budget_bytes"] >= 15 * GIB, "Unsafe runtime/disk limits")
    chroms = manifest["chromosomes"]
    if "source_max_bytes" in manifest:
        require(type(manifest["source_max_bytes"]) is int and 0 < manifest["source_max_bytes"] <= 80 * GIB,
                "Invalid bounded source admission ceiling")
    require(chroms and len(chroms) == len(set(chroms)) and set(chroms) <= {str(i) for i in range(1, 23)}, "Invalid chromosome assignment")
    require([e["chromosome"] for e in manifest["entries"]] == chroms, "Queue entries differ from assignment")
    required = {"frozen/bin/r02_region_support_queue.py", "frozen/bin/r02_structure_region_support.py",
                "frozen/bin/r02_structure_regions.py", "frozen/bin/r02_structure_materialize.py",
                "frozen/bin/r02_structure_increment.py", "frozen/bin/r02_structure_input_plan.py",
                "frozen/bin/r02_genomic_pair_evidence.py",
                "frozen/bin/r02_publish_evidence.py", "frozen/bin/r02_autosome_pipeline.py",
                "frozen/workflows/r02_structure_region_support.nf", manifest["config_path"]}
    require(set(manifest["files_sha256"]) >= required, "Frozen code/config hashes incomplete")
    for name, digest in manifest["files_sha256"].items():
        path = publication.safe_local(root / publication.safe_relative(name))
        require(sha(path) == publication.validate_sha(digest), "Frozen bundle file changed: " + name)
    for entry in manifest["entries"]:
        for key in ("map", "persons", "policy"):
            descriptor = entry["contract_template"][key]
            path = publication.safe_local(descriptor["path"])
            require(path.is_relative_to(root) and
                    manifest["files_sha256"].get(path.relative_to(root).as_posix()) == descriptor["sha256"],
                    "Support metadata missing from frozen bundle")
    verify_ownership(root, manifest)
    manifest["authenticated_launch_sha256"] = expected
    return manifest


def verify_completion(root, manifest):
    result = read(root / "complete.json")
    require(result["launch_sha256"] == manifest["authenticated_launch_sha256"] and
            result["chromosomes"] == manifest["chromosomes"] and len(result["results"]) == len(manifest["chromosomes"]), "Incomplete queue receipt")
    for chromosome, record in zip(manifest["chromosomes"], result["results"]):
        local = root / "chromosomes" / ("chr" + chromosome.zfill(2)) / "completed.json"
        require(sha(local) == record["sha256"], "Local completion receipt changed")
        for item in read(local)["files"]:
            publication.verify_remote(item, pinned=True)
        publication.verify_remote(record, pinned=True)
    receipt = read(root / "complete.publication.json")
    require(sha(root / "complete.json") == receipt["sha256"], "Campaign completion changed")
    publication.verify_remote(receipt, pinned=True)
    return dict(status="ALL_ASSIGNED_SUPPORT_REMOTE_REVERIFIED_SAFE_TO_RETIRE_OWN_VM", chromosomes=manifest["chromosomes"])


def run_queue(root, manifest, attempt):
    started = time.monotonic()
    results = {}
    while len(results) != len(manifest["entries"]):
        progressed = False
        for entry in manifest["entries"]:
            remaining(started, manifest)
            if entry["chromosome"] in results:
                continue
            result = run_chromosome(root, manifest, entry, started, attempt)
            if result is not None:
                results[entry["chromosome"]] = result
                progressed = True
                print(json.dumps(dict(chromosome=entry["chromosome"], status="FIT_SUPPORT_PUBLISHED_VERIFIED", completed=len(results))), flush=True)
        if not progressed:
            time.sleep(min(120, remaining(started, manifest)))
    path = root / "complete.json"
    save_once(path, dict(schema=SCHEMA, status="ALL_ASSIGNED_SUPPORT_PUBLISHED_VERIFIED", chromosomes=manifest["chromosomes"],
                         launch_sha256=manifest["authenticated_launch_sha256"], results=[results[c] for c in manifest["chromosomes"]]))
    receipt = dict(name="complete.json", path=str(path), uri=manifest["output_prefix"] + "/complete.json", **pipeline.publication_digests(path))
    final = root / "complete.publication.json"
    if final.exists():
        receipt = read(final)
        publication.verify_remote(receipt, pinned=True)
    else:
        publication.upload_one(receipt)
        publication.verify_remote(receipt, pinned=True)
        save_once(final, receipt)
    return verify_completion(root, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--attempt", default="initial")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-complete", action="store_true")
    args = parser.parse_args()
    require(re.fullmatch(r"initial|resume[1-9][0-9]*", args.attempt), "Explicit attempt required")
    require(not (args.verify_only and args.verify_complete), "Choose one verification action")
    os.umask(0o077)
    os.environ.update(CLOUDSDK_CORE_ACCOUNT="653458115080-compute@developer.gserviceaccount.com", CLOUDSDK_CORE_PROJECT="uspbr-242713")
    require(os.getuid() > 0 and os.getgid() > 0, "Run as the actual non-root worker user")
    os.environ.update(R02_WORKER_UID=str(os.getuid()), R02_WORKER_GID=str(os.getgid()))
    root = args.root.absolute()
    with (root / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_bundle(root, args.launch_sha256)
        if args.verify_only:
            print(json.dumps(dict(status="FROZEN_QUEUE_VERIFIED_NOT_RUNNING")))
            return
        if args.verify_complete:
            print(json.dumps(verify_completion(root, manifest)))
            return
        require(subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "--target", str(root)], text=True).strip() in {"ext4", "xfs"},
                "Local block filesystem required")
        def stopped(*_):
            raise RuntimeError("Stopped; preserve unfinished input/cache")
        signal.signal(signal.SIGTERM, stopped)
        print(json.dumps(run_queue(root, manifest, args.attempt)), flush=True)


if __name__ == "__main__":
    main()
