#!/usr/bin/env python3
"""Resume the original chromosome queue without changing scientific contracts.

The separate continuation contract authenticates the operational extension.
Completed chromosome products are reused. Completed task caches are archived
before cleanup; unfinished work stays on the original disk if execution fails.
This entry point never deletes or creates a VM, chooses markers, or fits a model.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import time

import r02_structure_materialize_queue as queue
import r02_publish_evidence as publication
import r02_autosome_pipeline as pipeline

SCHEMA = "r02_structure_queue_continuation_v1"
require, read, sha, save = queue.require, queue.read, queue.sha, queue.save


def validate(path, expected):
    require(sha(path) == expected, "Continuation contract changed")
    c = read(path)
    require(c["schema"] == SCHEMA and c["training_allowed"] is False,
            "Only the original data preparation may resume")
    require(re.fullmatch(r"[a-z0-9-]+", c["continuation_id"]), "Invalid continuation ID")
    require(c["resume_number"] == 1 and c["max_runtime_seconds"] == 27000,
            "Only one bounded continuation is authorized")
    require(c["max_cache_bytes"] > 0 and c["max_cache_bytes"] <= 30 * 2**30,
            "Cache archive allowance exceeds its bound")
    root = Path(c["root"])
    require(root == Path("/opt/r02-structure-materialize-20261005f"), "Wrong original root")
    require(c["cache_prefix"].startswith(publication.PREFIX + "r02-structure-continuation-"),
            "Wrong private retention prefix")
    for name, digest in c["code_sha256"].items():
        require(Path(name).name == name and sha(path.parent / "code" / name) == digest,
                "Continuation implementation changed: " + name)
    job = queue.validate(root, root / c["launch_file"], c["original_launch_sha256"])
    require(job["worker"] == c["worker"] and job["max_runtime_seconds"] == c["max_runtime_seconds"]
            and job["contract_sha256"] == c["scientific_contract_sha256"],
            "Continuation changed assignment, duration or scientific contract")
    return c, root, job


def preserve_cache(root, job, chrom, folder, contract):
    """Publish bounded immutable cache bytes, including symlink relationships.

    Source VCFs live outside each chromosome work folder, so symlinks are
    archived as links, never followed into those large authenticated sources.
    Their original staging receipts remain part of the archive.
    """
    base = root / "continuations" / contract["continuation_id"] / f"chr{int(chrom):02d}"
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt_path = base / "preservation.json"
    if receipt_path.exists():
        old = read(receipt_path)
        require(old["original_launch_sha256"] == job["launch_sha256"]
                and old["chromosome"] == chrom, "Preservation belongs to another chromosome")
        for r in old["files"]:
            publication.verify_remote(r, pinned=True)
        return old
    entries = []
    for p in sorted(folder.rglob("*")):
        name = p.relative_to(folder).as_posix()
        if p.is_symlink():
            target = p.resolve()
            require(target.is_relative_to(root), "Cache link escapes original owned runtime: " + name)
            entries.append(dict(path=name, kind="SYMLINK", target=os.readlink(p)))
        elif p.is_file():
            before = (p.stat().st_size, p.stat().st_mtime_ns)
            digest = sha(p)
            require(before == (p.stat().st_size, p.stat().st_mtime_ns), "Cache changed during inventory")
            entries.append(dict(path=name, kind="FILE", bytes=before[0], sha256=digest))
    total = sum(e.get("bytes", 0) for e in entries)
    require(0 < total <= contract["max_cache_bytes"], "Unexpected cache volume; preserve original disk")
    queue.staging.check_disk(root, job, total * 2)
    archive, inventory = base / "nextflow-state.tar.gz", base / "inventory.json"
    require(not archive.exists() and not inventory.exists(),
            "Interrupted cache publication requires inspection; do not overwrite")
    with tarfile.open(archive, "x:gz", dereference=False) as tar:
        for e in entries:
            tar.add(folder / e["path"], arcname=e["path"], recursive=False)
    with tarfile.open(archive, "r:gz") as tar:
        require(set(tar.getnames()) == {e["path"] for e in entries}, "Cache archive inventory mismatch")
        for e in entries:
            if e["kind"] == "FILE":
                with tar.extractfile(e["path"]) as stream:
                    require(hashlib.file_digest(stream, "sha256").hexdigest() == e["sha256"],
                            "Cache archive checksum mismatch")
    value = dict(schema="r02_materialize_cache_inventory_v1", chromosome=chrom,
                 original_launch_sha256=job["launch_sha256"],
                 nextflow_work_present=(folder / "work").exists(),
                 exact_cache_absent_in_original_completed_runs_not_recreated=True,
                 entries=entries, bytes=total, archive_sha256=sha(archive))
    save(inventory, value)
    files = []
    for p in (archive, inventory):
        r = dict(name=p.name, path=str(p),
                 uri=contract["cache_prefix"] + f"/worker-{job['worker']}/chr{int(chrom):02d}/" + p.name,
                 **pipeline.publication_digests(p))
        publication.upload_one(r)
        publication.verify_remote(r, pinned=True)
        files.append(r)
    receipt = dict(schema=SCHEMA, chromosome=chrom, status="CACHE_PUBLISHED_VERIFIED",
                   original_launch_sha256=job["launch_sha256"], files=files,
                   training_allowed=False)
    save(receipt_path, receipt)
    record = dict(name=receipt_path.name, path=str(receipt_path),
                  uri=contract["cache_prefix"] + f"/worker-{job['worker']}/chr{int(chrom):02d}/preservation.json",
                  **pipeline.publication_digests(receipt_path))
    publication.upload_one(record)
    publication.verify_remote(record, pinned=True)
    save(base / "preservation.publication.json", record)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--contract-sha256", required=True)
    args = p.parse_args()
    os.umask(0o077)
    c, root, job = validate(args.contract, args.contract_sha256)
    directory = root / "continuations" / c["continuation_id"]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / ("lock-" + job["worker"])).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(directory / "started.json", dict(schema=SCHEMA, started_unix=time.time(),
             continuation_contract_sha256=args.contract_sha256,
             original_launch_sha256=job["launch_sha256"], training_allowed=False))
        result = queue.execute(root, job, cache_preserver=lambda r, j, chrom, folder:
                               preserve_cache(r, j, chrom, folder, c))
        save(directory / "complete.json", dict(result=result,
             continuation_contract_sha256=args.contract_sha256, training_allowed=False))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
