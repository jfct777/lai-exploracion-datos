#!/usr/bin/env python3
"""Wait for authenticated slices, then assemble data-only genomic block bundles.

The readiness mode reads small completion records only. The execution mode
downloads sufficient statistics, never source VCFs, and delegates all scientific
calculations to the existing Nextflow materializer. It cannot fit models or
create, stop, or delete machines. Publication is generation-pinned and resumable.
"""
from __future__ import annotations

import argparse
import base64
import csv
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time

import r02_autosome_pipeline as pipeline
import r02_publish_evidence as publication
import r02_region_support_queue as staging

require, sha, read, save = staging.require, staging.sha, staging.read, staging.save_once
SCHEMA = "r02_structure_assembly_queue_v1"
READY = "EXACT_CHROMOSOME_PARTIALS_READY_NOT_SCORED"
COMPLETE = "BOTH_EXCLUSION_SCOPES_PUBLISHED_NOT_SCORED"
GIB = 2**30


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream,delimiter="\t"))


def code_hashes(root):
    planned = read(root/"assembly-plan.json")
    return {name:sha(root/"frozen/bin"/name) for name in planned["code_sha256"]}


def validate(root, launch, expected):
    require(root.is_absolute() and root.resolve() == root, "Canonical absolute root required")
    require(sha(launch) == expected, "Launch SHA256 differs")
    job = read(launch)
    require(job["schema"] == SCHEMA and job["training_allowed"] is False, "Data-only assembly required")
    require(job["output_prefix"].startswith(publication.PREFIX + "r02-structure-assembly-"), "Private output scope differs")
    for name, digest in job["files_sha256"].items():
        require(sha(root/publication.safe_relative(name)) == digest, "Frozen assembly input changed: " + name)
    require(sha(root/"bundle/materialize.json") == job["contract_sha256"], "Scientific contract changed")
    contract = read(root/"bundle/materialize.json")
    planned = read(root/"assembly-plan.json")
    require(planned["contract_sha256"] == job["contract_sha256"] and planned["code_sha256"] == code_hashes(root),
            "Frozen materializer plan/code differs")
    for name,digest in contract["metadata_sha256"].items():
        require(sha(root/"bundle"/publication.safe_relative(name)) == digest, "Scientific metadata changed")
    people, blocks = table(root/"bundle/persons.tsv"), table(root/"bundle/blocks.tsv")
    roles = {role:[i for i,p in enumerate(people) if p["role"] == role] for role in ("FIT","SELECT","EVALUATE")}
    require(sum(map(len,roles.values())) == len(people) and len({p["sample_id"] for p in people}) == len(people),
            "Unknown role or duplicated person")
    tasks, memberships = planned["tasks"], planned["blocks"]
    require(set(memberships) == {b["block_id"] for b in blocks} and planned["n_people"] == len(people),
            "Frozen plan population or blocks differ")
    require(contract["roles_state"] == "EXPLICITLY_FROZEN", "Production roles must be frozen")
    require(job["role_counts"] == {r:len(v) for r,v in roles.items()}, "Role counts changed")
    require(job["expected_chromosomes"] == sorted(contract["sources"], key=int), "Chromosome scope differs")
    require(set(job["upstream"]) == set(contract["sources"]), "Exactly one source receipt per chromosome required")
    require(job["expected_partials"] == len(tasks) and job["expected_blocks"] == len(blocks), "Task/block scope differs")
    require(job["scopes"] == [contract["exclusion_scope"], *contract.get("additional_exclusion_scopes", [])],
            "Both registered exclusion scopes must be assembled in their fixed order")
    require(len(set(v["uri"] for v in job["upstream"].values())) == len(job["upstream"]), "Duplicate upstream receipt")
    require(0 < job["max_runtime_seconds"] <= 43200 and 0 < job["wait_seconds"] <= 604800,
            "Bounded execution and waiting deadlines required")
    require(30 <= job["poll_seconds"] <= 3600 and job["min_free_bytes"] >= 12*GIB, "Unsafe polling or disk reserve")
    require(job["task_cpus"] == contract["resources"]["threads"], "Workflow resource contract differs")
    require(job["task_memory_gib"]*1024 >= contract["resources"]["max_array_mb"] +
            contract["resources"]["plink_memory_mb"] + 512, "Task memory below workflow admission")
    require(0 < job["max_forks"] <= job["vm_cpus"] and job["memory_reserve_gib"] >= 2 and
            job["max_forks"]*job["task_memory_gib"] + job["memory_reserve_gib"] <= job["vm_memory_gib"],
            "Assembly concurrency exceeds worker memory or CPUs")
    for chrom, source in job["upstream"].items():
        require(source["uri"].startswith(publication.PREFIX + "r02-structure-materialize-") and
                source["uri"].endswith(f"/chr{int(chrom):02d}/completed.json"), "Wrong chromosome receipt scope")
        publication.validate_sha(source["launch_sha256"])
    job["launch_sha256"] = expected
    return job, contract, people, roles, tasks, memberships


def fetch_json(uri, *, allow_missing=False):
    """Read one bounded immutable generation; distinguish absence from failure."""
    result = subprocess.run(["gcloud", "storage", "objects", "describe", uri, "--format=json"],
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        if allow_missing and staging.missing_gcs_object(result, uri):
            return None
        raise RuntimeError("Upstream receipt unavailable, not confirmed absent: " + result.stderr[-500:])
    meta = json.loads(result.stdout)
    require(0 < int(meta["size"]) <= 8*2**20, "Receipt exceeds eight MiB")
    payload = subprocess.check_output(["gcloud", "storage", "cat", uri + "#" + str(meta["generation"])], timeout=120)
    md5 = base64.b64encode(hashlib.md5(payload).digest()).decode()
    require(len(payload) == int(meta["size"]) and md5 == meta.get("md5_hash", meta.get("md5Hash")), "Receipt checksum differs")
    record = dict(uri=uri, bytes=len(payload), generation=str(meta["generation"]),
                  md5_base64=md5, sha256=hashlib.sha256(payload).hexdigest())
    return json.loads(payload), record


def dependency_records(job, chrom, document, tasks):
    origin = job["upstream"][chrom]
    require(document["schema"] == "r02_structure_materialize_queue_v1" and
            document["status"] == "CHROMOSOME_PARTIALS_PUBLISHED_NOT_SCORED" and
            document["chromosome"] == chrom and document["launch_sha256"] == origin["launch_sha256"] and
            document["contract_sha256"] == job["contract_sha256"] and document["training_allowed"] is False,
            "Upstream chromosome identity/contract differs")
    names = [row["name"] for row in document["files"]]
    require(len(names) == len(set(names)), "Duplicate upstream artifact")
    records = {}
    prefix = origin["uri"].rsplit("/", 1)[0] + "/"
    for row in document["files"]:
        name = publication.safe_relative(row["name"]).as_posix()
        require(row["uri"] == prefix + name, "Upstream artifact escaped chromosome prefix")
        if name.startswith("published/partial_"):
            task = Path(name).parts[1].removeprefix("partial_")
            require(task in tasks and tasks[task]["chromosome"] == chrom, "Unexpected partial task/chromosome")
            if len(Path(name).parts) > 3:
                require(Path(name).parts[2] == "scratch", "Unexpected nested partial artifact")
                continue
            allowed = {"manifest.json", "data.npz", "selection.json", "common_markers.tsv"}
            if tasks[task]["kind"] == "target":
                allowed.add("markers.tsv")
            # Scratch/tool logs are traceability only, never analysis inputs.
            require(Path(name).name in allowed, "Unexpected direct partial artifact")
            record = staging.normalize_record(row)
            require("sha256" in record, "Partial SHA256 required")
            records.setdefault(task, {})[Path(name).name] = record
    required = {t for t,r in tasks.items() if r["chromosome"] == chrom}
    require(set(records) == required, "Chromosome partial coverage incomplete")
    for task, files in records.items():
        expected = {"manifest.json", "data.npz", "selection.json", "common_markers.tsv"}
        if tasks[task]["kind"] == "target":
            expected.add("markers.tsv")
        require(set(files) == expected, "Partial artifact inventory incomplete")
    return records


def probe(root, job, tasks):
    folder = root/"dependencies"
    folder.mkdir(exist_ok=True, mode=0o700)
    documents, records, missing = {}, {}, []
    for chrom in job["expected_chromosomes"]:
        cached = folder/f"chr{int(chrom):02d}.json"
        if cached.exists():
            entry = read(cached)
            require(entry["receipt"]["uri"] == job["upstream"][chrom]["uri"], "Pinned dependency URI differs")
            current, receipt = fetch_json(entry["receipt"]["uri"])
            require(receipt == entry["receipt"] and current == entry["document"], "Pinned dependency generation/content changed")
        else:
            fetched = fetch_json(job["upstream"][chrom]["uri"], allow_missing=True)
            if fetched is None:
                missing.append(chrom)
                continue
            document, receipt = fetched
            entry = dict(document=document, receipt=receipt)
            # Validate before freezing even an incomplete-run dependency.
            dependency_records(job, chrom, document, tasks)
            save(cached, entry)
        selected = dependency_records(job, chrom, entry["document"], tasks)
        require(not (set(records) & set(selected)), "Duplicate task across chromosomes")
        records.update(selected)
        documents[chrom] = entry["receipt"]
    result = dict(schema=SCHEMA, status="WAITING" if missing else READY,
        launch_sha256=job["launch_sha256"], contract_sha256=job["contract_sha256"],
        chromosome_receipts=documents, missing_chromosomes=missing, partials=records,
        n_chromosomes=len(documents), n_partials=len(records), training_allowed=False)
    if not missing:
        require(set(records) == set(tasks), "Complete task coverage differs from frozen plan")
        save(root/"dependencies-ready.json", result)
    return result


def epoch_deadline(root, job, stage, amendment=None, amendment_sha256=None):
    """Keep the original clock, allowing a separately authenticated wait extension.

    A controller's polling deadline does not override a previously persisted
    waiting deadline. Both must use the same reviewed absolute endpoint. This
    amendment never modifies the launch, the old receipt, or execution time.
    """
    require(stage in {"wait", "execution"}, "Unknown deadline stage")
    require(bool(amendment) == bool(amendment_sha256), "Both wait amendment and SHA256 required")
    require(not amendment or stage == "wait", "Execution deadline cannot use a wait amendment")
    path = root/(stage + "-deadline.json")
    require(not amendment or path.exists(), "Wait amendment requires an existing original clock")
    if not path.exists():
        save(path, dict(launch_sha256=job["launch_sha256"], started_epoch=time.time(),
                       seconds=job["wait_seconds"] if stage == "wait" else job["max_runtime_seconds"]))
    value = read(path)
    require(value["launch_sha256"] == job["launch_sha256"], "Deadline belongs to another launch")
    endpoint = value["started_epoch"] + value["seconds"]
    if amendment:
        publication.validate_sha(amendment_sha256)
        require(sha(amendment) == amendment_sha256, "Wait amendment checksum differs")
        update = read(amendment)
        require(update["schema"] == "r02_assembly_wait_amendment_v1" and
                update["launch_sha256"] == job["launch_sha256"] and
                update["original_wait_receipt_sha256"] == sha(path),
                "Wait amendment belongs to another launch or clock")
        require(update["training_allowed"] is False and update["scientific_changes"] == [] and
                update["execution_runtime_seconds"] == job["max_runtime_seconds"],
                "Wait amendment cannot change science or execution budget")
        publication.validate_sha(update["authorization_receipt_sha256"])
        parsed = datetime.fromisoformat(update["wait_until_utc"].replace("Z", "+00:00"))
        require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
                "Wait amendment requires an absolute UTC deadline")
        endpoint = parsed.timestamp()
        require(value["started_epoch"] + value["seconds"] < endpoint <= value["started_epoch"] + 604800,
                "Wait extension must advance the old deadline within seven days")
        require(datetime.now(timezone.utc).timestamp() < endpoint, "Amended wait deadline has expired")
    remaining = endpoint - time.time()
    require(remaining > 0, "Absolute campaign deadline reached; resume does not reset it")
    return time.monotonic() - (time.time() - value["started_epoch"]), remaining


def estimate_storage(job, contract, people, roles, dependencies):
    """Conservative bound including duplicated task/index/publication outputs.

    Pair archives contain three nested arms, support counts and ordered pair
    keys. The 256-byte per-pair bound includes these arrays and metadata slack.
    All target archives are counted from actual upstream sizes. Three complete
    copies per regime cover block tasks, the index copy and publishDir copy.
    """
    pairs = sum(len(indices)*(len(indices)-1)//2 - sum(n*(n-1)//2 for n in
        Counter(people[i]["component"] for i in indices).values()) for indices in roles.values())
    staged = sum(r["bytes"] for files in dependencies["partials"].values() for r in files.values())
    targets = sum(files["data.npz"]["bytes"] for name,files in dependencies["partials"].items() if name.startswith("target_"))
    per_scope = targets + job["expected_blocks"]*(pairs*256 + len(people)*256 + 2**20)
    total = staged + 3*len(job["scopes"])*per_scope + 5*GIB
    return dict(schema="r02_structure_assembly_storage_v1",launch_sha256=job["launch_sha256"],
                contract_sha256=job["contract_sha256"],n_partials=len(dependencies["partials"]),
                n_pairs=pairs, staged_bytes=staged, one_scope_output_bound_bytes=per_scope,
                required_free_bytes=total+job["min_free_bytes"],
                intermediate_and_outputs_bound_bytes=total, raw_vcf_bytes=0,
                basis="Pinned partial sizes plus 256 bytes/pair/block; three complete copies per scope, five GiB logs/cache")


def stage_partials(root, job, dependencies, tasks, people, started):
    directory = root/"partials"
    directory.mkdir(exist_ok=True, mode=0o700)
    rows = []
    for task, files in dependencies["partials"].items():
        staging.remaining(started, job)
        target = directory/("partial_" + task)
        target.mkdir(exist_ok=True, mode=0o700)
        for filename, record in files.items():
            staging.check_disk(root, job, 0 if (target/filename).exists() else record["bytes"])
            staging.stage_one(record, target/filename, started, job)
        manifest = read(target/"manifest.json")
        require(manifest["request"] == tasks[task] and manifest["contract_sha256"] == job["contract_sha256"] and
                manifest["schema"] == "r02_structure_materialize_partial_v1" and manifest["stage"] == "COMPLETE_SLICE" and
                manifest["code_sha256"] == code_hashes(root) and manifest["selection_used"] == "FIT_ONLY" and
                manifest["evaluated_outcomes_for_selection"] is False and manifest["training_allowed"] is False and
                manifest["sample_ids_sha256"] == hashlib.sha256(("\n".join(p["sample_id"] for p in people)+"\n").encode()).hexdigest(),
                "Partial manifest scope/code/cohort changed")
        require(set(manifest["outputs_sha256"]) == set(files)-{"manifest.json"}, "Partial output contract differs")
        for filename, digest in manifest["outputs_sha256"].items():
            require(digest == files[filename]["sha256"] == sha(target/filename), "Partial internal/external SHA differs")
        rows.append(dict(path=str(target), manifest_sha256=sha(target/"manifest.json")))
    save(root/"partials.json", dict(schema="r02_structure_partial_locations_v1", partials=rows))
    return root/"partials.json"


def run_scope(root, job, scope, started):
    folder = root/"scopes"/scope
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    params = dict(stage="assemble", bundle=str(root/"bundle"), code=str(root/"frozen/bin"),
        output_dir=str(folder/"published"), contract_sha256=job["contract_sha256"],
        partials_manifest=str(root/"partials.json"), partials_manifest_sha256=sha(root/"partials.json"),
        task_cpus=job["task_cpus"], memory_gb=job["task_memory_gib"], max_forks=job["max_forks"],
        task_time=job["task_time"], exclusion_scope=scope)
    save(folder/"params.json", params)
    attempt = str(time.time_ns())
    logs = {k:folder/f"{k}.{attempt}.{ext}" for k,ext in (("trace","tsv"),("nextflow","log"),("console","log"))}
    command = [job["nextflow_path"], "-C", str(root/job["config_path"]), "-log", str(logs["nextflow"]),
        "run", str(root/"frozen/workflows/r02_structure_materialize.nf"), "-params-file", str(folder/"params.json"),
        "-work-dir", str(folder/"work"), "-with-trace", str(logs["trace"]), "-ansi-log", "false"]
    if (folder/".nextflow").exists():
        command.append("-resume")
    save(folder/f"command.{attempt}.json", dict(command=command, launch_sha256=job["launch_sha256"]))
    env = {**os.environ, "NXF_OFFLINE":"true", "NXF_DISABLE_CHECK_LATEST":"true", "NXF_SYNTAX_PARSER":"v1",
           "NXF_OPTS":"-Xms64m -Xmx768m", "PYTHONDONTWRITEBYTECODE":"1"}
    with logs["console"].open("x") as stream:
        process = subprocess.Popen(command, cwd=folder, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                staging.remaining(started, job)
                staging.check_disk(root, job)
                time.sleep(2)
            require(process.returncode == 0, "Assembly Nextflow failed; retain all inputs, outputs and cache")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            raise
    with logs["trace"].open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(rows and all(r["status"] in {"COMPLETED","CACHED"} and r["exit"] == "0" for r in rows),
            "Unsuccessful assembly task in trace")
    require(len(rows) == job["expected_blocks"]+2, "Assembly must execute PLAN, each block, and INDEX only")
    counts = {kind:sum(kind in row["name"] for row in rows) for kind in (
        "R02_STRUCTURE_MATERIALIZE_PLAN", "R02_STRUCTURE_MATERIALIZE_ASSEMBLE_BLOCK", "R02_STRUCTURE_MATERIALIZE_INDEX")}
    require(list(counts.values()) == [1,job["expected_blocks"],1], "Unexpected process in assembly DAG")
    return folder


def verify_prepared(root, job, scope, directory):
    receipt = read(directory/"manifest.json")
    require(receipt["stage"] == "COMPLETE_BLOCK_INDEX_NO_TRAINING" and receipt["training_allowed"] is False and
            receipt["source_contract_sha256"] == job["contract_sha256"] and receipt["code_sha256"] == code_hashes(root)
            and receipt["exclusion_scope"] == scope and receipt["n_blocks"] == job["expected_blocks"],
            "Prepared index scope differs")
    for name, digest in receipt["outputs_sha256"].items():
        require(Path(name).name == name and sha(directory/name) == digest, "Prepared index changed")
    index = read(directory/"block_index.json")
    expected = table(root/"bundle/blocks.tsv")
    require([r["block_id"] for r in index["blocks"]] == [r["block_id"] for r in expected], "Ordered block coverage differs")
    for row in index["blocks"]:
        require(row["path"] == "shards/"+row["block_id"], "Unsafe shard path")
        shard = directory/row["path"]
        require(sha(shard/"manifest.json") == row["manifest_sha256"] and
                sha(shard/"contract.json") == row["contract_sha256"], "Shard index linkage differs")
        manifest = read(shard/"manifest.json")
        require(manifest["source_contract_sha256"] == job["contract_sha256"] and
                manifest["exclusion_scope"] == scope and manifest["block_id"] == row["block_id"] and
                manifest["training_allowed"] is False, "Shard scope differs")
        for name, digest in manifest["outputs_sha256"].items():
            require(Path(name).name == name and sha(shard/name) == digest, "Shard output changed")
    return receipt


def publish_scope(root, job, scope):
    folder = root/"scopes"/scope
    prepared = folder/"published/prepared"
    verify_prepared(root, job, scope, prepared)
    if (folder/"completed.publication.json").exists():
        record = read(folder/"completed.publication.json")
        require(sha(folder/"completed.json") == record["sha256"], "Preserved completion changed")
        for artifact in read(folder/"completed.json")["files"]:
            publication.verify_remote(artifact, pinned=True)
        publication.verify_remote(record, pinned=True)
        return record
    plan_path = folder/"publication-plan.json"
    if not plan_path.exists():
        paths = sorted(p for p in prepared.rglob("*") if p.is_file())
        paths += sorted(p for p in folder.iterdir() if p.is_file() and
                        (p.name.startswith(("trace.", "nextflow.", "console.", "command.")) or p.name == "params.json"))
        records = []
        for path in paths:
            require(not path.is_symlink() and path.resolve().is_relative_to(folder), "Unsafe output path")
            name = path.relative_to(folder).as_posix()
            records.append(dict(name=name, path=str(path), uri=job["output_prefix"]+"/"+scope+"/"+name,
                                **pipeline.publication_digests(path)))
        save(plan_path, dict(launch_sha256=job["launch_sha256"], scope=scope, files=records))
    plan = read(plan_path)
    require(plan["launch_sha256"] == job["launch_sha256"] and plan["scope"] == scope, "Publication plan changed")
    completion = folder/"completed.json"
    if completion.exists():
        value = read(completion)
        require(value["publication_plan_sha256"] == sha(plan_path), "Completion plan changed")
        for record in value["files"]:
            publication.verify_remote(record, pinned=True)
    else:
        for record in plan["files"]:
            publication.upload_one(record)
            publication.verify_remote(record, pinned=True)
        value = dict(schema=SCHEMA, status="SCOPE_PUBLISHED_NOT_SCORED", scope=scope, training_allowed=False,
            launch_sha256=job["launch_sha256"], contract_sha256=job["contract_sha256"],
            dependencies_sha256=sha(root/"dependencies-ready.json"), publication_plan_sha256=sha(plan_path), files=plan["files"])
        save(completion, value)
    record = dict(name="completed.json", path=str(completion), uri=job["output_prefix"]+"/"+scope+"/completed.json",
                  **pipeline.publication_digests(completion))
    publication.upload_one(record)
    publication.verify_remote(record, pinned=True)
    save(folder/"completed.publication.json", record)
    return record


def preserve_worker(root, job, dependencies):
    """Preserve unique state; duplicate arrays point to verified bucket copies.

    Nothing is deleted here. Symlinks record their link text, while their owned
    targets are inventoried separately. Unknown large files prevent retirement
    instead of being silently discarded. A stopped/restarted analysis retains
    the same Nextflow cache and work-directory structure for manual recovery.
    """
    receipt_path = root/"preservation.publication.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(sha(root/"preservation.json") == receipt["sha256"], "Preservation receipt changed")
        for record in read(root/"preservation.json")["files"]:
            publication.verify_remote(record, pinned=True)
        publication.verify_remote(receipt, pinned=True)
        return receipt
    known = {}
    for files in dependencies["partials"].values():
        for record in files.values():
            known[record["sha256"]] = dict(record, md5_base64=record["md5Hash"])
    for scope in job["scopes"]:
        for record in read(root/"scopes"/scope/"completed.json")["files"]:
            known[record["sha256"]] = record
    directory = root/"preservation"
    directory.mkdir(exist_ok=True, mode=0o700)
    inventory_path, archive = directory/"inventory.json", directory/"unique-state.tar.gz"
    if not inventory_path.exists():
        rows, unique = [], []
        excluded = {"assembly.lock", "complete.json", "complete.publication.json",
                    "preservation.json", "preservation.publication.json"}
        for path in sorted(root.rglob("*")):
            if path.is_relative_to(directory) or path.relative_to(root).as_posix() in excluded:
                continue
            name = path.relative_to(root).as_posix()
            if path.is_symlink():
                rows.append(dict(path=name, kind="SYMLINK", target=os.readlink(path)))
            elif path.is_file():
                signature = (path.stat().st_size, path.stat().st_mtime_ns)
                digest = sha(path)
                require(signature == (path.stat().st_size,path.stat().st_mtime_ns), "File changed during preservation: "+name)
                if digest in known:
                    rows.append(dict(path=name, kind="REUSED_PUBLISHED_BYTES", sha256=digest,
                                     bytes=signature[0], source=known[digest]))
                else:
                    unique.append(path)
                    rows.append(dict(path=name, kind="ARCHIVED_UNIQUE_BYTES", sha256=digest, bytes=signature[0]))
        total = sum(p.stat().st_size for p in unique)
        require(total <= job["max_unique_state_bytes"], "Unexpected unique large files; keep worker disk and review")
        staging.check_disk(root, job, 2*total)
        temporary = directory/"unique-state.tar.gz.partial"
        require(not temporary.is_symlink(), "Unsafe preservation archive")
        # An incomplete archive has no publication receipt and may be rebuilt;
        # immutable original artifacts are never overwritten or removed.
        with tarfile.open(temporary,"w:gz",dereference=False) as tar:
            for path in unique:
                tar.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
            for row in rows:
                if row["kind"] == "SYMLINK":
                    tar.add(root/row["path"],arcname=row["path"],recursive=False)
        with tarfile.open(temporary,"r:gz") as tar:
            for row in rows:
                if row["kind"] == "ARCHIVED_UNIQUE_BYTES":
                    with tar.extractfile(row["path"]) as stream:
                        require(hashlib.file_digest(stream,"sha256").hexdigest() == row["sha256"], "Archived state checksum differs")
        if archive.exists():
            # A crash between archive rename and inventory publication may
            # leave equivalent gzip bytes with a different compression mtime.
            with tarfile.open(archive,"r:gz") as tar:
                require(set(tar.getnames()) == {r["path"] for r in rows if r["kind"] in
                        {"ARCHIVED_UNIQUE_BYTES","SYMLINK"}}, "Preserved archive content differs")
                for row in rows:
                    if row["kind"] == "ARCHIVED_UNIQUE_BYTES":
                        with tar.extractfile(row["path"]) as stream:
                            require(hashlib.file_digest(stream,"sha256").hexdigest() == row["sha256"], "Existing archive checksum differs")
            temporary.unlink()
        else:
            temporary.rename(archive)
        save(inventory_path, dict(schema="r02_worker_preservation_inventory_v1", training_allowed=False,
            launch_sha256=job["launch_sha256"], rows=rows,
            excluded_transient_paths=sorted(excluded), archive_sha256=sha(archive),
            explanation="Regular file bytes preserved once; repeated array copies reference pinned publications. Lock and later closure receipts excluded explicitly."))
    inventory = read(inventory_path)
    require(inventory["launch_sha256"] == job["launch_sha256"] and inventory["archive_sha256"] == sha(archive),
            "Preservation inventory differs")
    reused = {r["source"]["uri"]:r["source"] for r in inventory["rows"] if r["kind"] == "REUSED_PUBLISHED_BYTES"}
    for record in reused.values():
        publication.verify_remote(record,pinned=True)
    if (root/"preservation.json").exists():
        # A failed final receipt upload must not rewrite immutable records
        # merely because retry uploads would now be labelled REUSED.
        value = read(root/"preservation.json")
        require(value["launch_sha256"] == job["launch_sha256"] and value["inventory_sha256"] == sha(inventory_path),
                "Preserved state belongs to another inventory")
        for record in value["files"]:
            publication.verify_remote(record,pinned=True)
    else:
        files = []
        for path in (inventory_path,archive):
            record = dict(name=path.name,path=str(path),uri=job["output_prefix"]+"/preservation/"+path.name,
                          **pipeline.publication_digests(path))
            publication.upload_one(record)
            publication.verify_remote(record,pinned=True)
            files.append(record)
        value = dict(schema=SCHEMA,status="UNIQUE_STATE_AND_REUSED_BYTES_PRESERVED", training_allowed=False,
            launch_sha256=job["launch_sha256"],contract_sha256=job["contract_sha256"],
            inventory_sha256=sha(inventory_path), files=files)
        save(root/"preservation.json",value)
    record = dict(name="preservation.json",path=str(root/"preservation.json"),uri=job["output_prefix"]+"/preservation.json",
                  **pipeline.publication_digests(root/"preservation.json"))
    publication.upload_one(record)
    publication.verify_remote(record,pinned=True)
    save(receipt_path,record)
    return record


def verify_complete(root, job):
    result = read(root/"complete.json")
    require(result["schema"] == SCHEMA and result["status"] == COMPLETE and result["training_allowed"] is False and
            result["launch_sha256"] == job["launch_sha256"] and result["contract_sha256"] == job["contract_sha256"] and
            set(result["scopes"]) == set(job["scopes"]), "Assembly completion differs")
    require(sha(root/"dependencies-ready.json") == result["dependencies_sha256"], "Dependencies changed")
    preserved = result["preservation"]
    require(preserved == read(root/"preservation.publication.json") and preserved["uri"] == job["output_prefix"]+"/preservation.json"
            and sha(root/"preservation.json") == preserved["sha256"], "Unique state preservation missing or changed")
    publication.verify_remote(preserved,pinned=True)
    preservation = read(root/"preservation.json")
    require(preservation["status"] == "UNIQUE_STATE_AND_REUSED_BYTES_PRESERVED" and
            preservation["launch_sha256"] == job["launch_sha256"], "Preservation scope differs")
    for record in preservation["files"]:
        publication.verify_remote(record,pinned=True)
    inventory = read(root/"preservation/inventory.json")
    require(sha(root/"preservation/inventory.json") == preservation["inventory_sha256"], "Preservation inventory changed")
    reused = {row["source"]["uri"]:row["source"] for row in inventory["rows"] if row["kind"] == "REUSED_PUBLISHED_BYTES"}
    for record in reused.values():
        publication.verify_remote(record,pinned=True)
    for scope, record in result["scopes"].items():
        require(record["uri"] == job["output_prefix"]+"/"+scope+"/completed.json", "Completion escaped output prefix")
        publication.verify_remote(record, pinned=True)
        local = root/"scopes"/scope/"completed.json"
        require(sha(local) == record["sha256"], "Scope completion linkage differs")
        value = read(local)
        require(value["scope"] == scope and value["contract_sha256"] == job["contract_sha256"] and
                value["launch_sha256"] == job["launch_sha256"] and value["training_allowed"] is False,
                "Scope completion contract differs")
        for artifact in value["files"]:
            require(artifact["uri"] == job["output_prefix"]+"/"+scope+"/"+publication.safe_relative(artifact["name"]).as_posix(),
                    "Output escaped scope prefix")
            publication.verify_remote(artifact, pinned=True)
    final = read(root/"complete.publication.json")
    require(final["uri"] == job["output_prefix"]+"/complete.json" and final["sha256"] == sha(root/"complete.json"),
            "Final publication linkage differs")
    publication.verify_remote(final, pinned=True)
    return dict(status="SAFE_TO_RETIRE_ASSEMBLY_WORKER_AFTER_ID_CHECK", training_allowed=False,
                scopes=job["scopes"], blocks_per_scope=job["expected_blocks"])


def execute(root, job, contract, people, roles, tasks):
    if (root/"complete.publication.json").exists():
        return verify_complete(root, job)
    started, _ = epoch_deadline(root, job, "execution")
    dependencies = probe(root, job, tasks)
    require(dependencies["status"] == READY, "Worker execution requires all dependencies; wait on the lightweight controller")
    filesystem = subprocess.check_output(["findmnt","-n","-o","FSTYPE","-T",str(root)],text=True).strip()
    require(filesystem in {"ext4","xfs","btrfs"}, "Assembly needs local or block storage, never FUSE/NFS")
    budget = estimate_storage(job, contract, people, roles, dependencies)
    save(root/"storage-admission.json", budget)
    # Resume must count existing owned files as occupied budget, not demand the
    # entire campaign size again in addition to completed outputs.
    owned = sum(p.stat().st_size for top in (root/"partials", root/"scopes") if top.exists()
                for p in top.rglob("*") if p.is_file() and not p.is_symlink())
    staging.check_disk(root, job, max(0, budget["intermediate_and_outputs_bound_bytes"]-owned))
    stage_partials(root, job, dependencies, tasks, people, started)
    results = {}
    for scope in job["scopes"]:
        staging.remaining(started, job)
        folder = root/"scopes"/scope
        if not (folder/"completed.json").exists():
            run_scope(root, job, scope, started)
        results[scope] = publish_scope(root, job, scope)
    preserved = preserve_worker(root,job,dependencies)
    completion = dict(schema=SCHEMA, status=COMPLETE, training_allowed=False,
        launch_sha256=job["launch_sha256"], contract_sha256=job["contract_sha256"],
        dependencies_sha256=sha(root/"dependencies-ready.json"), scopes=results,
        chromosomes=job["expected_chromosomes"], n_partials=job["expected_partials"],
        blocks_per_scope=job["expected_blocks"], preservation=preserved)
    save(root/"complete.json", completion)
    record = dict(name="complete.json", path=str(root/"complete.json"), uri=job["output_prefix"]+"/complete.json",
                  **pipeline.publication_digests(root/"complete.json"))
    publication.upload_one(record)
    publication.verify_remote(record, pinned=True)
    save(root/"complete.publication.json", record)
    return verify_complete(root, job)


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--wait-deadline-amendment", type=Path)
    parser.add_argument("--wait-deadline-amendment-sha256")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--probe", action="store_true")
    modes.add_argument("--wait-ready", action="store_true")
    modes.add_argument("--verify-complete", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    require(not args.wait_deadline_amendment or args.probe or args.wait_ready,
            "Wait amendment applies only to dependency polling, never execution")
    require(bool(args.wait_deadline_amendment) == bool(args.wait_deadline_amendment_sha256),
            "Both wait amendment and SHA256 required")
    job, contract, people, roles, tasks, _ = validate(args.root, args.launch, args.launch_sha256)
    with (args.root/"assembly.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.verify_only:
            result = dict(status="ASSEMBLY_CONTRACT_VERIFIED_NOT_RUNNING",training_allowed=False,
                          n_partials=len(tasks),n_blocks=job["expected_blocks"])
        elif args.verify_complete:
            result = verify_complete(args.root, job)
        elif args.probe or args.wait_ready:
            while True:
                _, left = epoch_deadline(args.root, job, "wait", args.wait_deadline_amendment,
                                        args.wait_deadline_amendment_sha256)
                result = probe(args.root, job, tasks)
                if result["status"] == READY:
                    budget = estimate_storage(job,contract,people,roles,result)
                    save(args.root/"readiness-budget.json",budget)
                    result = dict(result,storage_admission=budget)
                print(json.dumps({k:v for k,v in result.items() if k not in {"partials", "chromosome_receipts"}}), flush=True)
                if result["status"] == READY or args.probe:
                    break
                time.sleep(min(job["poll_seconds"], left))
            if result["status"] != READY:
                raise SystemExit(3)
        else:
            result = execute(args.root, job, contract, people, roles, tasks)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
