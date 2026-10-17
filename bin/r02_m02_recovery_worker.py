#!/usr/bin/env python3
"""Recover only M02 from authenticated M01, one chromosome at a time per VM.

No common/rare selection, model fitting, M01 rerun or ancestry inference. Local
bulk files are removed only after create-only GCS publication has been verified.
Failure retains the unfinished chromosome and its Nextflow cache.
"""
from __future__ import annotations

import argparse
import base64
import csv
import fcntl
from datetime import datetime, timezone
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


require = publication.require
ACCOUNT = "653458115080-compute@developer.gserviceaccount.com"
PROJECT = "uspbr-242713"
GIB = 2**30
RECOVERY_PREFIX = ('gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/checkpoints/'
                   'R02_20260930/r02-m02-recovery-20261004q/')
DELEGATED_SCHEMA = 'r02_m02_recovery_launch_v2'
REQUIRED_FILES = {"sources.json", "resources.config", "frozen/bin/r02_m02_recovery_worker.py",
    "frozen/bin/r02_publish_evidence.py", "frozen/bin/r02_autosome_pipeline.py",
    "frozen/bin/r02_recover_m02_admission.py", "frozen/bin/preprocess_storage_guard.py",
    "frozen/workflows/r02_recover_m02.nf", "frozen/modules/02_preprocess_filter_snv_biallelic_pass.nf",
    "frozen/conf/r02_recover_m02.config"}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return publication.read_unique_json(Path(path))


def write(path, value):
    publication.write_receipt_new(Path(path), value)


def command(args, **kwargs):
    return subprocess.run(args, check=True, timeout=kwargs.pop("timeout", 300), **kwargs)


def corrected_resources_config(root):
    """Only repair a Docker option closure; all scientific configuration stays pinned."""
    original = (root / "resources.config").read_text()
    lines = original.splitlines()
    matches = [i for i, line in enumerate(lines) if line.startswith("docker.runOptions=")]
    require(len(matches) == 1, "Expected exactly one Docker options declaration")
    index = matches[0]
    line = lines[index]
    require(line.startswith('docker.runOptions={ "') and line.endswith('" }'),
            "Only the known Docker closure defect can be amended")
    lines[index] = "docker.runOptions=" + line[len("docker.runOptions={ "):-2]
    require(lines[0] == "includeConfig 'frozen/conf/r02_recover_m02.config'", "Unexpected original config include")
    lines[0] = "includeConfig '" + str(root / "frozen/conf/r02_recover_m02.config") + "'"
    return "\n".join(lines) + "\n"


def apply_runtime_amendment(root, manifest, path, expected_sha, *, runtime_worker_path=None):
    require(path is not None and expected_sha and sha(path) == expected_sha, "Runtime amendment checksum differs")
    amendment = read(path)
    require(amendment["schema"] in {"r02_m02_docker_string_amendment_v1", "r02_m02_legacy_count_amendment_v1"} and
            amendment["parent_launch_sha256"] == manifest["authenticated_launch_sha256"] and
            amendment["root"] == str(root) and amendment["instance"] == manifest["instance"],
            "Runtime amendment belongs to another launch")
    executed_worker = Path(__file__).resolve()
    if runtime_worker_path is not None:
        executed_worker = Path(runtime_worker_path)
        require(amendment['schema'] == 'r02_m02_legacy_count_amendment_v1' and
                executed_worker.resolve() == executed_worker and
                executed_worker == Path(amendment['source_root']).parent / 'bin/r02_m02_recovery_worker.py',
                'Historical finalization must authenticate the exact executed worker')
    require(amendment["worker_sha256"] == sha(executed_worker), "Amended worker checksum differs")
    config = Path(amendment["config_path"])
    require(config.resolve() == config and config.is_file() and sha(config) == amendment["config_sha256"] and
            config.read_text() == corrected_resources_config(root), "Runtime amendment changes more than Docker options")
    manifest["runtime_config"] = str(config)
    manifest["runtime_amendment"] = {"path": str(path), "sha256": expected_sha, **amendment}
    if amendment['schema'] == 'r02_m02_legacy_count_amendment_v1':
        changed = amendment['source_files_sha256']
        require(set(changed) == {'bin/r02_recover_m02_admission.py', 'bin/preprocess_count_validation.py',
            'bin/preprocess_storage_guard.py', 'workflows/r02_recover_m02.nf',
            'modules/02_preprocess_filter_snv_biallelic_pass.nf'}, 'Unexpected amended analysis files')
        source = Path(amendment['source_root'])
        require(source.resolve() == source and source == path.parent / 'source', 'Invalid amended source root')
        for name, digest in changed.items():
            require(sha(source / name) == digest, 'Amended source checksum differs')
        manifest['runtime_source'] = str(source)


def raw_count_evidence(entry, records):
    """Authenticate only a tiny existing M01 log; no genotype download/recount."""
    record = entry['norm_log']
    require(record['uri'] == entry['source']['uri'].removesuffix('.vcf.gz') + '.log',
            'Normalization log is not the sibling of this exact M01 source')
    authenticate_remote(record)
    data = command(['gcloud', 'storage', 'cat', record['uri'] + '#' + record['generation']],
                   capture_output=True, timeout=60).stdout
    require(len(data) == int(record['bytes']) and len(data) < 4096 and
            base64.b64encode(hashlib.md5(data).digest()).decode() == record['md5Hash'],
            'Pinned normalization log content differs')
    text = data.decode()
    require(text.strip() == entry['header_probe']['norm_log'].strip() and
            entry['header_probe']['norm_log_md5_verified'], 'Normalization log differs from prior source audit')
    match = re.fullmatch(r'Lines\s+total/split/realigned/skipped:\s*(\d+)/(\d+)/(\d+)/(\d+)\s*', text)
    require(match is not None and int(match[1]) > 0 and int(match[4]) == 0,
            'Invalid normalization input count or skipped records')
    return dict(schema='r02_m02_raw_count_evidence_v1', chromosome=entry['chromosome'],
        normalized_source_uri=entry['source']['uri'], normalized_source_generation=entry['source']['generation'],
        raw_index_sha256=records['raw_index']['sha256'], raw_records=int(match[1]),
        normalization_log={**record, 'text':text, 'sha256':hashlib.sha256(data).hexdigest()},
        interpretation='M01 normalization input count; original-site annotation preserves record count')


def publication_file_count(manifest):
    return 11 if manifest.get('runtime_source') else (9 if manifest.get('runtime_amendment') else 8)


def delegated_resources_config(root, label):
    """The tested v4 configuration, with only root and ownership label varying."""
    require(re.fullmatch(r'dnabr-r02=m02-recovery-[e-h]-1004q', label), 'Wrong delegated container label')
    require(root.is_absolute() and root.resolve() == root and not any(c in str(root) for c in "'\"\r\n"),
            'Invalid delegated config root')
    return ("includeConfig '" + str(root / 'frozen/conf/r02_recover_m02.config') + "'\n"
            "docker.enabled=true\n"
            "process.container='sha256:76a0fddd300d3634363b5a13a1d28983122644a42e7d6c18339f9c8ac08548a7'\n"
            "docker.writableInputMounts=false\n"
            'docker.runOptions="--network none --user ${System.getenv(\'R02_WORKER_UID\')}:${System.getenv(\'R02_WORKER_GID\')} '
            '--label ' + label + ' --cpus 8 --memory 16g -v ' + str(root) + ':' + str(root) + ':ro '
            '-v ${params.preprocess_large_temp_dir}:${params.preprocess_large_temp_dir}:rw"\n')


def lineage_document(root, manifest, key):
    descriptor = manifest[key]
    path = Path(descriptor['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_relative_to(root) and path.is_file(),
            'Lineage document must be frozen inside this launch')
    require(re.fullmatch(r'[a-f0-9]{64}', descriptor['sha256']) and sha(path) == descriptor['sha256'] and
            manifest['files_sha256'].get(str(path.relative_to(root))) == descriptor['sha256'],
            'Lineage document checksum differs')
    return read(path)


def validate_delegation(root, manifest, inventory):
    """A new VM takes only an authenticated, paused owner's pending tail.

    Parent inputs and per-chromosome destinations remain byte-for-byte
    compatible with the frozen FIT queue. The handoff receipt must already
    exist in GCS; this function never signals or authorizes stopping a VM.
    """
    letter = manifest['instance'].split('-')[-2]
    require(letter in 'efgh', 'Unexpected delegated VM identity')
    owner = chr(ord(letter) - 4)
    require(manifest['publication_owner'] == owner and manifest['output_prefix'] == RECOVERY_PREFIX + 'worker-' + owner,
            'Delegated publication must retain the original owner')
    require(manifest['campaign_output_prefix'] == manifest['output_prefix'] + '/delegated/' + manifest['instance'],
            'Delegated campaign cannot replace the original completion receipt')
    parent = lineage_document(root, manifest, 'parent_launch')
    original_sources = lineage_document(root, manifest, 'parent_sources')
    handoff = lineage_document(root, manifest, 'handoff_receipt')
    require(parent['schema'] == 'r02_m02_recovery_launch_v1' and
            parent['instance'] == 'dnabr-r02-m02-' + owner + '-1004' and
            parent['project'] == PROJECT and parent['zone'] == manifest['zone'] and
            parent['output_prefix'] == manifest['output_prefix'] and
            parent['source_stage'] == manifest['source_stage'] and parent['target_stage'] == manifest['target_stage'],
            'Original launch scope differs')
    require(parent['files_sha256']['sources.json'] == manifest['parent_sources']['sha256'] and
            handoff['schema'] == 'r02_m02_pending_handoff_v1' and
            handoff['status'] == 'PENDING_M02_DELEGATED_SUPERVISOR_PAUSED' and
            handoff['original_instance'] == parent['instance'] and
            handoff['delegate_instance'] == manifest['instance'] and
            handoff['original_launch_sha256'] == manifest['parent_launch']['sha256'] and
            handoff['source_inventory_sha256'] == manifest['parent_sources']['sha256'],
            'Pending handoff is not bound to the original source/owner')
    adopted = handoff['adopted_chromosome']
    require(adopted in parent['chromosomes'] and adopted not in manifest['chromosomes'] and
            handoff['chromosomes'] == manifest['chromosomes'] ==
            parent['chromosomes'][parent['chromosomes'].index(adopted) + 1:] and manifest['chromosomes'],
            'Delegated chromosomes must be exactly the pending tail, never the running chromosome')
    originals = {e['chromosome']: e for e in original_sources['entries']}
    require(len(originals) == len(original_sources['entries']) and list(originals) == parent['chromosomes'] and
            inventory['entries'] == [originals[c] for c in manifest['chromosomes']],
            'Delegated source entries differ from the authenticated original inventory')
    for name in ('supervisor', 'adopted_nextflow'):
        process = handoff[name]
        require(type(process['pid']) is int and process['pid'] > 1 and
                type(process['start_ticks']) is int and process['start_ticks'] > 0 and
                re.fullmatch(r'[a-f0-9]{64}', process['cmdline_sha256']), 'Incomplete handoff process identity')
    require(re.fullmatch(r'[a-f0-9]{64}', handoff['handoff_contract_sha256']) and
            manifest['original_deadline_utc'] == handoff['original_deadline_utc'], 'Handoff/deadline differs')
    deadline = datetime.fromisoformat(manifest['original_deadline_utc'].replace('Z', '+00:00'))
    require(deadline.tzinfo is not None and deadline > datetime.now(timezone.utc), 'Original campaign deadline reached')
    receipt = manifest['handoff_receipt']
    require(receipt['uri'].startswith(RECOVERY_PREFIX + 'worker-' + owner + '/handoff/') and
            receipt['uri'].endswith('/queue_transfer.json'), 'Handoff receipt outside original owner')
    require(pipeline.publication_digests(Path(receipt['path'])) ==
            {k: receipt[k] for k in ('bytes', 'sha256', 'md5_base64')}, 'Handoff receipt bytes changed')
    publication.verify_remote(receipt, pinned=True)
    require(manifest['container_label'] == 'dnabr-r02=m02-recovery-' + letter + '-1004q' and
            (root / 'resources.config').read_text() == delegated_resources_config(root, manifest['container_label']),
            'Delegated resources differ from the tested configuration')
    manifest['runtime_source'] = str(root / 'frozen')


def campaign_prefix(manifest):
    return manifest.get('campaign_output_prefix', manifest['output_prefix'])


def check_runtime(manifest, started):
    require(time.monotonic() - started < manifest['max_runtime_seconds'], 'Campaign runtime limit reached')
    if manifest.get('schema') == DELEGATED_SCHEMA:
        deadline = datetime.fromisoformat(manifest['original_deadline_utc'].replace('Z', '+00:00'))
        require((deadline - datetime.now(timezone.utc)).total_seconds() > 1800,
                'Original campaign deadline reached; preserve remaining publication margin')


def preflight_staged_inputs(root, manifest, inventory):
    """Run real admission on the already staged chromosome; no M02 or staging."""
    require(manifest.get('runtime_source') and manifest.get('runtime_amendment'),
            'Staged preflight requires the legacy-count amendment; new delegated launches use Nextflow admission')
    entry = inventory['entries'][0]
    folder = root / 'chromosomes' / ('chr' + entry['chromosome'].zfill(2))
    receipt = read(folder / 'source-receipt.json')
    require(receipt['source_inventory_sha256'] == sha(root / 'sources.json'), 'Source contract changed')
    records = {key:receipt[key] for key in ('source', 'index', 'raw_index')}
    for record in records.values():
        authenticate_remote(record)
        stat = Path(record['path']).stat()
        require((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) ==
                (record['bytes'], record['mtime_ns'], record['ctime_ns']), 'Staged source changed')
    amendment_root = Path(manifest['runtime_amendment']['path']).parent
    evidence = raw_count_evidence(entry, records)
    path = amendment_root / 'preflight-count.json'
    write(path, evidence)
    image = 'sha256:76a0fddd300d3634363b5a13a1d28983122644a42e7d6c18339f9c8ac08548a7'
    args = ['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
            '--cpus','1','--memory','1g','-v',f'{root}:{root}:ro',
            '-v',f'{amendment_root}:{amendment_root}:ro','-v',f'{folder}/bulk:{folder}/bulk:rw',image,
            'python3',str(Path(manifest['runtime_source']) / 'bin/r02_recover_m02_admission.py'),
            '--inventory',str(folder / 'source-receipt.json'),'--inventory-sha256',sha(folder / 'source-receipt.json'),
            '--chromosome',entry['chromosome'],'--vcf',records['source']['path'],'--index',records['index']['path'],
            '--raw-index',records['raw_index']['path'],'--scratch',str(folder / 'bulk'),
            '--minimum-free-gib','12','--required-free-bytes',str(3*entry['source']['bytes']+32*GIB),
            '--count-evidence',str(path),'--count-evidence-sha256',sha(path),
            '--expected-raw-count',str(evidence['raw_records'])]
    result = command(args,capture_output=True,text=True,timeout=180)
    report = json.loads(result.stdout)
    write(amendment_root / 'preflight-admission.json',report)
    print(json.dumps(dict(status='REAL_STAGED_INPUT_ADMISSION_PASSED_NO_M02',instance=manifest['instance'],
        chromosome=entry['chromosome'],raw_records=evidence['raw_records'],
        normalized_records=report['normalized_index_records'],raw_count_source=report['raw_count_source'],
        receipt_sha256=sha(amendment_root / 'preflight-admission.json'))),flush=True)


def verify_bundle(root, expected):
    require(sha(root / "launch.json") == expected, "Launch manifest changed")
    manifest = read(root / "launch.json")
    require(manifest["schema"] in {"r02_m02_recovery_launch_v1", DELEGATED_SCHEMA}, "Wrong launch schema")
    delegated = manifest['schema'] == DELEGATED_SCHEMA
    require(manifest["source_stage"] == "M01_NORMALIZED" and
            manifest["target_stage"] == "M02_PASS_SNV_BIALLELIC_NO_MAF", "Wrong recovery scope")
    require(re.fullmatch(r"dnabr-r02-m02-" + ('[e-h]' if delegated else '[a-d]') + r"-1004", manifest["instance"]), "Unexpected VM identity")
    require(manifest["project"] == PROJECT and manifest["zone"] == "us-central1-a", "Wrong project/zone")
    require(manifest["root"] == str(root) and root.resolve() == root, "Wrong canonical campaign path")
    require(len(set(manifest["chromosomes"])) == len(manifest["chromosomes"]) and
            set(manifest["chromosomes"]) <= {str(i) for i in range(1, 13)} | {"21"}, "Wrong chromosomes")
    require(type(manifest['max_runtime_seconds']) is int and
            (0 < manifest['max_runtime_seconds'] <= 82800 if delegated else manifest['max_runtime_seconds'] == 82800),
            "Runtime cap must leave publication margin")
    worker = manifest["instance"].split("-")[-2]
    if not delegated:
        require(manifest['output_prefix'] == RECOVERY_PREFIX + 'worker-' + worker, 'Wrong publication scope')
    required = REQUIRED_FILES | ({'frozen/bin/preprocess_count_validation.py'} if delegated else set())
    require(set(manifest["files_sha256"]) >= required, "Required source/code/config hashes omitted")
    for name, digest in manifest["files_sha256"].items():
        path = root / publication.safe_relative(name)
        require(path.resolve() == path and sha(path) == digest, "Frozen file changed: " + name)
    inventory = read(root / "sources.json")
    require([entry["chromosome"] for entry in inventory["entries"]] == manifest["chromosomes"], "Inventory assignment differs")
    if delegated:
        require(sha(root / 'frozen/bin/r02_m02_recovery_worker.py') == sha(Path(__file__).resolve()),
                'Delegated worker differs from its frozen launch')
        validate_delegation(root, manifest, inventory)
    manifest["authenticated_launch_sha256"] = expected
    return manifest, inventory


def authenticate_remote(record):
    actual = publication.remote_metadata(record["uri"])
    require(str(actual["generation"]) == record["generation"] and int(actual["size"]) == record["bytes"] and
            actual.get("md5_hash", actual.get("md5Hash")) == record["md5Hash"], "Remote input generation/size/MD5 changed")


def stage(entry, folder):
    folder.mkdir(mode=0o700)
    records = {}
    for key in ("source", "index", "raw_index"):
        record = entry[key]
        authenticate_remote(record)
        name = record["uri"].rsplit("/", 1)[-1]
        expected = f"dnabr.hg38.2723.chr{entry['chromosome']}" + {
            "source": ".norm.vcf.gz", "index": ".norm.vcf.gz.tbi", "raw_index": ".vcf.gz.tbi"}[key]
        require(name == expected, "Source chromosome/stage basename differs from assignment")
        target = folder / name
        require(not target.exists(), "Do not overwrite staged data")
        part = folder / (name + ".partial")
        command(["gcloud", "storage", "cp", record["uri"] + "#" + record["generation"], str(part)], timeout=7200)
        digest = pipeline.publication_digests(part)
        require(digest["bytes"] == record["bytes"] and digest["md5_base64"] == record["md5Hash"], "Downloaded data checksum differs")
        os.link(part, target)
        part.unlink()
        stat = target.stat()
        records[key] = {**record, **digest, "path": str(target),
                        "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
    require(records["index"]["path"] == records["source"]["path"] + ".tbi", "Normalized index name differs")
    return records


def stop_owned(label):
    ids = subprocess.check_output(["docker", "ps", "--filter", "label=" + label, "--format", "{{.ID}}"], text=True).split()
    require(all(re.fullmatch(r"[a-f0-9]{12,64}", value) for value in ids), "Unexpected container ID")
    for identity in ids:
        command(["docker", "stop", "--time", "30", identity], timeout=60)


def cleanup_bulk(root, folder):
    require(folder.parent == root / "chromosomes" and re.fullmatch(r"chr(?:0[1-9]|1[0-9]|2[0-2])", folder.name),
            "Cleanup folder outside assigned chromosome directory")
    require(folder.resolve() == folder and not folder.is_symlink(), "Unsafe chromosome cleanup")
    for name in ("inputs", "bulk", "work", "published"):
        target = folder / name
        require(target.resolve() == target and target.parent == folder and not target.is_symlink(), "Unsafe temporary cleanup")
        if target.exists():
            require(target.is_dir(), "Temporary path is not a directory")
            shutil.rmtree(target)


def publish_chromosome(root, manifest, folder, plan):
    """Resume the exact immutable publication plan, including receipt-only failures."""
    chrom = plan["chromosome"]
    require(plan["schema"] == "r02_m02_recovery_publication_plan_v1" and
            plan["launch_sha256"] == manifest["authenticated_launch_sha256"] and
            chrom in manifest["chromosomes"] and folder == root / "chromosomes" / ("chr" + chrom.zfill(2)),
            "Publication plan assignment differs")
    expected_count = publication_file_count(manifest)
    require(len(plan["files"]) == expected_count and len({f["name"] for f in plan["files"]}) == expected_count,
            "Incomplete publication plan")
    for record in plan["files"]:
        require(record["uri"] == manifest["output_prefix"] + "/chr" + chrom.zfill(2) + "/" + record["name"],
                "Publication target outside chromosome assignment")
    completed_path = folder / "completed.json"
    if not completed_path.exists():
        for record in plan["files"]:
            publication.upload_one(record)
        for record in plan["files"]:
            publication.verify_remote(record, pinned=True)
        result = dict(schema="r02_m02_recovered_v1", status="M02_RECOVERED_PUBLISHED_VERIFIED_NOT_FIT_SELECTED",
            chromosome=chrom, files=plan["files"], source_receipt_sha256=sha(folder / "source-receipt.json"),
            launch_sha256=manifest["authenticated_launch_sha256"], source_stage="M01_NORMALIZED",
            no_maf_filter=True, no_new_original_allele_filter=True, preserve_all_format=True,
            publication_plan_sha256=sha(folder / "publication-plan.json"), created_utc=datetime.now(timezone.utc).isoformat())
        write(completed_path, result)
    else:
        result = read(completed_path)
        require(result["chromosome"] == chrom and result["launch_sha256"] == manifest["authenticated_launch_sha256"] and
                result["publication_plan_sha256"] == sha(folder / "publication-plan.json"), "Completion receipt changed")
        for record in result["files"]:
            publication.verify_remote(record, pinned=True)
    record = dict(name="completed.json", path=str(completed_path),
        uri=manifest["output_prefix"] + "/chr" + chrom.zfill(2) + "/completed.json", **pipeline.publication_digests(completed_path))
    publication.upload_one(record); publication.verify_remote(record, pinned=True)
    receipt = folder / "completed.publication.json"
    if receipt.exists():
        old = read(receipt)
        require(all(old[k] == record[k] for k in ("uri", "generation", "sha256", "bytes", "md5_base64")), "Receipt changed")
    else:
        write(receipt, record)
    cleanup_bulk(root, folder)
    return record


def publish_complete(root, manifest, results):
    path = root / "complete.json"
    if not path.exists():
        write(path, {"schema": "r02_m02_recovery_campaign_v1", "status": "ALL_ASSIGNED_M02_REMOTE_VERIFIED",
            "instance": manifest["instance"], "chromosomes": manifest["chromosomes"], "results": results,
            "launch_sha256": manifest["authenticated_launch_sha256"], "created_utc": datetime.now(timezone.utc).isoformat()})
    else:
        old = read(path)
        require(old["launch_sha256"] == manifest["authenticated_launch_sha256"] and
                old["chromosomes"] == manifest["chromosomes"] and
                [(r["uri"], r["sha256"], r["generation"]) for r in old["results"]] ==
                [(r["uri"], r["sha256"], r["generation"]) for r in results], "Campaign receipt differs")
    record = dict(name="complete.json", path=str(path), uri=campaign_prefix(manifest) + "/complete.json",
                  **pipeline.publication_digests(path))
    publication.upload_one(record); publication.verify_remote(record, pinned=True)
    receipt = root / "complete.publication.json"
    if receipt.exists():
        old = read(receipt)
        require(all(old[k] == record[k] for k in ("uri", "generation", "sha256", "bytes", "md5_base64")), "Final receipt changed")
    else:
        write(receipt, record)
    return record


def run_chromosome(root, manifest, entry, started, attempt):
    chrom = entry["chromosome"]
    folder = root / "chromosomes" / ("chr" + chrom.zfill(2))
    if (folder / "publication-plan.json").exists():
        return publish_chromosome(root, manifest, folder, read(folder / "publication-plan.json"))
    resume = folder.exists()
    require(not resume or attempt != "initial", "Interrupted chromosome requires explicit resumeN; preserve cache")
    folder.mkdir(mode=0o700, parents=True, exist_ok=resume)
    # Four compressed copies (input, SNV, PASS, published PASS) plus reserve.
    # This is an admission estimate, not a compression-ratio guarantee; watch disk.
    required = 4 * entry["source"]["bytes"] + 32 * GIB
    if resume:
        require((folder / "source-receipt.json").is_file(), "Interrupted staging requires explicit transfer recovery; no overwrite")
        receipt = read(folder / "source-receipt.json")
        require(receipt["source_inventory_sha256"] == sha(root / "sources.json") and receipt["chromosome"] == chrom,
                "Resume source contract changed")
        records = {key: receipt[key] for key in ("source", "index", "raw_index")}
        for record in records.values():
            authenticate_remote(record)
            stat = Path(record["path"]).stat()
            require((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) ==
                    (record["bytes"], record["mtime_ns"], record["ctime_ns"]), "Resume source changed")
        require(shutil.disk_usage(root).free >= 12 * GIB, "Insufficient remaining disk reserve")
    else:
        require(shutil.disk_usage(root).free >= required, "Insufficient local disk for this chromosome")
        records = stage(entry, folder / "inputs")
    receipt = {"schema": "r02_m02_recovery_staged_v1", "chromosome": chrom,
               "source_stage": "M01_NORMALIZED", "source": records["source"],
               "index": records["index"], "raw_index": records["raw_index"],
               "source_inventory_sha256": sha(root / "sources.json"),
               "source_header_evidence": entry["header_probe"]}
    if not resume:
        write(folder / "source-receipt.json", receipt)
    params = {"r02_chrom": chrom, "r02_normalized_vcf": records["source"]["path"],
        "r02_normalized_index": records["index"]["path"], "r02_raw_index": records["raw_index"]["path"],
        "r02_source_receipt": str(folder / "source-receipt.json"),
        "r02_source_receipt_sha256": sha(folder / "source-receipt.json"),
        "r02_bin_dir": str(root / "frozen/bin"), "outdir": str(folder / "published"),
        "preprocess_large_temp_dir": str(folder / "bulk"), "preprocess_minimum_free_gib": 12,
        "preprocess_required_free_bytes": max(0, required - entry["source"]["bytes"]),
        "preprocess_raw_index_only": True,
        "cpus": 8, "memory": "16 GB", "time": "8h",
        "resources": {"preprocess_filter_snv_biallelic_pass": {"threads": 6}},
        "bcftools_min_alleles": 2, "plink_max_alleles": 2, "plink_snps_only": True,
        "keep_pass": True, "max_maf": None}
    (folder / "bulk").mkdir(mode=0o700, exist_ok=resume)
    if resume:
        require(read(folder / "params.json") == params, "Resume parameters differ")
    else:
        write(folder / "params.json", params)
    params_name = 'params.json'
    count_name = None
    if manifest.get('runtime_source'):
        count_name = 'raw-count-evidence.' + attempt + '.json'
        evidence = raw_count_evidence(entry, records)
        write(folder / count_name, evidence)
        params = {**params, 'r02_bin_dir': str(Path(manifest['runtime_source']) / 'bin'),
            'r02_raw_count_evidence':str(folder / count_name),
            'r02_raw_count_evidence_sha256':sha(folder / count_name),
            'preprocess_authenticated_raw_count':evidence['raw_records']}
        params_name = 'params.' + attempt + '.json'
        write(folder / params_name, params)
    trace_name, log_name, console_name = "trace." + attempt + ".tsv", "nextflow." + attempt + ".log", "console." + attempt + ".log"
    require(not any((folder / name).exists() for name in (trace_name, log_name, console_name)), "Attempt logs already exist")
    amendment_name = "execution-amendment." + attempt + ".json"
    if manifest.get("runtime_amendment"):
        write(folder / amendment_name, manifest["runtime_amendment"])
    if manifest.get('schema') == DELEGATED_SCHEMA:
        delegation_path = folder / 'delegation.json'
        if delegation_path.exists():
            require(sha(delegation_path) == manifest['handoff_receipt']['sha256'], 'Stored delegation receipt changed')
        else:
            with delegation_path.open('xb') as handle:
                handle.write(Path(manifest['handoff_receipt']['path']).read_bytes())
    args = ["/usr/local/bin/nextflow", "-C", manifest.get("runtime_config", str(root / "resources.config")), "-log", str(folder / log_name),
            "run", str(Path(manifest.get('runtime_source', str(root / 'frozen'))) / 'workflows/r02_recover_m02.nf'), "-params-file", str(folder / params_name),
            "-work-dir", str(folder / "work"), "-with-trace", str(folder / trace_name), "-ansi-log", "false"]
    if resume:
        args.append("-resume")
    env = {**os.environ, "NXF_VER": "26.04.6", "NXF_OFFLINE": "true", "NXF_DISABLE_CHECK_LATEST": "true",
           "NXF_SYNTAX_PARSER": "v1", "NXF_OPTS": "-Xms64m -Xmx768m", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with (folder / console_name).open("x") as log:
        process = subprocess.Popen(args, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                require(shutil.disk_usage(root).free >= 12 * GIB, "Disk reserve reached")
                check_runtime(manifest, started)
                for record in records.values():
                    stat = Path(record["path"]).stat()
                    require((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) ==
                            (record["bytes"], record["mtime_ns"], record["ctime_ns"]), "Staged source changed during analysis")
                time.sleep(10)
            if process.returncode != 0:
                # Bounded diagnostic logs, never genotype records or credentials.
                with (folder / console_name).open() as console:
                    from collections import deque
                    print("".join(deque(console, maxlen=80)), flush=True)
                require(False, "Nextflow failed; preserve data/cache")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            stop_owned(manifest["container_label"])
            raise
    return finalize_chromosome(root, manifest, entry, attempt)


def finalize_chromosome(root, manifest, entry, attempt, *, runtime_worker_path=None):
    """Publish one genuinely completed M02 attempt without staging or filtering.

    Boundary handoff may call this only after the existing Nextflow process
    exits successfully. The exact task trace and frozen sources are checked
    again here; an unfinished chromosome never receives a completion marker.
    """
    chrom = entry['chromosome']
    require(chrom in manifest['chromosomes'] and re.fullmatch(r'initial|resume[1-9][0-9]*', attempt),
            'Invalid finalization assignment or attempt')
    folder = root / 'chromosomes' / ('chr' + chrom.zfill(2))
    plan_path = folder / 'publication-plan.json'
    if plan_path.exists():
        return publish_chromosome(root, manifest, folder, read(plan_path))
    trace_name, log_name, console_name = 'trace.' + attempt + '.tsv', 'nextflow.' + attempt + '.log', 'console.' + attempt + '.log'
    amendment_name = 'execution-amendment.' + attempt + '.json'
    count_name = 'raw-count-evidence.' + attempt + '.json'
    params_name = 'params.' + attempt + '.json'
    with (folder / trace_name).open() as handle:
        trace = list(csv.DictReader(handle, delimiter="\t"))
    require(len(trace) == 2 and all(row["status"] in {"COMPLETED", "CACHED"} and row["exit"] == "0" for row in trace) and
            {row["name"].split(" (")[0].split(":")[-1] for row in trace} ==
            {"RECOVER_M02_ADMISSION", "PREPROCESS_FILTER_SNV_BIALLELIC_PASS"}, "Unexpected process scope/status")
    verify_bundle(root, manifest["authenticated_launch_sha256"])
    if manifest.get('runtime_amendment'):
        apply_runtime_amendment(root, manifest, Path(manifest['runtime_amendment']['path']), manifest['runtime_amendment']['sha256'],
                                runtime_worker_path=runtime_worker_path)
    output = folder / "published/02_filter"
    base = f"dnabr.hg38.2723.chr{chrom}"
    names = [base + ".snv.bi.pass.vcf.gz", base + ".snv.bi.pass.vcf.gz.tbi", base + ".counts.tsv"]
    require(all((output / name).is_file() and not (output / name).is_symlink() for name in names), "Missing/nonlocal final outputs")
    files = []
    for name in names:
        path = output / name
        files.append(dict(name=name, path=str(path), uri=manifest["output_prefix"] + "/chr" + chrom.zfill(2) + "/" + name,
                          **pipeline.publication_digests(path)))
    # Publish metadata and exact command too, before the chromosome completion marker.
    for name in ("source-receipt.json", "params.json", trace_name, log_name, console_name):
        path = folder / name
        files.append(dict(name=name, path=str(path), uri=manifest["output_prefix"] + "/chr" + chrom.zfill(2) + "/" + name,
                          **pipeline.publication_digests(path)))
    if manifest.get("runtime_amendment"):
        path = folder / amendment_name
        files.append(dict(name=amendment_name, path=str(path), uri=manifest["output_prefix"] + "/chr" + chrom.zfill(2) + "/" + amendment_name,
                          **pipeline.publication_digests(path)))
    if manifest.get('runtime_source'):
        for name in (count_name, params_name):
            path = folder / name
            files.append(dict(name=name, path=str(path), uri=manifest['output_prefix'] + '/chr' + chrom.zfill(2) + '/' + name,
                              **pipeline.publication_digests(path)))
    if manifest.get('schema') == DELEGATED_SCHEMA:
        path = folder / 'delegation.json'
        require(sha(path) == manifest['handoff_receipt']['sha256'], 'Published delegation differs from launch')
        files.append(dict(name=path.name, path=str(path), uri=manifest['output_prefix']+'/chr'+chrom.zfill(2)+'/'+path.name,
                          **pipeline.publication_digests(path)))
    plan = dict(schema="r02_m02_recovery_publication_plan_v1", chromosome=chrom,
                launch_sha256=manifest["authenticated_launch_sha256"], files=files)
    write(folder / "publication-plan.json", plan)
    return publish_chromosome(root, manifest, folder, plan)


def verify_completion(root, manifest):
    result = read(root / "complete.json")
    require(result["schema"] == "r02_m02_recovery_campaign_v1" and
            result["status"] == "ALL_ASSIGNED_M02_REMOTE_VERIFIED" and
            result["launch_sha256"] == manifest["authenticated_launch_sha256"] and
            result["chromosomes"] == manifest["chromosomes"] and result["instance"] == manifest["instance"],
            "Campaign completion differs from assignment")
    require(len(result["results"]) == len(manifest["chromosomes"]), "Incomplete result inventory")
    for chrom, remote in zip(manifest["chromosomes"], result["results"]):
        folder = root / "chromosomes" / ("chr" + chrom.zfill(2))
        local = read(folder / "completed.json")
        require(local["chromosome"] == chrom and local["launch_sha256"] == manifest["authenticated_launch_sha256"] and
                local["status"] == "M02_RECOVERED_PUBLISHED_VERIFIED_NOT_FIT_SELECTED" and
                len(local["files"]) == publication_file_count(manifest),
                "Chromosome completion differs from assignment")
        require(sha(folder / "completed.json") == remote["sha256"], "Chromosome receipt changed")
        for record in local["files"]:
            publication.verify_remote(record, pinned=True)
        publication.verify_remote(remote, pinned=True)
    final = read(root / "complete.publication.json")
    require(sha(root / "complete.json") == final["sha256"], "Final receipt changed")
    publication.verify_remote(final, pinned=True)
    return {"status": "ALL_OUTPUTS_REMOTE_REVERIFIED_SAFE_TO_RETIRE_OWN_VM", "instance": manifest["instance"],
            "chromosomes": manifest["chromosomes"], "launch_sha256": manifest["authenticated_launch_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-complete", action="store_true")
    parser.add_argument("--attempt", default="initial")
    parser.add_argument("--runtime-amendment", type=Path)
    parser.add_argument("--runtime-amendment-sha256")
    parser.add_argument("--admit-only", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CLOUDSDK_CORE_ACCOUNT"] = ACCOUNT
    os.environ["CLOUDSDK_CORE_PROJECT"] = PROJECT
    root = args.root.absolute()
    require(re.fullmatch(r"initial|resume[1-9][0-9]*", args.attempt), "Explicit attempt initial or resumeN required")
    with (root / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(root, args)


def execute(root, args):
    # The GCP guest agent allocates users on each VM; numeric IDs from the
    # coordinator cannot safely be reused in another VM's containers.
    require(os.getuid() > 0 and os.getgid() > 0, "Run as the verified non-root worker account")
    os.environ["R02_WORKER_UID"] = str(os.getuid())
    os.environ["R02_WORKER_GID"] = str(os.getgid())
    manifest, inventory = verify_bundle(root, args.launch_sha256)
    amendment = getattr(args, "runtime_amendment", None)
    amendment_sha = getattr(args, "runtime_amendment_sha256", None)
    require(bool(amendment) == bool(amendment_sha), "Runtime amendment needs both path and checksum")
    require(not (manifest.get('schema') == DELEGATED_SCHEMA and amendment),
            'Delegated launch is already frozen with native count provenance; no runtime amendment allowed')
    if amendment:
        apply_runtime_amendment(root, manifest, amendment, amendment_sha)
    require(not (args.verify_only and args.verify_complete), "Choose one verification action")
    if args.verify_complete:
        print(json.dumps(verify_completion(root, manifest))); return
    if args.verify_only:
        print(json.dumps({"status": "FROZEN_RECOVERY_VERIFIED_NOT_RUNNING"})); return
    if getattr(args,'admit_only',False):
        preflight_staged_inputs(root,manifest,inventory); return
    require(subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "--target", str(root)], text=True).strip() == "ext4",
            "Dedicated local ext4 block disk required")
    def stopped(*_):
        raise RuntimeError("Stopped: preserve unfinished chromosome/cache")
    signal.signal(signal.SIGTERM, stopped)
    started = time.monotonic()
    results = []
    for entry in inventory["entries"]:
        check_runtime(manifest, started)
        results.append(run_chromosome(root, manifest, entry, started, args.attempt))
    for record in results:
        publication.verify_remote(record, pinned=True)
    publish_complete(root, manifest, results)
    print(json.dumps({"status": "ALL_ASSIGNED_M02_REMOTE_VERIFIED", "chromosomes": manifest["chromosomes"]}))


if __name__ == "__main__":
    main()
