#!/usr/bin/env python3
"""Publish one completed R02 evidence bundle, create-only, with API verification.

This does not run Nextflow, interpret scientific tables, grant publication
authority, or promote descriptive evidence to validation. The exact input
manifest hash and a new receipt path are required. An interrupted publication
can be retried with the same immutable inputs: only HTTP 412 plus matching
remote size/MD5 permits reuse. The manifest is uploaded last; no remote object
is deleted or overwritten, and no local scientific file is modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import r02_autosome_pipeline as pipeline


SCHEMA = "r02_evidence_publication_v1"
PREFIX = ("gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/"
          "biologico/R02_20260930/")
ALLOWED_MANIFESTS = {
    "r02_lai_genotype_support_v1": "COMPLETE_GENOTYPE_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION",
    "r02_biological_evaluation_v1": "BIOLOGICAL_DESCRIPTIVE_NOT_VALIDATED",
    "r02_segment_evidence_v1": "COMPLETE_DESCRIPTIVE_NOT_VALIDATED",
    "r02_community_evidence_v1": "COMPLETE_DESCRIPTIVE_COMMUNITY_EVIDENCE_NOT_VALIDATED",
    "r02_community_summary_v1": "COMPLETE_DESCRIPTIVE_COMMUNITY_SUMMARY_NOT_VALIDATED",
    "r02_lai_catalogue_correspondence_v1": "COMPLETE_CATALOGUE_CORRESPONDENCE_ONLY_NOT_SUPPORT",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(name):
    require(isinstance(name, str) and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*", name),
        "Unsafe relative publication name")
    return Path(name)


def safe_local(path, *, directory=False, new=False):
    path = Path(path).absolute()
    require(".." not in path.parts and path.resolve() == path,
            "Local path must be canonical and contain no symlinks")
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            "Symlink in local publication path")
    if new:
        require(not os.path.lexists(path), "Receipt already exists; never overwrite")
        require(path.parent.is_dir(), "Receipt parent directory is missing")
    else:
        require(path.is_dir() if directory else path.is_file(), "Missing local publication input")
    return path


def validate_sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256")
    return value


def destination_prefix(value):
    require(isinstance(value, str) and value.startswith(PREFIX), "Destination outside authorized R02 prefix")
    value = value.removesuffix("/")
    relative = value.removeprefix(PREFIX)
    path = safe_relative(relative)
    require(len(path.parts) >= 2, "Destination must identify a run and an evidence subdirectory")
    return value


def read_unique_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def authenticated_bundle(output_dir, expected_manifest_sha256):
    """Authenticate the declared inventory, not the semantics of its tables."""
    folder = safe_local(output_dir, directory=True)
    manifest_path = safe_local(folder / "manifest.json")
    manifest_digest = pipeline.publication_digests(manifest_path)
    require(manifest_digest["sha256"] == validate_sha(expected_manifest_sha256), "Manifest SHA256 mismatch")
    manifest = read_unique_json(manifest_path)
    require(isinstance(manifest, dict), "Manifest must be an object")
    require(isinstance(manifest.get("schema"), str) and manifest.get("schema") in ALLOWED_MANIFESTS and
            manifest.get("status") == ALLOWED_MANIFESTS[manifest["schema"]],
            "Unsupported schema or incomplete/descriptive status")
    expected = {}
    for field in ("outputs_sha256", "operational_outputs_sha256"):
        entries = manifest.get(field, {} if field == "operational_outputs_sha256" else None)
        require(isinstance(entries, dict) and (entries or field == "operational_outputs_sha256"),
                "Missing or invalid output hash inventory")
        for name, checksum in entries.items():
            safe_relative(name)
            require(name != "manifest.json" and name not in expected, "Repeated or self-referential output")
            expected[name] = validate_sha(checksum)
    expected["manifest.json"] = expected_manifest_sha256
    selected = {p.relative_to(folder).as_posix() for p in pipeline.publication_files(folder)}
    require(selected == set(expected), "Publication inventory differs from manifest (missing, excluded, or unlisted file)")
    records = []
    # A published manifest is the final marker, never the first uploaded file.
    for name in sorted(set(expected) - {"manifest.json"}) + ["manifest.json"]:
        path = safe_local(folder / safe_relative(name))
        require(path.is_relative_to(folder), "Publication input escaped output directory")
        digest = pipeline.publication_digests(path)
        require(digest["sha256"] == expected[name], "Output SHA256 mismatch: " + name)
        records.append(dict(name=name, path=str(path), **digest))
    return folder, manifest, records


def local_recheck(record):
    path = safe_local(record["path"])
    require(pipeline.publication_digests(path) == {k: record[k] for k in ("bytes", "sha256", "md5_base64")},
            "Local source changed during publication: " + record["name"])


def remote_metadata(uri):
    # Reuse the established API reader; this is only its logical mount-path
    # adapter. No FUSE lookup, stat, copy, or mount access is performed.
    return pipeline.object_metadata(pipeline.MOUNT / uri.removeprefix(pipeline.BUCKET))


def verify_remote(record, *, pinned=False):
    metadata = remote_metadata(record["uri"])
    require(isinstance(metadata, dict), "Invalid GCS object metadata")
    size, generation = metadata.get("size"), metadata.get("generation")
    require(not isinstance(size, bool) and re.fullmatch(r"[0-9]+", str(size)) and
            int(size) == record["bytes"], "Published GCS size differs from source")
    require(not isinstance(generation, bool) and re.fullmatch(r"[0-9]+", str(generation)) and
            int(generation) > 0, "Published object has no valid generation")
    require(metadata.get("md5_hash", metadata.get("md5Hash")) == record["md5_base64"],
            "Published GCS MD5 differs from source or is missing")
    if pinned:
        require(str(generation) == record["generation"], "Published GCS generation changed")
    return str(generation)


def upload_one(record):
    local_recheck(record)
    result = subprocess.run([
        "gcloud", "storage", "cp", "--if-generation-match=0",
        "--content-md5=" + record["md5_base64"], record["path"], record["uri"]],
        capture_output=True, text=True, timeout=7200,
        env={**os.environ, "CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED": "false"})
    reused = result.returncode != 0
    if reused and "HTTPError 412:" not in (result.stderr or ""):
        raise subprocess.CalledProcessError(result.returncode, result.args,
                                            output=result.stdout, stderr=result.stderr)
    record["generation"] = verify_remote(record)
    record["operation"] = "REUSED_EQUIVALENT_HTTP_412" if reused else "CREATED_GENERATION_0"
    local_recheck(record)


def write_receipt_new(path, value):
    """Expose either one complete receipt or no receipt, without replacing it."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".r02-publication-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # link is atomic on the local filesystem and fails if the receipt path
        # appeared concurrently; unlike replace(), it never overwrites.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def publish(output_dir, expected_manifest_sha256, destination, receipt):
    destination = destination_prefix(destination)
    receipt = safe_local(receipt, new=True)
    folder, manifest, records = authenticated_bundle(output_dir, expected_manifest_sha256)
    require(not receipt.is_relative_to(folder), "Receipt must be outside immutable output directory")
    sources = {Path(__file__).name: pipeline.sha(__file__),
               Path(pipeline.__file__).name: pipeline.sha(pipeline.__file__)}
    for record in records:
        record["uri"] = destination + "/" + record["name"]
    for record in records:
        if record["name"] == "manifest.json":
            # Recheck all previous objects before publishing the completion
            # marker, including generations observed during this invocation.
            authenticated_bundle(folder, expected_manifest_sha256)
            for previous in records[:-1]:
                verify_remote(previous, pinned=True)
        upload_one(record)
    for record in records:
        verify_remote(record, pinned=True)
    # Inventory may have changed without changing any previously declared file.
    _, _, final_records = authenticated_bundle(folder, expected_manifest_sha256)
    require([r["name"] for r in final_records] == [r["name"] for r in records], "Local inventory changed")
    require(sources == {Path(__file__).name: pipeline.sha(__file__),
                        Path(pipeline.__file__).name: pipeline.sha(pipeline.__file__)},
            "Publisher source changed during publication")
    result = dict(schema=SCHEMA, status="PUBLISHED_VERIFIED", completed_utc=pipeline.now(),
        output_dir=str(folder), destination=destination, manifest_sha256=expected_manifest_sha256,
        evidence_schema=manifest["schema"], evidence_status=manifest["status"],
        verification="GCS API size + MD5 + pinned generation; all local SHA256 rechecked",
        manifest_published_last=True, no_remote_overwrite=True, no_scientific_validation=True,
        source_sha256=sources, files=records)
    write_receipt_new(receipt, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "expected-manifest-sha256", "destination", "receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        result = publish(**vars(args))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print("R02 evidence publication FAILED: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(dict(status=result["status"], files=len(result["files"]), receipt=args.receipt)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
