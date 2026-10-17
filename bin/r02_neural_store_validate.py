#!/usr/bin/env python3
"""Stage and authenticate one DNABR event store, then release the derived copy.

The command is an input-validation adapter.  It never starts training.  It
downloads one generation-pinned chromosome archive, extracts only files named
by the authenticated store manifest, and validates every common/rare chunk
through the mmap reader used by the context trainer.  A compact receipt may be
published create-only before the derived store is removed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone


PROJECT_PREFIX = "gs://projects-usp/dnaBr-lai/datalake/"
SCHEMA = "r02_neural_store_validation_request_v1"
RECEIPT_SCHEMA = "r02_neural_store_validation_receipt_v1"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    require(not path.is_absolute() and value == path.as_posix(), "non-canonical store path")
    require(path.parts and all(part not in ("", ".", "..") for part in path.parts),
            "store path escapes its root")
    return path


def manifest_files(manifest: dict) -> dict[str, dict]:
    """Return the exact hashed file inventory declared by a store manifest."""
    files: dict[str, dict] = {}
    for channel in ("common", "rare"):
        entries = manifest.get("channels", {}).get(channel)
        require(isinstance(entries, list) and entries, f"missing {channel} channel")
        for entry in entries:
            require(isinstance(entry, dict), "invalid channel entry")
            name = safe_relative(entry.get("path", "")).as_posix()
            require(name not in files, "duplicate store file")
            require(type(entry.get("bytes")) is int and entry["bytes"] > 0,
                    "invalid store file size")
            require(re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")) is not None,
                    "invalid store file SHA256")
            files[name] = {"bytes": entry["bytes"], "sha256": entry["sha256"],
                           "channel": channel, "n_sites": entry.get("n_sites")}
    outputs = manifest.get("outputs_sha256")
    require(isinstance(outputs, dict) and outputs, "complete store output inventory required")
    for name, digest in outputs.items():
        name = safe_relative(name).as_posix()
        require(re.fullmatch(r"[0-9a-f]{64}", digest or "") is not None,
                "invalid store output SHA256")
        if name in files:
            require(files[name]["sha256"] == digest, "channel/output digest differs")
        else:
            files[name] = {"bytes": None, "sha256": digest, "channel": "metadata", "n_sites": None}
    return files


def validate_request(request: dict, request_path: Path) -> tuple[dict, dict, Path]:
    require(request.get("schema") == SCHEMA, "validation request schema differs")
    required = {"schema", "chromosome", "archive", "manifest_path", "manifest_sha256",
                "stage_root", "minimum_free_bytes", "reserve_free_bytes", "publish_uri"}
    require(set(request) == required, "validation request fields differ")
    chrom = str(request["chromosome"])
    require(chrom in tuple(map(str, range(1, 23))), "invalid autosome")
    archive = request["archive"]
    require(set(archive) >= {"uri", "generation", "bytes", "sha256"},
            "archive descriptor is incomplete")
    require(isinstance(archive["uri"], str) and archive["uri"].startswith(PROJECT_PREFIX)
            and "#" not in archive["uri"], "archive is outside the DNABR datalake")
    require(re.fullmatch(r"[1-9][0-9]*", str(archive["generation"])) is not None,
            "archive generation is not pinned")
    require(type(archive["bytes"]) is int and archive["bytes"] > 0 and
            re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]) is not None,
            "archive authentication differs")
    manifest_path = Path(request["manifest_path"]).resolve()
    require(manifest_path.is_file() and sha256(manifest_path) == request["manifest_sha256"],
            "store manifest SHA256 differs")
    manifest = load_json(manifest_path)
    require(str(manifest.get("chromosome")) == chrom and
            manifest.get("schema") == "r02_genomic_event_chromosome_v1",
            "store manifest chromosome/schema differs")
    stage_root = Path(request["stage_root"]).resolve()
    require(stage_root.is_absolute() and stage_root != Path("/") and not stage_root.is_symlink(),
            "unsafe stage root")
    require(type(request["minimum_free_bytes"]) is int and request["minimum_free_bytes"] > 0,
            "explicit positive free-space threshold required")
    require(type(request["reserve_free_bytes"]) is int and request["reserve_free_bytes"] > 0 and
            request["minimum_free_bytes"] >= archive["bytes"] + request["reserve_free_bytes"],
            "explicit positive post-task free-space reserve required")
    publish_uri = request["publish_uri"]
    require(publish_uri is None or (isinstance(publish_uri, str) and
            publish_uri.startswith(PROJECT_PREFIX) and "#" not in publish_uri and
            publish_uri.endswith(".json")), "unsafe publication URI")
    return archive, manifest, stage_root


def _tar_prefix(archive: tarfile.TarFile, manifest_digest: str) -> PurePosixPath:
    candidates = []
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if member.isfile() and path.name in ("events-manifest.json", "manifest.json"):
            extracted = archive.extractfile(member)
            require(extracted is not None, "cannot read archived manifest")
            if hashlib.sha256(extracted.read()).hexdigest() == manifest_digest:
                candidates.append(path.parent)
    require(len(candidates) == 1, "archive lacks one authenticated store manifest")
    return candidates[0]


def extract_allowlisted(archive_path: Path, destination: Path, manifest_path: Path,
                        manifest: dict, reserve_free_bytes: int) -> dict[str, dict]:
    expected = manifest_files(manifest)
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / ".r02-neural-staging-owned"
    marker.write_text(RECEIPT_SCHEMA + "\n")
    shutil.copyfile(manifest_path, destination / "manifest.json")
    with tarfile.open(archive_path, "r:gz") as archive:
        prefix = _tar_prefix(archive, sha256(manifest_path))
        by_name = {PurePosixPath(member.name): member for member in archive.getmembers()}
        selected = [by_name.get(prefix / safe_relative(name)) for name in expected]
        require(all(member is not None and member.isfile() and not member.issym() and not member.islnk()
                    for member in selected), "archive lacks a safe complete store inventory")
        require(shutil.disk_usage(destination.parent).free >=
                sum(member.size for member in selected) + reserve_free_bytes,
                "measured free space cannot hold the allowlisted store payload plus reserve")
        for name, descriptor in expected.items():
            member = by_name.get(prefix / safe_relative(name))
            require(member is not None and member.isfile() and not member.issym() and
                    not member.islnk() and (descriptor["bytes"] is None or member.size == descriptor["bytes"]),
                    "archive member differs: " + name)
            target = destination.joinpath(*safe_relative(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            require(source is not None, "cannot extract: " + name)
            with target.open("xb") as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
            require(sha256(target) == descriptor["sha256"], "extracted SHA256 differs: " + name)
            descriptor["bytes"] = member.size
    return expected


def validate_with_real_reader(store: Path, manifest_path: Path, manifest: dict) -> dict:
    # Local import keeps the CLI's help usable outside the repository environment.
    from r02_event_mmap_reader import MappedEventChromosome, VerifiedChunkCache

    chunks = [entry for channel in ("common", "rare")
              for entry in manifest["channels"][channel]]
    cache = VerifiedChunkCache(len(chunks))
    reader = MappedEventChromosome(store, sha256(manifest_path), cache=cache)
    require(len(reader.persons) == manifest["n_people"], "person axis size differs")
    summary = {}
    for channel in ("common", "rare"):
        loci = observed = missing = 0
        min_bp = max_bp = None
        for chunk in manifest["channels"][channel]:
            with cache.open(reader.safe_path(chunk["path"]), chunk, len(reader.persons),
                            reader.r["max_array_mb"]) as arrays:
                n_sites = int(arrays["position_bp"].shape[0])
                loci += n_sites
                observed += int(arrays["observed"].sum())
                missing += int(arrays["observed"].size - arrays["observed"].sum())
                lo, hi = int(arrays["position_bp"][0]), int(arrays["position_bp"][-1])
                min_bp = lo if min_bp is None else min(min_bp, lo)
                max_bp = hi if max_bp is None else max(max_bp, hi)
        summary[channel] = {"chunks": len(manifest["channels"][channel]), "loci": loci,
                            "observed_genotypes": observed, "missing_genotypes": missing,
                            "first_bp": min_bp, "last_bp": max_bp}
    return {"reader": "bin/r02_event_mmap_reader.py:MappedEventChromosome",
            "reader_backend": "ZIP_STORED_MMAP_V1", "bytes_hashed": cache.bytes_hashed,
            "n_people": len(reader.persons), "channels": summary}


def publish_create_only(path: Path, uri: str) -> dict:
    digest = {"bytes": path.stat().st_size, "sha256": sha256(path),
              "md5_base64": md5_base64(path)}
    result = subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0",
                             "--content-md5=" + digest["md5_base64"], str(path), uri],
                            text=True, capture_output=True)
    reused = result.returncode != 0
    if reused and "HTTPError 412:" not in (result.stderr or ""):
        raise subprocess.CalledProcessError(result.returncode, result.args,
                                            output=result.stdout, stderr=result.stderr)
    metadata = json.loads(subprocess.check_output(
        ["gcloud", "storage", "objects", "describe", uri, "--format=json"], text=True))
    require(int(metadata["size"]) == digest["bytes"] and
            metadata.get("md5_hash", metadata.get("md5Hash")) == digest["md5_base64"],
            "published receipt differs")
    return {**digest, "uri": uri, "generation": str(metadata["generation"]),
            "operation": "REUSED_EQUIVALENT_HTTP_412" if reused else "CREATED_GENERATION_0"}


def validate_in_pinned_container(store: Path, image: str) -> dict:
    require(isinstance(image, str) and "@sha256:" in image, "digest-pinned validation image required")
    code = Path(__file__).resolve().parent
    command = ["docker", "run", "--rm", "--cpus", "1", "--memory", "3g",
               "-v", f"{code}:/code:ro", "-v", f"{store.resolve()}:/store:ro",
               "--entrypoint", "python3", image, "/code/r02_neural_store_validate.py",
               "--validate-only-store", "/store", "--validate-only-manifest", "/store/manifest.json"]
    return json.loads(subprocess.check_output(command, text=True))


def run(request_path: Path, request_sha256: str, output: Path, keep_staged: bool,
        manifest_override: Path | None = None, recover_owned_stage: bool = False,
        validation_container: str | None = None) -> dict:
    require(re.fullmatch(r"[0-9a-f]{64}", request_sha256) is not None and
            sha256(request_path) == request_sha256, "request SHA256 differs")
    request = load_json(request_path)
    if manifest_override is not None:
        override = manifest_override.resolve()
        require(override.is_file() and sha256(override) == request.get("manifest_sha256"),
                "staged manifest override differs")
        request["manifest_path"] = str(override)
    archive, manifest, stage_root = validate_request(request, request_path)
    stage_root.mkdir(parents=True, exist_ok=True)
    chrom = str(request["chromosome"])
    work = stage_root / ("chr" + chrom + ".neural-input-validation")
    if work.exists():
        require(recover_owned_stage and
                (work / ".r02-neural-staging-owned").read_text().strip() == RECEIPT_SCHEMA,
                "staging destination already exists; explicit owned-stage recovery required")
    else:
        work.mkdir()
        (work / ".r02-neural-staging-owned").write_text(RECEIPT_SCHEMA + "\n")
    archive_path = work / "source.tar.gz"
    store = work / "store"
    try:
        expected = manifest_files(manifest)
        validation = None
        if store.exists():
            require(recover_owned_stage and
                    (store / ".r02-neural-staging-owned").read_text().strip() == RECEIPT_SCHEMA,
                    "existing store requires explicit owned-stage recovery")
            # The complete store already consumes its payload bytes.  Requiring
            # archive+payload+reserve again would make an authenticated recovery
            # impossible.  Keep the reserve and let the real reader authenticate
            # every file.  A failed validation preserves the store for inspection;
            # rebuilding it requires a separate deliberate action.
            require(shutil.disk_usage(stage_root).free >= request["reserve_free_bytes"],
                    "measured free space is below the recovery reserve")
            validation = validate_in_pinned_container(store, validation_container)
            for name, descriptor in expected.items():
                descriptor["bytes"] = (store / name).stat().st_size
        if validation is None:
            archive_ready = archive_path.exists()
            if archive_ready:
                require(archive_path.stat().st_size == archive["bytes"] and
                        sha256(archive_path) == archive["sha256"],
                        "existing staged archive differs")
            required_free = (request["minimum_free_bytes"] - archive["bytes"]
                             if archive_ready else request["minimum_free_bytes"])
            require(shutil.disk_usage(stage_root).free >= required_free,
                    "measured free space is below the admitted threshold")
            if not archive_path.exists():
                subprocess.run(["gcloud", "storage", "cp",
                                archive["uri"] + "#" + str(archive["generation"]), str(archive_path)],
                               check=True)
            require(archive_path.stat().st_size == archive["bytes"] and
                    sha256(archive_path) == archive["sha256"], "downloaded archive differs")
            expected = extract_allowlisted(archive_path, store, Path(request["manifest_path"]), manifest,
                                           request["reserve_free_bytes"])
            archive_path.unlink()
            validation = validate_in_pinned_container(store, validation_container)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "COMPLETE_INPUT_VALIDATION_NO_TRAINING",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "chromosome": chrom,
            "request_sha256": request_sha256,
            "archive": {k: archive[k] for k in ("uri", "generation", "bytes", "sha256")},
            "manifest": {"path": str(Path(request["manifest_path"]).resolve()),
                         "sha256": request["manifest_sha256"]},
            "file_count": len(expected),
            "extracted_bytes": sum(value["bytes"] for value in expected.values()),
            "validation": validation,
            "training_started": False,
            "evaluate_opened": False,
            "derived_store_released": False,
        }
        temp = work / "validation-receipt.json"
        temp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        if request["publish_uri"] is not None:
            receipt["publication"] = publish_create_only(temp, request["publish_uri"])
        elif not keep_staged:
            raise ValueError("derived store release requires a verified publication")
        if not keep_staged:
            require((store / ".r02-neural-staging-owned").read_text().strip() == RECEIPT_SCHEMA,
                    "staging ownership marker differs")
            shutil.rmtree(store)
            require(not store.exists(), "derived store release did not complete")
            receipt["derived_store_released"] = True
        temp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt
    except Exception:
        # Preserve the chromosome copy on failure for inspection/recovery.
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-override", type=Path)
    parser.add_argument("--keep-staged", action="store_true")
    parser.add_argument("--recover-owned-stage", action="store_true")
    parser.add_argument("--validation-container")
    parser.add_argument("--validate-only-store", type=Path)
    parser.add_argument("--validate-only-manifest", type=Path)
    args = parser.parse_args(argv)
    if args.validate_only_store is not None or args.validate_only_manifest is not None:
        require(args.validate_only_store is not None and args.validate_only_manifest is not None,
                "both validate-only paths are required")
        manifest = load_json(args.validate_only_manifest)
        print(json.dumps(validate_with_real_reader(args.validate_only_store,
              args.validate_only_manifest, manifest), sort_keys=True))
        return 0
    require(args.request is not None and args.request_sha256 is not None and args.output is not None,
            "request, request SHA256 and output are required")
    run(args.request.resolve(), args.request_sha256, args.output.resolve(), args.keep_staged,
        args.manifest_override, args.recover_owned_stage, args.validation_container)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
