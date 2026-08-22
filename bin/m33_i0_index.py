#!/usr/bin/env python3
"""Build and authenticate an M33 Tabix index without modifying its source VCF."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


EXPECTED_TABIX_VERSION = "tabix (htslib) 1.16"
EXPECTED_IMAGE = (
    "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-tabix@sha256:"
    "e730c35759e3851a92d7f3a6619333105331d97f1ae44a50dfa8d59745c43e54"
)
EXPECTED_CONTRACT_SHA256 = "fb74cd610a36b22fe54b8681238a13b48a0243642c9a24cd036597d161361614"
EXPECTED_FIXTURE_SHA256 = "52868ddaad1bb9641ecfe499d61817736af36169d9d51e717b1d9112bf06a108"
EXPECTED_TBI_SHA256 = "ecab3b3f84174efb992be57a46e237c764791d0f81387a16a063baca71b7cc3b"
EXPECTED_RECORD_COUNT = 4
EXPECTED_RECORD_SHA256 = "a3bbc3a262733a3017c1ffd7faf8adaeea063ad81f8a935a4d790f4478b6f3cf"
SOURCE_FILES = {
    "bin/m33_i0_index.py",
    "conf/m33_i0_fixture.config",
    "conf/m33_i0_fixture_authorization.json",
    "conf/m33_m0_materializer_contract.json",
    "modules/33_I0_FIXTURE.nf",
    "tests/test_m33_i0_index.py",
    "tests/test_m33_i0_nextflow.py",
    "workflows/m33_i0_fixture.nf",
}
VCF_TEXT = """##fileformat=VCFv4.2
##contig=<ID=22,length=50818468>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSYNTHETIC
22\t101\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1
22\t205\trs2\tC\tT\t.\tPASS\t.\tGT\t1|0
22\t1001\trs3\tG\tA\t.\tPASS\t.\tGT\t1|1
22\t5000\trs4\tT\tC\t.\tPASS\t.\tGT\t0|0
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def line_digest(lines: Iterable[bytes]) -> tuple[int, str]:
    count = 0
    digest = hashlib.sha256()
    for line in lines:
        require(line.endswith(b"\n"), "record stream contains an unterminated line")
        digest.update(line)
        count += 1
    return count, digest.hexdigest()


def load_json_strict(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate,
    )
    require(isinstance(payload, dict), f"JSON object required: {path}")
    return payload


def load_authorization(path: Path, contract: Path) -> dict[str, Any]:
    payload = load_json_strict(path)
    require(set(payload) == {
        "schema_version", "stage", "status", "base_m0_contract_sha256", "tabix_image",
        "tabix_version", "fixture_source_vcf_sha256", "fixture_descriptor", "chromosome",
        "root_label", "root_seed",
        "contains_real_genomic_data", "real_asset_read", "safe_bridge", "materialize", "training",
        "completion_marker", "global_ready_forbidden",
    }, "I0 fixture authorization keys differ")
    require(payload["schema_version"] == "1.0.0", "authorization version differs")
    require(payload["stage"] == "M33_I0_FIXTURE_AUTHORIZATION", "authorization stage differs")
    require(payload["status"] == "AUTHORIZED_FIXTURE_ONLY_NO_REAL_ASSET_READ", "fixture is not authorized")
    require(payload["base_m0_contract_sha256"] == EXPECTED_CONTRACT_SHA256, "authorized contract hash differs")
    require(sha256_file(contract) == EXPECTED_CONTRACT_SHA256, "base M0 contract bytes drifted")
    require(payload["tabix_image"] == EXPECTED_IMAGE, "Tabix OCI image differs")
    require(payload["tabix_version"] == "1.16", "authorized Tabix version differs")
    require(payload["fixture_source_vcf_sha256"] == EXPECTED_FIXTURE_SHA256, "fixture source hash differs")
    require(payload["fixture_descriptor"] == {
        "logical_id": "synthetic_m33_i0_fixture_v1",
        "uri": "synthetic://m33/i0/fixture/v1",
        "generation": "1",
        "size_bytes": 270,
        "sha256": EXPECTED_FIXTURE_SHA256,
    }, "fixture descriptor differs")
    require(payload["chromosome"] == "22", "fixture chromosome differs")
    require(payload["root_label"] == "fixture" and payload["root_seed"] == 0, "fixture root differs")
    require(payload["contains_real_genomic_data"] is False, "real data entered fixture authorization")
    require(all(payload[key] is False for key in ("real_asset_read", "safe_bridge", "materialize", "training")),
            "fixture authorization opened a forbidden stage")
    require(payload["completion_marker"] == "I0_FIXTURE_PASS", "fixture marker differs")
    require(payload["global_ready_forbidden"] is True, "global READY was not forbidden")
    return payload


def load_source_auth(path: Path, repo_root: Path) -> str:
    payload = load_json_strict(path)
    require(set(payload) == {"schema_version", "stage", "status", "files"}, "source-auth keys differ")
    require(payload["schema_version"] == "1.0.0", "source-auth version differs")
    require(payload["stage"] == "M33_I0_FIXTURE_SOURCE_AUTH", "source-auth stage differs")
    require(payload["status"] == "AUTHORIZED_EXACT_FIXTURE_ONLY_SOURCES", "sources are not authorized")
    files = payload["files"]
    require(isinstance(files, dict) and set(files) == SOURCE_FILES, "source-auth inventory differs")
    for relative, expected in files.items():
        require(isinstance(expected, str) and len(expected) == 64, f"invalid source hash: {relative}")
        require(sha256_file(repo_root / relative) == expected, f"source drifted: {relative}")
    return sha256_file(path)


def tabix_version() -> str:
    completed = subprocess.run(["tabix", "--version"], check=True, capture_output=True, text=True)
    version = completed.stdout.splitlines()[0]
    require(version == EXPECTED_TABIX_VERSION, f"Tabix version drifted: {version}")
    return version


def validate_container_image(container_image: str, authorization: dict[str, Any]) -> str:
    require(container_image == EXPECTED_IMAGE, "effective Tabix OCI image differs")
    require(container_image == authorization["tabix_image"], "effective image is not authorized")
    return container_image


def path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes_no_overwrite(destination: Path, data: bytes) -> None:
    require(not path_lexists(destination), f"refusing to overwrite {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output_handle:
            output_handle.write(data)
            output_handle.flush()
            os.fsync(output_handle.fileno())
            os.fchmod(output_handle.fileno(), 0o444)
            os.fsync(output_handle.fileno())
        os.link(temporary, destination)
        fsync_directory(destination.parent)
        temporary.unlink()
        fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_bytes_no_overwrite(path, serialized.encode("utf-8"))


def atomic_copy_no_overwrite(source: Path, destination: Path) -> None:
    require(not path_lexists(destination), f"refusing to overwrite {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1 << 20)
            output_handle.flush()
            os.fsync(output_handle.fileno())
            os.fchmod(output_handle.fileno(), 0o444)
            os.fsync(output_handle.fileno())
        os.link(temporary, destination)
        fsync_directory(destination.parent)
        temporary.unlink()
        fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def make_fixture(
    output_vcf: Path, manifest: Path, authorization: Path, contract: Path,
    source_auth: Path, repo_root: Path, container_image: str,
) -> dict[str, Any]:
    require(output_vcf.name == "flare.anc.vcf.gz", "fixture VCF basename differs")
    require(manifest.name == "fixture_source.receipt.json", "fixture manifest basename differs")
    require(output_vcf.absolute().parent == manifest.absolute().parent,
            "fixture outputs must share one task-local directory")
    require(not path_lexists(output_vcf) and not path_lexists(manifest), "fixture output already exists")
    auth = load_authorization(authorization, contract)
    effective_image = validate_container_image(container_image, auth)
    source_auth_sha = load_source_auth(source_auth, repo_root)
    version = tabix_version()
    completed = subprocess.run(
        ["bgzip", "--threads", "1", "--stdout"], input=VCF_TEXT.encode(), check=True, capture_output=True
    )
    atomic_bytes_no_overwrite(output_vcf, completed.stdout)
    observed = sha256_file(output_vcf)
    require(observed == auth["fixture_source_vcf_sha256"], "generated fixture hash differs")
    require(output_vcf.stat().st_size == auth["fixture_descriptor"]["size_bytes"], "generated fixture size differs")
    payload = {
        "schema_version": "1.0.0",
        "stage": "M33_I0_FIXTURE_SOURCE",
        "status": "PASS_AUTHENTICATED_SYNTHETIC_SOURCE",
        "source_vcf_sha256": observed,
        "source_descriptor": auth["fixture_descriptor"],
        "tabix_version": version,
        "tabix_oci_repository_digest": effective_image,
        "source_auth_sha256": source_auth_sha,
        "contains_real_genomic_data": False,
    }
    write_json_exclusive(manifest, payload)
    return payload


def validate_source_manifest(
    path: Path, source_sha256: str, source_auth_sha256: str, source_descriptor: dict[str, Any],
    container_image: str,
) -> None:
    payload = load_json_strict(path)
    require(payload == {
        "schema_version": "1.0.0",
        "stage": "M33_I0_FIXTURE_SOURCE",
        "status": "PASS_AUTHENTICATED_SYNTHETIC_SOURCE",
        "source_vcf_sha256": source_sha256,
        "source_descriptor": source_descriptor,
        "tabix_version": EXPECTED_TABIX_VERSION,
        "tabix_oci_repository_digest": container_image,
        "source_auth_sha256": source_auth_sha256,
        "contains_real_genomic_data": False,
    }, "fixture source manifest differs")


def build_one_index(source: Path, directory: Path, expected_source_sha256: str) -> tuple[Path, int, str]:
    staged = directory / "flare.anc.vcf.gz"
    shutil.copyfile(source, staged)
    require(sha256_file(staged) == expected_source_sha256, "staged source copy differs before indexing")
    staged.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    subprocess.run(["tabix", "-p", "vcf", staged.name], cwd=directory, check=True)
    index = Path(f"{staged}.tbi")
    require(index.is_file() and index.stat().st_size > 0, "Tabix did not create a non-empty index")
    indexed = subprocess.run(
        ["tabix", staged.name, "22"], cwd=directory, check=True, capture_output=True
    ).stdout.splitlines(keepends=True)
    count, digest = line_digest(indexed)
    require(count > 0, "chr22 query returned no records")
    require(sha256_file(staged) == expected_source_sha256, "staged source copy differs after indexing")
    return index, count, digest


def sequential_chr22(source: Path) -> tuple[int, str]:
    with gzip.open(source, "rb") as handle:
        records = []
        for line in handle:
            if line.startswith(b"#"):
                continue
            fields = line.rstrip(b"\r\n").split(b"\t", 1)
            require(len(fields) == 2, "malformed VCF record")
            if fields[0] == b"22":
                records.append(line)
    return line_digest(records)


def validate_index_evidence(
    *, tbi_a: str, tbi_b: str,
    indexed_count_a: int, indexed_count_b: int, indexed_sha_a: str, indexed_sha_b: str,
    sequential_count: int, sequential_sha: str,
    source_size_before: int, source_size_after: int,
    source_sha_before: str, source_sha_after: str,
) -> None:
    require(tbi_a == tbi_b, "independent Tabix index hashes differ")
    require(indexed_count_a == indexed_count_b == sequential_count, "indexed/sequential counts differ")
    require(indexed_sha_a == indexed_sha_b == sequential_sha, "indexed/sequential record hashes differ")
    require(tbi_a == EXPECTED_TBI_SHA256, "Tabix index known answer differs")
    require(sequential_count == EXPECTED_RECORD_COUNT, "record-count known answer differs")
    require(sequential_sha == EXPECTED_RECORD_SHA256, "record-stream known answer differs")
    require(source_size_after == source_size_before, "source VCF size changed during indexing")
    require(source_sha_after == source_sha_before, "source VCF changed during indexing")


def derive(
    *, source: Path, source_manifest: Path, authorization: Path, contract: Path,
    source_auth: Path, repo_root: Path,
    output_tbi: Path, receipt: Path, marker: Path, run_id: str, container_image: str,
) -> dict[str, Any]:
    auth = load_authorization(authorization, contract)
    effective_image = validate_container_image(container_image, auth)
    source_auth_sha = load_source_auth(source_auth, repo_root)
    require(run_id and run_id.isascii(), "run ID is required")
    require(source.is_file(), "source VCF is missing")
    require((source.stat().st_mode & 0o222) == 0, "source VCF must be staged read-only")
    require(output_tbi.name == "flare.anc.vcf.gz.tbi", "I0 index basename differs")
    require(receipt.name == "i0_fixture.receipt.json", "I0 receipt basename differs")
    require(marker.name == auth["completion_marker"] == "I0_FIXTURE_PASS", "I0 marker basename differs")
    parents = {path.absolute().parent for path in (source, output_tbi, receipt, marker)}
    require(len(parents) == 1, "I0 inputs and outputs must share one task-local directory")
    require(not any(path_lexists(path) for path in (output_tbi, receipt, marker)), "I0 output already exists")
    source_size_before = source.stat().st_size
    source_before = sha256_file(source)
    require(source_before == auth["fixture_source_vcf_sha256"], "source VCF hash differs")
    require(source_size_before == auth["fixture_descriptor"]["size_bytes"], "source VCF size differs")
    validate_source_manifest(
        source_manifest, source_before, source_auth_sha, auth["fixture_descriptor"], effective_image
    )
    version = tabix_version()
    sequential_count, sequential_sha = sequential_chr22(source)

    with tempfile.TemporaryDirectory(prefix="m33-i0-a-", dir=output_tbi.parent) as tmp_a, \
         tempfile.TemporaryDirectory(prefix="m33-i0-b-", dir=output_tbi.parent) as tmp_b:
        index_a, indexed_count_a, indexed_sha_a = build_one_index(source, Path(tmp_a), source_before)
        index_b, indexed_count_b, indexed_sha_b = build_one_index(source, Path(tmp_b), source_before)
        tbi_a = sha256_file(index_a)
        tbi_b = sha256_file(index_b)
        validate_index_evidence(
            tbi_a=tbi_a, tbi_b=tbi_b,
            indexed_count_a=indexed_count_a, indexed_count_b=indexed_count_b,
            indexed_sha_a=indexed_sha_a, indexed_sha_b=indexed_sha_b,
            sequential_count=sequential_count, sequential_sha=sequential_sha,
            source_size_before=source_size_before, source_size_after=source.stat().st_size,
            source_sha_before=source_before, source_sha_after=sha256_file(source),
        )
        atomic_copy_no_overwrite(index_a, output_tbi)

    output_sha = sha256_file(output_tbi)
    require(output_sha == tbi_a and output_tbi.stat().st_size > 0, "reopened output index differs")
    payload = {
        "schema_version": "1.0.0",
        "stage": "M33_I0_FIXTURE_INDEX",
        "status": "PASS_DOUBLE_INDEX_AND_QUERY_PARITY_FIXTURE_ONLY",
        "run_id": run_id,
        "root_label": auth["root_label"],
        "root_seed": auth["root_seed"],
        "source_flare_sha256": source_before,
        "source_flare_size_bytes": source_size_before,
        "source_descriptor": auth["fixture_descriptor"],
        "source_auth_sha256": source_auth_sha,
        "tabix_version": version,
        "tabix_oci_repository_digest": effective_image,
        "independent_tbi_sha256": tbi_a,
        "query_parity_sha256": sequential_sha,
        "indexed_record_count": indexed_count_a,
        "sequential_record_count": sequential_count,
        "output_tbi_sha256": output_sha,
        "append_only": True,
        "reopen_verified": True,
        "contains_real_genomic_data": False,
        "real_asset_read": False,
        "safe_bridge": False,
        "materialize": False,
        "training": False,
    }
    write_json_exclusive(receipt, payload)
    receipt_sha = sha256_file(receipt)
    marker_payload = {
        "stage": "M33_I0_FIXTURE_PASS",
        "status": "PASS_FIXTURE_ONLY_NON_CONSUMABLE",
        "run_id": run_id,
        "receipt_sha256": receipt_sha,
        "contains_real_genomic_data": False,
    }
    write_json_exclusive(marker, marker_payload)
    reopened = load_json_strict(receipt)
    reopened_marker = load_json_strict(marker)
    require(reopened == payload, "I0 receipt reopen failed")
    require(reopened_marker == marker_payload, "I0 marker reopen failed")
    require(reopened_marker["receipt_sha256"] == sha256_file(receipt), "I0 marker receipt hash differs")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("make-fixture")
    fixture.add_argument("--output-vcf", type=Path, required=True)
    fixture.add_argument("--manifest", type=Path, required=True)
    fixture.add_argument("--authorization", type=Path, required=True)
    fixture.add_argument("--contract", type=Path, required=True)
    fixture.add_argument("--source-auth", type=Path, required=True)
    fixture.add_argument("--repo-root", type=Path, required=True)
    fixture.add_argument("--container-image", required=True)
    derive_parser = subparsers.add_parser("derive")
    derive_parser.add_argument("--source", type=Path, required=True)
    derive_parser.add_argument("--source-manifest", type=Path, required=True)
    derive_parser.add_argument("--authorization", type=Path, required=True)
    derive_parser.add_argument("--contract", type=Path, required=True)
    derive_parser.add_argument("--source-auth", type=Path, required=True)
    derive_parser.add_argument("--repo-root", type=Path, required=True)
    derive_parser.add_argument("--output-tbi", type=Path, required=True)
    derive_parser.add_argument("--receipt", type=Path, required=True)
    derive_parser.add_argument("--marker", type=Path, required=True)
    derive_parser.add_argument("--run-id", required=True)
    derive_parser.add_argument("--container-image", required=True)
    args = parser.parse_args()
    if args.command == "make-fixture":
        result = make_fixture(
            args.output_vcf, args.manifest, args.authorization, args.contract,
            args.source_auth, args.repo_root, args.container_image,
        )
    else:
        result = derive(
            source=args.source, source_manifest=args.source_manifest,
            authorization=args.authorization, contract=args.contract,
            source_auth=args.source_auth, repo_root=args.repo_root,
            output_tbi=args.output_tbi, receipt=args.receipt, marker=args.marker,
            run_id=args.run_id, container_image=args.container_image,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
