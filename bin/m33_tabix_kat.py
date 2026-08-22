#!/usr/bin/env python3
"""Build and compare independent synthetic Tabix indexes for M33."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable


EXPECTED_VERSION = "tabix (htslib) 1.16"
EXPECTED_RUNTIME_SERVICE_ACCOUNT = "dnabr-m33-frank@uspbr-242713.iam.gserviceaccount.com"
EXPECTED_SOURCE_VCF_SHA256 = "52868ddaad1bb9641ecfe499d61817736af36169d9d51e717b1d9112bf06a108"
EXPECTED_TBI_SHA256 = "ecab3b3f84174efb992be57a46e237c764791d0f81387a16a063baca71b7cc3b"
EXPECTED_RECORD_SHA256 = "a3bbc3a262733a3017c1ffd7faf8adaeea063ad81f8a935a4d790f4478b6f3cf"
METADATA_EMAIL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/default/email"
)
REPLICAS = {"A", "B"}
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
        digest.update(line.rstrip(b"\r\n") + b"\n")
        count += 1
    return count, digest.hexdigest()


def tabix_version() -> str:
    result = subprocess.run(
        ["tabix", "--version"], check=True, capture_output=True, text=True
    )
    version = result.stdout.splitlines()[0]
    require(version == EXPECTED_VERSION, f"Tabix version drifted: {version}")
    return version


def runtime_service_account() -> str:
    request = urllib.request.Request(METADATA_EMAIL, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=5) as response:
        require(
            response.headers.get("Metadata-Flavor") == "Google",
            "metadata response is unauthenticated",
        )
        return response.read().decode("ascii").strip()


def validate_cloud_context(
    task_work_uri: str,
    expected_work_prefix: str,
    expected_runtime_service_account: str,
) -> str:
    require(
        task_work_uri.startswith(expected_work_prefix)
        and task_work_uri != expected_work_prefix,
        "task work directory escapes the exact run work prefix",
    )
    observed = runtime_service_account()
    require(
        observed == expected_runtime_service_account == EXPECTED_RUNTIME_SERVICE_ACCOUNT,
        "worker is not using the authorized M33 runtime service account",
    )
    return observed


def write_json_exclusive(path: Path, payload: dict) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def build(
    replica: str,
    task_hash: str,
    task_work_uri: str,
    output_dir: Path,
    *,
    expected_work_prefix: str | None = None,
    expected_runtime_service_account: str | None = None,
    local_kat: bool = False,
) -> dict:
    require(replica in REPLICAS, "replica must be A or B")
    require(task_hash and task_hash.isascii(), "task hash is required")
    require(task_work_uri and task_work_uri.isascii(), "task work URI is required")
    require(output_dir.is_dir(), "output directory must already exist")
    if local_kat:
        require(
            expected_work_prefix is None and expected_runtime_service_account is None,
            "local KAT must not pretend to authenticate a cloud context",
        )
        observed_runtime_service_account = "LOCAL_KAT_NOT_CLOUD_AUTHENTICATED"
    else:
        require(expected_work_prefix is not None, "expected cloud work prefix is required")
        require(
            expected_runtime_service_account is not None,
            "expected runtime service account is required",
        )
        observed_runtime_service_account = validate_cloud_context(
            task_work_uri,
            expected_work_prefix,
            expected_runtime_service_account,
        )
    version = tabix_version()
    vcf = output_dir / f"fixture.{replica}.vcf.gz"
    tbi = Path(f"{vcf}.tbi")
    receipt = output_dir / f"build.{replica}.json"
    require(not any(path.exists() for path in (vcf, tbi, receipt)), "build outputs already exist")

    completed = subprocess.run(
        ["bgzip", "--threads", "1", "--stdout"],
        input=VCF_TEXT.encode(),
        check=True,
        capture_output=True,
    )
    vcf.write_bytes(completed.stdout)
    subprocess.run(["tabix", "-p", "vcf", vcf.name], cwd=output_dir, check=True)

    indexed = subprocess.run(
        ["tabix", vcf.name, "22"], cwd=output_dir, check=True, capture_output=True
    ).stdout.splitlines()
    with gzip.open(vcf, "rb") as handle:
        sequential = [line for line in handle if not line.startswith(b"#")]
    indexed_count, indexed_sha = line_digest(indexed)
    sequential_count, sequential_sha = line_digest(sequential)
    require(indexed_count == sequential_count, "indexed and sequential counts differ")
    require(indexed_sha == sequential_sha, "indexed and sequential records differ")
    require(indexed_count == 4, "synthetic fixture cardinality drifted")

    source_sha = sha256_file(vcf)
    tbi_sha = sha256_file(tbi)
    require(source_sha == EXPECTED_SOURCE_VCF_SHA256, "synthetic VCF known-answer drifted")
    require(tbi_sha == EXPECTED_TBI_SHA256, "synthetic TBI known-answer drifted")
    require(indexed_sha == EXPECTED_RECORD_SHA256, "synthetic record known-answer drifted")
    payload = {
        "stage": "M33_TABIX_SYNTHETIC_KAT_BUILD",
        "status": "PASS",
        "replica": replica,
        "task_hash": task_hash,
        "task_work_dir": task_work_uri,
        "runtime_service_account": observed_runtime_service_account,
        "cloud_context_authenticated": not local_kat,
        "tabix_version": version,
        "source_vcf_sha256": source_sha,
        "tbi_sha256": tbi_sha,
        "indexed_record_count": indexed_count,
        "indexed_record_sha256": indexed_sha,
        "sequential_record_count": sequential_count,
        "sequential_record_sha256": sequential_sha,
        "contains_real_genomic_data": False,
    }
    write_json_exclusive(receipt, payload)
    return payload


def compare(
    receipt_a: Path,
    receipt_b: Path,
    vcf_a: Path,
    tbi_a: Path,
    vcf_b: Path,
    tbi_b: Path,
    output: Path,
    ready: Path,
    *,
    require_cloud: bool = False,
) -> dict:
    require(output.parent == ready.parent, "receipt and READY must share a directory")
    require(not ready.exists(), "READY already exists")
    a = json.loads(receipt_a.read_text(encoding="utf-8"))
    b = json.loads(receipt_b.read_text(encoding="utf-8"))
    require({a.get("replica"), b.get("replica")} == REPLICAS, "replica pair drifted")
    require(a.get("status") == b.get("status") == "PASS", "a build did not pass")
    require(a["task_hash"] != b["task_hash"], "replicas reused the same Nextflow task hash")
    require(a["task_work_dir"] != b["task_work_dir"], "replicas reused a work directory")
    require(a["tabix_version"] == b["tabix_version"] == EXPECTED_VERSION, "version drifted")
    require(
        a["source_vcf_sha256"] == b["source_vcf_sha256"] == EXPECTED_SOURCE_VCF_SHA256,
        "fixture bytes differ from the known answer",
    )
    require(
        a["tbi_sha256"] == b["tbi_sha256"] == EXPECTED_TBI_SHA256,
        "independent TBI hashes differ from the known answer",
    )
    require(sha256_file(vcf_a) == a["source_vcf_sha256"], "replica A VCF differs from receipt")
    require(sha256_file(tbi_a) == a["tbi_sha256"], "replica A TBI differs from receipt")
    require(sha256_file(vcf_b) == b["source_vcf_sha256"], "replica B VCF differs from receipt")
    require(sha256_file(tbi_b) == b["tbi_sha256"], "replica B TBI differs from receipt")
    for item in (a, b):
        require(item["contains_real_genomic_data"] is False, "real data entered the KAT")
        require(
            item["indexed_record_count"] == item["sequential_record_count"] == 4,
            "record-count parity failed",
        )
        require(
            item["indexed_record_sha256"]
            == item["sequential_record_sha256"]
            == EXPECTED_RECORD_SHA256,
            "record-stream parity failed",
        )
    cloud_contexts = {bool(a.get("cloud_context_authenticated")), bool(b.get("cloud_context_authenticated"))}
    require(len(cloud_contexts) == 1, "replicas disagree about cloud authentication")
    if require_cloud:
        require(cloud_contexts == {True}, "cloud KAT receipts are not cloud-authenticated")
        require(
            a.get("runtime_service_account")
            == b.get("runtime_service_account")
            == EXPECTED_RUNTIME_SERVICE_ACCOUNT,
            "cloud KAT used the wrong runtime service account",
        )
    payload = {
        "stage": "M33_TABIX_SYNTHETIC_KAT",
        "status": "PASS_INDEPENDENT_INDEX_AND_QUERY_PARITY",
        "replicas": [a["replica"], b["replica"]],
        "build_receipts": [
            {
                "replica": item["replica"],
                "task_hash": item["task_hash"],
                "task_work_dir": item["task_work_dir"],
                "runtime_service_account": item.get("runtime_service_account"),
                "source_vcf_sha256": item["source_vcf_sha256"],
                "tbi_sha256": item["tbi_sha256"],
            }
            for item in (a, b)
        ],
        "source_vcf_sha256": a["source_vcf_sha256"],
        "independent_tbi_sha256": a["tbi_sha256"],
        "record_count": 4,
        "record_sha256": a["indexed_record_sha256"],
        "runtime_service_account": a.get("runtime_service_account"),
        "cloud_context_authenticated": cloud_contexts == {True},
        "tabix_version": EXPECTED_VERSION,
        "contains_real_genomic_data": False,
        "local_candidate_ready_only": True,
    }
    write_json_exclusive(output, payload)
    ready.write_text(sha256_file(output) + "\n", encoding="ascii")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--replica", choices=sorted(REPLICAS), required=True)
    build_parser.add_argument("--task-hash", required=True)
    build_parser.add_argument("--task-work-uri", required=True)
    build_parser.add_argument("--output-dir", type=Path, default=Path("."))
    build_parser.add_argument("--expected-work-prefix")
    build_parser.add_argument("--expected-runtime-service-account")
    build_parser.add_argument("--local-kat", action="store_true")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--receipt-a", required=True, type=Path)
    compare_parser.add_argument("--receipt-b", required=True, type=Path)
    compare_parser.add_argument("--vcf-a", required=True, type=Path)
    compare_parser.add_argument("--tbi-a", required=True, type=Path)
    compare_parser.add_argument("--vcf-b", required=True, type=Path)
    compare_parser.add_argument("--tbi-b", required=True, type=Path)
    compare_parser.add_argument("--output", required=True, type=Path)
    compare_parser.add_argument("--local-candidate-ready", required=True, type=Path)
    compare_parser.add_argument("--require-cloud", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        payload = build(
            args.replica,
            args.task_hash,
            args.task_work_uri,
            args.output_dir,
            expected_work_prefix=args.expected_work_prefix,
            expected_runtime_service_account=args.expected_runtime_service_account,
            local_kat=args.local_kat,
        )
    else:
        payload = compare(
            args.receipt_a, args.receipt_b,
            args.vcf_a, args.tbi_a, args.vcf_b, args.tbi_b,
            args.output, args.local_candidate_ready,
            require_cloud=args.require_cloud,
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
