#!/usr/bin/env python3
"""Project frozen DISCOVERY, REF and VALID samples without genotype-based decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path


ROLES = ("DISCOVERY_CORE", "REF_TRAIN", "SOURCE_VALID", "SOURCE_TEST")
TARGET_ANCESTRIES = ("African", "European", "Native_American")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument("--source-panel-vcf", type=Path, required=True)
    parser.add_argument("--split-private", type=Path, required=True)
    parser.add_argument("--split-public", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(command: list[str]) -> list[str]:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout.splitlines()


def variant_key_digest(bcftools: str, path: Path) -> tuple[int, str]:
    process = subprocess.Popen(
        [bcftools, "query", "-f", r"%CHROM\t%POS\t%REF\t%ALT\n", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Could not open bcftools query stream")
    digest = hashlib.sha256()
    count = 0
    for line in process.stdout:
        digest.update(line)
        count += 1
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    if process.wait() != 0:
        raise RuntimeError(f"bcftools query failed for {path.name}: {stderr[:1000]}")
    return count, digest.hexdigest()


def write_allowlist(path: Path, samples: list[str]) -> None:
    path.write_text("\n".join(samples) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def project(bcftools: str, source: Path, allowlist: Path, output: Path) -> None:
    subset = output.with_name(f"{output.stem}.subset.work.bcf")
    subprocess.run(
        [
            bcftools,
            "view",
            "--samples-file",
            str(allowlist),
            "--force-samples",
            "--no-update",
            "--no-version",
            "--output-type",
            "b",
            "--output",
            str(subset),
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        [
            bcftools,
            "annotate",
            "--remove",
            "INFO",
            "--no-version",
            "--output-type",
            "b",
            "--output",
            str(output),
            str(subset),
        ],
        check=True,
    )
    subset.unlink()
    subprocess.run([bcftools, "index", "--force", "--csi", str(output)], check=True)
    os.chmod(output, 0o600)
    os.chmod(Path(f"{output}.csi"), 0o600)


def nonempty_info_records(bcftools: str, path: Path) -> int:
    values = run_text([bcftools, "query", "-f", r"%INFO\n", str(path)])
    return sum(value not in ("", ".") for value in values)


def samples_by_role(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return {
        "DISCOVERY_CORE": [
            row["sample_id"] for row in rows if row["exclusion_reason"] == "DISCOVERY_CORE"
        ],
        "REF_TRAIN": [row["sample_id"] for row in rows if row["role"] == "REF_TRAIN"],
        "SOURCE_VALID": [row["sample_id"] for row in rows if row["role"] == "SOURCE_VALID"],
        "SOURCE_TEST": [row["sample_id"] for row in rows if row["role"] == "SOURCE_TEST"],
    }


def ancestry_counts(rows: list[dict[str, str]], role: str) -> dict[str, int]:
    return {
        ancestry: sum(row["role"] == role and row["ancestry"] == ancestry for row in rows)
        for ancestry in TARGET_ANCESTRIES
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27F_REF_VALID_SUPPORT_AUDIT" or prereg.get("version") != 2:
        raise ValueError("Invalid M27F-b preregistration")
    contract = prereg["upstream_contract"]
    observed_hashes = {
        "m27f_split_manifest_sha256": sha256_file(args.split_manifest),
        "m27f_split_private_sha256": sha256_file(args.split_private),
        "m27f_split_public_sha256": sha256_file(args.split_public),
        "phased_panel_vcf_sha256": sha256_file(args.source_panel_vcf),
    }
    if any(observed_hashes[key] != contract[key] for key in observed_hashes):
        raise ValueError("STOP_PROVENANCE: an upstream hash differs")

    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split_public = json.loads(args.split_public.read_text(encoding="utf-8"))
    if (
        split_manifest.get("git_commit") != contract["m27f_split_generator_commit"]
        or split_public.get("decision") != "GO_OPEN_TRAIN_VALID_SUPPORT_AUDIT"
        or split_public.get("private_manifest_sha256")
        != contract["m27f_split_private_sha256"]
    ):
        raise ValueError("STOP_PROVENANCE: split decision, generator or private hash differs")

    with args.split_private.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != int(contract["expected_panel_samples"]):
        raise ValueError("STOP_ALLOWLIST: unexpected split row count")

    panel_samples = run_text([args.bcftools, "query", "-l", str(args.source_panel_vcf)])
    if panel_samples != [row["sample_id"] for row in rows]:
        raise ValueError("STOP_ALLOWLIST: split order differs from the panel header")
    if len(panel_samples) != len(set(panel_samples)):
        raise ValueError("STOP_ALLOWLIST: duplicate source-panel sample")

    roles = samples_by_role(rows)
    expected_counts = {
        "DISCOVERY_CORE": int(contract["expected_discovery_core_samples"]),
        "REF_TRAIN": int(contract["expected_ref_samples"]),
        "SOURCE_VALID": int(contract["expected_valid_samples"]),
        "SOURCE_TEST": int(contract["expected_test_samples"]),
    }
    for role in ROLES:
        if len(roles[role]) != expected_counts[role] or len(roles[role]) != len(set(roles[role])):
            raise ValueError(f"STOP_ALLOWLIST: unexpected or duplicate {role} membership")
    for left_index, left in enumerate(ROLES):
        for right in ROLES[left_index + 1 :]:
            if set(roles[left]) & set(roles[right]):
                raise ValueError(f"STOP_ALLOWLIST: {left} overlaps {right}")

    expected_by_ancestry = contract["expected_samples_by_ancestry_and_role"]
    for role in ("REF_TRAIN", "SOURCE_VALID", "SOURCE_TEST"):
        observed = ancestry_counts(rows, role)
        expected = {key: int(value) for key, value in expected_by_ancestry[role].items()}
        if observed != expected:
            raise ValueError(f"STOP_ALLOWLIST: ancestry counts differ for {role}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, tuple[Path, Path]] = {}
    file_stems = {
        "DISCOVERY_CORE": "m27f_discovery_core",
        "REF_TRAIN": "m27f_ref",
        "SOURCE_VALID": "m27f_valid",
    }
    for role, stem in file_stems.items():
        allowlist = args.outdir / f"{stem}.samples.private.txt"
        bcf = args.outdir / f"{stem}.chr22.private.bcf"
        write_allowlist(allowlist, roles[role])
        project(args.bcftools, args.source_panel_vcf, allowlist, bcf)
        paths[role] = (allowlist, bcf)

    test_set = set(roles["SOURCE_TEST"])
    source_count, source_digest = variant_key_digest(args.bcftools, args.source_panel_vcf)
    projected_receipts: dict[str, dict[str, object]] = {}
    for role, (allowlist, bcf) in paths.items():
        observed_samples = run_text([args.bcftools, "query", "-l", str(bcf)])
        if observed_samples != roles[role]:
            raise ValueError(f"STOP_OUTPUT_MEMBERSHIP: {role} header differs from allowlist")
        if set(observed_samples) & test_set:
            raise ValueError(f"STOP_TEST_LEAKAGE: SOURCE_TEST appeared in {role}")
        count, digest = variant_key_digest(args.bcftools, bcf)
        if count != source_count or digest != source_digest:
            raise ValueError(f"STOP_VARIANT_PROJECTION: {role} changed keys or order")
        nonempty_info = nonempty_info_records(args.bcftools, bcf)
        if nonempty_info:
            raise ValueError(f"STOP_TEST_LEAKAGE: {role} retained INFO fields")
        projected_receipts[role] = {
            "n_samples": len(observed_samples),
            "allowlist_sha256": sha256_file(allowlist),
            "bcf_sha256": sha256_file(bcf),
            "bcf_index_sha256": sha256_file(Path(f"{bcf}.csi")),
            "n_records_with_nonempty_info": nonempty_info,
        }

    if source_count != int(contract["expected_panel_variant_records"]):
        raise ValueError("STOP_VARIANT_PROJECTION: unexpected source variant count")
    version = run_text([args.bcftools, "--version"])[0]
    if version != f"bcftools {prereg['projection_contract']['version']}":
        raise ValueError("STOP_PROVENANCE: bcftools version differs")

    public = {
        "stage": "M27F_REF_VALID_MECHANICAL_PROJECTION",
        "decision": "GO_REF_SUPPORT_AUDIT",
        "gates": {"P0": "PASS", "P1": "PASS", "P2": "PASS", "P3": "PASS"},
        "bcftools_version": version,
        "n_source_samples": len(panel_samples),
        "n_source_test_samples_sealed": len(roles["SOURCE_TEST"]),
        "n_variant_records": source_count,
        "variant_key_order_sha256": source_digest,
        "authenticated_upstream_sha256": observed_hashes,
        "projections": projected_receipts,
        "source_test_projection_created": False,
        "source_valid_bcf_mechanically_projected": True,
        "source_test_samples_in_projected_outputs": 0,
        "variant_filters_applied": [],
        "genotype_filters_applied": [],
        "genotype_statistics_computed": False,
        "source_info_fields_removed": True,
        "info_ac_an_updated": False,
        "sample_ids_emitted_publicly": False,
        "population_labels_emitted_publicly": False,
    }
    (args.outdir / "m27f_projection.public.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return public


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
