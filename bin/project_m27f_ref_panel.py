#!/usr/bin/env python3
"""Create audited DISCOVERY_CORE and REF-only BCF projections without variant filtering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path


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


def project(
    bcftools: str,
    source: Path,
    allowlist: Path,
    output: Path,
) -> None:
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
            str(output),
            str(source),
        ],
        check=True,
    )
    subprocess.run([bcftools, "index", "--force", "--csi", str(output)], check=True)
    os.chmod(output, 0o600)
    os.chmod(Path(f"{output}.csi"), 0o600)


def run(args: argparse.Namespace) -> dict[str, object]:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27F_REF_SUPPORT_AUDIT" or prereg.get("version") != 1:
        raise ValueError("Invalid M27F REF preregistration")
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
        or split_public.get("decision") != "GO_REF_EXTRACTION_ONLY"
    ):
        raise ValueError("STOP_PROVENANCE: split decision or generator differs")
    with args.split_private.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != int(contract["expected_panel_samples"]):
        raise ValueError("STOP_ALLOWLIST: unexpected split row count")

    panel_samples = run_text([args.bcftools, "query", "-l", str(args.source_panel_vcf)])
    if panel_samples != [row["sample_id"] for row in rows]:
        raise ValueError("STOP_ALLOWLIST: split order differs from the panel header")
    discovery = [row["sample_id"] for row in rows if row["exclusion_reason"] == "DISCOVERY_CORE"]
    ref = [row["sample_id"] for row in rows if row["role"] == "REF_TRAIN"]
    non_ref = {row["sample_id"] for row in rows if row["role"] != "REF_TRAIN"}
    if (
        len(discovery) != int(contract["expected_discovery_core_samples"])
        or len(ref) != int(contract["expected_ref_samples"])
        or len(discovery) != len(set(discovery))
        or len(ref) != len(set(ref))
        or set(discovery) & set(ref)
        or set(ref) & non_ref
    ):
        raise ValueError("STOP_ALLOWLIST: identity, size or disjunction failed")

    args.outdir.mkdir(parents=True, exist_ok=True)
    discovery_list = args.outdir / "m27f_discovery_core.samples.private.txt"
    ref_list = args.outdir / "m27f_ref.samples.private.txt"
    discovery_bcf = args.outdir / "m27f_discovery_core.chr22.work.bcf"
    ref_bcf = args.outdir / "m27f_ref.chr22.private.bcf"
    write_allowlist(discovery_list, discovery)
    write_allowlist(ref_list, ref)
    project(args.bcftools, args.source_panel_vcf, discovery_list, discovery_bcf)
    project(args.bcftools, args.source_panel_vcf, ref_list, ref_bcf)

    if run_text([args.bcftools, "query", "-l", str(discovery_bcf)]) != discovery:
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: discovery projection header differs")
    if run_text([args.bcftools, "query", "-l", str(ref_bcf)]) != ref:
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: REF projection header differs")

    source_count, source_digest = variant_key_digest(args.bcftools, args.source_panel_vcf)
    discovery_count, discovery_digest = variant_key_digest(args.bcftools, discovery_bcf)
    ref_count, ref_digest = variant_key_digest(args.bcftools, ref_bcf)
    if (
        source_count != int(contract["expected_panel_variant_records"])
        or not source_count == discovery_count == ref_count
        or len({source_digest, discovery_digest, ref_digest}) != 1
    ):
        raise ValueError("STOP_VARIANT_PROJECTION: variant keys or order changed")

    version = run_text([args.bcftools, "--version"])[0]
    if version != f"bcftools {prereg['projection_contract']['version']}":
        raise ValueError("STOP_PROVENANCE: bcftools version differs")
    public = {
        "stage": "M27F_REF_PROJECTION",
        "decision": "GO_REF_SUPPORT_AUDIT",
        "gates": {"R0": "PASS", "R1": "PASS", "R2": "PASS", "R3": "PASS"},
        "bcftools_version": version,
        "n_source_samples": len(panel_samples),
        "n_discovery_core_samples": len(discovery),
        "n_ref_samples": len(ref),
        "n_variant_records": source_count,
        "variant_key_order_sha256": source_digest,
        "ref_allowlist_sha256": sha256_file(ref_list),
        "discovery_bcf_sha256": sha256_file(discovery_bcf),
        "ref_bcf_sha256": sha256_file(ref_bcf),
        "ref_bcf_index_sha256": sha256_file(Path(f"{ref_bcf}.csi")),
        "non_ref_samples_emitted": 0,
        "variant_filters_applied": [],
        "genotype_filters_applied": [],
        "info_ac_an_updated": False,
        "discovery_projection_published": False,
        "sample_ids_emitted_publicly": False,
    }
    (args.outdir / "m27f_ref_projection.public.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return public


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
