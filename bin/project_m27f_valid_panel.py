#!/usr/bin/env python3
"""Project only the frozen SOURCE_VALID allow-list from the phased chr22 panel."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from project_m27f_ref_panel import project, run_text, sha256_file, variant_key_digest


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


def run(args: argparse.Namespace) -> dict[str, object]:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27F_VALID_LOCAL_TRANSFER" or prereg.get("version") != 1:
        raise ValueError("Invalid M27F VALID preregistration")
    contract = prereg["upstream_contract"]
    observed = {
        "m27f_split_manifest_sha256": sha256_file(args.split_manifest),
        "m27f_split_private_sha256": sha256_file(args.split_private),
        "m27f_split_public_sha256": sha256_file(args.split_public),
        "phased_panel_vcf_sha256": sha256_file(args.source_panel_vcf),
    }
    if any(observed[key] != contract[key] for key in observed):
        raise ValueError("STOP_PROVENANCE: an upstream hash differs")

    split_public = json.loads(args.split_public.read_text(encoding="utf-8"))
    if split_public.get("decision") != "GO_REF_EXTRACTION_ONLY":
        raise ValueError("STOP_PROVENANCE: split decision differs")
    with args.split_private.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != int(contract["expected_panel_samples"]):
        raise ValueError("STOP_ALLOWLIST: unexpected split row count")

    panel_samples = run_text([args.bcftools, "query", "-l", str(args.source_panel_vcf)])
    if panel_samples != [row["sample_id"] for row in rows]:
        raise ValueError("STOP_ALLOWLIST: split order differs from panel header")
    valid = [row["sample_id"] for row in rows if row["role"] == "SOURCE_VALID"]
    non_valid = {row["sample_id"] for row in rows if row["role"] != "SOURCE_VALID"}
    expected_by_ancestry = contract["expected_valid_samples_by_ancestry"]
    observed_by_ancestry = {
        ancestry: sum(row["role"] == "SOURCE_VALID" and row["ancestry"] == ancestry for row in rows)
        for ancestry in expected_by_ancestry
    }
    if (
        len(valid) != int(contract["expected_valid_samples"])
        or len(valid) != len(set(valid))
        or set(valid) & non_valid
        or observed_by_ancestry != expected_by_ancestry
    ):
        raise ValueError("STOP_ALLOWLIST: SOURCE_VALID identity or size differs")

    args.outdir.mkdir(parents=True, exist_ok=True)
    allowlist = args.outdir / "m27f_valid.samples.private.txt"
    allowlist.write_text("\n".join(valid) + "\n", encoding="utf-8")
    os.chmod(allowlist, 0o600)
    valid_bcf = args.outdir / "m27f_valid.chr22.private.bcf"
    project(args.bcftools, args.source_panel_vcf, allowlist, valid_bcf)
    if run_text([args.bcftools, "query", "-l", str(valid_bcf)]) != valid:
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: VALID BCF header differs")

    source_count, source_digest = variant_key_digest(args.bcftools, args.source_panel_vcf)
    valid_count, valid_digest = variant_key_digest(args.bcftools, valid_bcf)
    if (
        source_count != int(contract["expected_panel_variant_records"])
        or source_count != valid_count
        or source_digest != valid_digest
    ):
        raise ValueError("STOP_VARIANT_PROJECTION: variant keys or order changed")
    version = run_text([args.bcftools, "--version"])[0]
    if version != f"bcftools {prereg['projection_contract']['version']}":
        raise ValueError("STOP_PROVENANCE: bcftools version differs")

    public = {
        "stage": "M27F_VALID_PROJECTION",
        "decision": "GO_VALID_TRANSFER_AUDIT",
        "gates": {"V0": "PASS", "V1": "PASS", "V2": "PASS"},
        "bcftools_version": version,
        "n_source_samples": len(panel_samples),
        "n_valid_samples": len(valid),
        "valid_samples_by_ancestry": observed_by_ancestry,
        "n_variant_records": source_count,
        "variant_key_order_sha256": source_digest,
        "valid_allowlist_sha256": sha256_file(allowlist),
        "valid_bcf_sha256": sha256_file(valid_bcf),
        "valid_bcf_index_sha256": sha256_file(Path(f"{valid_bcf}.csi")),
        "non_valid_samples_emitted": 0,
        "variant_filters_applied": [],
        "genotype_filters_applied": [],
        "info_ac_an_updated": False,
        "source_test_opened": False,
        "sample_ids_emitted_publicly": False,
    }
    (args.outdir / "m27f_valid_projection.public.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return public


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
