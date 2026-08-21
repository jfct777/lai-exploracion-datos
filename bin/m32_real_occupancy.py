#!/usr/bin/env python3
"""Audit and report real chr22 occupancy without truth, genotypes or predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from m32_locus_contract import sha256_file, validate_git_commit
from m32_locus_occupancy import occupancy_report
from m32_locus_smoke import authenticate_sources, write_json_atomic


REQUIRED_SOURCES = {
    "bin/m32_locus_contract.py", "bin/m32_locus_occupancy.py", "bin/m32_locus_smoke.py",
    "bin/m32_locus_tensor.py", "bin/m32_prepare_coordinates.py", "bin/m32_real_occupancy.py",
    "conf/m32_real_occupancy_preregistration.json", "conf/m32_real_occupancy.config",
    "modules/32_REAL_OCCUPANCY.nf", "workflows/m32_real_occupancy.nf",
}


def parse_sources(values: Sequence[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        relative, separator, staged = value.partition("=")
        if not separator or not relative or relative.startswith("/") or ".." in Path(relative).parts or relative in output:
            raise ValueError("invalid source specification")
        output[relative] = Path(staged)
    if set(output) != REQUIRED_SOURCES:
        raise ValueError("source set does not authenticate the complete real-occupancy implementation")
    return output


def load_coordinates(path: Path) -> tuple[list[int], list[float], list[str]]:
    bp, cm, identifiers = [], [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["chrom", "bp", "cm", "locus_id"]:
            raise ValueError("coordinate table header or field order drifted")
        for row in reader:
            if row["chrom"] != "chr22":
                raise ValueError("coordinate table chromosome mismatch")
            bp.append(int(row["bp"]))
            cm.append(float(row["cm"]))
            identifiers.append(row["locus_id"])
    if not bp or any(right <= left for left, right in zip(bp, bp[1:])) or len(set(identifiers)) != len(identifiers):
        raise ValueError("coordinate identity/order is invalid")
    if any(right < left for left, right in zip(cm, cm[1:])):
        raise ValueError("coordinate cM are decreasing")
    return bp, cm, identifiers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--root-label", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--rare", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], required=True)
    parser.add_argument("--nextflow-version", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if contract.get("stage") != "M32_REAL_CHR22_OCCUPANCY_SCREEN" or contract.get("select_radius") is not False or contract.get("scientific_run_authorized") is not False:
        raise ValueError("invalid real-occupancy contract")
    if tuple(float(value) for value in contract["radii_cm"]) != (0.05, 0.1, 0.2, 0.5):
        raise ValueError("occupancy radii drifted")
    materialization = json.loads(args.materialization.read_text(encoding="utf-8"))
    if materialization.get("root_label") != args.root_label or materialization.get("truth_accessed") is not False:
        raise ValueError("materialization root or truth contract is invalid")
    expected = {
        args.grid.name: materialization["flare_grid"]["sha256"],
        args.rare.name: materialization["rare_loci"]["sha256"],
    }
    for path in (args.grid, args.rare):
        if sha256_file(path) != expected[path.name]:
            raise ValueError("coordinate table hash differs from materialization")

    git_commit = validate_git_commit(args.git_commit)
    sources = parse_sources(args.source)
    source_hashes = authenticate_sources(args.repository_root.resolve(), git_commit, sources)
    grid_bp, grid_cm, grid_ids = load_coordinates(args.grid)
    rare_bp, rare_cm, rare_ids = load_coordinates(args.rare)
    report = occupancy_report(grid_cm, rare_cm, contract["radii_cm"])
    report.update({
        "stage": contract["stage"], "status": contract["status"], "root_label": args.root_label,
        "selects_radius": False, "scientific_evidence": False,
        "grid": {"sha256": sha256_file(args.grid), "count": len(grid_bp), "first_bp": grid_bp[0], "last_bp": grid_bp[-1], "identity_count": len(grid_ids)},
        "rare": {"sha256": sha256_file(args.rare), "count": len(rare_bp), "first_bp": rare_bp[0], "last_bp": rare_bp[-1], "identity_count": len(rare_ids)},
    })
    args.outdir.mkdir(parents=True, exist_ok=False)
    report_path = args.outdir / "m32_real_occupancy.json"
    provenance_path = args.outdir / "m32_real_occupancy.provenance.json"
    manifest_path = args.outdir / "m32_real_occupancy.manifest.json"
    receipt_path = args.outdir / "m32_real_occupancy.receipt.json"
    write_json_atomic(report_path, report)
    write_json_atomic(provenance_path, {
        "git_commit": git_commit, "nextflow_version": args.nextflow_version,
        "source_sha256": source_hashes, "input_sha256": materialization["input_sha256"],
        "coordinate_materialization_sha256": sha256_file(args.materialization),
        "execution_interface": "workflows/m32_real_occupancy.nf with conf/m32_real_occupancy.config",
    })
    manifest = {"files": {path.name: sha256_file(path) for path in (report_path, provenance_path)}, "sources": source_hashes}
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(receipt_path, {
        "stage": contract["stage"], "root_label": args.root_label,
        "status": "PASS_TRUTH_FREE_OCCUPANCY_ONLY", "git_commit": git_commit,
        "manifest_sha256": sha256_file(manifest_path), "scientific_run_authorized": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
