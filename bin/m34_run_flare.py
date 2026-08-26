#!/usr/bin/env python3
"""Run the truth-blind AFR/EUR/NAM FLARE baseline for M34."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PARAMETER_MEMBERS = {
    "array", "probs", "em", "min-mac", "min-maf", "gen",
    "update-p", "panel-probs", "seed", "nthreads",
}
FIXED_PARAMETERS = {
    "array": False, "probs": True, "em": True, "min-mac": 1,
    "min-maf": 0.0, "update-p": False, "panel-probs": False,
}
INPUT_MEMBERS = {
    "reference_vcf", "reference_tbi", "target_vcf", "target_tbi",
    "sample_map", "genetic_map", "flare_jar",
}


class FlareContractError(ValueError):
    """Raised when an input or frozen FLARE setting differs from the contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FlareContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" \
        else path.open(encoding="utf-8")


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    require(set(contract) == {"schema_version", "stage", "status", "chromosome",
                              "ancestry_names", "parameters", "expected_sha256"},
            "FLARE contract members differ")
    require(contract["stage"] == "M34_AFR_EUR_NAM_FLARE" and
            contract["status"] == "EXPLORATORY_CONTRACT_BLINDED_TO_LABELS",
            "FLARE contract identity differs")
    require(contract["ancestry_names"] == ["AFR", "EUR", "NAM"],
            "M34 FLARE ancestry order must be AFR/EUR/NAM")
    require(str(contract["chromosome"]).removeprefix("chr") == "22",
            "M34 FLARE is restricted to chromosome 22")
    parameters = contract["parameters"]
    require(set(parameters) == PARAMETER_MEMBERS, "FLARE parameter members differ")
    for name, value in FIXED_PARAMETERS.items():
        require(parameters[name] == value, f"FLARE parameter drift: {name}")
    require(type(parameters["gen"]) in (int, float) and
            math.isfinite(float(parameters["gen"])) and float(parameters["gen"]) > 0,
            "FLARE gen must be finite and positive")
    require(float(parameters["gen"]) == 12.0,
            "primary M34 contract must match the 12-generation mosaic scenario")
    require(type(parameters["seed"]) is int and parameters["seed"] >= 0,
            "FLARE seed differs")
    require(type(parameters["nthreads"]) is int and parameters["nthreads"] >= 1,
            "FLARE nthreads differs")
    hashes = contract["expected_sha256"]
    require(set(hashes) == INPUT_MEMBERS and
            all(isinstance(value, str) and len(value) == 64 and
                all(character in "0123456789abcdef" for character in value)
                for value in hashes.values()), "input SHA-256 contract differs")
    return contract


def verify_hashes(contract: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, str]:
    require(set(paths) == INPUT_MEMBERS, "runtime input members differ")
    observed: dict[str, str] = {}
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"missing or linked input: {name}")
        observed[name] = sha256_file(path)
        require(observed[name] == contract["expected_sha256"][name],
                f"SHA-256 mismatch for {name}")
    return observed


def scan_vcf(path: Path, chromosome: str) -> dict[str, Any]:
    samples: list[str] | None = None
    loci: list[tuple[str, int, str, str]] = []
    vcf_chromosome: str | None = None
    minimum_mac: int | None = None
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                require(len(fields) >= 10, f"VCF has no samples: {path}")
                samples = fields[9:]
                require(len(samples) == len(set(samples)), f"duplicate VCF samples: {path}")
                continue
            if line.startswith("#"):
                continue
            require(samples is not None, f"VCF header missing at {path}:{line_number}")
            fields = line.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(samples), f"malformed VCF row at {path}:{line_number}")
            chrom, position_text, _identifier, ref, alt = fields[:5]
            require(chrom.removeprefix("chr") == chromosome.removeprefix("chr"),
                    f"unexpected chromosome at {path}:{line_number}")
            if vcf_chromosome is None:
                vcf_chromosome = chrom
            require(chrom == vcf_chromosome,
                    f"inconsistent chromosome spelling at {path}:{line_number}")
            require(alt != "." and "," not in alt, f"non-biallelic locus at {path}:{line_number}")
            position = int(position_text)
            require(not loci or position > loci[-1][1],
                    f"VCF positions are not strictly increasing at {path}:{line_number}")
            fmt = fields[8].split(":")
            require("GT" in fmt, f"GT missing at {path}:{line_number}")
            gt_index = fmt.index("GT")
            alternative_copies = 0
            for sample_index, sample_field in enumerate(fields[9:]):
                values = sample_field.split(":")
                require(gt_index < len(values),
                        f"GT missing for {samples[sample_index]} at {path}:{line_number}")
                genotype = values[gt_index]
                require("|" in genotype and "/" not in genotype,
                        f"unphased GT at {path}:{line_number}")
                alleles = genotype.split("|")
                require(len(alleles) == 2 and all(value in {"0", "1"} for value in alleles),
                        f"missing or invalid diploid GT at {path}:{line_number}")
                alternative_copies += sum(value == "1" for value in alleles)
            copies = 2 * len(samples)
            mac = min(alternative_copies, copies - alternative_copies)
            minimum_mac = mac if minimum_mac is None else min(minimum_mac, mac)
            loci.append((chrom.removeprefix("chr"), position, ref, alt))
    require(samples is not None and loci, f"VCF is empty or lacks a header: {path}")
    return {"samples": samples, "loci": loci, "minimum_mac": minimum_mac,
            "vcf_chromosome": vcf_chromosome,
            "first_bp": loci[0][1], "last_bp": loci[-1][1]}


def normalize_sample_map(source: Path, output: Path, reference_samples: Sequence[str],
                         ancestry_names: Sequence[str]) -> dict[str, Any]:
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        require(len(fields) == 2, f"sample-map row differs at line {line_number}")
        rows.append((fields[0], fields[1]))
    require([sample for sample, _ancestry in rows] == list(reference_samples),
            "sample-map order differs from REF")
    require(all(ancestry in ancestry_names for _sample, ancestry in rows),
            "sample-map contains an undeclared ancestry")
    counts = Counter(ancestry for _sample, ancestry in rows)
    require(all(counts[name] > 0 for name in ancestry_names),
            "sample-map leaves an ancestry without REF samples")
    require(list(dict.fromkeys(ancestry for _sample, ancestry in rows)) == list(ancestry_names),
            "sample-map ancestry order differs")
    output.write_text("".join(f"{sample}\t{ancestry}\n" for sample, ancestry in rows),
                      encoding="utf-8")
    return {"sample_count": len(rows), "ancestry_counts": dict(counts),
            "sha256": sha256_file(output)}


def normalize_genetic_map(source: Path, output: Path, chromosome: str,
                          output_chromosome: str, first_marker: int,
                          last_marker: int) -> dict[str, Any]:
    rows: list[tuple[str, int, float]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        require(len(fields) in {3, 4}, f"genetic-map row differs at line {line_number}")
        if len(fields) == 3:
            chrom, position_text, cm_text = fields
        else:
            chrom, _marker_id, cm_text, position_text = fields
        require(chrom.removeprefix("chr") == chromosome.removeprefix("chr"),
                f"genetic-map chromosome differs at line {line_number}")
        position, cm = int(position_text), float(cm_text)
        require(math.isfinite(cm), f"non-finite cM at line {line_number}")
        require(not rows or (position > rows[-1][1] and cm >= rows[-1][2]),
                f"genetic-map order differs at line {line_number}")
        rows.append((chrom, position, cm))
    require(rows and rows[0][1] <= first_marker and rows[-1][1] >= last_marker,
            "genetic map does not cover the complete marker axis")
    # FLARE requires the map chromosome label to match the VCF contig exactly.
    # The source map may use 22 while the VCF uses chr22 (or vice versa).
    output.write_text("".join(
        f"{output_chromosome}\t{output_chromosome}:{position}\t{cm:g}\t{position}\n"
        for _chrom, position, cm in rows), encoding="utf-8")
    return {"row_count": len(rows), "first_bp": rows[0][1], "last_bp": rows[-1][1],
            "first_cM": rows[0][2], "last_cM": rows[-1][2],
            "output_chromosome": output_chromosome,
            "sha256": sha256_file(output)}


def build_command(java: str, jar: Path, reference_vcf: Path, target_vcf: Path,
                  sample_map: Path, genetic_map: Path, output_prefix: Path,
                  parameters: Mapping[str, Any]) -> list[str]:
    return [
        java, "-jar", str(jar), f"ref={reference_vcf}", f"ref-panel={sample_map}",
        f"gt={target_vcf}", f"map={genetic_map}", f"out={output_prefix}",
        f"array={str(parameters['array']).lower()}",
        f"probs={str(parameters['probs']).lower()}",
        f"em={str(parameters['em']).lower()}", f"min-mac={parameters['min-mac']}",
        f"min-maf={float(parameters['min-maf']):g}",
        f"gen={float(parameters['gen']):g}",
        f"update-p={str(parameters['update-p']).lower()}",
        f"panel-probs={str(parameters['panel-probs']).lower()}",
        f"seed={parameters['seed']}", f"nthreads={parameters['nthreads']}",
    ]


def audit_ancestry_vcf(path: Path, target: Mapping[str, Any],
                       ancestry_names: Sequence[str]) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, "FLARE ancestry VCF is missing")
    samples: list[str] | None = None
    loci: list[tuple[str, int, str, str]] = []
    ancestry_header: str | None = None
    format_headers: set[str] = set()
    cells = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##ANCESTRY=<"):
                ancestry_header = line.strip()
            elif line.startswith("##FORMAT=<ID="):
                format_headers.add(line.split("##FORMAT=<ID=", 1)[1].split(",", 1)[0])
            elif line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
            elif not line.startswith("#"):
                require(samples is not None, "FLARE VCF header is missing")
                fields = line.rstrip("\n").split("\t")
                require(len(fields) == 9 + len(samples),
                        f"FLARE row width differs at line {line_number}")
                loci.append((fields[0].removeprefix("chr"), int(fields[1]),
                             fields[3], fields[4]))
                fmt = fields[8].split(":")
                require(len(fmt) == len(set(fmt)) and
                        all(name in fmt for name in ("AN1", "AN2", "ANP1", "ANP2")),
                        f"FLARE ancestry FORMAT differs at line {line_number}")
                indexes = {name: fmt.index(name) for name in ("AN1", "AN2", "ANP1", "ANP2")}
                for value in fields[9:]:
                    parts = value.split(":")
                    require(len(parts) == len(fmt),
                            f"FLARE sample FORMAT width differs at line {line_number}")
                    for hard_name, probability_name in (("AN1", "ANP1"), ("AN2", "ANP2")):
                        hard = int(parts[indexes[hard_name]])
                        probabilities = [float(item) for item in
                                         parts[indexes[probability_name]].split(",")]
                        require(0 <= hard < len(ancestry_names),
                                f"invalid {hard_name} at line {line_number}")
                        require(len(probabilities) == len(ancestry_names) and
                                all(math.isfinite(item) and 0 <= item <= 1
                                    for item in probabilities),
                                f"invalid {probability_name} at line {line_number}")
                        require(abs(sum(probabilities) - 1.0) <= 0.010000001,
                                f"invalid {probability_name} mass at line {line_number}")
                        require(probabilities[hard] >= max(probabilities) - 1e-9,
                                f"{hard_name} is not an argmax of {probability_name}")
                        cells += 1
    require(samples == target["samples"], "FLARE/TARGET sample axes differ")
    require(loci == target["loci"], "FLARE/TARGET locus axes differ")
    require({"AN1", "AN2", "ANP1", "ANP2"}.issubset(format_headers),
            "FLARE ancestry FORMAT headers differ")
    expected_header = ",".join(f"{name}={index}" for index, name in enumerate(ancestry_names))
    require(ancestry_header == f"##ANCESTRY=<{expected_header}>",
            "FLARE ancestry order differs")
    return {"sample_count": len(samples), "marker_count": len(loci),
            "haplotype_probability_cells": cells, "sha256": sha256_file(path)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    require(not args.outdir.exists(), "refusing to overwrite the output directory")
    contract = load_contract(args.contract)
    inputs = {
        "reference_vcf": args.reference_vcf, "reference_tbi": args.reference_tbi,
        "target_vcf": args.target_vcf, "target_tbi": args.target_tbi,
        "sample_map": args.sample_map, "genetic_map": args.genetic_map,
        "flare_jar": args.flare_jar,
    }
    observed_hashes = verify_hashes(contract, inputs)
    reference = scan_vcf(args.reference_vcf, contract["chromosome"])
    target = scan_vcf(args.target_vcf, contract["chromosome"])
    require(reference["loci"] == target["loci"], "REF/TARGET locus and allele axes differ")
    require(set(reference["samples"]).isdisjoint(target["samples"]),
            "REF and TARGET sample sets overlap")
    require(reference["minimum_mac"] >= contract["parameters"]["min-mac"],
            "REF contains a marker removed by min-mac")
    args.outdir.mkdir(parents=True, exist_ok=False)
    normalized_panel = args.outdir / "flare.ref-panel.tsv"
    normalized_map = args.outdir / "flare.map"
    panel_audit = normalize_sample_map(args.sample_map, normalized_panel,
                                       reference["samples"], contract["ancestry_names"])
    map_audit = normalize_genetic_map(args.genetic_map, normalized_map,
                                      contract["chromosome"], target["vcf_chromosome"],
                                      target["first_bp"],
                                      target["last_bp"])
    prefix = args.outdir / "m34"
    command = build_command(args.java, args.flare_jar, args.reference_vcf,
                            args.target_vcf, normalized_panel, normalized_map,
                            prefix, contract["parameters"])
    ancestry_audit = None
    status = "PASS_PREFLIGHT_ONLY"
    if not args.preflight_only:
        subprocess.run(command, check=True)
        ancestry_path = Path(f"{prefix}.anc.vcf.gz")
        ancestry_audit = audit_ancestry_vcf(ancestry_path, target,
                                            contract["ancestry_names"])
        status = "PASS_TRUTH_BLIND_FLARE"
    receipt = {
        "schema_version": "1.0.0", "stage": "M34_AFR_EUR_NAM_FLARE",
        "status": status, "claim_level": "exploratory",
        "chromosome": contract["chromosome"],
        "ancestry_names": contract["ancestry_names"],
        "shape": {"marker_count": len(target["loci"]),
                  "reference_sample_count": len(reference["samples"]),
                  "target_sample_count": len(target["samples"]),
                  "reference_panel_counts": panel_audit["ancestry_counts"]},
        "parameters": contract["parameters"], "command_argv": command,
        "input_sha256": observed_hashes, "contract_sha256": sha256_file(args.contract),
        "derived_input_audit": {"sample_map": panel_audit, "genetic_map": map_audit},
        "ancestry_vcf_audit": ancestry_audit,
        "truth_argument_available": False, "truth_accessed": False,
        "scoring_performed": False, "preflight_only": bool(args.preflight_only),
        "wall_seconds": time.monotonic() - started,
    }
    (args.outdir / "m34_flare.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--reference-vcf", type=Path, required=True)
    parser.add_argument("--reference-tbi", type=Path, required=True)
    parser.add_argument("--target-vcf", type=Path, required=True)
    parser.add_argument("--target-tbi", type=Path, required=True)
    parser.add_argument("--sample-map", type=Path, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--flare-jar", type=Path, required=True)
    parser.add_argument("--java", default="java")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "shape": result["shape"]},
                     sort_keys=True))
