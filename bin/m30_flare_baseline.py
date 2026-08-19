#!/usr/bin/env python3
"""Preflight and run the M30 direct-FLARE baseline without accessing truth."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


EXPERIMENT_ID = "M30_FLARE_BASELINE"
EXPECTED_FLARE = {
    "version": "0.6.0",
    "reported_build": "616fcc9d4 03-Nov-2025",
    "jar_sha256": "8c804341b555f302591b12cd72e870b1ca7849055d1dcd2b5cfa09b725bd9420",
}
EXPECTED_PARAMS = {
    "array": False,
    "probs": True,
    "em": True,
    "min-mac": 1,
    "min-maf": 0.0,
    "gen": 10.0,
    "update-p": False,
    "panel-probs": False,
    "seed": 3001701,
    "nthreads": 4,
}
EXPECTED_ROOTS = {"root17": 20260817, "root18": 20260818}


class ContractError(RuntimeError):
    """Raised when a frozen scientific or provenance invariant is violated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def verify_preregistration(prereg: dict, root_label: str, root_seed: int) -> dict:
    require(prereg.get("experiment_id") == EXPERIMENT_ID, "Unexpected experiment_id")
    require(prereg.get("status") == "PREREGISTERED_NOT_RUN", "Preregistration is not frozen as NOT_RUN")
    require(root_label in EXPECTED_ROOTS, f"Unexpected root: {root_label}")
    require(root_seed == EXPECTED_ROOTS[root_label], f"Unexpected seed for {root_label}")
    truth = prereg.get("truth_policy", {})
    require(not any(truth.get(key, True) for key in (
        "truth_permitted_in_preflight",
        "truth_permitted_in_inference",
        "truth_permitted_in_this_workflow",
    )), "Truth is not permitted in M30 preflight or inference")
    flare = prereg.get("methods", {}).get("flare", {})
    for key, expected in EXPECTED_FLARE.items():
        require(flare.get(key) == expected, f"FLARE {key} drift: {flare.get(key)!r} != {expected!r}")
    require(flare.get("parameters") == EXPECTED_PARAMS, "FLARE parameter drift")
    flare2 = prereg.get("methods", {}).get("flare2", {})
    require(flare2.get("status") == "DEFERRED", "FLARE2 must remain deferred in M30")
    root = prereg.get("roots", {}).get(root_label, {})
    require(root.get("root_seed") == root_seed, "Root seed does not match preregistration")
    return root


def verify_file(path: Path, expected_sha: str, label: str) -> str:
    require(path.is_file(), f"Missing {label}: {path}")
    observed = sha256(path)
    require(observed == expected_sha, f"SHA-256 mismatch for {label}: {observed} != {expected_sha}")
    return observed


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def scan_vcf(path: Path, expected_chrom: str) -> dict:
    samples: list[str] | None = None
    loci: list[tuple[str, int, str, str]] = []
    first_bp = None
    last_bp = None
    minimum_mac = None
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                require(len(fields) >= 10, f"VCF has no samples: {path}")
                samples = fields[9:]
                require(len(samples) == len(set(samples)), f"Duplicate VCF sample IDs: {path}")
                continue
            if line.startswith("#"):
                continue
            require(samples is not None, f"VCF header missing before data at {path}:{line_number}")
            fields = line.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(samples), f"Malformed VCF row at {path}:{line_number}")
            chrom, pos_text, _id, ref, alt = fields[:5]
            require(chrom.removeprefix("chr") == expected_chrom.removeprefix("chr"),
                    f"Unexpected chromosome at {path}:{line_number}: {chrom}")
            require("," not in alt and alt != ".", f"Non-biallelic record at {path}:{line_number}")
            pos = int(pos_text)
            if last_bp is not None:
                require(pos > last_bp, f"VCF positions are not strictly increasing at {path}:{line_number}")
            first_bp = pos if first_bp is None else first_bp
            last_bp = pos
            fmt = fields[8].split(":")
            require("GT" in fmt, f"GT missing at {path}:{line_number}")
            gt_index = fmt.index("GT")
            alt_copies = 0
            for sample_index, sample_field in enumerate(fields[9:]):
                values = sample_field.split(":")
                require(gt_index < len(values), f"GT missing for sample {samples[sample_index]} at {path}:{line_number}")
                gt = values[gt_index]
                require("|" in gt and "/" not in gt, f"Unphased GT {gt!r} at {path}:{line_number}")
                alleles = gt.split("|")
                require(len(alleles) == 2 and all(allele in {"0", "1"} for allele in alleles),
                        f"Missing, non-diploid, or invalid GT {gt!r} at {path}:{line_number}")
                alt_copies += sum(allele == "1" for allele in alleles)
            mac = min(alt_copies, 2 * len(samples) - alt_copies)
            minimum_mac = mac if minimum_mac is None else min(minimum_mac, mac)
            loci.append((chrom.removeprefix("chr"), pos, ref, alt))
    require(samples is not None, f"VCF header not found: {path}")
    require(loci, f"VCF contains no records: {path}")
    return {
        "samples": samples,
        "loci": loci,
        "first_bp": first_bp,
        "last_bp": last_bp,
        "minimum_mac": minimum_mac,
    }


def normalize_panel_map(source: Path, output: Path, reference_samples: list[str], ancestry_order: list[str],
                        expected_counts: dict[str, int]) -> dict:
    rows: list[tuple[str, str]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.split()
            require(len(fields) == 2, f"Panel map row must have two columns at {source}:{line_number}")
            rows.append((fields[0], fields[1]))
    require([sample for sample, _panel in rows] == reference_samples,
            "Panel-map sample order does not exactly match the reference VCF")
    counts = Counter(panel for _sample, panel in rows)
    require(dict(counts) == expected_counts, f"Panel counts differ: {dict(counts)} != {expected_counts}")
    require(list(dict.fromkeys(panel for _sample, panel in rows)) == ancestry_order,
            "Panel order differs from preregistered ancestry order")
    output.write_text("".join(f"{sample}\t{panel}\n" for sample, panel in rows), encoding="utf-8")
    return {"row_count": len(rows), "panel_counts": dict(counts), "sha256": sha256(output)}


def convert_genetic_map(source: Path, output: Path, chromosome: str, expected_rows: int,
                        marker_first_bp: int, marker_last_bp: int) -> dict:
    previous_bp = None
    previous_cm = None
    first_bp = first_cm = last_bp = last_cm = None
    row_count = 0
    with source.open(encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            fields = line.split()
            require(len(fields) == 3, f"Expected three map columns at {source}:{line_number}")
            chrom, bp_text, cm_text = fields
            require(chrom.removeprefix("chr") == chromosome.removeprefix("chr"),
                    f"Unexpected map chromosome at {source}:{line_number}")
            bp = int(bp_text)
            cm = float(cm_text)
            if previous_bp is not None:
                require(bp > previous_bp, f"Map bp is not strictly increasing at {source}:{line_number}")
                require(cm >= previous_cm, f"Map cM decreases at {source}:{line_number}")
            first_bp = bp if first_bp is None else first_bp
            first_cm = cm if first_cm is None else first_cm
            last_bp, last_cm = bp, cm
            previous_bp, previous_cm = bp, cm
            row_count += 1
            dst.write(f"{chrom}\t{chrom}:{bp}\t{cm_text}\t{bp}\n")
    require(row_count == expected_rows, f"Unexpected map row count: {row_count} != {expected_rows}")
    require(first_bp <= marker_first_bp and last_bp >= marker_last_bp,
            "Genetic map does not cover all target markers")
    return {
        "row_count": row_count,
        "first_bp": first_bp,
        "last_bp": last_bp,
        "first_cm": first_cm,
        "last_cm": last_cm,
        "sha256": sha256(output),
    }


def verify_gnomix_binding(binding_path: Path, fb_path: Path, msp_path: Path, root_seed: int) -> dict:
    binding = load_json(binding_path)
    require(binding.get("stage") == "M29_AUTHENTICATED_B0_BINDING", "Unexpected Gnomix binding stage")
    require(binding.get("root_seed") == root_seed, "Gnomix binding root seed mismatch")
    observed = {"fb": sha256(fb_path), "msp": sha256(msp_path)}
    require(binding.get("sha256") == observed, "Gnomix binding does not authenticate staged FB/MSP files")
    return observed


def preflight(args: argparse.Namespace) -> None:
    prereg = load_json(args.preregistration)
    root = verify_preregistration(prereg, args.root_label, args.root_seed)
    require("@sha256:" in args.container_image, "Container image must be pinned by @sha256 digest")
    require(args.container_image.endswith(args.container_digest), "Container image and digest differ")
    require(args.flare_jar_sha256 == EXPECTED_FLARE["jar_sha256"], "Configured FLARE JAR hash drift")
    prior_root_audit_sha = None
    if args.root_label == "root18":
        require(args.prior_root_audit is not None, "root18 requires the passed root17 output audit")
        prior = load_json(args.prior_root_audit)
        require(prior.get("experiment_id") == EXPERIMENT_ID, "Unexpected root17 gate experiment")
        require(prior.get("stage") == "M30_FLARE_INFERENCE_AUDIT", "Unexpected root17 gate stage")
        require(prior.get("root_label") == "root17" and prior.get("status") == "PASS",
                "root17 did not pass its FLARE output audit")
        require(prior.get("truth_accessed") is False, "root17 gate is not truth-blind")
        prior_root_audit_sha = sha256(args.prior_root_audit)
    else:
        require(args.prior_root_audit is None, "root17 must not receive a prior-root audit")
    outdir = args.outdir
    require(not outdir.exists() or not any(outdir.iterdir()), f"Refusing to overwrite non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    actual_paths = {
        "reference_vcf": args.reference_vcf,
        "reference_tbi": args.reference_tbi,
        "target_vcf": args.target_vcf,
        "target_tbi": args.target_tbi,
        "sample_map": args.sample_map,
        "gnomix_binding": args.gnomix_binding,
        "gnomix_fb": args.gnomix_fb,
        "gnomix_msp": args.gnomix_msp,
    }
    observed_hashes = {}
    for label, path in actual_paths.items():
        observed_hashes[label] = verify_file(path, root["inputs"][label]["sha256"], label)
    contract = prereg["input_contract"]
    verify_file(args.genetic_map, contract["map_sha256"], "genetic_map")

    reference = scan_vcf(args.reference_vcf, contract["chromosome"])
    target = scan_vcf(args.target_vcf, contract["chromosome"])
    require(len(reference["loci"]) == contract["marker_count"], "Reference marker count mismatch")
    require(len(target["loci"]) == contract["marker_count"], "Target marker count mismatch")
    require(reference["loci"] == target["loci"], "Reference and target CHROM/POS/REF/ALT do not match exactly")
    require(len(reference["samples"]) == contract["reference_sample_count"], "Reference sample count mismatch")
    require(len(target["samples"]) == contract["target_sample_count"], "Target sample count mismatch")
    require(set(reference["samples"]).isdisjoint(target["samples"]), "Reference and target samples overlap")
    require(reference["minimum_mac"] >= EXPECTED_PARAMS["min-mac"],
            "At least one reference locus would be removed by FLARE min-mac")
    require([target["first_bp"], target["last_bp"]] == root["coordinate_range_bp"], "Coordinate range mismatch")

    panel_path = outdir / f"{args.root_label}.flare.ref-panel.tsv"
    panel_report = normalize_panel_map(
        args.sample_map,
        panel_path,
        reference["samples"],
        prereg["ancestry_order"],
        contract["reference_panel_counts"],
    )
    map_path = outdir / f"{args.root_label}.flare.map"
    map_report = convert_genetic_map(
        args.genetic_map,
        map_path,
        contract["chromosome"],
        contract["map_row_count"],
        target["first_bp"],
        target["last_bp"],
    )
    gnomix_hashes = verify_gnomix_binding(args.gnomix_binding, args.gnomix_fb, args.gnomix_msp, args.root_seed)

    runtime_contract = {
        "schema_version": "1.0.0",
        "experiment_id": EXPERIMENT_ID,
        "stage": "M30_FLARE_RUNTIME_CONTRACT",
        "status": "PREFLIGHT_PASS",
        "root_label": args.root_label,
        "root_seed": args.root_seed,
        "analysis_role": "development",
        "preregistration_sha256": sha256(args.preregistration),
        "source_uris": {key: value["uri"] for key, value in root["inputs"].items()},
        "inputs_sha256": observed_hashes | {"genetic_map": contract["map_sha256"]},
        "derived_inputs_sha256": {"ref_panel": panel_report["sha256"], "genetic_map_4col": map_report["sha256"]},
        "shape": {
            "markers": len(target["loci"]),
            "reference_samples": len(reference["samples"]),
            "target_samples": len(target["samples"]),
            "coordinate_range_bp": [target["first_bp"], target["last_bp"]],
            "reference_panel_counts": panel_report["panel_counts"],
            "reference_minimum_mac": reference["minimum_mac"],
        },
        "flare": prereg["methods"]["flare"],
        "runtime": {
            "container_image": args.container_image,
            "container_digest": args.container_digest,
            "flare_jar_sha256": args.flare_jar_sha256,
        },
        "flare2": prereg["methods"]["flare2"],
        "gnomix": {"action": "reuse_frozen_predictions", "sha256": gnomix_hashes},
        "truth_accessed": False,
        "scoring_implemented": False,
        "provenance_scope": "inference_contract_only",
        "scoring_permitted_in_inference": False,
        "separate_scoring_stage_implemented": True,
        "planned_metrics": prereg["planned_metrics_for_separate_scoring_stage"],
        "future_scoring_decision_rule": prereg["future_scoring_decision_rule"],
        "prior_root17_audit_sha256": prior_root_audit_sha,
        "preflight_script_sha256": sha256(Path(__file__)),
    }
    runtime_path = outdir / f"{args.root_label}.m30.run_contract.json"
    write_json(runtime_path, runtime_contract)
    report = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "M30_FLARE_PREFLIGHT",
        "status": "PASS",
        "root_label": args.root_label,
        "checks": {
            "exact_hashes": True,
            "exact_locus_and_allele_parity": True,
            "phased_diploid_no_missing": True,
            "sample_disjunction": True,
            "panel_counts_and_order": True,
            "deterministic_four_column_map": map_report,
            "frozen_gnomix_binding": True,
            "truth_accessed": False,
        },
        "runtime_contract_sha256": sha256(runtime_path),
    }
    write_json(outdir / f"{args.root_label}.m30.preflight.json", report)


def java_major(java: str) -> int:
    completed = subprocess.run([java, "-version"], check=True, text=True, capture_output=True)
    version_text = completed.stderr + completed.stdout
    match = re.search(r'version "([0-9]+)(?:\.([0-9]+))?', version_text)
    require(match is not None, "Could not parse Java version")
    major = int(match.group(1))
    return int(match.group(2)) if major == 1 and match.group(2) else major


def build_flare_command(java: str, jar: Path, reference_vcf: Path, target_vcf: Path,
                        panel_map: Path, genetic_map: Path, output_prefix: Path, params: dict) -> list[str]:
    return [
        java,
        "-jar",
        str(jar),
        f"ref={reference_vcf}",
        f"ref-panel={panel_map}",
        f"gt={target_vcf}",
        f"map={genetic_map}",
        f"out={output_prefix}",
        f"array={str(params['array']).lower()}",
        f"probs={str(params['probs']).lower()}",
        f"em={str(params['em']).lower()}",
        f"min-mac={params['min-mac']}",
        f"min-maf={params['min-maf']:g}",
        f"gen={params['gen']:g}",
        f"update-p={str(params['update-p']).lower()}",
        f"panel-probs={str(params['panel-probs']).lower()}",
        f"seed={params['seed']}",
        f"nthreads={params['nthreads']}",
    ]


def audit_flare_vcf(path: Path, target_vcf: Path, ancestry_order: list[str]) -> dict:
    target = scan_vcf(target_vcf, "22")
    samples = None
    loci = []
    ancestry_header = None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##ANCESTRY=<"):
                ancestry_header = line.strip()
            elif line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
            elif not line.startswith("#"):
                require(samples is not None, "FLARE VCF header is missing")
                fields = line.rstrip("\n").split("\t")
                chrom, pos, _id, ref, alt = fields[:5]
                loci.append((chrom.removeprefix("chr"), int(pos), ref, alt))
                fmt = fields[8].split(":")
                for required in ("AN1", "AN2", "ANP1", "ANP2"):
                    require(required in fmt, f"FLARE output lacks {required} at line {line_number}")
                indices = {name: fmt.index(name) for name in ("AN1", "AN2", "ANP1", "ANP2")}
                for sample_field in fields[9:]:
                    values = sample_field.split(":")
                    for hap in ("1", "2"):
                        name = f"AN{hap}"
                        prob_name = f"ANP{hap}"
                        require(values[indices[name]] in {"0", "1", "2"}, f"Invalid {name} in FLARE output")
                        hard_call = int(values[indices[name]])
                        probs = [float(value) for value in values[indices[prob_name]].split(",")]
                        require(len(probs) == len(ancestry_order), f"Invalid {prob_name} length in FLARE output")
                        require(all(0.0 <= value <= 1.0 for value in probs), f"Invalid {prob_name} probability")
                        # FLARE serializes ANP to two decimals, so a valid vector
                        # can sum to 0.99 or 1.01. Scoring renormalizes it.
                        require(0.99 - 1e-9 <= sum(probs) <= 1.01 + 1e-9,
                                f"{prob_name} rounded probabilities have invalid mass")
                        maximum = max(probs)
                        require(abs(probs[hard_call] - maximum) <= 1e-9,
                                f"{name} is not an argmax of {prob_name}")
    require(samples == target["samples"], "FLARE output sample order differs from target VCF")
    require(loci == target["loci"], "FLARE output loci differ from target VCF")
    expected_header = ",".join(f"{ancestry}={index}" for index, ancestry in enumerate(ancestry_order))
    require(ancestry_header == f"##ANCESTRY=<{expected_header}>", "FLARE ancestry order differs from contract")
    return {"markers": len(loci), "samples": len(samples), "sha256": sha256(path)}


def audit_flare_log(path: Path, expected_params: dict) -> dict:
    text = path.read_text(encoding="utf-8")
    expected_lines = {
        "array": str(expected_params["array"]).lower(),
        "min-maf": str(expected_params["min-maf"]),
        "min-mac": str(expected_params["min-mac"]),
        "probs": str(expected_params["probs"]).lower(),
        "gen": str(expected_params["gen"]),
        "em": str(expected_params["em"]).lower(),
        "update-p": str(expected_params["update-p"]).lower(),
        "nthreads": str(expected_params["nthreads"]),
        "seed": str(expected_params["seed"]),
    }
    for name, value in expected_lines.items():
        require(re.search(rf"(?m)^\s*{re.escape(name)}\s*:\s*{re.escape(value)}\s*$", text) is not None,
                f"FLARE log does not confirm {name}={value}")
    require(re.search(r"(?m)^\s*(model|gt-ancestries)\s*:", text) is None,
            "FLARE log shows a forbidden optional input")
    # FLARE 0.6.0 only prints panel-probs when it is true. Its absence is
    # therefore the effective-log confirmation of the frozen false value.
    require(re.search(r"(?m)^\s*panel-probs\s*:", text) is None,
            "FLARE log shows panel-probs mode despite the direct-panel contract")
    require("flare version 0.6.0 [616fcc9d4 03-Nov-2025]" in text, "FLARE log version mismatch")
    return {"effective_parameters": expected_lines, "sha256": sha256(path)}


def audit_flare_model(path: Path, ancestry_order: list[str]) -> dict:
    """Validate the complete numeric FLARE model, ignoring comments/blanks."""
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    n_anc = len(ancestry_order)
    require(len(rows) == 2 + 1 + 1 + n_anc + n_anc + 1,
            f"Unexpected FLARE model shape: {len(rows)} data rows")
    require(rows[0] == ancestry_order, "FLARE model ancestry order differs from contract")
    require(rows[1] == ancestry_order, "FLARE reference-panel order differs from contract")
    numeric = [[float(value) for value in row] for row in rows[2:]]
    require(all(math.isfinite(value) for row in numeric for value in row),
            "FLARE model contains a non-finite parameter")
    generation, mu = numeric[0], numeric[1]
    p_rows = numeric[2:2 + n_anc]
    theta_rows = numeric[2 + n_anc:2 + 2 * n_anc]
    rho = numeric[-1]
    require(len(generation) == 1 and generation[0] > 0, "Invalid FLARE generation parameter")
    require(len(mu) == n_anc and all(0 <= value <= 1 for value in mu)
            and abs(sum(mu) - 1.0) <= 2e-5, "Invalid FLARE ancestry proportions")
    for label, matrix in (("p", p_rows), ("theta", theta_rows)):
        require(len(matrix) == n_anc and all(len(row) == n_anc for row in matrix),
                f"Invalid FLARE {label} matrix shape")
        require(all(0 <= value <= 1 for row in matrix for value in row),
                f"Invalid FLARE {label} parameter")
    require(all(abs(sum(row) - 1.0) <= 2e-5 for row in p_rows),
            "FLARE p rows do not sum to one")
    require(len(rho) == n_anc and all(value > 0 for value in rho), "Invalid FLARE rho parameters")
    return {
        "ancestry_order": rows[0],
        "reference_panel_order": rows[1],
        "generations": generation[0],
        "mu": mu,
        "finite_parameters": True,
        "sha256": sha256(path),
    }


def run_flare(args: argparse.Namespace) -> None:
    contract = load_json(args.runtime_contract)
    require(contract.get("experiment_id") == EXPERIMENT_ID, "Unexpected runtime contract")
    require(contract.get("status") == "PREFLIGHT_PASS", "Runtime contract did not pass preflight")
    require(contract.get("root_label") == args.root_label, "Runtime contract root mismatch")
    require(contract.get("truth_accessed") is False, "Runtime contract is not truth-blind")
    require(contract.get("scoring_implemented") is False, "Scoring must remain outside M30 inference")
    preflight_report = load_json(args.preflight_report)
    require(preflight_report.get("experiment_id") == EXPERIMENT_ID, "Unexpected preflight report")
    require(preflight_report.get("root_label") == args.root_label and preflight_report.get("status") == "PASS",
            "Preflight report does not release this root")
    require(preflight_report.get("runtime_contract_sha256") == sha256(args.runtime_contract),
            "Preflight report does not authenticate the runtime contract")
    provenance = load_json(args.run_provenance)
    require(provenance.get("experiment_id") == EXPERIMENT_ID, "Unexpected run provenance")
    require(provenance.get("truth_accessed") is False and provenance.get("scoring_implemented") is False,
            "Run provenance is not truth-blind")
    require(provenance.get("container_image") == contract["runtime"]["container_image"],
            "Run provenance container differs from preflight contract")
    require(provenance.get("container_digest") == contract["runtime"]["container_digest"],
            "Run provenance container digest differs from preflight contract")
    require(provenance.get("flare_jar_sha256") == contract["runtime"]["flare_jar_sha256"],
            "Run provenance FLARE JAR differs from preflight contract")
    require(args.scorer_receipt is not None and args.scoring_contract is not None and args.scorer is not None,
            f"{args.root_label} inference requires the frozen scorer known-answer gate")
    scoring_contract = load_json(args.scoring_contract)
    frozen_scorer_sha = scoring_contract.get("scoring_contract", {}).get("scorer_sha256")
    require(frozen_scorer_sha == sha256(args.scorer), "Frozen scorer SHA-256 mismatch before inference")
    receipt = load_json(args.scorer_receipt)
    require(receipt.get("stage") == "M30_SCORER_KNOWN_ANSWERS" and receipt.get("status") == "PASS",
            "M30 scorer known-answer gate did not pass")
    require(receipt.get("real_truth_accessed") is False, "Scorer known-answer gate accessed real truth")
    require(receipt.get("scorer_sha256") == frozen_scorer_sha, "Known-answer receipt scorer mismatch")
    require(receipt.get("contract_sha256") == sha256(args.scoring_contract),
            "Known-answer receipt scoring-contract mismatch")
    require(receipt.get("base_scorer_sha256") == scoring_contract["scoring_contract"]["base_metric_library"]["sha256"],
            "Known-answer receipt base-scorer mismatch")
    flare = contract.get("flare", {})
    for key, expected in EXPECTED_FLARE.items():
        require(flare.get(key) == expected, f"Runtime FLARE {key} drift")
    require(flare.get("parameters") == EXPECTED_PARAMS, "Runtime FLARE parameter drift")
    require(sha256(args.flare_jar) == EXPECTED_FLARE["jar_sha256"], "FLARE JAR SHA-256 mismatch")
    require(java_major(args.java) >= flare["minimum_java_major"], "Java version is too old for FLARE")

    banner = subprocess.run([args.java, "-jar", str(args.flare_jar)], text=True, capture_output=True, check=True)
    require("flare version 0.6.0 [616fcc9d4 03-Nov-2025]" in banner.stdout + banner.stderr,
            "FLARE executable banner differs from preregistration")
    expected_inputs = contract["inputs_sha256"]
    verify_file(args.reference_vcf, expected_inputs["reference_vcf"], "reference_vcf")
    verify_file(args.target_vcf, expected_inputs["target_vcf"], "target_vcf")
    verify_file(args.panel_map, contract["derived_inputs_sha256"]["ref_panel"], "ref_panel")
    verify_file(args.genetic_map, contract["derived_inputs_sha256"]["genetic_map_4col"], "genetic_map_4col")

    args.outdir.mkdir(parents=True, exist_ok=False)
    output_prefix = args.outdir / f"{args.root_label}.flare"
    command = build_flare_command(
        args.java,
        args.flare_jar,
        args.reference_vcf,
        args.target_vcf,
        args.panel_map,
        args.genetic_map,
        output_prefix,
        flare["parameters"],
    )
    subprocess.run(command, check=True)
    output_paths = {
        "ancestry_vcf": Path(f"{output_prefix}.anc.vcf.gz"),
        "global_ancestry": Path(f"{output_prefix}.global.anc.gz"),
        "model": Path(f"{output_prefix}.model"),
        "log": Path(f"{output_prefix}.log"),
    }
    for label, path in output_paths.items():
        require(path.is_file() and path.stat().st_size > 0, f"Missing or empty FLARE {label}: {path}")
    ancestry_audit = audit_flare_vcf(output_paths["ancestry_vcf"], args.target_vcf, args.ancestry_order)
    log_audit = audit_flare_log(output_paths["log"], flare["parameters"])
    model_audit = audit_flare_model(output_paths["model"], args.ancestry_order)
    report = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "M30_FLARE_INFERENCE_AUDIT",
        "status": "PASS",
        "root_label": args.root_label,
        "runtime_contract_sha256": sha256(args.runtime_contract),
        "preflight_report_sha256": sha256(args.preflight_report),
        "run_provenance_sha256": sha256(args.run_provenance),
        "scorer_known_answer_receipt_sha256": sha256(args.scorer_receipt),
        "scoring_contract_sha256": sha256(args.scoring_contract),
        "frozen_scorer_sha256": sha256(args.scorer),
        "flare_jar_sha256": sha256(args.flare_jar),
        "fixed_parameters": flare["parameters"],
        "output_sha256": {label: sha256(path) for label, path in output_paths.items()},
        "ancestry_vcf_audit": ancestry_audit,
        "effective_log_audit": log_audit,
        "model_audit": model_audit,
        "truth_accessed": False,
        "target_truth_accuracy_computed": False,
    }
    write_json(args.outdir / f"{args.root_label}.m30.flare_audit.json", report)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="Validate one root and emit immutable run inputs")
    pre.add_argument("--root-label", required=True, choices=sorted(EXPECTED_ROOTS))
    pre.add_argument("--root-seed", required=True, type=int)
    pre.add_argument("--preregistration", required=True, type=Path)
    pre.add_argument("--container-image", required=True)
    pre.add_argument("--container-digest", required=True)
    pre.add_argument("--flare-jar-sha256", required=True)
    pre.add_argument("--reference-vcf", required=True, type=Path)
    pre.add_argument("--reference-tbi", required=True, type=Path)
    pre.add_argument("--target-vcf", required=True, type=Path)
    pre.add_argument("--target-tbi", required=True, type=Path)
    pre.add_argument("--sample-map", required=True, type=Path)
    pre.add_argument("--genetic-map", required=True, type=Path)
    pre.add_argument("--gnomix-binding", required=True, type=Path)
    pre.add_argument("--gnomix-fb", required=True, type=Path)
    pre.add_argument("--gnomix-msp", required=True, type=Path)
    pre.add_argument("--prior-root-audit", type=Path)
    pre.add_argument("--outdir", required=True, type=Path)
    pre.set_defaults(func=preflight)

    run = sub.add_parser("run", help="Run and audit direct FLARE using a passed preflight contract")
    run.add_argument("--root-label", required=True, choices=sorted(EXPECTED_ROOTS))
    run.add_argument("--runtime-contract", required=True, type=Path)
    run.add_argument("--preflight-report", required=True, type=Path)
    run.add_argument("--run-provenance", required=True, type=Path)
    run.add_argument("--reference-vcf", required=True, type=Path)
    run.add_argument("--target-vcf", required=True, type=Path)
    run.add_argument("--panel-map", required=True, type=Path)
    run.add_argument("--genetic-map", required=True, type=Path)
    run.add_argument("--flare-jar", required=True, type=Path)
    run.add_argument("--java", default="java")
    run.add_argument("--ancestry-order", nargs="+", default=["AFR", "EUR", "ASIA"])
    run.add_argument("--scorer-receipt", type=Path)
    run.add_argument("--scoring-contract", type=Path)
    run.add_argument("--scorer", type=Path)
    run.add_argument("--outdir", required=True, type=Path)
    run.set_defaults(func=run_flare)
    return ap


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except (ContractError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
