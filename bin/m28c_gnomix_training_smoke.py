#!/usr/bin/env python3
"""Prepare, run and audit the blinded M28C Gnomix training smoke."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


STAGE = "M28C_GNOMIX_TRAINING_SMOKE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> str:
    stdout_handle = stdout_path.open("w", encoding="utf-8") if stdout_path else subprocess.PIPE
    stderr_handle = stderr_path.open("w", encoding="utf-8") if stderr_path else subprocess.PIPE
    try:
        result = subprocess.run(
            command,
            check=True,
            cwd=cwd,
            env=env,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    except subprocess.CalledProcessError as exc:
        stderr = ""
        if stderr_path and stderr_path.exists():
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        elif isinstance(exc.stderr, str):
            stderr = exc.stderr[-4000:]
        raise RuntimeError(
            f"Command failed with exit code {exc.returncode}: {command!r}; stderr={stderr!r}"
        ) from exc
    finally:
        if stdout_path:
            stdout_handle.close()
        if stderr_path:
            stderr_handle.close()
    return result.stdout.strip() if isinstance(result.stdout, str) else ""


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("stage") != STAGE:
        raise ValueError(f"Unexpected contract stage: {contract.get('stage')!r}")
    if contract.get("status") != "PRE_FROZEN_AMENDED_BEFORE_SUCCESSFUL_TRAINING":
        raise ValueError("Training-smoke contract is not frozen")
    return contract


def require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, observed {observed}")
    return observed


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(
        "r", encoding="utf-8", newline=""
    )


def read_b0_markers(path: Path, expected_count: int) -> list[dict]:
    with open_text(path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} B0 markers, observed {len(rows)}")
    parsed = []
    previous = None
    for row in rows:
        marker = {
            "chrom": row["chrom"],
            "position": int(row["position"]),
            "cm": float(row["cm"]),
        }
        if marker["chrom"] != "chr22":
            raise ValueError(f"Unexpected chromosome: {marker['chrom']}")
        key = (marker["position"], marker["cm"])
        if previous is not None and key <= previous:
            raise ValueError("B0 marker table is not strictly ordered")
        previous = key
        parsed.append(marker)
    return parsed


def allocate_subset(markers: list[dict], contract: dict) -> tuple[list[dict], list[dict]]:
    specification = contract["subset"]
    target = int(specification["markers"])
    width = float(specification["bin_width_cm"])
    origin = float(specification["bin_origin_cm"])
    expected_bins = int(specification["bins"])
    groups: dict[int, list[dict]] = defaultdict(list)
    for marker in markers:
        bin_index = min(int(math.floor((marker["cm"] - origin) / width)), expected_bins - 1)
        if bin_index < 0:
            raise ValueError("Marker precedes the frozen bin origin")
        groups[bin_index].append(marker)
    if sorted(groups) != list(range(expected_bins)):
        missing = sorted(set(range(expected_bins)) - set(groups))
        raise ValueError(f"Frozen 0.2 cM coverage is incomplete; missing bins: {missing}")

    remaining = target - expected_bins
    residual_total = sum(len(groups[index]) - 1 for index in range(expected_bins))
    if remaining < 0 or residual_total <= 0:
        raise ValueError("Invalid subset allocation budget")
    allocations = {}
    remainders = []
    assigned = expected_bins
    for index in range(expected_bins):
        exact = remaining * (len(groups[index]) - 1) / residual_total
        base = math.floor(exact)
        allocations[index] = 1 + base
        assigned += base
        remainders.append((exact - base, index))
    for _, index in sorted(remainders, key=lambda item: (-item[0], item[1]))[: target - assigned]:
        allocations[index] += 1

    selected = []
    bin_audit = []
    for index in range(expected_bins):
        group = groups[index]
        count = allocations[index]
        if not 1 <= count <= len(group):
            raise ValueError(f"Invalid allocation {count} for bin {index} with {len(group)} markers")
        indices = [math.floor((rank + 0.5) * len(group) / count) for rank in range(count)]
        if len(set(indices)) != count or indices[-1] >= len(group):
            raise ValueError(f"Non-unique within-bin selection for bin {index}")
        for rank, marker_index in enumerate(indices):
            marker = dict(group[marker_index])
            marker["bin_index"] = index
            marker["within_bin_rank"] = rank
            selected.append(marker)
        bin_audit.append(
            {
                "bin_index": index,
                "available": len(group),
                "selected": count,
                "start_cm": origin + index * width,
                "end_cm": origin + (index + 1) * width,
            }
        )
    selected.sort(key=lambda marker: marker["position"])
    if len(selected) != target or len({marker["position"] for marker in selected}) != target:
        raise ValueError("Subset cardinality or uniqueness failed")
    return selected, bin_audit


def bcftools_positions(path: Path) -> list[int]:
    output = run_command(["bcftools", "query", "-f", "%POS\n", str(path)])
    return [int(value) for value in output.splitlines()]


def bcftools_contigs(path: Path) -> list[str]:
    output = run_command(["bcftools", "query", "-f", "%CHROM\n", str(path)])
    return sorted(set(output.splitlines()))


def canonical_autosome(value: str) -> str:
    return value.removeprefix("chr")


def bcftools_samples(path: Path) -> list[str]:
    output = run_command(["bcftools", "query", "-l", str(path)])
    return output.splitlines()


def assert_phased_complete(path: Path, expected_markers: int, expected_samples: int) -> None:
    output = run_command(["bcftools", "query", "-f", "[%GT\t]\n", str(path)])
    rows = output.splitlines()
    if len(rows) != expected_markers:
        raise ValueError(f"Unexpected genotype row count in {path.name}: {len(rows)}")
    allowed = {"0|0", "0|1", "1|0", "1|1"}
    for row_number, row in enumerate(rows, start=1):
        calls = row.rstrip("\t").split("\t")
        if len(calls) != expected_samples or any(call not in allowed for call in calls):
            raise ValueError(f"Unphased, missing or non-binary genotype at {path.name}:{row_number}")


def read_sample_map(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [tuple(row) for row in csv.reader(handle, delimiter="\t") if row]
    if any(len(row) != 2 for row in rows):
        raise ValueError("Sample map must have exactly two tab-separated columns")
    return [(sample, ancestry) for sample, ancestry in rows]


def write_subset_inputs(
    source: Path,
    destination: Path,
    regions_path: Path,
) -> Path:
    run_command(
        [
            "bcftools",
            "view",
            "--no-version",
            "--regions-file",
            str(regions_path),
            "--output-type",
            "z",
            "--output",
            str(destination),
            str(source),
        ]
    )
    run_command(["bcftools", "index", "--tbi", str(destination)])
    run_command(["bcftools", "index", "--nrecords", str(destination)])
    return Path(f"{destination}.tbi")


def prepare(args: argparse.Namespace) -> dict:
    contract = load_contract(args.preregistration)
    authenticated = contract["authenticated_inputs"]
    observed_inputs = {
        "reference_vcf": require_hash(args.reference_vcf, authenticated["reference_vcf_sha256"], "reference VCF"),
        "reference_tbi": require_hash(args.reference_tbi, authenticated["reference_tbi_sha256"], "reference TBI"),
        "target_vcf": require_hash(args.target_vcf, authenticated["target_vcf_sha256"], "target VCF"),
        "target_tbi": require_hash(args.target_tbi, authenticated["target_tbi_sha256"], "target TBI"),
        "sample_map": require_hash(args.sample_map, authenticated["sample_map_sha256"], "sample map"),
        "b0_markers": require_hash(args.b0_markers, authenticated["b0_marker_table_sha256"], "B0 table"),
        "genetic_map": require_hash(args.genetic_map, authenticated["genetic_map_sha256"], "genetic map"),
        "gnomix_config": require_hash(args.gnomix_config, authenticated["gnomix_config_sha256"], "Gnomix config"),
    }
    source_panel = contract["source_panel"]
    markers = read_b0_markers(args.b0_markers, int(source_panel["full_b0_markers"]))
    full_positions = [marker["position"] for marker in markers]
    if bcftools_positions(args.reference_vcf) != full_positions:
        raise ValueError("Reference VCF positions differ from the authenticated B0 table")
    if bcftools_positions(args.target_vcf) != full_positions:
        raise ValueError("Target VCF positions differ from the authenticated B0 table")
    reference_contigs = bcftools_contigs(args.reference_vcf)
    target_contigs = bcftools_contigs(args.target_vcf)
    if len(reference_contigs) != 1 or target_contigs != reference_contigs:
        raise ValueError(
            f"REF/TARGET must share exactly one contig; REF={reference_contigs}, TARGET={target_contigs}"
        )
    vcf_contig = reference_contigs[0]
    table_contigs = sorted({marker["chrom"] for marker in markers})
    if len(table_contigs) != 1 or canonical_autosome(table_contigs[0]) != canonical_autosome(vcf_contig):
        raise ValueError(f"B0 table and VCF contigs differ: table={table_contigs}, VCF={vcf_contig}")

    sample_map = read_sample_map(args.sample_map)
    reference_samples = bcftools_samples(args.reference_vcf)
    target_samples = bcftools_samples(args.target_vcf)
    expected_counts = source_panel["reference_samples_per_ancestry"]
    observed_counts = {ancestry: sum(value == ancestry for _, value in sample_map) for ancestry in expected_counts}
    if [sample for sample, _ in sample_map] != reference_samples:
        raise ValueError("Sample-map IDs/order differ from the reference VCF")
    if observed_counts != expected_counts:
        raise ValueError(f"Reference ancestry counts differ: {observed_counts}")
    if len(target_samples) != int(source_panel["target_samples"]):
        raise ValueError("Unexpected target sample count")
    if set(reference_samples) & set(target_samples):
        raise ValueError("Reference and target sample IDs overlap")

    selected, bin_audit = allocate_subset(markers, contract)
    args.outdir.mkdir(parents=True, exist_ok=False)
    markers_path = args.outdir / "m28c_b0_smoke_markers.tsv"
    with markers_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["chrom", "position", "cm", "bin_index", "within_bin_rank"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(selected)
    regions_path = args.outdir / "m28c_b0_smoke_regions.tsv"
    with regions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows((vcf_contig, marker["position"], marker["position"]) for marker in selected)

    reference_subset = args.outdir / "m28c_b0_smoke_reference.vcf.gz"
    target_subset = args.outdir / "m28c_b0_smoke_target.vcf.gz"
    reference_subset_tbi = write_subset_inputs(args.reference_vcf, reference_subset, regions_path)
    target_subset_tbi = write_subset_inputs(args.target_vcf, target_subset, regions_path)
    selected_positions = [marker["position"] for marker in selected]
    if bcftools_positions(reference_subset) != selected_positions:
        raise ValueError("Reference subset positions differ from the frozen selection")
    if bcftools_positions(target_subset) != selected_positions:
        raise ValueError("Target subset positions differ from the frozen selection")
    assert_phased_complete(reference_subset, len(selected), len(reference_samples))
    assert_phased_complete(target_subset, len(selected), len(target_samples))

    derived = contract["gnomix_parameters"]["derived_expected"]
    map_cm = []
    with args.genetic_map.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row:
                map_cm.append(float(row[2]))
    calculated_m = int(round(float(contract["gnomix_parameters"]["window_size_cM"]) * (len(selected) / max(map_cm))))
    if len(selected) % calculated_m == 0:
        calculated_m += 1
    calculated = {
        "C": len(selected),
        "M": calculated_m,
        "W": len(selected) // calculated_m,
        "A": len(expected_counts),
        "S": int(contract["gnomix_parameters"]["smooth_size_windows"]),
        "context_markers_each_side": int(calculated_m * float(contract["gnomix_parameters"]["context_ratio"])),
    }
    if calculated != derived:
        raise ValueError(f"Derived Gnomix dimensions differ: expected {derived}, observed {calculated}")

    outputs = [markers_path, regions_path, reference_subset, reference_subset_tbi, target_subset, target_subset_tbi]
    report = {
        "stage": f"{STAGE}_PREPARE",
        "scope": contract["scope"],
        "contract_sha256": sha256(args.preregistration),
        "authenticated_input_sha256": observed_inputs,
        "selection": {
            "markers": len(selected),
            "bins_total": len(bin_audit),
            "bins_covered": sum(item["selected"] > 0 for item in bin_audit),
            "available_per_bin_min": min(item["available"] for item in bin_audit),
            "available_per_bin_max": max(item["available"] for item in bin_audit),
            "selected_per_bin_min": min(item["selected"] for item in bin_audit),
            "selected_per_bin_max": max(item["selected"] for item in bin_audit),
            "first_position": selected[0]["position"],
            "last_position": selected[-1]["position"],
            "b0_table_contig": table_contigs[0],
            "vcf_contig": vcf_contig,
        },
        "reference_samples": len(reference_samples),
        "target_samples": len(target_samples),
        "reference_ancestry_counts": observed_counts,
        "derived_gnomix_dimensions": calculated,
        "output_sha256": {path.name: sha256(path) for path in outputs},
        "target_used_for_marker_selection": False,
        "truth_accessed": False,
        "model_training_performed": False,
        "gates": {
            "T0_AUTH": True,
            "T1_SUBSET": True,
            "T2_BOUNDARY": True,
        },
        "decision": "GO_PARALLEL_TRAINING_REPLICATES",
    }
    report_path = args.outdir / "m28c_gnomix_smoke_prepare.public.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def authenticate_prepared(reference: Path, prepare_report: Path) -> dict:
    report = json.loads(prepare_report.read_text(encoding="utf-8"))
    if report.get("decision") != "GO_PARALLEL_TRAINING_REPLICATES":
        raise ValueError("Preparation stage did not pass")
    expected = report["output_sha256"].get(reference.name)
    observed = sha256(reference)
    if observed != expected:
        raise ValueError("Prepared reference hash mismatch")
    return report


def audit_model(model_path: Path, config_path: Path, contract: dict) -> dict:
    config = load_yaml(config_path)
    expected = contract["gnomix_parameters"]["derived_expected"]
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    observed = {
        "C": int(model.C),
        "M": int(model.M),
        "W": int(model.W),
        "A": int(model.A),
        "S": int(model.S),
        "context_markers_each_side": int(model.context),
    }
    if observed != expected:
        raise ValueError(f"Serialized model dimensions differ: {observed}")
    population_order = [str(value) for value in model.population_order]
    if population_order != ["AFR", "EUR", "ASIA"]:
        raise ValueError(f"Unexpected population order: {population_order}")
    if model.seed != int(contract["gnomix_parameters"]["seed"]):
        raise ValueError("Serialized model seed differs from contract")
    if type(model.base).__name__ != "LogisticRegressionBase" or type(model.smooth).__name__ != "XGB_Smoother":
        raise ValueError("Serialized model classes differ from the frozen default architecture")
    if len(model.base.models) != expected["W"]:
        raise ValueError("Unexpected number of fitted base models")
    if not all(hasattr(base_model, "classes_") for base_model in model.base.models):
        raise ValueError("At least one base model is not fitted")
    if not hasattr(model.smooth.model, "classes_"):
        raise ValueError("Smoother is not fitted")
    frozen = contract["gnomix_parameters"]
    checks = {
        "seed": config["seed"] == frozen["seed"],
        "mode": config["model"]["inference"] == frozen["mode"],
        "window_size_cM": config["model"]["window_size_cM"] == frozen["window_size_cM"],
        "smooth_size": config["model"]["smooth_size"] == frozen["smooth_size_windows"],
        "context_ratio": config["model"]["context_ratio"] == frozen["context_ratio"],
        "n_cores": config["model"]["n_cores"] == frozen["n_cores"],
        "calibrate": config["model"]["calibrate"] is False,
        "retrain_base": config["model"]["retrain_base"] is True,
    }
    if not all(checks.values()):
        raise ValueError(f"Resolved config differs from contract: {checks}")
    return {
        "derived_dimensions": observed,
        "population_order": population_order,
        "architecture": {
            "base": type(model.base).__name__,
            "smoother": type(model.smooth).__name__,
            "base_models": len(model.base.models),
        },
        "config_checks": checks,
    }


def training(args: argparse.Namespace) -> dict:
    contract = load_contract(args.preregistration)
    if args.replicate not in contract["execution"]["replicates"]:
        raise ValueError(f"Unexpected replicate: {args.replicate}")
    prepared = authenticate_prepared(args.reference_vcf, args.prepare_report)
    authenticated = contract["authenticated_inputs"]
    require_hash(args.sample_map, authenticated["sample_map_sha256"], "sample map")
    require_hash(args.genetic_map, authenticated["genetic_map_sha256"], "genetic map")
    require_hash(args.gnomix_config, authenticated["gnomix_config_sha256"], "Gnomix config")
    runtime_audit_path = args.gnomix_root / "GNOMIX_RUNTIME_AUDIT.json"
    require_hash(
        runtime_audit_path,
        contract["software"]["runtime_known_answer_sha256"],
        "Gnomix runtime known-answer audit",
    )
    runtime_audit = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
    if runtime_audit.get("decision") != "PASS_GNOMIX_LIBLINEAR_OVR_RUNTIME":
        raise ValueError("Gnomix runtime known-answer audit did not pass")
    if runtime_audit.get("scikit_learn_version") != contract["software"]["scikit_learn"]:
        raise ValueError("Gnomix runtime scikit-learn version differs from the contract")
    args.outdir.mkdir(parents=True, exist_ok=False)
    stdout_path = args.outdir / "gnomix_train.stdout.log"
    stderr_path = args.outdir / "gnomix_train.stderr.log"
    command = [
        sys.executable,
        str(args.gnomix_root / "gnomix.py"),
        "None",
        str(args.outdir.resolve()),
        "22",
        "false",
        str(args.genetic_map.resolve()),
        str(args.reference_vcf.resolve()),
        str(args.sample_map.resolve()),
        str(args.gnomix_config.resolve()),
    ]
    environment = os.environ.copy()
    threads = str(contract["gnomix_parameters"]["n_cores"])
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = threads
    started = time.monotonic()
    run_command(command, env=environment, stdout_path=stdout_path, stderr_path=stderr_path)
    duration = time.monotonic() - started
    model_name = load_yaml(args.gnomix_config)["model"]["name"]
    model_dir = args.outdir / "models" / f"{model_name}_chm_22"
    model_path = model_dir / f"{model_name}_chm_22.pkl"
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise ValueError("Gnomix did not create a nonempty model pickle")
    model_audit = audit_model(model_path, args.gnomix_config, contract)
    report = {
        "stage": f"{STAGE}_TRAIN",
        "replicate": args.replicate,
        "scope": contract["scope"],
        "contract_sha256": sha256(args.preregistration),
        "prepare_report_sha256": sha256(args.prepare_report),
        "prepared_reference_sha256": prepared["output_sha256"][args.reference_vcf.name],
        "gnomix_config_sha256": sha256(args.gnomix_config),
        "runtime_known_answer_sha256": sha256(runtime_audit_path),
        "runtime_known_answer_decision": runtime_audit["decision"],
        "model_relative_path": str(model_path.relative_to(args.outdir)),
        "model_sha256": sha256(model_path),
        "duration_seconds_internal": duration,
        "model_audit": model_audit,
        "target_input_present": False,
        "truth_accessed": False,
        "internal_synthetic_validation_generated_by_gnomix": True,
        "internal_synthetic_validation_used_for_decision": False,
        "target_truth_accuracy_computed": False,
        "gates": {
            "T0_AUTH": True,
            "T2_BOUNDARY": True,
            "T3_MODEL": True,
            "T4_SERIALIZATION": True,
        },
        "decision": "GO_FROZEN_MODEL_INFERENCE_NO_TRUTH",
    }
    report_path = args.outdir / "m28c_gnomix_smoke_train.public.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_msp(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as handle:
        population_line = handle.readline().rstrip("\n")
        header = handle.readline().rstrip("\n").lstrip("#").split("\t")
        rows = list(csv.reader(handle, delimiter="\t"))
    populations = [item.split("=", 1)[0] for item in population_line.split(": ", 1)[1].split("\t")]
    return {"populations": populations, "header": header, "rows": rows}


def parse_fb(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as handle:
        population_line = handle.readline().rstrip("\n")
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        header = reader.fieldnames or []
    populations = population_line.split("\t")[1:]
    metadata_header = header[:4]
    probability_header = header[4:]
    metadata = [[row[field] for field in metadata_header] for row in rows]
    probabilities = [[float(row[field]) for field in probability_header] for row in rows]
    return {
        "populations": populations,
        "metadata_header": metadata_header,
        "probability_header": probability_header,
        "metadata": metadata,
        "probabilities": probabilities,
    }


def audit_predictions(msp_path: Path, fb_path: Path, target_vcf: Path, contract: dict) -> dict:
    expected = contract["gnomix_parameters"]["derived_expected"]
    samples = bcftools_samples(target_vcf)
    msp = parse_msp(msp_path)
    fb = parse_fb(fb_path)
    expected_populations = ["AFR", "EUR", "ASIA"]
    if msp["populations"] != expected_populations or fb["populations"] != expected_populations:
        raise ValueError("Prediction ancestry order differs from the frozen order")
    expected_sample_columns = [f"{sample}.{hap}" for sample in samples for hap in (0, 1)]
    if msp["header"][6:] != expected_sample_columns:
        raise ValueError("MSP sample/haplotype columns differ from TARGET")
    if len(msp["rows"]) != expected["W"] or len(fb["metadata"]) != expected["W"]:
        raise ValueError("Prediction window count differs from the serialized model")
    labels = [[int(value) for value in row[6:]] for row in msp["rows"]]
    if any(value not in (0, 1, 2) for row in labels for value in row):
        raise ValueError("MSP contains an invalid ancestry label")
    if sum(int(float(row[5])) for row in msp["rows"]) != int(contract["subset"]["markers"]):
        raise ValueError("MSP marker counts do not sum to the 10000-marker subset")
    expected_probability_columns = len(samples) * 2 * len(expected_populations)
    if len(fb["probability_header"]) != expected_probability_columns:
        raise ValueError("FB probability-column count differs from TARGET x haplotypes x ancestries")
    max_sum_error = 0.0
    for row in fb["probabilities"]:
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in row):
            raise ValueError("FB contains a nonfinite or out-of-range probability")
        for offset in range(0, len(row), len(expected_populations)):
            max_sum_error = max(max_sum_error, abs(sum(row[offset : offset + 3]) - 1.0))
    if max_sum_error > 1e-6:
        raise ValueError(f"FB ancestry probabilities do not sum to one: {max_sum_error}")
    return {
        "target_samples": len(samples),
        "windows": len(msp["rows"]),
        "population_order": expected_populations,
        "msp_haplotype_columns": len(expected_sample_columns),
        "fb_probability_columns": expected_probability_columns,
        "max_probability_sum_error": max_sum_error,
    }


def inference(args: argparse.Namespace) -> dict:
    contract = load_contract(args.preregistration)
    if args.replicate not in contract["execution"]["replicates"]:
        raise ValueError(f"Unexpected replicate: {args.replicate}")
    train_report = json.loads(args.train_report.read_text(encoding="utf-8"))
    if train_report.get("decision") != "GO_FROZEN_MODEL_INFERENCE_NO_TRUTH":
        raise ValueError("Training stage did not pass")
    prepared = json.loads(args.prepare_report.read_text(encoding="utf-8"))
    if sha256(args.target_vcf) != prepared["output_sha256"].get(args.target_vcf.name):
        raise ValueError("Prepared TARGET subset hash mismatch")
    model_path = args.training_dir / train_report["model_relative_path"]
    if sha256(model_path) != train_report["model_sha256"]:
        raise ValueError("Frozen model hash differs before inference")
    require_hash(
        args.gnomix_config,
        contract["authenticated_inputs"]["gnomix_config_sha256"],
        "Gnomix config",
    )
    args.outdir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.gnomix_config, args.outdir / "config.yaml")
    stdout_path = args.outdir / "gnomix_infer.stdout.log"
    stderr_path = args.outdir / "gnomix_infer.stderr.log"
    result_dir = args.outdir / "results"
    command = [
        sys.executable,
        str(args.gnomix_root / "gnomix.py"),
        str(args.target_vcf.resolve()),
        str(result_dir.resolve()),
        "22",
        "false",
        str(model_path.resolve()),
    ]
    environment = os.environ.copy()
    threads = str(contract["gnomix_parameters"]["n_cores"])
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = threads
    started = time.monotonic()
    run_command(
        command,
        cwd=args.outdir,
        env=environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    duration = time.monotonic() - started
    msp_path = result_dir / "query_results.msp"
    fb_path = result_dir / "query_results.fb"
    if not msp_path.is_file() or not fb_path.is_file():
        raise ValueError("Gnomix did not create both MSP and FB outputs")
    prediction_audit = audit_predictions(msp_path, fb_path, args.target_vcf, contract)
    report = {
        "stage": f"{STAGE}_INFERENCE",
        "replicate": args.replicate,
        "scope": contract["scope"],
        "contract_sha256": sha256(args.preregistration),
        "train_report_sha256": sha256(args.train_report),
        "model_sha256_verified": sha256(model_path),
        "target_subset_sha256": sha256(args.target_vcf),
        "duration_seconds_internal": duration,
        "prediction_audit": prediction_audit,
        "output_sha256": {msp_path.name: sha256(msp_path), fb_path.name: sha256(fb_path)},
        "truth_accessed": False,
        "target_truth_accuracy_computed": False,
        "gates": {
            "T4_SERIALIZATION": True,
            "T5_INFERENCE": True,
            "T8_SCOPE": True,
        },
        "decision": "GO_REPLICATE_COMPARISON_NO_TRUTH",
    }
    report_path = args.outdir / "m28c_gnomix_smoke_inference.public.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def comparison(args: argparse.Namespace) -> dict:
    contract = load_contract(args.preregistration)
    report_a = json.loads(args.report_a.read_text(encoding="utf-8"))
    report_b = json.loads(args.report_b.read_text(encoding="utf-8"))
    if report_a.get("replicate") != "A" or report_b.get("replicate") != "B":
        raise ValueError("Inference reports are not ordered A then B")
    if report_a.get("decision") != "GO_REPLICATE_COMPARISON_NO_TRUTH" or report_b.get("decision") != "GO_REPLICATE_COMPARISON_NO_TRUTH":
        raise ValueError("At least one inference replicate did not pass")
    msp_a = args.inference_a / "results" / "query_results.msp"
    msp_b = args.inference_b / "results" / "query_results.msp"
    fb_a = args.inference_a / "results" / "query_results.fb"
    fb_b = args.inference_b / "results" / "query_results.fb"
    msp_exact = msp_a.read_bytes() == msp_b.read_bytes()
    parsed_a = parse_fb(fb_a)
    parsed_b = parse_fb(fb_b)
    metadata_exact = (
        parsed_a["populations"] == parsed_b["populations"]
        and parsed_a["metadata_header"] == parsed_b["metadata_header"]
        and parsed_a["probability_header"] == parsed_b["probability_header"]
        and parsed_a["metadata"] == parsed_b["metadata"]
    )
    if len(parsed_a["probabilities"]) != len(parsed_b["probabilities"]):
        raise ValueError("FB replicate row counts differ")
    max_abs = 0.0
    for row_a, row_b in zip(parsed_a["probabilities"], parsed_b["probabilities"], strict=True):
        if len(row_a) != len(row_b):
            raise ValueError("FB replicate column counts differ")
        for value_a, value_b in zip(row_a, row_b, strict=True):
            max_abs = max(max_abs, abs(value_a - value_b))
    tolerance = contract["execution"]["prediction_comparison"]
    predictions_equal = max_abs <= float(tolerance["probabilities_atol"])
    gates = {
        "T5_INFERENCE": True,
        "T6_REPRODUCIBILITY": msp_exact and metadata_exact and predictions_equal,
        "T8_SCOPE": True,
    }
    if not all(gates.values()):
        raise ValueError(
            f"Prediction reproducibility failed: msp={msp_exact}, metadata={metadata_exact}, max_abs={max_abs}"
        )
    args.outdir.mkdir(parents=True, exist_ok=False)
    report = {
        "stage": f"{STAGE}_COMPARE",
        "scope": contract["scope"],
        "contract_sha256": sha256(args.preregistration),
        "replicate_report_sha256": {"A": sha256(args.report_a), "B": sha256(args.report_b)},
        "msp_byte_identical": msp_exact,
        "fb_metadata_exact": metadata_exact,
        "fb_probability_max_abs_difference": max_abs,
        "fb_probability_atol": float(tolerance["probabilities_atol"]),
        "model_pickle_hash_equality_required": False,
        "truth_accessed": False,
        "target_truth_accuracy_computed": False,
        "resource_review_pending_from_nextflow_trace": True,
        "gates": gates,
        "decision": "PASS_PREDICTIONS_PENDING_RESOURCE_TRACE_REVIEW",
    }
    report_path = args.outdir / "m28c_gnomix_smoke_compare.public.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preregistration", required=True, type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    add_common(prepare_parser)
    prepare_parser.add_argument("--reference-vcf", required=True, type=Path)
    prepare_parser.add_argument("--reference-tbi", required=True, type=Path)
    prepare_parser.add_argument("--target-vcf", required=True, type=Path)
    prepare_parser.add_argument("--target-tbi", required=True, type=Path)
    prepare_parser.add_argument("--sample-map", required=True, type=Path)
    prepare_parser.add_argument("--b0-markers", required=True, type=Path)
    prepare_parser.add_argument("--genetic-map", required=True, type=Path)
    prepare_parser.add_argument("--gnomix-config", required=True, type=Path)
    prepare_parser.add_argument("--outdir", required=True, type=Path)
    prepare_parser.set_defaults(function=prepare)

    train_parser = subparsers.add_parser("train")
    add_common(train_parser)
    train_parser.add_argument("--reference-vcf", required=True, type=Path)
    train_parser.add_argument("--sample-map", required=True, type=Path)
    train_parser.add_argument("--genetic-map", required=True, type=Path)
    train_parser.add_argument("--gnomix-config", required=True, type=Path)
    train_parser.add_argument("--prepare-report", required=True, type=Path)
    train_parser.add_argument("--gnomix-root", required=True, type=Path)
    train_parser.add_argument("--replicate", required=True)
    train_parser.add_argument("--outdir", required=True, type=Path)
    train_parser.set_defaults(function=training)

    infer_parser = subparsers.add_parser("infer")
    add_common(infer_parser)
    infer_parser.add_argument("--training-dir", required=True, type=Path)
    infer_parser.add_argument("--train-report", required=True, type=Path)
    infer_parser.add_argument("--target-vcf", required=True, type=Path)
    infer_parser.add_argument("--prepare-report", required=True, type=Path)
    infer_parser.add_argument("--gnomix-config", required=True, type=Path)
    infer_parser.add_argument("--gnomix-root", required=True, type=Path)
    infer_parser.add_argument("--replicate", required=True)
    infer_parser.add_argument("--outdir", required=True, type=Path)
    infer_parser.set_defaults(function=inference)

    compare_parser = subparsers.add_parser("compare")
    add_common(compare_parser)
    compare_parser.add_argument("--inference-a", required=True, type=Path)
    compare_parser.add_argument("--report-a", required=True, type=Path)
    compare_parser.add_argument("--inference-b", required=True, type=Path)
    compare_parser.add_argument("--report-b", required=True, type=Path)
    compare_parser.add_argument("--outdir", required=True, type=Path)
    compare_parser.set_defaults(function=comparison)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = args.function(args)
    print(json.dumps({"decision": report["decision"], "stage": report["stage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
