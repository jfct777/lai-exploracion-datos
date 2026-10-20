#!/usr/bin/env python3
"""Run one authenticated K06 Refined IBD case and normalize its complete calls.

This program does not choose detector parameters.  It requires every Refined
IBD analysis parameter, its scientific status, and its provenance in a pinned
JSON specification.  A diagnostic historical profile may produce reusable
calls, but it is never marked as an admitted calibration.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(value: dict, label: str, staged_input_dir: Path | None = None) -> Path:
    require(set(value) == {"path", "sha256"}, f"Exact descriptor required: {label}")
    original = Path(value["path"])
    require(original.is_absolute(), f"Non-absolute source descriptor: {label}")
    path = original
    if staged_input_dir is not None:
        candidates = [candidate for candidate in staged_input_dir.iterdir()
                      if candidate.is_file() and sha256(candidate) == value["sha256"]]
        require(len(candidates) == 1, f"Staged input identity differs: {label}")
        path = candidates[0]
    require(path.is_file(), f"Missing input: {label}")
    require(sha256(path) == value["sha256"], f"Input changed: {label}")
    return path


def read_vcf_roster(path: Path, chromosome: str) -> tuple[list[str], int]:
    opener = gzip.open if path.suffix == ".gz" else open
    samples: list[str] | None = None
    markers = 0
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM\t"):
                fields = line.rstrip("\n").split("\t")
                require(fields[8] == "FORMAT" and len(fields) > 9, "VCF sample header missing")
                samples = fields[9:]
                require(len(samples) == len(set(samples)), "Duplicate VCF sample identifier")
                continue
            require(samples is not None, "VCF data precede sample header")
            fields = line.rstrip("\n").split("\t")
            require(fields[0].removeprefix("chr") == chromosome, "VCF chromosome differs")
            require(len(fields) == 9 + len(samples), "VCF row/sample count differs")
            formats = fields[8].split(":")
            require("GT" in formats, "VCF GT field missing")
            gt_index = formats.index("GT")
            for sample_field in fields[9:]:
                values = sample_field.split(":")
                require(gt_index < len(values) and re.fullmatch(r"[0-9]+\|[0-9]+", values[gt_index]) is not None,
                        "Refined IBD requires phased, nonmissing diploid genotypes")
            markers += 1
    require(samples is not None and markers > 1, "VCF has no usable markers")
    return samples, markers


def read_parameters(spec_path: Path, spec_sha256: str) -> tuple[dict, bool]:
    require(sha256(spec_path) == spec_sha256, "Parameter specification changed")
    spec = json.loads(spec_path.read_text())
    require(spec.get("schema") == "r02_k06_refined_ibd_parameters_v1", "Parameter schema differs")
    status = spec.get("status")
    require(status in {"ADMITTED", "DIAGNOSTIC_HISTORICAL_NOT_ADMITTED"}, "Parameter status differs")
    expected = {"window_cm", "lod_min", "length_cm_min", "trim_cm", "scale_input", "nthreads"}
    values = spec.get("parameters", {})
    require(set(values) == expected, "All Refined IBD parameters must be explicit")
    for name in expected - {"nthreads"}:
        require(isinstance(values[name], (int, float)) and math.isfinite(values[name]), f"Invalid {name}")
    require(isinstance(values["nthreads"], int) and 1 <= values["nthreads"] <= 2, "nthreads exceeds task limit")
    require(values["window_cm"] >= 3 * values["length_cm_min"], "window must be at least 3*length")
    require(values["lod_min"] > 0 and values["length_cm_min"] > 0, "Positive lod/length required")
    require(values["trim_cm"] >= 0 and values["scale_input"] >= 0, "Negative trim/scale")
    provenance = spec.get("provenance", {})
    require(provenance.get("refined_ibd_source_level") == "CODIGO_PRIMARIO", "Primary detector source required")
    require(len(provenance.get("refined_ibd_source_sha256", "")) == 64, "Primary source hash required")
    if status != "ADMITTED":
        require(spec.get("scientific_scope") == "TECHNICAL_DETECTOR_DIAGNOSTIC_ONLY",
                "Diagnostic profile cannot claim calibration scope")
    return spec, status == "ADMITTED"


def prepare_plink_map(source: Path, output: Path, chromosome: str) -> int:
    """Convert project chrom/bp/cM rows to Refined IBD's PLINK 4 columns."""
    opener = gzip.open if source.suffix == ".gz" else open
    rows = 0
    previous_bp, previous_cm = None, None
    with opener(source, "rt") as src, output.open("x") as dst:
        for line in src:
            fields = line.split()
            require(len(fields) == 3, "Project genetic map must contain chrom, bp, cM")
            chrom, bp_text, cm_text = fields
            require(chrom.removeprefix("chr") == chromosome, "Map chromosome differs")
            bp, cm = int(bp_text), float(cm_text)
            require(bp > 0 and math.isfinite(cm) and cm >= 0, "Invalid map coordinate")
            require(previous_bp is None or (bp > previous_bp and cm >= previous_cm),
                    "Map coordinates are not strictly increasing in bp and nondecreasing in cM")
            dst.write(f"{chrom}\tk06_{chrom}_{bp}\t{cm_text}\t{bp_text}\n")
            previous_bp, previous_cm = bp, cm
            rows += 1
    require(rows >= 2 and previous_cm is not None, "Map requires at least two rows")
    return rows


def normalize_calls(native_path: Path | list[Path], output_path: Path, chromosome: str,
                    sample_ids: list[str], max_rows: int) -> int:
    """Translate Refined IBD's documented nine fields to a named stable table."""
    people = set(sample_ids)
    rows: list[tuple] = []
    paths = [native_path] if isinstance(native_path, Path) else native_path
    for path in paths:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                require(len(rows) < max_rows, "Detector call row budget exceeded")
                fields = line.split()
                require(len(fields) == 9, "Refined IBD output must contain nine fields")
                sample_a, hap_a, sample_b, hap_b, chrom, start, end, lod, length = fields
                hap_a, hap_b = int(hap_a), int(hap_b)
                start, end = int(start), int(end)
                lod, length = float(lod), float(length)
                require(sample_a in people and sample_b in people,
                        "Detector call contains an unknown person")
                require(hap_a in (1, 2) and hap_b in (1, 2), "Detector haplotype index differs")
                require((sample_a, hap_a) != (sample_b, hap_b),
                        "Detector call repeats one chromosome copy")
                require(chrom.removeprefix("chr") == chromosome, "Detector chromosome differs")
                require(0 < start <= end and math.isfinite(lod) and math.isfinite(length) and length > 0,
                        "Invalid detector interval or score")
                left, right = (sample_a, hap_a), (sample_b, hap_b)
                if right < left:
                    left, right = right, left
                rows.append((chromosome, left[0], left[1], right[0], right[1], start, end, lod, length))
    rows.sort()
    fields = ["chromosome", "sample_a", "haplotype_a", "sample_b", "haplotype_b",
              "start_bp_inclusive", "end_bp_inclusive", "lod_score", "reported_length_cm"]
    with gzip.open(output_path, "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)
    return len(rows)


def build_command(java: str, jar: Path, vcf: Path, genetic_map: Path,
                  prefix: Path, parameters: dict, java_stack_mib: int) -> list[str]:
    require(isinstance(java_stack_mib, int) and not isinstance(java_stack_mib, bool),
            "JVM stack size must be an integer number of MiB")
    require(1 <= java_stack_mib <= 16, "JVM stack size must be between 1 and 16 MiB")
    return [java, f"-Xss{java_stack_mib}m", "-Xmx1536m", "-jar", str(jar),
            f"gt={vcf}", f"map={genetic_map}",
            f"out={prefix}", f"nthreads={parameters['nthreads']}",
            f"window={parameters['window_cm']}", f"lod={parameters['lod_min']}",
            f"length={parameters['length_cm_min']}", f"trim={parameters['trim_cm']}",
            f"scale={parameters['scale_input']}"]


def run(request_path: Path, expected_sha256: str, output: Path,
        staged_input_dir: Path | None, java_stack_mib: int) -> dict:
    require(sha256(request_path) == expected_sha256, "Request changed")
    request = json.loads(request_path.read_text())
    require(request.get("schema") == "r02_k06_refined_ibd_request_v1", "Request schema differs")
    require(request.get("genome_build") == "GRCh38" and request.get("coordinate_system") == "1-based-inclusive",
            "Genome build or coordinate system differs")
    require(request.get("case_id") and request.get("chromosome") in {"1", "22"}, "Case/chromosome differs")
    if staged_input_dir is not None:
        require(staged_input_dir.is_dir(), "Staged input directory missing")
    jar = descriptor(request["detector_jar"], "detector_jar", staged_input_dir)
    vcf = descriptor(request["genotypes"], "genotypes", staged_input_dir)
    genetic_map = descriptor(request["genetic_map"], "genetic_map", staged_input_dir)
    positive = descriptor(request["positive_territory"], "positive_territory", staged_input_dir)
    spec_path = descriptor(request["parameter_spec"], "parameter_spec", staged_input_dir)
    spec, admitted = read_parameters(spec_path, request["parameter_spec"]["sha256"])
    sample_ids, marker_count = read_vcf_roster(vcf, request["chromosome"])
    require(len(sample_ids) == request["expected_samples"], "VCF sample count differs")
    require(marker_count == request["expected_markers"], "VCF marker count differs")
    require(request["max_call_rows"] > 0, "Positive call row budget required")
    require(not output.exists(), "Output exists; preserve prior result")
    output.parent.mkdir(parents=True, exist_ok=True)
    java = shutil.which(request.get("java_executable", "java"))
    require(java is not None, "Java executable unavailable")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="k06-refined-ibd-", dir=output.parent) as temporary:
        work = Path(temporary)
        prefix = work / "refined-ibd"
        effective_map = work / "refined-ibd.effective.plink.map"
        effective_map_rows = prepare_plink_map(genetic_map, effective_map, request["chromosome"])
        command = build_command(java, jar, vcf, effective_map, prefix, spec["parameters"],
                                java_stack_mib)
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=request["timeout_seconds"], check=False)
        require(completed.returncode == 0, "Refined IBD failed: " + completed.stdout[-2000:])
        native = prefix.with_suffix(".ibd.gz")
        native_hbd = prefix.with_suffix(".hbd.gz")
        log = prefix.with_suffix(".log")
        require(native.is_file() and native_hbd.is_file() and log.is_file(), "Refined IBD complete outputs missing")
        require(log.read_text().rstrip().endswith("refined-ibd.17Jan20.102.jar finished"),
                "Refined IBD terminal marker missing")
        destination = work / "result"
        destination.mkdir()
        native_final = destination / "refined-ibd.native.ibd.gz"
        native_hbd_final = destination / "refined-ibd.native.hbd.gz"
        log_final = destination / "refined-ibd.log"
        calls_final = destination / "refined-ibd.detected.private.tsv.gz"
        effective_map_final = destination / "refined-ibd.effective.plink.map"
        shutil.copy2(native, native_final)
        shutil.copy2(native_hbd, native_hbd_final)
        shutil.copy2(log, log_final)
        shutil.copy2(effective_map, effective_map_final)
        call_rows = normalize_calls([native_final, native_hbd_final], calls_final, request["chromosome"],
                                    sample_ids, request["max_call_rows"])
        receipt = {
            "schema": "r02_k06_refined_ibd_receipt_v1", "status": "COMPLETE",
            "case_id": request["case_id"], "chromosome": request["chromosome"],
            "request_sha256": expected_sha256,
            "scientific_scope": spec["scientific_scope"], "detector_parameters_admitted": admitted,
            "parameter_spec_sha256": request["parameter_spec"]["sha256"],
            "detector_sha256": request["detector_jar"]["sha256"],
            "genotypes_sha256": request["genotypes"]["sha256"],
            "genetic_map_sha256": request["genetic_map"]["sha256"],
            "effective_plink_map_sha256": sha256(effective_map_final),
            "effective_plink_map_rows": effective_map_rows,
            "effective_plink_map_conversion": "source chrom,bp,cM -> chrom,k06_{chrom}_{bp},cM,bp; coordinate strings preserved",
            "positive_territory_sha256": request["positive_territory"]["sha256"],
            "sample_ids": sample_ids, "sample_count": len(sample_ids), "marker_count": marker_count,
            "phased_nonmissing_genotypes_verified": True,
            "phase_accuracy_validated": False,
            "call_rows": call_rows, "calls_sha256": sha256(calls_final),
            "native_calls_sha256": sha256(native_final),
            "native_hbd_calls_sha256": sha256(native_hbd_final), "log_sha256": sha256(log_final),
            "java_stack_mib": java_stack_mib, "java_max_heap_mib": 1536,
            "elapsed_seconds": time.monotonic() - started,
            "negative_truth_available": False, "precision": None, "specificity": None,
            "detector_calibration_complete": False, "demographic_inference": False,
        }
        (destination / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        shutil.move(destination, output)
    require(sha256(request_path) == expected_sha256 and sha256(vcf) == request["genotypes"]["sha256"],
            "Input changed while detector ran")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--staged-input-dir", type=Path)
    parser.add_argument("--java-stack-mib", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.request, args.request_sha256, args.output,
                         args.staged_input_dir, args.java_stack_mib), sort_keys=True))
