#!/usr/bin/env python3
"""Project an existing FLARE posterior onto a truth-blind LAI query grid.

The source FLARE VCF is read-only and must contain ANP1/ANP2 haplotype
probabilities.  Query coordinates and people come from the published overlay;
the truth arrays in that archive are deliberately never loaded.  Exact bp
matches override genetic-map interpolation.  Between markers the established
M39 linear-cM rule is used, and queries outside the marker span are clamped.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


SCHEMA = "r02_lai_flare_query_projection_request_v1"
POLICY = "M39_LINEAR_CM_EXACT_BP_OVERRIDE_MEAN_CM_TIES_CLAMP_ENDS"
STATES = ("AA", "AE", "AN", "EE", "EN", "NN")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve(item, name, staged=None):
    require(isinstance(item, dict) and set(item) == {"path", "sha256"}, "invalid descriptor: " + name)
    path = Path(item["path"])
    if staged:
        matches = [p for p in Path(staged).iterdir() if p.is_file() and sha(p) == item["sha256"]]
        require(len(matches) == 1, "staged input absent or ambiguous: " + name)
        path = matches[0]
    require(path.is_file() and sha(path) == item["sha256"], "input differs: " + name)
    return path


def genetic_map(path, chromosome, np):
    bp, cm = [], []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            require(len(fields) == 4 and fields[0].removeprefix("chr") == chromosome,
                    f"genetic map row {number} differs")
            bp.append(int(fields[3])); cm.append(float(fields[2]))
    bp = np.asarray(bp, dtype=np.int64); cm = np.asarray(cm, dtype=np.float64)
    require(len(bp) >= 2 and np.all(np.diff(bp) > 0) and np.all(np.diff(cm) >= 0)
            and np.isfinite(cm).all(), "genetic map is not finite/monotone")
    return bp, cm


def joint(first, second, np):
    a = np.asarray(first, dtype=np.float32).astype(np.float64)
    b = np.asarray(second, dtype=np.float32).astype(np.float64)
    require(a.shape == b.shape and a.ndim == 2 and a.shape[1] == 3
            and np.isfinite(a).all() and np.isfinite(b).all()
            and (a >= 0).all() and (b >= 0).all(), "invalid FLARE haplotype probability")
    residual = max(float(np.abs(a.sum(1) - 1).max()), float(np.abs(b.sum(1) - 1).max()))
    require(residual <= 5e-7, "stored FLARE haplotype probability mass differs")
    result = np.stack((a[:, 0] * b[:, 0], a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0],
        a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0], a[:, 1] * b[:, 1],
        a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1], a[:, 2] * b[:, 2]), axis=1)
    require(np.abs(result.sum(1) - 1).max() <= 1e-6, "six-state probability mass differs")
    return result.astype(np.float32), residual


def scan_vcf(path):
    samples = None; markers = 0
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.startswith("#CHROM\t"):
                require(samples is None, "duplicate VCF header")
                samples = line.rstrip("\n").split("\t")[9:]
            elif not line.startswith("#"):
                require(samples is not None, "VCF data precedes header")
                markers += 1
    require(samples and len(samples) == len(set(samples)) and markers >= 2, "invalid FLARE VCF axes")
    return samples, markers


def load_vcf(path, chromosome, samples, markers, work, map_bp, map_cm, np):
    probability = np.lib.format.open_memmap(work / "source-probabilities.npy", mode="w+",
                                             dtype=np.float32, shape=(markers, len(samples), 6))
    positions = np.empty(markers, dtype=np.int64); maximum_residual = 0.0
    row_index = 0; ancestry_header = False
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.startswith("##ANCESTRY=<AFR=0,EUR=1,NAM=2>"):
                ancestry_header = True
            if line.startswith("#CHROM\t"):
                require(line.rstrip("\n").split("\t")[9:] == samples and ancestry_header,
                        "FLARE sample/ancestry header differs")
                continue
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(samples) and fields[0].removeprefix("chr") == chromosome,
                    "FLARE row chromosome/sample count differs")
            position = int(fields[1])
            require(row_index < markers and (row_index == 0 or position > positions[row_index - 1]),
                    "FLARE marker bp axis is not unique and ordered")
            fmt = fields[8].split(":")
            require(len(set(fmt)) == len(fmt) and {"ANP1", "ANP2"} <= set(fmt), "FLARE posterior fields absent")
            first, second = [], []; ix, iy = fmt.index("ANP1"), fmt.index("ANP2")
            for cell in fields[9:]:
                values = cell.split(":")
                require(len(values) == len(fmt), "FLARE posterior cell differs")
                first.append([float(x) for x in values[ix].split(",")])
                second.append([float(x) for x in values[iy].split(",")])
            probability[row_index], residual = joint(first, second, np)
            maximum_residual = max(maximum_residual, residual)
            positions[row_index] = position; row_index += 1
    require(row_index == markers and map_bp[0] <= positions[0] < positions[-1] <= map_bp[-1],
            "FLARE marker count or genetic-map support differs")
    probability.flush()
    return probability, positions, np.interp(positions, map_bp, map_cm), maximum_residual


def project(source, source_bp, source_cm, query_bp, query_cm, output, chunk, np):
    require(len(source_bp) == len(source_cm) == source.shape[0] and np.all(np.diff(source_bp) > 0),
            "source axes differ")
    require(np.all(np.diff(query_bp) > 0) and np.all(np.diff(query_cm) >= 0), "query axes differ")
    # Average tied cM markers before interpolation, as in the established M39 projector.
    unique, starts, counts = np.unique(source_cm, return_index=True, return_counts=True)
    grouped = np.lib.format.open_memmap(output.parent / "grouped-probabilities.npy", mode="w+",
                                        dtype=np.float32, shape=(len(unique), source.shape[1], 6))
    for index, (start, count) in enumerate(zip(starts, counts)):
        grouped[index] = np.asarray(source[start:start + count], dtype=np.float64).mean(0).astype(np.float32)
    grouped.flush()
    result = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32,
                                       shape=(source.shape[1], len(query_bp), 6))
    exact_lookup = {int(bp): i for i, bp in enumerate(source_bp)}
    audit = {"exact_locus": 0, "exact_cm_group": 0, "interpolated": 0,
             "clamped_left": 0, "clamped_right": 0}
    for lo in range(0, len(query_bp), chunk):
        hi = min(len(query_bp), lo + chunk)
        block = np.empty((source.shape[1], hi - lo, 6), dtype=np.float32)
        for offset, q in enumerate(range(lo, hi)):
            exact = exact_lookup.get(int(query_bp[q]))
            if exact is not None:
                value = source[exact]; audit["exact_locus"] += 1
            else:
                right = int(np.searchsorted(unique, query_cm[q], side="left"))
                if right == 0:
                    value = grouped[0]; audit["clamped_left"] += 1
                elif right == len(unique):
                    value = grouped[-1]; audit["clamped_right"] += 1
                elif unique[right] == query_cm[q]:
                    value = grouped[right]; audit["exact_cm_group"] += 1
                else:
                    fraction = (query_cm[q] - unique[right - 1]) / (unique[right] - unique[right - 1])
                    value = ((1 - fraction) * grouped[right - 1].astype(np.float64)
                             + fraction * grouped[right].astype(np.float64)).astype(np.float32)
                    audit["interpolated"] += 1
            block[:, offset] = value
        result[:, lo:hi] = block
    result.flush()
    mass = float(np.abs(np.asarray(result).sum(-1) - 1).max())
    require(np.isfinite(result).all() and (result >= 0).all() and mass <= 5e-5,
            "projected probability domain/mass differs")
    return audit, mass


def run(request_path, digest, output_dir, staged=None):
    import numpy as np
    require(sha(request_path) == digest, "request SHA-256 differs")
    request = read_json(request_path)
    require(request.get("schema") == SCHEMA
            and request.get("operation") == "PROJECT_EXISTING_FLARE_POSTERIOR_TO_TRUTHBLIND_GRID"
            and request.get("projection_policy") == POLICY
            and request.get("state_order") == list(STATES), "projection request differs")
    inputs = request.get("inputs", {})
    require(set(inputs) == {"predictions", "genetic_map", "overlay", "overlay_receipt"},
            "exact projection inputs required")
    paths = {name: resolve(item, name, staged) for name, item in inputs.items()}
    receipt = read_json(paths["overlay_receipt"])
    require(receipt.get("schema") == "r02_lai_truthblind_context_receipt_v2"
            and receipt.get("output_sha256") == inputs["overlay"]["sha256"]
            and receipt.get("truth_read_for_query_geometry") is False, "overlay receipt differs")
    with np.load(paths["overlay"], allow_pickle=False) as overlay:
        require({"person_id", "query_cm", "query_pos_bp", "query_chrom", "query_valid"} <= set(overlay.files),
                "overlay predictor axes absent")
        people = overlay["person_id"].astype(str)
        query_cm = np.asarray(overlay["query_cm"][0], dtype=np.float64)
        query_bp = np.asarray(overlay["query_pos_bp"][0], dtype=np.int64)
        query_chrom = np.asarray(overlay["query_chrom"])
        query_valid = np.asarray(overlay["query_valid"])
        require(np.all(query_chrom == query_chrom[0, 0]) and query_valid.all()
                and np.all(overlay["query_cm"] == query_cm)
                and np.all(overlay["query_pos_bp"] == query_bp), "overlay query axes differ by person")
    chromosome = str(int(query_chrom[0, 0]))
    require(request.get("case_id") and request.get("chromosome") == chromosome, "case/chromosome differs")
    samples, markers = scan_vcf(paths["predictions"])
    sample_keys = np.asarray([hashlib.sha256(x.encode()).hexdigest() for x in samples])
    require(np.array_equal(sample_keys, people), "FLARE and overlay person axes differ")
    limits = request.get("resources", {})
    require(set(limits) == {"max_markers", "max_people", "max_queries", "max_output_bytes",
                            "minimum_free_bytes", "query_chunk"}
            and all(type(x) is int and x > 0 for x in limits.values()), "positive resource bounds required")
    require(markers <= limits["max_markers"] and len(people) <= limits["max_people"]
            and len(query_bp) <= limits["max_queries"], "projection shape exceeds bound")
    # Source and cM-tie-group memmaps can each approach M×N×6×float32;
    # add the N×Q×6 output and fixed metadata headroom.
    projected = 48 * markers * len(people) + 24 * len(people) * len(query_bp) + 16 * 2**20
    out = Path(output_dir).resolve()
    require(not out.exists() and out.parent.is_dir() and projected <= limits["max_output_bytes"]
            and shutil.disk_usage(out.parent).free >= projected + limits["minimum_free_bytes"],
            "fresh output/disk admission failed")
    out.mkdir(mode=0o700)
    work = Path(tempfile.mkdtemp(prefix="flare-projection-", dir=out))
    try:
        map_bp, map_cm = genetic_map(paths["genetic_map"], chromosome, np)
        source, source_bp, source_cm, residual = load_vcf(paths["predictions"], chromosome, samples,
                                                          markers, work, map_bp, map_cm, np)
        output = out / "baseline-probabilities.npy"
        audit, mass = project(source, source_bp, source_cm, query_bp, query_cm, output,
                              limits["query_chunk"], np)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    require(output.stat().st_size <= limits["max_output_bytes"], "probability output exceeds byte bound")
    result = {
        "schema": "r02_lai_flare_query_projection_receipt_v1",
        "status": "EXISTING_FLARE_POSTERIOR_PROJECTED_NO_TRAINING_NO_SELECTION",
        "case_id": request["case_id"], "chromosome": chromosome,
        "initialization_label": request["initialization_label"], "marker_condition": request["marker_condition"],
        "projection_policy": POLICY, "state_order": list(STATES), "people": len(people),
        "source_markers": markers, "queries": len(query_bp), "projection_audit": audit,
        "maximum_source_haplotype_mass_residual": residual, "maximum_projected_mass_residual": mass,
        "truth_arrays_loaded": False, "truth_used_for_projection": False,
        "selection_performed": False, "source_prediction_sha256": inputs["predictions"]["sha256"],
        "request_sha256": digest, "genetic_map_sha256": inputs["genetic_map"]["sha256"],
        "view_kind": "INTERPOLATED_DERIVED_NEURAL_INPUT",
        "overlay_sha256": inputs["overlay"]["sha256"], "probabilities_sha256": sha(output),
        "probability_semantics": "UNORDERED_DIPLOID_PRODUCT_OF_FLARE_HAPLOTYPE_MARGINALS",
        "scientific_limit": "Interpolation is a derived neural-input view; it does not replace exact-marker FLARE evaluation.",
    }
    (out / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True); parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--staged-inputs")
    args = parser.parse_args(); os.umask(0o077)
    print(json.dumps(run(args.request, args.request_sha256, args.output_dir, args.staged_inputs), sort_keys=True))


if __name__ == "__main__":
    main()
