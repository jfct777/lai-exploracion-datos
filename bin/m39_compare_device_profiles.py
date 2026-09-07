"""Compare authenticated device profiles without reopening genomic arrays.

The ratio compares the two measured execution environments, not GPU silicon in
isolation. Full-pass durations are stratified projections, never completed epochs
or estimates of time to convergence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(path, device):
    report = load(path)
    require(report["decision"] == "PASS_ORDERED_THROUGHPUT_TECHNICAL_ONLY", "profile did not pass")
    require(report["runtime"]["device"] == device, "unexpected device")
    require(report["scope"]["biological_training"] is False, "not a technical profile")
    require(report["scope"]["truth_opened"] is False, "truth opened during profiling")
    sampling_path = path.with_name("sampling-manifest.json")
    require(digest(sampling_path) == report["sampling_manifest_sha256"], "sampling bytes changed")
    sampling = load(sampling_path)
    rows = report["steps"]
    require(len([r for r in rows if r["warmup"]]) == 1, "one excluded warmup required")
    files = list((path.parent / "steps").glob("step-*.json"))
    require(len(files) == len(rows), "primary step inventory differs")
    for row in rows:
        require(load(path.parent / "steps" / f"step-{row['step']:04d}.json") == row,
                "step disagrees with aggregate")
    measured = [row for row in rows if not row["warmup"]]
    batch = report["case"]["batch_size"]
    require(len(measured) == 32 // batch, "incomplete measurement")
    pairs = [tuple(pair) for row in measured for pair in row["pairs"]]
    expected = [tuple(pair) for a in sampling["anchor_samples"] for pair in a["pairs"]]
    require(sorted(pairs) == sorted(expected) and len(set(pairs)) == 32, "sample differs")
    times = [row["end_to_end_seconds"] for row in measured]
    require(all(math.isfinite(t) and t > 0 for t in times), "invalid step duration")
    total = sum(times)
    require(math.isclose(total, report["summary"]["measured_end_to_end_seconds"], rel_tol=1e-12),
            "aggregate elapsed time differs")
    estimate = None
    if report["case"]["policy"] == "grouped":
        terms = []
        for h, stratum in enumerate(sampling["strata"]):
            values = [r["end_to_end_seconds"] for r in measured if r["stratum"] == h]
            require(len(values) == 8 // batch, "stratum incomplete")
            terms.append(stratum["pair_count"] / batch * statistics.mean(values))
        estimate = sum(terms)
        require(math.isclose(estimate, report["summary"]["grouped_fullpass_estimate"]["seconds"],
                             rel_tol=1e-12), "weighted projection differs")
    else:
        require(report["summary"]["grouped_fullpass_estimate"] is None, "random replay is not a pass")
    peak = None
    if device == "cuda:0":
        parity = load(path.with_name("device-parity.json"))
        require(parity == report["device_parity"], "parity receipt differs")
        require(parity["decision"] == "PASS_CPU_DEVICE_WARMUP_GRADIENT_PARITY", "parity failed")
        require(parity["relative_tolerance_reference"] == "cpu", "wrong numerical reference")
        require(bool(parity["max_tolerance_ratio_by_tensor"]), "missing parity comparisons")
        require(all(math.isfinite(r) and 0 <= r <= 1
                    for r in parity["max_tolerance_ratio_by_tensor"].values()), "parity tolerance failed")
        require(report["runtime"]["torch"] == "2.12.1+cu126", "unexpected CUDA torch")
        peak = report["resources"]["device_memory"]["peak_reserved_bytes"]
        require(0 < peak <= 8 * 1024**3, "GPU allocation limit exceeded")
    return report, {"sample_seconds": total, "pass_hours": None if estimate is None else estimate / 3600,
                    "gpu_peak_reserved_gib": None if peak is None else peak / 1024**3,
                    "profile_sha256": digest(path)}


def compare(cpu_root, gpu_root):
    indexed = []
    for root, device in ((cpu_root, "cpu"), (gpu_root, "cuda:0")):
        cases = {}
        for path in root.rglob("profile.json"):
            report, result = inspect(path, device)
            name = report["case"]["id"]
            require(name not in cases, "duplicate case report")
            cases[name] = (report, result)
        require(len(cases) == 6, "six complete case reports required")
        indexed.append(cases)
    cpu, gpu = indexed
    require(cpu.keys() == gpu.keys(), "case inventories differ")
    rows = []
    for name in sorted(cpu):
        c, cr = cpu[name]
        g, gr = gpu[name]
        for key in ("case", "architecture", "parameter_count", "store_shape", "sampling_manifest_sha256"):
            require(c[key] == g[key], f"CPU/GPU {key} differs")
        for key in ("parent_sha256", "manifest_sha256", "folds_sha256"):
            require(c["provenance"][key] == g["provenance"][key], f"CPU/GPU source {key} differs")
        require(c["provenance"]["source_sha256"]["m39_ordered_models.py"] ==
                g["provenance"]["source_sha256"]["m39_ordered_models.py"], "model changed")
        rows.append({"case": name, "cpu_sample_seconds": cr["sample_seconds"],
                     "gpu_sample_seconds": gr["sample_seconds"],
                     "observed_cpu_gpu_time_ratio": cr["sample_seconds"] / gr["sample_seconds"],
                     "cpu_projected_pass_hours": cr["pass_hours"],
                     "gpu_projected_pass_hours": gr["pass_hours"],
                     "gpu_peak_reserved_gib": gr["gpu_peak_reserved_gib"],
                     "cpu_profile_sha256": cr["profile_sha256"],
                     "gpu_profile_sha256": gr["profile_sha256"]})
    return {"decision": "PASS_SIX_DEVICE_PROFILE_COMPARISONS_TECHNICAL_ONLY", "rows": rows,
            "limits": ["synthetic labels; no LAI efficacy", "different host CPUs and CPU concurrency",
                       "no convergence estimate", "no random-policy full-pass estimate",
                       "GPU optimizer uses explicit foreach=False/fused=False",
                       "setup, cloud provisioning and receipts excluded from projected passes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-root", type=Path, required=True)
    parser.add_argument("--gpu-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.cpu_root, args.gpu_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "comparison.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({"decision": result["decision"], "comparisons": len(result["rows"])}))


if __name__ == "__main__":
    main()
