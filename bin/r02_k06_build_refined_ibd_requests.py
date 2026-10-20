#!/usr/bin/env python3
"""Build the twelve hash-pinned K06 Refined IBD requests after profile admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import r02_k06_refined_ibd_calls as calls


def run(report_path: Path, report_sha: str, jar_path: Path, jar_sha: str,
        spec_path: Path, spec_sha: str, output: Path, timeout_seconds: int,
        max_call_rows: int) -> dict:
    calls.require(calls.sha256(report_path) == report_sha, "Case report changed")
    calls.require(calls.sha256(jar_path) == jar_sha, "Detector changed")
    _, admitted = calls.read_parameters(spec_path, spec_sha)
    calls.require(admitted, "Only an ADMITTED profile can produce runnable requests")
    source = json.loads(report_path.read_text())
    cases = source.get("cases", [])
    calls.require(len(cases) == 12 and len({c["case_id"] for c in cases}) == 12,
                  "Exact twelve-case family required")
    calls.require(timeout_seconds > 0 and max_call_rows > 0, "Positive limits required")
    calls.require(not output.exists(), "Output exists; preserve prior requests")
    output.mkdir(parents=True)
    manifest = {"schema": "r02_k06_refined_ibd_request_manifest_v1", "requests": []}
    for case in sorted(cases, key=lambda value: (value["chromosome"] != "22", value["case_id"])):
        simulation = json.loads(Path(case["files"]["manifest"]["path"]).read_text())
        chromosome = case["chromosome"]
        request = {
            "schema": "r02_k06_refined_ibd_request_v1", "case_id": case["case_id"],
            "genome_build": "GRCh38", "coordinate_system": "1-based-inclusive",
            "chromosome": chromosome,
            "detector_jar": {"path": str(jar_path.resolve()), "sha256": jar_sha},
            "genotypes": {k: case["files"]["genotypes"][k] for k in ("path", "sha256")},
            "genetic_map": {k: case["files"]["genetic_map"][k] for k in ("path", "sha256")},
            "positive_territory": {k: case["files"]["positive_territory"][k] for k in ("path", "sha256")},
            "parameter_spec": {"path": str(spec_path.resolve()), "sha256": spec_sha},
            "expected_samples": case["simulated_individuals"],
            "expected_markers": simulation["summary"]["variants_by_chromosome"][chromosome],
            "max_call_rows": max_call_rows, "timeout_seconds": timeout_seconds,
            "java_executable": "java"
        }
        target = output / f"{case['case_id']}.request.json"
        target.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
        manifest["requests"].append({"case_id": case["case_id"],
                                     "path": str(target.resolve()),
                                     "sha256": calls.sha256(target)})
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"status": "RUNNABLE_REQUESTS_BUILT", "cases": 12,
            "manifest": str(manifest_path), "manifest_sha256": calls.sha256(manifest_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--jar", required=True, type=Path)
    parser.add_argument("--jar-sha256", required=True)
    parser.add_argument("--parameter-spec", required=True, type=Path)
    parser.add_argument("--parameter-spec-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-call-rows", type=int, default=10000000)
    args = parser.parse_args()
    print(json.dumps(run(args.report, args.report_sha256, args.jar, args.jar_sha256,
                         args.parameter_spec, args.parameter_spec_sha256, args.output,
                         args.timeout_seconds, args.max_call_rows), sort_keys=True))
