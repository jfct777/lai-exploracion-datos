#!/usr/bin/env python3
"""Create an immutable M29 binding for authenticated root-specific B0 output."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-seed", type=int, required=True, choices=[20260817, 20260818])
    parser.add_argument("--fb", type=Path, required=True)
    parser.add_argument("--msp", type=Path, required=True)
    parser.add_argument("--inference-report", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--training-rss-gate", type=Path, required=True)
    parser.add_argument("--run-provenance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.fb, args.msp, args.inference_report, args.inference_manifest, args.runtime_contract, args.training_rss_gate, args.run_provenance):
        if not path.is_file():
            raise SystemExit(f"missing required input: {path}")
    runtime = json.loads(args.runtime_contract.read_text(encoding="utf-8"))
    binding = runtime.get("m29_binding", {})
    label = "root17" if args.root_seed == 20260817 else "root18"
    if binding.get("stage") != "M29_ROOT_GNOMIX_B0" or binding.get("root_label") != label or binding.get("root_seed") != args.root_seed:
        raise SystemExit("runtime contract belongs to another root")
    resources = runtime.get("resources", {})
    if resources.get("memory_per_training") != "8 GB" or float(resources.get("peak_rss_stop_gib", -1)) != 6.4:
        raise SystemExit("runtime contract has unexpected resource limits")
    report = json.loads(args.inference_report.read_text(encoding="utf-8"))
    if report.get("decision") != "GO_REPLICATE_COMPARISON_NO_TRUTH" or report.get("replicate") != label or report.get("contract_sha256") != sha256(args.runtime_contract):
        raise SystemExit("inference report failed or belongs to another contract/root")
    observed = {"fb": sha256(args.fb), "msp": sha256(args.msp)}
    if report.get("output_sha256", {}).get(args.fb.name) != observed["fb"] or report.get("output_sha256", {}).get(args.msp.name) != observed["msp"]:
        raise SystemExit("inference report does not authenticate FB/MSP")
    manifest = json.loads(args.inference_manifest.read_text(encoding="utf-8"))
    if manifest.get("stage") != f"M29_ROOT_GNOMIX_INFER_{label}" or manifest.get("sha256", {}).get(args.fb.name) != observed["fb"] or manifest.get("sha256", {}).get(args.msp.name) != observed["msp"]:
        raise SystemExit("inference manifest does not authenticate FB/MSP")
    rss = json.loads(args.training_rss_gate.read_text(encoding="utf-8"))
    if rss.get("stage") != "M29_PROCESS_TREE_RSS_GATE" or rss.get("decision") != "PASS_RSS_GATE" or rss.get("threshold_exceeded") is not False:
        raise SystemExit("training RSS gate did not pass")
    if float(rss.get("max_rss_gib", float("inf"))) != 6.4 or float(rss.get("peak_rss_gib", float("inf"))) > 6.4:
        raise SystemExit("training RSS gate differs from the frozen 6.4 GiB limit")
    if manifest.get("inputs", {}).get(args.training_rss_gate.name) != sha256(args.training_rss_gate):
        raise SystemExit("inference manifest does not authenticate the training RSS gate")
    provenance = json.loads(args.run_provenance.read_text(encoding="utf-8"))
    if provenance.get("scientific_scope") != "M29 root-specific B0 training and inference; no truth or effect estimation":
        raise SystemExit("run provenance has unexpected scope")
    payload = {
        "stage": "M29_AUTHENTICATED_B0_BINDING",
        "root_seed": args.root_seed,
        "sha256": observed,
        "runtime_contract_sha256": sha256(args.runtime_contract),
        "inference_report_sha256": sha256(args.inference_report),
        "inference_manifest_sha256": sha256(args.inference_manifest),
        "training_rss_gate_sha256": sha256(args.training_rss_gate),
        "run_provenance_sha256": sha256(args.run_provenance),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
