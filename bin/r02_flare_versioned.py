#!/usr/bin/env python3
"""Prepare and execute recoverable FLARE2 stages on the exact FLARE 0.6.0 inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r02_flare2_model_io as model_io  # noqa: E402
from r02_flare2_model_io import Flare2ModelError, parse_model, require, sha256_file  # noqa: E402
import m35_flare2_paired as m35  # noqa: E402


SCHEMA = "r02_flare_versioned_config_v1"
PLAN_SCHEMA = "r02_flare_versioned_plan_v1"
INPUTS = {
    "reference_vcf", "reference_index", "target_vcf", "target_index",
    "direct_panel_map", "fine_panel_map", "panel_to_ancestry", "genetic_map",
    "excludemarkers", "flare_jar", "official_builder", "seeded_builder", "roles",
}
COMMON_PARAMETERS = {"array", "min-maf", "min-mac", "gen", "seed", "nthreads"}
INTERNAL_PINS = {
    "panel-weight", "states", "ibs-step", "ibs-buffer", "ibs-haps", "ibs-recycle",
    "em-its", "em-haps", "delta-mu", "delta-p", "panel-markers", "panel-haps",
    "panel-ne",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def descriptor(path: Path, expected: str, label: str) -> dict[str, Any]:
    require(path.is_file(), f"invalid {label}")
    resolved = path.resolve(strict=True)
    require(resolved.is_file(), f"invalid resolved {label}")
    observed = sha256_file(resolved)
    require(observed == expected, f"{label} SHA-256 differs")
    return {"path": str(resolved), "sha256": observed, "bytes": resolved.stat().st_size}


def current_code_sha256() -> dict[str, str]:
    """Hash every local module whose behavior is reused after prepare."""
    return {
        "launcher": sha256_file(Path(__file__).resolve(strict=True)),
        "model_io": sha256_file(Path(model_io.__file__).resolve(strict=True)),
        "m35_runner": sha256_file(Path(m35.__file__).resolve(strict=True)),
        "m34_dependency": sha256_file(Path(m35.flare_common.__file__).resolve(strict=True)),
    }


def vcf_samples(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                require(len(fields) > 9 and fields[8] == "FORMAT", "VCF sample header differs")
                samples = fields[9:]
                require(samples and len(samples) == len(set(samples)),
                        "VCF sample identifiers are empty or duplicated")
                return samples
    raise Flare2ModelError("VCF lacks #CHROM sample header")


def two_column(path: Path, label: str) -> list[tuple[str, str]]:
    rows = [tuple(line.split()) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    require(rows and all(len(row) == 2 for row in rows), f"{label} must have two columns")
    first = [row[0] for row in rows]
    require(len(first) == len(set(first)), f"{label} first column is duplicated")
    return [(row[0], row[1]) for row in rows]


def role_axis(path: Path, target_samples: list[str], fit_role: str) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames is not None and {"sample_id", "role"} <= set(reader.fieldnames),
                "roles TSV requires sample_id and role columns")
        rows = list(reader)
    by_sample = {row["sample_id"]: row["role"] for row in rows}
    require(len(by_sample) == len(rows), "roles TSV has duplicate sample identifiers")
    require(all(sample in by_sample for sample in target_samples),
            "target sample is absent from roles TSV")
    observed = {by_sample[sample] for sample in target_samples}
    require(observed == {fit_role},
            "panel-probability model input must contain only the declared FIT role")
    return {"model_fit_role": fit_role, "target_roles": sorted(observed),
            "target_sample_count": len(target_samples)}


def map_audit(path: Path, chromosome: str) -> dict[str, Any]:
    rows: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        require(len(fields) >= 3, "genetic map row has fewer than three columns")
        if len(fields) >= 4:
            chrom, cm_text, bp_text = fields[0], fields[2], fields[3]
        else:
            chrom, bp_text, cm_text = chromosome, fields[0], fields[2]
        require(chrom.removeprefix("chr") == chromosome.removeprefix("chr"),
                "genetic map chromosome differs")
        bp, cm = int(bp_text), float(cm_text)
        require(bp >= 0 and math.isfinite(cm), "invalid genetic map coordinate")
        rows.append((bp, cm))
    require(len(rows) >= 2, "genetic map requires at least two points")
    require(all(b[0] > a[0] and b[1] >= a[1] for a, b in zip(rows, rows[1:])),
            "genetic map positions must increase and cM must not decrease")
    return {"points": len(rows), "first_bp": rows[0][0], "last_bp": rows[-1][0],
            "first_cm": rows[0][1], "last_cm": rows[-1][1], "input_unit": "cM",
            "hmm_distance_conversion": "d_m_Morgans=0.01*d_m_cM"}


def validate_config(config: Mapping[str, Any]) -> None:
    require(config.get("schema") == SCHEMA, "FLARE versioned config schema differs")
    require(config.get("method") == "FLARE2_CLUSTERED_PROCEDURE_ON_FLARE_0_6_0",
            "method must distinguish the FLARE2 procedure from the JAR version")
    require(config.get("flare_version") == "0.6.0", "only FLARE 0.6.0 is supported")
    require(str(config.get("chromosome")) in {str(value) for value in range(1, 23)},
            "one autosome 1-22 is required")
    ancestries = config.get("ancestry_order")
    require(isinstance(ancestries, list) and len(ancestries) > 1
            and len(ancestries) == len(set(ancestries)), "ancestry order differs")
    require(config.get("n_ancestries") == len(ancestries), "cluster count differs from ancestry order")
    require(config.get("model_fit_role") == "FIT", "cluster model must be learned from FIT only")
    require(config.get("evaluation_claim") == "FIT_ONLY_TRANSDUCTIVE_MODEL_BUILD_NOT_EVALUATION",
            "transductive FLARE2 fitting cannot be declared inductive evaluation")
    require(type(config.get("java_heap_mib")) is int and config["java_heap_mib"] > 0,
            "positive Java heap MiB is required")
    require(set(config.get("inputs", {})) == INPUTS, "FLARE input descriptors differ")
    for name, value in config["inputs"].items():
        require(isinstance(value, dict) and set(value) == {"sha256"}
                and len(value["sha256"]) == 64, f"invalid input descriptor: {name}")
    shared = config.get("shared_parameters", {})
    require(set(shared) == COMMON_PARAMETERS, "shared FLARE parameters differ")
    require(all(type(shared[name]) is bool for name in ("array",)), "array must be boolean")
    require(type(shared["min-mac"]) is int and shared["min-mac"] >= 0, "invalid min-mac")
    require(type(shared["min-maf"]) in {int, float} and 0 <= shared["min-maf"] < .5,
            "invalid min-maf")
    require(type(shared["gen"]) in {int, float} and shared["gen"] >= 1, "invalid gen")
    require(type(shared["seed"]) is int and type(shared["nthreads"]) is int
            and shared["nthreads"] >= 1, "invalid FLARE seed or threads")
    pins = config.get("flare_0_6_0_internal_pins", {})
    require(set(pins) == INTERNAL_PINS, "FLARE 0.6.0 internal pins differ")
    stages = config.get("stage_parameters", {})
    require(set(stages) == {"direct", "panel_probabilities", "clustered_final"},
            "FLARE stage parameters differ")
    expected = {
        "direct": (True, True, False, False),
        "panel_probabilities": (False, True, False, True),
        "clustered_final": (True, True, True, False),
    }
    for name, (probs, em, update_p, panel_probs) in expected.items():
        value = stages[name]
        require(set(value) == {"probs", "em", "update-p", "panel-probs"},
                f"{name} parameter members differ")
        require((value["probs"], value["em"], value["update-p"], value["panel-probs"]) ==
                (probs, em, update_p, panel_probs), f"{name} FLARE semantics differ")
    window = config.get("panel_probability_window", {})
    require(set(window) == {"panel_cm", "provenance", "claim"}
            and type(window["panel_cm"]) in {int, float} and window["panel_cm"] > 0,
            "panel-probability window differs")
    require(window["claim"] == "HISTORICAL_OR_PUBLISHED_SETTING_NOT_UNIVERSAL_OPTIMUM",
            "panel-cm must not be presented as a universal optimum")
    mapping = config.get("cluster_mapping", {})
    require(set(mapping) == {"source", "minimum_probability", "minimum_log_margin", "justification"}
            and mapping["source"] == "FIXED_REFERENCE_PANEL_MAP_ONLY"
            and str(mapping["justification"]).strip(), "cluster-mapping policy differs")
    require(0 < mapping["minimum_probability"] <= 1 and mapping["minimum_log_margin"] > 0,
            "cluster-mapping thresholds must be positive and predeclared")
    gmm = config.get("gmm", {})
    require(set(gmm) == {"seed", "n_init", "init_params", "center_normalization_tolerance"}
            and 0 <= gmm["seed"] < 2**32 and gmm["n_init"] == 10
            and gmm["init_params"] == "k-means++"
            and gmm["center_normalization_tolerance"] == 1e-8,
            "GMM seed or official clustering settings differ")


def command(jar: Path, reference: Path, target: Path, panel: Path, genetic_map: Path,
            excludemarkers: Path, prefix: Path, config: Mapping[str, Any], stage: str,
            model: Path | None = None) -> list[str]:
    shared, stage_p = config["shared_parameters"], config["stage_parameters"][stage]
    argv = ["java", f"-Xmx{config['java_heap_mib']}m", "-jar", str(jar),
            f"ref={reference}", f"ref-panel={panel}", f"gt={target}", f"map={genetic_map}",
            f"excludemarkers={excludemarkers}", f"out={prefix}"]
    for key in ("array", "min-maf", "min-mac", "gen", "seed", "nthreads"):
        value = shared[key]
        argv.append(f"{key}={bool_text(value) if type(value) is bool else value}")
    for key in ("probs", "em", "update-p", "panel-probs"):
        argv.append(f"{key}={bool_text(stage_p[key])}")
    argv.append(f"panel-cm={config['panel_probability_window']['panel_cm']}")
    for key in sorted(INTERNAL_PINS):
        argv.append(f"{key}={config['flare_0_6_0_internal_pins'][key]}")
    if model is not None:
        argv.append(f"model={model}")
    return argv


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    validate_config(config)
    supplied = {name: getattr(args, name) for name in INPUTS}
    observed = {name: descriptor(path, config["inputs"][name]["sha256"], name)
                for name, path in supplied.items()}
    reference_samples = vcf_samples(supplied["reference_vcf"])
    target_samples = vcf_samples(supplied["target_vcf"])
    require(set(reference_samples).isdisjoint(target_samples), "reference and target samples overlap")
    direct_rows = two_column(supplied["direct_panel_map"], "direct panel map")
    fine_rows = two_column(supplied["fine_panel_map"], "fine panel map")
    require([row[0] for row in direct_rows] == reference_samples
            and [row[0] for row in fine_rows] == reference_samples,
            "both reference panel maps must follow the exact reference VCF sample axis")
    require(set(row[1] for row in direct_rows) == set(config["ancestry_order"]),
            "direct panel map must use the declared ancestry labels")
    panel_macro = dict(two_column(supplied["panel_to_ancestry"], "panel-to-ancestry map"))
    require(set(row[1] for row in fine_rows) == set(panel_macro),
            "fine panels differ from the fixed panel-to-ancestry map")
    require(set(panel_macro.values()) == set(config["ancestry_order"]),
            "fixed panel-to-ancestry map does not cover ancestries")
    direct_by_sample = dict(direct_rows)
    fine_by_sample = dict(fine_rows)
    require(all(panel_macro[fine_by_sample[sample]] == direct_by_sample[sample]
                for sample in reference_samples),
            "fine-panel ancestry differs from direct ancestry for a reference sample")
    roles = role_axis(supplied["roles"], target_samples, config["model_fit_role"])
    code_sha256 = current_code_sha256()
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "PREPARED_NOT_EXECUTED",
        "method": config["method"],
        "flare_version": config["flare_version"],
        "chromosome": config["chromosome"],
        "ancestry_order": config["ancestry_order"],
        "n_ancestries": config["n_ancestries"],
        "config": config,
        "config_sha256": sha256_file(args.config),
        "code_sha256": code_sha256,
        "reused_m35_code": {
            "path": str(Path(m35.__file__).resolve()),
            "sha256": code_sha256["m35_runner"],
            "functions": ["cluster_assignment_evidence_from_model", "relabel_flare2_vcf"],
            "m34_dependency_sha256": code_sha256["m34_dependency"],
        },
        "inputs": observed,
        "reference_sample_count": len(reference_samples),
        "target": roles,
        "genetic_map": map_audit(supplied["genetic_map"], str(config["chromosome"])),
        "panel_to_ancestry": panel_macro,
        "label_policy": "REFERENCE_P_MATRIX_ONLY_TARGET_TRUTH_FORBIDDEN",
        "evaluation_claim": config["evaluation_claim"],
        "same_inputs_for_direct_and_clustered": True,
        "scientific_execution_performed": False,
    }
    write_json(args.output, plan)
    return plan


def load_plan(path: Path) -> dict[str, Any]:
    plan = read_json(path)
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("status") == "PREPARED_NOT_EXECUTED",
            "prepared FLARE plan differs")
    validate_config(plan["config"])
    require(plan.get("code_sha256") == current_code_sha256(),
            "launcher or reused FLARE2/M35/M34 code differs from prepare")
    return plan


def verify_stage_inputs(plan: Mapping[str, Any], args: argparse.Namespace, names: set[str]) -> None:
    for name in names:
        descriptor(getattr(args, name), plan["inputs"][name]["sha256"], name)


def run_flare_stage(args: argparse.Namespace, stage: str, panel_name: str,
                    output_suffix: str, model: Path | None = None,
                    write_receipt: bool = True) -> dict[str, Any]:
    plan = load_plan(args.plan)
    needed = {"flare_jar", "reference_vcf", "reference_index", "target_vcf",
              "target_index", panel_name,
              "genetic_map", "excludemarkers"}
    verify_stage_inputs(plan, args, needed)
    argv = command(args.flare_jar, args.reference_vcf, args.target_vcf, getattr(args, panel_name),
                   args.genetic_map, args.excludemarkers, args.output_prefix,
                   plan["config"], stage, model)
    subprocess.run(argv, check=True)
    output = Path(str(args.output_prefix) + output_suffix)
    require(output.is_file(), f"FLARE did not create {output.name}")
    receipt = {
        "schema": f"r02_flare_{stage}_receipt_v1",
        "status": "PASS_COMPLETED",
        "plan_sha256": sha256_file(args.plan),
        "command_argv": argv,
        "inputs_sha256": {name: plan["inputs"][name]["sha256"] for name in sorted(needed)},
        "output": {"path": output.name, "sha256": sha256_file(output), "bytes": output.stat().st_size},
        "model_sha256": sha256_file(model) if model is not None else None,
        "roles": plan["target"],
        "evaluation_claim": plan["evaluation_claim"],
        "target_truth_accessed": False,
    }
    if write_receipt:
        write_json(args.receipt, receipt)
    return receipt


def build_model(args: argparse.Namespace) -> dict[str, Any]:
    plan = load_plan(args.plan)
    verify_stage_inputs(plan, args, {"official_builder", "seeded_builder"})
    panel_receipt = read_json(args.panel_receipt)
    require(panel_receipt.get("schema") == "r02_flare_panel_probabilities_receipt_v1"
            and panel_receipt.get("status") == "PASS_COMPLETED"
            and panel_receipt.get("plan_sha256") == sha256_file(args.plan),
            "panel-probability receipt differs")
    require(sha256_file(args.panels) == panel_receipt["output"]["sha256"],
            "panel-probability file differs from receipt")
    builder_receipt = args.output_prefix.parent / "flare2.seeded_builder.receipt.json"
    command_argv = [sys.executable, str(args.seeded_builder),
                    "--seed", str(plan["config"]["gmm"]["seed"]),
                    "--builder", str(args.official_builder),
                    "--builder-sha256", plan["inputs"]["official_builder"]["sha256"],
                    "--receipt", str(builder_receipt), str(plan["n_ancestries"]),
                    str(args.panels), str(args.output_prefix)]
    subprocess.run(command_argv, check=True)
    model_path = Path(str(args.output_prefix) + ".model")
    parsed = parse_model(model_path, plan["n_ancestries"])
    policy = plan["config"]["cluster_mapping"]
    mapping = m35.cluster_assignment_evidence_from_model(
        model_path, plan["ancestry_order"], plan["panel_to_ancestry"],
        policy["minimum_probability"], policy["minimum_log_margin"],
    )
    write_json(args.mapping, mapping)
    receipt = {
        "schema": "r02_flare_clustered_model_receipt_v1",
        "status": mapping["status"],
        "plan_sha256": sha256_file(args.plan),
        "panel_receipt_sha256": sha256_file(args.panel_receipt),
        "command_argv": command_argv,
        "panels_sha256": sha256_file(args.panels),
        "model_sha256": parsed["model_sha256"],
        "model_log_sha256": sha256_file(Path(str(args.output_prefix) + ".log")),
        "seeded_builder_receipt_sha256": sha256_file(builder_receipt),
        "mapping_sha256": sha256_file(args.mapping),
        "p_row_sums": parsed["p_row_sums"],
        "target_truth_accessed": False,
        "mapping_implementation": "REUSED_M35_TRUTH_BLIND_REFERENCE_P_MATRIX_GATE",
        "final_inference_enabled": mapping["status"] == "PASS_UNAMBIGUOUS_TRUTH_BLIND_CLUSTER_ASSIGNMENT",
    }
    write_json(args.receipt, receipt)
    require(receipt["final_inference_enabled"],
            "cluster labels are not identifiable from fixed reference panels; final inference stopped")
    return receipt


def infer(args: argparse.Namespace) -> dict[str, Any]:
    plan = load_plan(args.plan)
    model_receipt = read_json(args.model_receipt)
    mapping = read_json(args.mapping)
    require(model_receipt.get("status") == "PASS_UNAMBIGUOUS_TRUTH_BLIND_CLUSTER_ASSIGNMENT"
            and model_receipt.get("plan_sha256") == sha256_file(args.plan),
            "clustered-model receipt does not enable final inference")
    require(mapping.get("status") == "PASS_UNAMBIGUOUS_TRUTH_BLIND_CLUSTER_ASSIGNMENT"
            and sha256_file(args.mapping) == model_receipt["mapping_sha256"],
            "fixed reference mapping differs")
    require(sha256_file(args.model) == model_receipt["model_sha256"], "clustered model differs")
    parsed_initial = parse_model(args.model, plan["n_ancestries"])
    raw_receipt = run_flare_stage(args, "clustered_final", "fine_panel_map",
                                  ".anc.vcf.gz", args.model, write_receipt=False)
    raw_path = Path(str(args.output_prefix) + ".anc.vcf.gz")
    raw_final_model = Path(str(args.output_prefix) + ".model")
    raw_final_log = Path(str(args.output_prefix) + ".log")
    require(raw_final_model.is_file() and raw_final_log.is_file(),
            "FLARE EM did not create its final model and log")
    parsed_final = parse_model(raw_final_model, plan["n_ancestries"])
    require(parsed_final["clusters"] == parsed_initial["clusters"],
            "post-EM model changed the ordered cluster identifiers")
    policy = plan["config"]["cluster_mapping"]
    final_mapping = m35.cluster_assignment_evidence_from_model(
        raw_final_model, plan["ancestry_order"], plan["panel_to_ancestry"],
        policy["minimum_probability"], policy["minimum_log_margin"],
    )
    write_json(args.final_mapping, final_mapping)
    require(final_mapping["status"] == "PASS_UNAMBIGUOUS_TRUTH_BLIND_CLUSTER_ASSIGNMENT",
            "post-EM clusters collapsed or are no longer identifiable from reference panels")
    require(final_mapping["cluster_to_ancestry"] == mapping["cluster_to_ancestry"],
            "post-EM cluster labels drifted relative to the truth-blind initial mapping")
    require(not args.final_model.exists() and not args.final_log.exists(),
            "refusing to overwrite final FLARE model or log")
    shutil.copy2(raw_final_model, args.final_model)
    shutil.copy2(raw_final_log, args.final_log)
    m35.relabel_flare2_vcf(raw_path, args.canonical_output, mapping, plan["ancestry_order"])
    raw_receipt["status"] = "PASS_COMPLETED_AND_REFERENCE_CANONICALIZED"
    raw_receipt["raw_output_sha256"] = raw_receipt["output"]["sha256"]
    raw_receipt["output"] = {"path": args.canonical_output.name,
                             "sha256": sha256_file(args.canonical_output),
                             "bytes": args.canonical_output.stat().st_size}
    raw_receipt["mapping_sha256"] = sha256_file(args.mapping)
    raw_receipt["post_em"] = {
        "model": {"path": args.final_model.name, "sha256": parsed_final["model_sha256"],
                  "bytes": args.final_model.stat().st_size},
        "log": {"path": args.final_log.name, "sha256": sha256_file(args.final_log),
                "bytes": args.final_log.stat().st_size},
        "mapping_sha256": sha256_file(args.final_mapping),
        "mapping_status": final_mapping["status"],
        "cluster_order_consistent_with_initial": True,
        "cluster_identifiers_consistent_with_initial": True,
        "mapping_uses_target_truth": False,
    }
    write_json(args.receipt, raw_receipt)
    return raw_receipt


def add_files(parser: argparse.ArgumentParser, names: set[str]) -> None:
    for name in sorted(names):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_p = sub.add_parser("prepare")
    prepare_p.add_argument("--config", type=Path, required=True)
    add_files(prepare_p, INPUTS)
    prepare_p.add_argument("--output", type=Path, required=True)

    direct_p = sub.add_parser("direct")
    direct_p.add_argument("--plan", type=Path, required=True)
    add_files(direct_p, {"flare_jar", "reference_vcf", "reference_index", "target_vcf",
                         "target_index", "direct_panel_map", "genetic_map", "excludemarkers"})
    direct_p.add_argument("--output-prefix", type=Path, required=True)
    direct_p.add_argument("--receipt", type=Path, required=True)

    panel_p = sub.add_parser("panel-probabilities")
    panel_p.add_argument("--plan", type=Path, required=True)
    add_files(panel_p, {"flare_jar", "reference_vcf", "reference_index", "target_vcf",
                        "target_index", "fine_panel_map", "genetic_map", "excludemarkers"})
    panel_p.add_argument("--output-prefix", type=Path, required=True)
    panel_p.add_argument("--receipt", type=Path, required=True)

    model_p = sub.add_parser("build-model")
    model_p.add_argument("--plan", type=Path, required=True)
    model_p.add_argument("--panels", type=Path, required=True)
    model_p.add_argument("--panel-receipt", type=Path, required=True)
    add_files(model_p, {"official_builder", "seeded_builder"})
    model_p.add_argument("--output-prefix", type=Path, required=True)
    model_p.add_argument("--mapping", type=Path, required=True)
    model_p.add_argument("--receipt", type=Path, required=True)

    infer_p = sub.add_parser("infer")
    infer_p.add_argument("--plan", type=Path, required=True)
    infer_p.add_argument("--model", type=Path, required=True)
    infer_p.add_argument("--model-receipt", type=Path, required=True)
    infer_p.add_argument("--mapping", type=Path, required=True)
    add_files(infer_p, {"flare_jar", "reference_vcf", "reference_index", "target_vcf",
                        "target_index", "fine_panel_map", "genetic_map", "excludemarkers"})
    infer_p.add_argument("--output-prefix", type=Path, required=True)
    infer_p.add_argument("--canonical-output", type=Path, required=True)
    infer_p.add_argument("--final-model", type=Path, required=True)
    infer_p.add_argument("--final-log", type=Path, required=True)
    infer_p.add_argument("--final-mapping", type=Path, required=True)
    infer_p.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        result = prepare(args)
    elif args.action == "direct":
        result = run_flare_stage(args, "direct", "direct_panel_map", ".anc.vcf.gz")
    elif args.action == "panel-probabilities":
        result = run_flare_stage(args, "panel_probabilities", "fine_panel_map", ".panels")
    elif args.action == "build-model":
        result = build_model(args)
    else:
        result = infer(args)
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
