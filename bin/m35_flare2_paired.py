#!/usr/bin/env python3
"""Prepare and run a paired direct-FLARE versus FLARE2 baseline on one marker spine."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m34_run_flare as flare_common


METHOD_KEYS = {"array", "probs", "em", "min-mac", "min-maf", "gen",
               "update-p", "panel-probs", "seed", "nthreads"}
INPUT_NAMES = ("reference_vcf", "reference_tbi", "target_vcf", "target_tbi",
               "sample_map", "panel_macro_map", "genetic_map", "flare_jar", "flare2_model_builder",
               "flare2_upstream_model_builder")


class PairedContractError(ValueError):
    """Raised when a paired comparison no longer has identical scientific inputs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PairedContractError(message)


def sha256_file(path: Path) -> str:
    return flare_common.sha256_file(path)


def _axis_digest(rows: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def marker_axis_sha256(loci: list[tuple[str, int, str, str]]) -> str:
    """Hash only the canonical CHROM/POS/REF/ALT marker axis, never VCF bytes."""
    return _axis_digest([f"{chrom}\t{position}\t{ref}\t{alt}" for chrom, position, ref, alt in loci])


def normalize_panel_maps(sample_map: Path, panel_macro_map: Path, output: Path,
                         reference_samples: list[str], ancestry_order: list[str]) -> dict[str, Any]:
    """Validate explicit fine/coarse FLARE panels and their macro-ancestry map."""
    sample_rows = [line.split() for line in sample_map.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
    require(all(len(row) == 2 for row in sample_rows), "M35 reference panel rows must be sample and panel")
    sample_to_panel = {sample: panel for sample, panel in sample_rows}
    require(len(sample_to_panel) == len(sample_rows), "M35 reference panel map has duplicate samples")
    require(set(sample_to_panel) == set(reference_samples),
            "M35 reference panel samples differ from REF")
    # FLARE requires the panel map to follow the VCF sample axis.  The blind
    # M27F role table is ordered by its own deterministic split, so normalize
    # that authenticated mapping here instead of treating row order as biology.
    sample_rows = [[sample, sample_to_panel[sample]] for sample in reference_samples]
    panel_rows = [line.split() for line in panel_macro_map.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    require(all(len(row) == 2 for row in panel_rows), "M35 panel-macro map rows must be panel and ancestry")
    panel_to_ancestry = {panel: ancestry for panel, ancestry in panel_rows}
    require(len(panel_to_ancestry) == len(panel_rows), "M35 panel-macro map has duplicate panels")
    require(set(panel_to_ancestry) == {row[1] for row in sample_rows},
            "M35 panel-macro map and reference panel labels differ")
    require(set(panel_to_ancestry.values()) == set(ancestry_order),
            "M35 panel-macro map must cover AFR/EUR/NAM exactly")
    panel_counts: dict[str, int] = {}
    for _sample, panel in sample_rows:
        panel_counts[panel] = panel_counts.get(panel, 0) + 1
    macro_counts = {ancestry: sum(panel_counts[panel] for panel, macro in panel_to_ancestry.items()
                                  if macro == ancestry) for ancestry in ancestry_order}
    require(all(count > 0 for count in panel_counts.values()), "M35 panel is empty")
    mode = ("COARSE_MACROANCESTRY_EXPLICIT" if
            set(panel_to_ancestry) == set(ancestry_order) and
            all(panel_to_ancestry[name] == name for name in ancestry_order)
            else "FINE_POPULATION_OR_SUBPOPULATION")
    output.write_text("".join(f"{sample}\t{panel}\n" for sample, panel in sample_rows), encoding="utf-8")
    return {"mode": mode, "sample_count": len(sample_rows), "panel_count": len(panel_counts),
            "panel_counts": dict(sorted(panel_counts.items())), "macro_counts": macro_counts,
            "panel_to_ancestry": dict(sorted(panel_to_ancestry.items())),
            "sample_map_sha256": sha256_file(output), "panel_macro_map_sha256": sha256_file(panel_macro_map)}


def sample_phase_axes(path: Path, chromosome: str) -> dict[str, str]:
    """Return independent sample-order and physical-haplotype-axis hashes.

    ``scan_vcf`` validates phasing.  This second streaming pass records the
    ordered GT fields so that a rephasing cannot masquerade as the same target.
    """
    samples: list[str] | None = None
    phase_rows: list[str] = []
    expected_chromosome = chromosome.removeprefix("chr")
    with flare_common.open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                require(samples is None and len(fields) > 9 and fields[8] == "FORMAT",
                        "M35 VCF sample axis differs")
                samples = fields[9:]
                require(all(samples) and len(samples) == len(set(samples)),
                        "M35 VCF sample axis is empty or duplicated")
                continue
            if line.startswith("#") or not line.strip():
                continue
            require(samples is not None, "M35 VCF record precedes sample axis")
            fields = line.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(samples), f"M35 malformed VCF row {line_number}")
            require(fields[0].removeprefix("chr") == expected_chromosome,
                    "M35 VCF chromosome differs")
            format_fields = fields[8].split(":")
            require("GT" in format_fields, "M35 VCF lacks GT")
            gt_index = format_fields.index("GT")
            genotypes = []
            for value in fields[9:]:
                parts = value.split(":")
                require(len(parts) > gt_index and "|" in parts[gt_index] and "/" not in parts[gt_index],
                        "M35 VCF has unphased GT")
                genotypes.append(parts[gt_index])
            phase_rows.append("\t".join([
                fields[0].removeprefix("chr"), fields[1], fields[3].upper(), fields[4].upper(), *genotypes
            ]))
    require(samples is not None, "M35 VCF lacks sample axis")
    return {
        "sample_axis_sha256": _axis_digest(samples),
        "phase_axis_sha256": _axis_digest([*samples, *phase_rows]),
    }


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == "1.0.0", "M35 contract schema differs")
    require(payload.get("experiment_id") == "M35_FLARE2_PAIRED_CHR22",
            "M35 contract identity differs")
    require(payload.get("status") == "PLAN_ONLY_PRECHECK_DEFAULT",
            "M35 contract status differs")
    require(str(payload.get("chromosome", "")).removeprefix("chr") == "22",
            "M35 is restricted to chromosome 22")
    require(payload.get("ancestry_order") == ["AFR", "EUR", "NAM"],
            "M35 ancestry order differs")
    panel_policy = payload.get("reference_panel_policy", {})
    require(panel_policy.get("fine_mode") == "population_or_subpopulation_panels_with_explicit_panel_to_macro_map" and
            panel_policy.get("coarse_mode") == "macroancestry_panels_allowed_only_when_manifest_labels_COARSE_MACROANCESTRY_EXPLICIT" and
            panel_policy.get("cluster_mapping") == "aggregate_model_p_matrix_over_all_panels_of_each_macroancestry_then_assign_one_to_one",
            "M35 reference-panel policy differs")
    methods = payload.get("methods", {})
    direct = methods.get("flare_0_6", {}).get("parameters")
    flare2 = methods.get("flare2", {})
    stages = [flare2.get("panel_probability_parameters"), flare2.get("final_parameters")]
    require(all(isinstance(stage, dict) and set(stage) == METHOD_KEYS
                for stage in [direct, *stages]), "M35 method parameters differ")
    require(direct["panel-probs"] is False and direct["update-p"] is False,
            "direct FLARE parameters differ")
    require(stages[0]["panel-probs"] is True and stages[0]["probs"] is False,
            "FLARE2 panel-probability parameters differ")
    require(stages[1]["panel-probs"] is False and stages[1]["probs"] is True and
            stages[1]["update-p"] is True, "FLARE2 final parameters differ")
    cluster_assignment = flare2.get("cluster_assignment")
    require(isinstance(cluster_assignment, dict) and
            cluster_assignment.get("min_probability") == 0.5 and
            cluster_assignment.get("min_log_margin") == 0.25,
            "M35 cluster-assignment gate differs")
    for name in ("array", "min-mac", "min-maf", "gen", "seed", "nthreads"):
        require(direct[name] == stages[0][name] == stages[1][name],
                f"paired parameter differs: {name}")
    return payload


def build_command(jar: Path, reference_vcf: Path, target_vcf: Path, panel: Path,
                  genetic_map: Path, output_prefix: Path,
                  parameters: Mapping[str, Any], model: Path | None = None) -> list[str]:
    command = flare_common.build_command(
        "java", jar, reference_vcf, target_vcf, panel, genetic_map, output_prefix, parameters,
    )
    if model is not None:
        command.append(f"model={model}")
    return command


def cluster_assignment_evidence_from_model(model_path: Path, ancestry_order: list[str],
                                           panel_to_ancestry: Mapping[str, str],
                                           minimum_probability: float,
                                           minimum_log_margin: float) -> dict[str, Any]:
    """Audit an unlabeled FLARE2 cluster map, including a terminal NO_GO state.

    This function deliberately does not raise for a failed scientific gate: a
    valid, truth-blind model matrix that fails support/margin is evidence for
    stopping before final FLARE2 inference, not a crashed computation.
    """
    lines = model_path.read_text(encoding="utf-8").splitlines()
    try:
        panel_index = lines.index("# list of reference panels") + 1
        panels = lines[panel_index].split()
        p_index = lines.index("# p[i][j]: probability that a model state haplotype is in reference panel j") + 2
    except ValueError as exc:
        raise PairedContractError("FLARE2 model lacks a readable panel-mixture matrix") from exc
    require(set(panels) == set(panel_to_ancestry) and len(panels) == len(panel_to_ancestry),
            "FLARE2 model panels differ from the authenticated panel-macro map")
    matrix: list[list[float]] = []
    for row in lines[p_index:p_index + len(ancestry_order)]:
        values = [float(value) for value in row.split()]
        require(len(values) == len(panels) and all(math.isfinite(value) and value >= 0 for value in values),
                "FLARE2 model panel-mixture row differs")
        require(abs(sum(values) - 1.0) <= 1e-6, "FLARE2 model panel-mixture mass differs")
        matrix.append(values)
    require(len(matrix) == len(ancestry_order), "FLARE2 model cluster count differs")
    panel_column = {panel: index for index, panel in enumerate(panels)}
    aggregate = {
        ancestry: [panel_column[panel] for panel, macro in panel_to_ancestry.items()
                   if macro == ancestry]
        for ancestry in ancestry_order
    }
    require(all(aggregate[ancestry] for ancestry in ancestry_order),
            "FLARE2 panel-macro map leaves a macro-ancestry without panels")
    scored: list[tuple[float, tuple[str, ...]]] = []
    for assignment in itertools.permutations(ancestry_order):
        values = [sum(matrix[cluster][column] for column in aggregate[ancestry])
                  for cluster, ancestry in enumerate(assignment)]
        score = sum(math.log(value) if value > 0 else float("-inf") for value in values)
        scored.append((score, assignment))
    scored.sort(key=lambda row: row[0], reverse=True)
    best_score, best_assignment = scored[0]
    runner_up = scored[1][0]
    selected = [sum(matrix[cluster][column] for column in aggregate[ancestry])
                for cluster, ancestry in enumerate(best_assignment)]
    macro_matrix = {
        str(cluster): {ancestry: sum(matrix[cluster][column] for column in aggregate[ancestry])
                       for ancestry in ancestry_order}
        for cluster in range(len(matrix))
    }
    failure_reasons = []
    if not math.isfinite(best_score):
        failure_reasons.append("nonfinite_assignment_score")
    if min(selected) < minimum_probability:
        failure_reasons.append("insufficient_panel_support")
    if best_score - runner_up < minimum_log_margin:
        failure_reasons.append("ambiguous_assignment_margin")
    status = ("PASS_UNAMBIGUOUS_TRUTH_BLIND_CLUSTER_ASSIGNMENT" if not failure_reasons else
              "NO_GO_TRUTH_BLIND_CLUSTER_ASSIGNMENT")
    return {
        "schema_version": "1.0.0",
        "status": status,
        "source": "FLARE2_model_p_matrix_aggregated_over_declared_reference_panels",
        "model_sha256": sha256_file(model_path),
        "panels": panels,
        "panel_to_ancestry": dict(panel_to_ancestry),
        "cluster_to_ancestry": {str(cluster): ancestry for cluster, ancestry in enumerate(best_assignment)},
        "selected_panel_probability": {str(cluster): selected[cluster] for cluster in range(len(selected))},
        "aggregate_panel_probability_AFR_EUR_NAM": macro_matrix,
        "assignment_log_score": best_score,
        "runner_up_log_score": runner_up,
        "log_margin": best_score - runner_up,
        "minimum_probability": minimum_probability,
        "minimum_log_margin": minimum_log_margin,
        "failure_reasons": failure_reasons,
    }


def cluster_assignment_from_model(model_path: Path, ancestry_order: list[str],
                                  panel_to_ancestry: Mapping[str, str],
                                  minimum_probability: float,
                                  minimum_log_margin: float) -> dict[str, Any]:
    """Return a canonicalization map only when the audited gate passes."""
    evidence = cluster_assignment_evidence_from_model(
        model_path, ancestry_order, panel_to_ancestry, minimum_probability, minimum_log_margin,
    )
    require(evidence["status"] == "PASS_UNAMBIGUOUS_TRUTH_BLIND_CLUSTER_ASSIGNMENT",
            "FLARE2 cluster-to-ancestry assignment has insufficient panel support or margin")
    return evidence


def relabel_flare2_vcf(source: Path, output: Path, mapping: Mapping[str, Any],
                       ancestry_order: list[str]) -> None:
    """Canonicalize FLARE2 arbitrary cluster IDs before the M34 parser sees them."""
    require(not output.exists(), "refusing to overwrite canonicalized FLARE2 VCF")
    cluster_to_ancestry = mapping["cluster_to_ancestry"]
    old_to_new = {int(cluster): ancestry_order.index(ancestry)
                  for cluster, ancestry in cluster_to_ancestry.items()}
    require(set(old_to_new) == set(range(len(ancestry_order))) and set(old_to_new.values()) == set(range(len(ancestry_order))),
            "FLARE2 cluster assignment is not one-to-one")
    with gzip.open(source, "rt", encoding="utf-8") as reader, gzip.open(output, "xt", encoding="utf-8") as writer:
        for line in reader:
            if line.startswith("##ANCESTRY=<"):
                writer.write("##ANCESTRY=<" + ",".join(f"{name}={index}" for index, name in enumerate(ancestry_order)) + ">\n")
                continue
            if line.startswith("#"):
                writer.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            fmt = fields[8].split(":")
            required = {name: fmt.index(name) for name in ("AN1", "AN2", "ANP1", "ANP2")}
            for sample_index in range(9, len(fields)):
                values = fields[sample_index].split(":")
                for hard in ("AN1", "AN2"):
                    values[required[hard]] = str(old_to_new[int(values[required[hard]])])
                for probability in ("ANP1", "ANP2"):
                    old = values[required[probability]].split(",")
                    require(len(old) == len(ancestry_order), "FLARE2 probability width differs")
                    new = ["0"] * len(ancestry_order)
                    for old_index, value in enumerate(old):
                        new[old_to_new[old_index]] = value
                    values[required[probability]] = ",".join(new)
                fields[sample_index] = ":".join(values)
            writer.write("\t".join(fields) + "\n")


def preflight(contract: Mapping[str, Any], paths: Mapping[str, Path], outdir: Path) -> dict[str, Any]:
    require(set(paths) == set(INPUT_NAMES), "M35 input members differ")
    require(not outdir.exists(), "refusing to overwrite an M35 output directory")
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"invalid M35 input: {name}")
    try:
        reference = flare_common.scan_vcf(paths["reference_vcf"], contract["chromosome"])
        target = flare_common.scan_vcf(paths["target_vcf"], contract["chromosome"])
    except flare_common.FlareContractError as exc:
        raise PairedContractError(str(exc)) from exc
    require(reference["loci"] == target["loci"], "REF/TARGET marker and REF/ALT axes differ")
    require(set(reference["samples"]).isdisjoint(target["samples"]), "REF/TARGET samples overlap")
    outdir.mkdir(parents=True, exist_ok=False)
    panel_path = outdir / "m35.shared.ref-panel.tsv"
    coarse_panel_path = outdir / "m35.direct.coarse.ref-panel.tsv"
    map_path = outdir / "m35.shared.map"
    panel_audit = normalize_panel_maps(
        paths["sample_map"], paths["panel_macro_map"], panel_path,
        reference["samples"], contract["ancestry_order"],
    )
    coarse_rows = []
    for line in panel_path.read_text(encoding="utf-8").splitlines():
        sample, panel = line.split()
        coarse_rows.append(f"{sample}\t{panel_audit['panel_to_ancestry'][panel]}\n")
    coarse_panel_path.write_text("".join(coarse_rows), encoding="utf-8")
    map_audit = flare_common.normalize_genetic_map(
        paths["genetic_map"], map_path, contract["chromosome"], target["vcf_chromosome"],
        target["first_bp"], target["last_bp"],
    )
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    methods = contract["methods"]
    flare2 = methods["flare2"]
    direct = methods["flare_0_6"]["parameters"]
    panel_probs = methods["flare2"]["panel_probability_parameters"]
    final = methods["flare2"]["final_parameters"]
    direct_prefix = outdir / "m35.flare060"
    panels_prefix = outdir / "m35.flare2.panel_probs"
    model_prefix = outdir / "m35.flare2.model"
    final_prefix = outdir / "m35.flare2.raw"
    # Canonical FLARE 0.6 consumes the three macro-ancestry labels used by M34.
    # FLARE2 alone receives the finer population panels from which it learns
    # three latent clusters.  The sample-level macro assignment remains fixed.
    direct_command = build_command(paths["flare_jar"], paths["reference_vcf"], paths["target_vcf"],
                                   coarse_panel_path, map_path, direct_prefix, direct)
    panels_command = build_command(paths["flare_jar"], paths["reference_vcf"], paths["target_vcf"],
                                   panel_path, map_path, panels_prefix, panel_probs)
    panels_path = Path(f"{panels_prefix}.panels")
    model_path = Path(f"{model_prefix}.model")
    model_command = ["python3", str(paths["flare2_model_builder"]), "--seed", str(final["seed"]),
                     "--builder", str(paths["flare2_upstream_model_builder"]),
                     str(len(contract["ancestry_order"])), str(panels_path), str(model_prefix)]
    final_command = build_command(paths["flare_jar"], paths["reference_vcf"], paths["target_vcf"],
                                  panel_path, map_path, final_prefix, final, model_path)
    delta = {
        "schema_version": "1.0.0",
        "experiment_id": contract["experiment_id"],
        "status": "PREFLIGHT_PAIRED_MARKER_SPINE",
        "comparison": contract["comparison"],
        "shared_axes": {
            "chromosome": contract["chromosome"],
            "ancestry_order": contract["ancestry_order"],
            "marker_count": len(target["loci"]),
            "marker_axis_sha256": marker_axis_sha256(target["loci"]),
            "reference_sample_count": len(reference["samples"]),
            "target_sample_count": len(target["samples"]),
            "reference_axes": sample_phase_axes(paths["reference_vcf"], contract["chromosome"]),
            "target_axes": sample_phase_axes(paths["target_vcf"], contract["chromosome"]),
            "reference_panel_mode": panel_audit["mode"],
            "reference_panel_count": panel_audit["panel_count"],
            "reference_panel_counts": panel_audit["panel_counts"],
            "reference_macro_counts": panel_audit["macro_counts"],
            "panel_to_ancestry": panel_audit["panel_to_ancestry"],
            "normalized_panel_sha256": panel_audit["sample_map_sha256"],
            "normalized_fine_panel_sha256": panel_audit["sample_map_sha256"],
            "normalized_direct_coarse_panel_sha256": sha256_file(coarse_panel_path),
            "panel_macro_map_sha256": panel_audit["panel_macro_map_sha256"],
            "normalized_map_sha256": map_audit["sha256"],
            "target_phase_verified": True,
            "reference_phase_verified": True
        },
        "input_sha256": input_hashes,
        "method_delta": {
            "flare_0_6": {"command_argv": direct_command, "parameters": direct},
            "flare2": {
                "panel_probability_command_argv": panels_command,
                "model_builder_command_argv": model_command,
                "final_command_argv": final_command,
                "panel_probability_parameters": panel_probs,
                "final_parameters": final,
                "additional_information": "clustered reference-panel model only",
                "cluster_assignment": flare2["cluster_assignment"],
                "builder_adapter": {
                    "adapter_sha256": input_hashes["flare2_model_builder"],
                    "upstream_builder_sha256": input_hashes["flare2_upstream_model_builder"],
                    "random_seed": final["seed"],
                }
            }
        },
        "shared_scorer_required": True,
        "label_input_present": False
    }
    resources = {
        "schema_version": "1.0.0",
        "experiment_id": contract["experiment_id"],
        "status": "UNMEASURED_PLANNING_ESTIMATE",
        "request": contract["resource_plan"]["request"],
        "basis": contract["resource_plan"]["basis"],
        "promotion_gate": contract["resource_plan"]["promotion_gate"],
        "staged_input_bytes": sum(path.stat().st_size for path in paths.values()),
        "marker_count": len(target["loci"])
    }
    (outdir / "m35_paired.delta_manifest.json").write_text(
        json.dumps(delta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (outdir / "m35_paired.resource_estimate.json").write_text(
        json.dumps(resources, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"delta": delta, "resources": resources, "commands": {
        "direct": direct_command, "panels": panels_command,
        "model": model_command, "final": final_command,
    }}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    contract = load_contract(args.contract)
    paths = {name: getattr(args, name) for name in INPUT_NAMES}
    prepared = preflight(contract, paths, args.outdir)
    status = "PASS_PREFLIGHT_ONLY"
    output_audits: dict[str, Any] = {}
    if not args.preflight_only:
        for command in (prepared["commands"]["direct"], prepared["commands"]["panels"],
                        prepared["commands"]["model"]):
            subprocess.run(command, check=True)
        cluster_evidence = cluster_assignment_evidence_from_model(
            args.outdir / "m35.flare2.model.model", contract["ancestry_order"],
            prepared["delta"]["shared_axes"]["panel_to_ancestry"],
            contract["methods"]["flare2"]["cluster_assignment"]["min_probability"],
            contract["methods"]["flare2"]["cluster_assignment"]["min_log_margin"],
        )
        evidence_path = args.outdir / "m35.flare2.cluster_assignment.evidence.json"
        evidence_path.write_text(json.dumps(cluster_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if cluster_evidence["status"] == "PASS_UNAMBIGUOUS_TRUTH_BLIND_CLUSTER_ASSIGNMENT":
            cluster_map = cluster_evidence
            cluster_map_path = args.outdir / "m35.flare2.cluster_map.json"
            cluster_map_path.write_text(json.dumps(cluster_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            subprocess.run(prepared["commands"]["final"], check=True)
            relabel_flare2_vcf(args.outdir / "m35.flare2.raw.anc.vcf.gz",
                               args.outdir / "m35.flare2.anc.vcf.gz", cluster_map,
                               contract["ancestry_order"])
            target = flare_common.scan_vcf(paths["target_vcf"], contract["chromosome"])
            for method, prefix in (("flare_0_6", args.outdir / "m35.flare060"),
                                   ("flare2", args.outdir / "m35.flare2")):
                output_audits[method] = flare_common.audit_ancestry_vcf(
                    Path(f"{prefix}.anc.vcf.gz"), target, contract["ancestry_order"],
                )
            status = "PASS_PAIRED_TRUTH_BLIND_INFERENCE"
        else:
            cluster_map = None
            status = "NO_GO_CLUSTER_ASSIGNMENT_BEFORE_FINAL_FLARE2"
    receipt = {
        "schema_version": "1.0.0", "experiment_id": contract["experiment_id"],
        "status": status, "preflight_only": bool(args.preflight_only),
        "contract_sha256": sha256_file(args.contract),
        "delta_manifest_sha256": sha256_file(args.outdir / "m35_paired.delta_manifest.json"),
        "resource_estimate_sha256": sha256_file(args.outdir / "m35_paired.resource_estimate.json"),
        "output_audits": output_audits, "scoring_performed": False,
        "label_input_present": False, "wall_seconds": time.monotonic() - started,
        "cluster_assignment": (cluster_map if not args.preflight_only else None),
        "cluster_assignment_evidence": (cluster_evidence if not args.preflight_only else None),
        "cluster_assignment_evidence_sha256": (sha256_file(args.outdir / "m35.flare2.cluster_assignment.evidence.json")
                                                if not args.preflight_only else None),
        "final_flare2_launched": status == "PASS_PAIRED_TRUTH_BLIND_INFERENCE",
    }
    (args.outdir / "m35_paired.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    for name in INPUT_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"]}, sort_keys=True))
