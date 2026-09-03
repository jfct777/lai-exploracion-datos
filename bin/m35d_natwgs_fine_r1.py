#!/usr/bin/env python3
"""Run the preregistered M35D NatWGS fine-panel R1 experiment stages."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m34_run_flare as flare_common
import m35_flare2_paired as m35
import m35b_prepare_balanced_reference as m35b
import m35c_prepare_source_comparison as source_common


ANCESTRIES = ["AFR", "EUR", "NAM"]
SELECTION_SEEDS = [350101, 350202, 350303]
GMM_SEEDS = [351103, 352207, 353301]


class M35DError(ValueError):
    """Raised when a frozen M35D invariant differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M35DError(message)


def sha256_file(path: Path) -> str:
    return m35.sha256_file(path)


def locus_axis_sha256(loci: list[tuple[str, int, str, str]]) -> str:
    """Hash an ordered CHROM/POS/REF/ALT axis without exposing genotypes."""
    digest = hashlib.sha256()
    for chrom, pos, ref, alt in loci:
        digest.update(f"{chrom}\t{pos}\t{ref}\t{alt}\n".encode("ascii"))
    return digest.hexdigest()


def ancestry_vcf_axis(path: Path) -> tuple[list[str], list[tuple[str, int, str, str]]]:
    """Read only sample and locus axes from a bgzipped FLARE ancestry VCF."""
    samples: list[str] | None = None
    loci: list[tuple[str, int, str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
            elif not line.startswith("#"):
                fields = line.split("\t", 5)
                require(len(fields) >= 5, "M35D ancestry VCF row is truncated")
                loci.append((fields[0].removeprefix("chr"), int(fields[1]),
                             fields[3], fields[4]))
    require(samples is not None and loci and len(loci) == len(set(loci)),
            "M35D ancestry VCF axis differs")
    return samples, loci


def ancestry_header_mapping(path: Path) -> dict[str, dict[str, str]]:
    """Recover FLARE's data-dependent ancestry index order for canonicalization."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##ANCESTRY=<"):
                body = line.strip().removeprefix("##ANCESTRY=<").removesuffix(">")
                entries = [item.rsplit("=", 1) for item in body.split(",")]
                require(all(len(item) == 2 for item in entries),
                        "M35D direct ancestry header differs")
                mapping = {index: name for name, index in entries}
                require(set(mapping) == {"0", "1", "2"} and
                        set(mapping.values()) == set(ANCESTRIES),
                        "M35D direct ancestry order is incomplete")
                return {"cluster_to_ancestry": mapping}
    raise M35DError("M35D direct ancestry header is missing")


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema_version") == "1.0.0", "M35D schema differs")
    require(value.get("experiment_id") == "M35D_NATWGS_FINE_R1_EXPLORATORY_CHR22",
            "M35D experiment identity differs")
    require(value.get("status") == "PREREGISTERED_BEFORE_R1_TRUTH_OPENING",
            "M35D contract was not frozen before R1 truth")
    scope = value["scope"]
    require(scope["chromosome"] == "22" and scope["marker_count"] == 42986 and
            scope["target_set"] == "M34_R1_VALID_32" and
            scope["reserved_root_policy"] == "R2_must_not_be_read_referenced_or_scored",
            "M35D target scope differs")
    design = value["reference_design"]
    require(design["arm"] == "NATWGS" and design["counts"] ==
            {"AFR": 23, "EUR": 23, "NAM": 23}, "M35D reference design differs")
    require(design["selection_seeds"] == SELECTION_SEEDS, "M35D selection seeds differ")
    screen = value["cluster_screen"]
    require(screen["primary_granularity"] == "fine" and
            screen["diagnostic_granularity"] == "coarse" and
            screen["gmm_seeds"] == GMM_SEEDS and
            screen["primary_gate"] == "all_9_NATWGS_fine_combinations_must_pass" and
            screen["nam_support_minimum"] == 0.5 and
            screen["assignment_log_margin_minimum"] == 0.25,
            "M35D prospective gate differs")
    primary = value["preassigned_final_pair"]
    require(primary == {
        "selection_seed": 350101,
        "gmm_seed": 351103,
        "granularity": "fine",
        "direct_flare_panel": "coarse_AFR_EUR_NAM_on_the_same_69_reference_people",
        "flare2_panel": "fine_population_labels_clustered_and_relabelled_to_AFR_EUR_NAM",
        "launch_condition": "all_9_NATWGS_fine_combinations_pass",
    }, "M35D final pair differs")
    require(value["relatedness_policy"]["method"] == "PC_Relate_without_KING",
            "M35D must use PC-Relate without KING")
    return value


def verify_hash(path: Path, expected: str, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"invalid M35D {label}")
    require(sha256_file(path) == expected, f"M35D {label} hash differs")


def _preparation_inputs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "roles": args.roles,
        "phased_scaffold_vcf": args.phased_scaffold_vcf,
        "target_vcf": args.target_vcf,
        "target_tbi": args.target_tbi,
        "m27d_manifest": args.m27d_manifest,
        "m27d_strata": args.m27d_strata,
        "m27d_training_set": args.m27d_training_set,
        "m27d_related_pairs": args.m27d_related_pairs,
        "m34_r1_donor_audit": args.m34_r1_donor_audit,
        "m34_r1_mosaic_receipt": args.m34_r1_mosaic_receipt,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract)
    require(args.selection_seed in SELECTION_SEEDS, "M35D selection seed differs")
    paths = _preparation_inputs(args)
    for name, path in paths.items():
        verify_hash(path, contract["inputs"][name]["sha256"], name)

    manifest = json.loads(args.m27d_manifest.read_text(encoding="utf-8"))
    require(manifest.get("stage") == "M27D_PASS0_PCRELATE" and
            manifest.get("params", {}).get("king_executed") is False,
            "M35D relatedness source is not PC-Relate without KING")
    mosaic = json.loads(args.m34_r1_mosaic_receipt.read_text(encoding="utf-8"))
    require(mosaic.get("stage") == "M34_NAM_EXPLORATORY_MOSAICS" and
            mosaic["parameters"]["target_prefix"] == "M34_R1_VALID" and
            mosaic["parameters"]["target_individuals"] == 32 and
            mosaic["inputs"]["phased_vcf"]["sha256"] ==
            contract["inputs"]["phased_scaffold_vcf"]["sha256"] and
            mosaic["outputs"]["m34_donor_audit.private.tsv"]["sha256"] ==
            contract["inputs"]["m34_r1_donor_audit"]["sha256"],
            "M35D R1 mosaic lineage differs")

    # Reuse the audited source-selection implementation while replacing only
    # the target-root-specific donor audit expected by this new hypothesis.
    compatibility = {
        "relatedness_policy": {
            "historical_23_audit": contract["relatedness_policy"]["historical_23_audit"],
            "eligible_natwgs": {
                "sample_count": contract["relatedness_policy"]["eligible_natwgs"]["sample_count"],
                "sample_axis_sha256": contract["relatedness_policy"]["eligible_natwgs"]["sample_axis_sha256"],
            },
        }
    }
    natwgs, relatedness = source_common.derive_natwgs_candidates(
        compatibility, args.m27d_strata, args.m27d_training_set,
        args.m27d_related_pairs, args.m34_r1_donor_audit,
    )
    expected_relatedness = contract["relatedness_policy"]["eligible_natwgs"]
    require(relatedness["eligible_natwgs"]["internal_edge_count"] == 0 and
            relatedness["eligible_natwgs"]["R0_donor_component_overlap_count"] == 0 and
            relatedness["R0_donors"]["unique_count"] == expected_relatedness["R1_donor_count"] and
            relatedness["R0_donors"]["sample_axis_sha256"] ==
            expected_relatedness["R1_donor_axis_sha256"],
            "M35D R1 donor disjunction differs")

    roles = m35b.load_ref_train(args.roles)
    external, external_selection = m35b.deterministic_subset(roles, args.selection_seed, 23)
    shared = {sample for sample in external
              if m35b.ANCESTRY_MAP[roles[sample]["ancestry"]] in {"AFR", "EUR"}}
    require(len(shared) == 46, "M35D shared AFR/EUR selection differs")
    natwgs_chosen, natwgs_selection = source_common.select_natwgs(
        natwgs, args.selection_seed, 23)
    external_nam = set(external) - shared
    selected = {"EXTERNAL_NAM": set(external), "NATWGS": shared | natwgs_chosen}
    require(len(selected["NATWGS"]) == 69 and external_nam.isdisjoint(natwgs_chosen),
            "M35D source sample selection differs")
    annotations: dict[str, dict[str, tuple[str, str]]] = {
        "EXTERNAL_NAM": {}, "NATWGS": {}
    }
    for arm in annotations:
        for sample in shared:
            annotations[arm][sample] = (
                m35b.ANCESTRY_MAP[roles[sample]["ancestry"]], roles[sample]["population"])
    for sample in external_nam:
        annotations["EXTERNAL_NAM"][sample] = ("NAM", roles[sample]["population"])
    for sample in natwgs_chosen:
        annotations["NATWGS"][sample] = ("NAM", natwgs[sample]["Population"])

    target = m35b.scan_target(args.target_vcf, "22")
    require(target["marker_axis_sha256"] == contract["scope"]["marker_axis_sha256"] and
            len(target["loci"]) == 42986 and len(target["samples"]) == 32 and
            source_common.axis_sha256(target["samples"]) ==
            contract["scope"]["target_sample_axis_sha256"],
            "M35D R1 target axes differ")
    require(selected["NATWGS"].isdisjoint(target["samples"]),
            "M35D NatWGS reference overlaps R1 target")
    references = source_common.materialize_reference_arms(
        args.phased_scaffold_vcf, target["loci"], args.output_prefix,
        selected, annotations,
    )
    for arm in selected:
        path = args.output_prefix.with_name(
            f"{args.output_prefix.name}.{arm.lower()}.selected_samples.txt")
        path.write_text("".join(f"{sample}\n" for sample in sorted(selected[arm])),
                        encoding="utf-8")
        references[arm]["selected_samples_file"] = path.name
        references[arm]["selected_samples_file_sha256"] = sha256_file(path)

    relatedness["R1_donors"] = relatedness.pop("R0_donors")
    relatedness["eligible_natwgs"]["R1_donor_component_overlap_count"] = \
        relatedness["eligible_natwgs"].pop("R0_donor_component_overlap_count")
    relatedness["eligible_natwgs"]["direct_R1_donor_edge_count"] = \
        relatedness["eligible_natwgs"].pop("direct_R0_donor_edge_count")
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M35D_R1_NATWGS_REFERENCE_PREPARATION",
        "status": "PASS_M35D_R1_NATWGS_23_23_23",
        "selection_seed": args.selection_seed,
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "relatedness_audit": relatedness,
        "selection": {
            "shared_AFR_EUR": external_selection,
            "natwgs_NAM": natwgs_selection,
            "shared_AFR_EUR_count": 46,
        },
        "target": {
            "sample_count": 32,
            "sample_axis_sha256": source_common.axis_sha256(target["samples"]),
            "marker_count": 42986,
            "marker_axis_sha256": target["marker_axis_sha256"],
            "truth_opened": False,
        },
        "reference_arms": references,
        "king_executed": False,
        "R2_referenced": False,
    }
    receipt_path = args.output_prefix.with_suffix(".prepare_receipt.json")
    require(not receipt_path.exists(), "refusing to overwrite M35D preparation receipt")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return receipt


def screen(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    contract = load_contract(args.contract)
    require(args.selection_seed in SELECTION_SEEDS and args.gmm_seed in GMM_SEEDS,
            "M35D screen seed differs")
    require(args.granularity in {"fine", "coarse"}, "M35D granularity differs")
    require(not args.outdir.exists(), "refusing to overwrite M35D screen output")
    args.outdir.mkdir(parents=True)
    verify_hash(args.target_vcf, contract["inputs"]["target_vcf"]["sha256"], "target VCF")
    verify_hash(args.target_tbi, contract["inputs"]["target_tbi"]["sha256"], "target index")
    verify_hash(args.genetic_map, contract["inputs"]["genetic_map"]["sha256"], "genetic map")
    prep = json.loads(args.prepare_receipt.read_text(encoding="utf-8"))
    require(prep.get("status") == "PASS_M35D_R1_NATWGS_23_23_23" and
            prep.get("selection_seed") == args.selection_seed and
            prep["target"]["truth_opened"] is False and
            prep.get("king_executed") is False and prep.get("R2_referenced") is False,
            "M35D preparation receipt differs")
    arm = prep["reference_arms"]["NATWGS"]
    require(arm["maps"][args.granularity]["sample_map_sha256"] ==
            sha256_file(args.sample_map) and
            arm["maps"][args.granularity]["panel_macro_map_sha256"] ==
            sha256_file(args.panel_macro_map), "M35D panel maps differ")

    reference = flare_common.scan_vcf(args.reference_vcf, "22")
    target = flare_common.scan_vcf(args.target_vcf, "22")
    require(len(reference["samples"]) == 69 and reference["loci"] == target["loci"] and
            len(target["loci"]) == 42986 and
            set(reference["samples"]).isdisjoint(target["samples"]),
            "M35D screen axes differ")
    panel_path = args.outdir / "m35d.ref-panel.tsv"
    map_path = args.outdir / "m35d.map"
    panel = m35.normalize_panel_maps(args.sample_map, args.panel_macro_map, panel_path,
                                     reference["samples"], ANCESTRIES)
    expected_mode = ("FINE_POPULATION_OR_SUBPOPULATION" if args.granularity == "fine"
                     else "COARSE_MACROANCESTRY_EXPLICIT")
    require(panel["mode"] == expected_mode and
            panel["macro_counts"] == {"AFR": 23, "EUR": 23, "NAM": 23},
            "M35D panel mode or balance differs")
    map_audit = flare_common.normalize_genetic_map(
        args.genetic_map, map_path, "22", target["vcf_chromosome"],
        target["first_bp"], target["last_bp"])

    panel_prefix = args.outdir / "m35d.panel_probs"
    subprocess.run(m35.build_command(
        args.flare_jar, args.reference_vcf, args.target_vcf, panel_path, map_path,
        panel_prefix, contract["flare_parameters"]["panel_probability"]), check=True)
    panels_path = Path(f"{panel_prefix}.panels")
    model_prefix = args.outdir / "m35d.cluster"
    subprocess.run([
        "python3", str(args.model_wrapper), "--seed", str(args.gmm_seed),
        "--builder", str(args.upstream_builder), "3", str(panels_path), str(model_prefix),
    ], check=True)
    model_path = Path(f"{model_prefix}.model")
    raw = m35.cluster_assignment_evidence_from_model(
        model_path, ANCESTRIES, panel["panel_to_ancestry"], 0.0,
        contract["cluster_screen"]["assignment_log_margin_minimum"])
    support_by_ancestry = {
        ancestry: raw["selected_panel_probability"][cluster]
        for cluster, ancestry in raw["cluster_to_ancestry"].items()
    }
    reasons = []
    if support_by_ancestry["NAM"] < contract["cluster_screen"]["nam_support_minimum"]:
        reasons.append("NAM_support_below_0.5")
    if raw["log_margin"] < contract["cluster_screen"]["assignment_log_margin_minimum"]:
        reasons.append("assignment_log_margin_below_0.25")
    status = "PASS_M35D_CLUSTER_SEPARATION" if not reasons else "NO_GO_M35D_CLUSTER_SEPARATION"
    evidence = {
        **raw,
        "experiment_id": contract["experiment_id"],
        "status": status,
        "failure_reasons": reasons,
        "selection_seed": args.selection_seed,
        "gmm_seed": args.gmm_seed,
        "granularity": args.granularity,
        "NAM_support": support_by_ancestry["NAM"],
        "selected_panel_probability_by_ancestry": support_by_ancestry,
        "balanced_macro_counts": panel["macro_counts"],
        "marker_axis_sha256": contract["scope"]["marker_axis_sha256"],
        "target_sample_axis_sha256": contract["scope"]["target_sample_axis_sha256"],
        "target_truth_opened": False,
        "R2_referenced": False,
    }
    evidence_path = args.outdir / "m35d.cluster_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M35D_R1_TRUTH_BLIND_CLUSTER_SCREEN",
        "status": status,
        "selection_seed": args.selection_seed,
        "gmm_seed": args.gmm_seed,
        "granularity": args.granularity,
        "wall_seconds": time.monotonic() - started,
        "children_max_rss_kib": usage.ru_maxrss,
        "contract_sha256": sha256_file(args.contract),
        "prepare_receipt_sha256": sha256_file(args.prepare_receipt),
        "evidence_sha256": sha256_file(evidence_path),
        "model_sha256": sha256_file(model_path),
        "panels_sha256": sha256_file(panels_path),
        "normalized_map_sha256": map_audit["sha256"],
        "truth_input_present": False,
        "final_inference_performed": False,
        "R2_referenced": False,
    }
    (args.outdir / "m35d.screen_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    support = sorted(row["NAM_support"] for row in rows)
    margin = sorted(row["log_margin"] for row in rows)
    return {
        "passed": sum(row["status"] == "PASS_M35D_CLUSTER_SEPARATION" for row in rows),
        "total": len(rows),
        "NAM_support_min": support[0],
        "NAM_support_median": support[len(support) // 2],
        "NAM_support_max": support[-1],
        "log_margin_min": margin[0],
        "log_margin_median": margin[len(margin) // 2],
        "log_margin_max": margin[-1],
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract)
    require(not args.output.exists() and not args.go_token.exists(),
            "refusing to overwrite M35D gate")
    expected = {(selection, granularity, gmm) for selection in SELECTION_SEEDS
                for granularity in ("fine", "coarse") for gmm in GMM_SEEDS}
    rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for directory in args.screen_dir:
        evidence_path = directory / "m35d.cluster_evidence.json"
        receipt_path = directory / "m35d.screen_receipt.json"
        require(evidence_path.is_file() and receipt_path.is_file(), "M35D screen is incomplete")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        key = (evidence["selection_seed"], evidence["granularity"], evidence["gmm_seed"])
        require(key in expected and key not in rows, "M35D grid is unexpected or duplicated")
        require(evidence.get("target_truth_opened") is False and
                evidence.get("R2_referenced") is False and
                receipt.get("truth_input_present") is False and
                receipt.get("final_inference_performed") is False and
                receipt["evidence_sha256"] == sha256_file(evidence_path),
                "M35D blind screen receipt differs")
        rows[key] = evidence
        hashes[":".join(map(str, key))] = sha256_file(evidence_path)
    require(set(rows) == expected, "M35D grid is incomplete")
    fine = _summary([rows[(selection, "fine", gmm)] for selection in SELECTION_SEEDS
                     for gmm in GMM_SEEDS])
    coarse = _summary([rows[(selection, "coarse", gmm)] for selection in SELECTION_SEEDS
                       for gmm in GMM_SEEDS])
    passed = fine["passed"] == fine["total"] == 9
    status = ("PASS_M35D_FINE_9_OF_9_GO_PREASSIGNED_R1_FINAL" if passed else
              "NO_GO_M35D_FINE_NOT_9_OF_9_STOP_BEFORE_R1_TRUTH")
    result = {
        "schema_version": "1.0.0",
        "stage": "M35D_R1_CLUSTER_GATE",
        "status": status,
        "claim_level": contract["claim_level"],
        "primary": {"granularity": "fine", **fine},
        "diagnostic": {"granularity": "coarse", **coarse},
        "primary_rule": contract["cluster_screen"]["primary_gate"],
        "truth_opened": False,
        "R2_referenced": False,
        "post_hoc_seed_selection": False,
        "preassigned_final_pair": contract["preassigned_final_pair"],
        "screen_rows": [
            {"selection_seed": key[0], "granularity": key[1], "gmm_seed": key[2],
             "status": rows[key]["status"], "NAM_support": rows[key]["NAM_support"],
             "log_margin": rows[key]["log_margin"]}
            for key in sorted(rows)
        ],
        "input_sha256": {"contract": sha256_file(args.contract),
                          "screen_evidence": hashes},
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    if passed:
        primary = contract["preassigned_final_pair"]
        args.go_token.write_text(json.dumps({
            "status": "GO_M35D_PREASSIGNED_R1_FINAL_ONLY",
            "gate_sha256": sha256_file(args.output),
            "selection_seed": primary["selection_seed"],
            "gmm_seed": primary["gmm_seed"],
            "granularity": primary["granularity"],
            "allowed_target": "M34_R1_VALID_32",
            "R2_allowed": False,
        }, sort_keys=True) + "\n", encoding="utf-8")
    return result


def final_inference(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    contract = load_contract(args.contract)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    token = json.loads(args.go_token.read_text(encoding="utf-8"))
    primary = contract["preassigned_final_pair"]
    require(gate.get("status") == "PASS_M35D_FINE_9_OF_9_GO_PREASSIGNED_R1_FINAL" and
            gate.get("truth_opened") is False and gate.get("R2_referenced") is False,
            "M35D final is blocked by its blind gate")
    require(token.get("status") == "GO_M35D_PREASSIGNED_R1_FINAL_ONLY" and
            token.get("gate_sha256") == sha256_file(args.gate) and
            token.get("R2_allowed") is False and
            {key: token[key] for key in ("selection_seed", "gmm_seed", "granularity")} ==
            {key: primary[key] for key in ("selection_seed", "gmm_seed", "granularity")},
            "M35D final token differs")
    require(not args.outdir.exists(), "refusing to overwrite M35D final output")
    args.outdir.mkdir(parents=True)
    evidence_path = args.screen_dir / "m35d.cluster_evidence.json"
    model_path = args.screen_dir / "m35d.cluster.model"
    fine_panel = args.screen_dir / "m35d.ref-panel.tsv"
    map_path = args.screen_dir / "m35d.map"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    require(evidence.get("status") == "PASS_M35D_CLUSTER_SEPARATION" and
            evidence["selection_seed"] == primary["selection_seed"] and
            evidence["gmm_seed"] == primary["gmm_seed"] and
            evidence["granularity"] == "fine" and
            evidence["model_sha256"] == sha256_file(model_path),
            "M35D preassigned screen differs")
    reference = flare_common.scan_vcf(args.reference_vcf, "22")
    target = flare_common.scan_vcf(args.target_vcf, "22")
    require(len(reference["samples"]) == 69 and len(target["samples"]) == 32 and
            reference["loci"] == target["loci"] and len(target["loci"]) == 42986 and
            set(reference["samples"]).isdisjoint(target["samples"]),
            "M35D final axes differ")
    coarse_panel = args.outdir / "m35d.direct.coarse.ref-panel.tsv"
    coarse_audit = m35.normalize_panel_maps(
        args.coarse_sample_map, args.coarse_panel_macro_map, coarse_panel,
        reference["samples"], ANCESTRIES)
    require(coarse_audit["mode"] == "COARSE_MACROANCESTRY_EXPLICIT" and
            coarse_audit["macro_counts"] == {"AFR": 23, "EUR": 23, "NAM": 23},
            "M35D direct-FLARE panel differs")
    direct_prefix = args.outdir / "m35d.direct.raw"
    flare2_prefix = args.outdir / "m35d.flare2.raw"
    subprocess.run(m35.build_command(
        args.flare_jar, args.reference_vcf, args.target_vcf, coarse_panel, map_path,
        direct_prefix, contract["flare_parameters"]["direct"]), check=True)
    subprocess.run(m35.build_command(
        args.flare_jar, args.reference_vcf, args.target_vcf, fine_panel, map_path,
        flare2_prefix, contract["flare_parameters"]["flare2_final"], model_path), check=True)
    direct_raw_path = Path(f"{direct_prefix}.anc.vcf.gz")
    direct_path = args.outdir / "m35d.direct.anc.vcf.gz"
    m35.relabel_flare2_vcf(direct_raw_path, direct_path,
                           ancestry_header_mapping(direct_raw_path), ANCESTRIES)
    flare2_path = args.outdir / "m35d.flare2.anc.vcf.gz"
    m35.relabel_flare2_vcf(Path(f"{flare2_prefix}.anc.vcf.gz"), flare2_path,
                           evidence, ANCESTRIES)
    direct_samples, direct_loci = ancestry_vcf_axis(direct_path)
    flare2_samples, flare2_loci = ancestry_vcf_axis(flare2_path)
    retained = set(direct_loci)
    require(direct_samples == flare2_samples == target["samples"] and
            direct_loci == flare2_loci and
            direct_loci == [locus for locus in target["loci"] if locus in retained],
            "M35D paired output axes differ or are not an ordered TARGET subset")
    excluded_loci = [locus for locus in target["loci"] if locus not in retained]
    require(len(direct_loci) == 42732 and len(excluded_loci) == 254,
            "M35D retained marker axis differs from the pre-score technical amendment")
    excluded_path = args.outdir / "m35d.excluded_monomorphic_reference_loci.tsv"
    excluded_path.write_text(
        "chrom\tpos\tref\talt\treason\n" + "".join(
            f"{chrom}\t{pos}\t{ref}\t{alt}\tMONOMORPHIC_IN_PREASSIGNED_69_REFERENCE\n"
            for chrom, pos, ref, alt in excluded_loci), encoding="utf-8")
    output_target = dict(target)
    output_target["loci"] = direct_loci
    audits = {
        "FLARE_F0_SAME_69": flare_common.audit_ancestry_vcf(
            direct_path, output_target, ANCESTRIES),
        "FLARE2_NATWGS_FINE_SAME_69": flare_common.audit_ancestry_vcf(
            flare2_path, output_target, ANCESTRIES),
    }
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M35D_R1_PREASSIGNED_TRUTH_BLIND_PAIRED_INFERENCE",
        "status": "PASS_M35D_R1_PAIRED_INFERENCE_READY_FOR_SEPARATE_SCORE",
        "preassigned_final_pair": primary,
        "reference_sample_count": 69,
        "target_sample_count": 32,
        "input_marker_count": len(target["loci"]),
        "marker_count": len(direct_loci),
        "excluded_marker_count": len(excluded_loci),
        "marker_axis_sha256": locus_axis_sha256(direct_loci),
        "excluded_locus_axis_sha256": locus_axis_sha256(excluded_loci),
        "excluded_loci_tsv_sha256": sha256_file(excluded_path),
        "output_audits": audits,
        "output_sha256": {
            "FLARE_F0_SAME_69": sha256_file(direct_path),
            "FLARE2_NATWGS_FINE_SAME_69": sha256_file(flare2_path),
        },
        "contract_sha256": sha256_file(args.contract),
        "gate_sha256": sha256_file(args.gate),
        "cluster_evidence_sha256": sha256_file(evidence_path),
        "wall_seconds": time.monotonic() - started,
        "truth_input_present": False,
        "scoring_performed": False,
        "R2_referenced": False,
    }
    (args.outdir / "m35d.final_inference_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract)
    direct = json.loads(args.direct_metrics.read_text(encoding="utf-8"))
    flare2 = json.loads(args.flare2_metrics.read_text(encoding="utf-8"))
    canonical = json.loads(args.canonical_metrics.read_text(encoding="utf-8"))
    inference = json.loads(args.inference_receipt.read_text(encoding="utf-8"))
    truth_subset = json.loads(args.truth_subset_receipt.read_text(encoding="utf-8"))
    require(all(value.get("status") == "PASS_SCORED" and
                value.get("truth_opened_only_by_scorer") is True
                for value in (direct, flare2, canonical)), "M35D scorer inputs differ")
    require(inference.get("status") ==
            "PASS_M35D_R1_PAIRED_INFERENCE_READY_FOR_SEPARATE_SCORE" and
            inference.get("truth_input_present") is False,
            "M35D inference receipt differs")
    require(truth_subset.get("status") ==
            "PASS_M35D_R1_TRUTH_SUBSET_EXACT_COMMON_AXIS" and
            truth_subset.get("R2_referenced") is False and
            direct["input_sha256"]["truth"] == flare2["input_sha256"]["truth"] ==
            truth_subset["output_sha256"] and
            canonical["input_sha256"]["truth"] ==
            truth_subset["input_sha256"]["full_R1_truth"],
            "M35D full/subset truth provenance differs")
    for key in ("sample_count", "haplotype_count", "marker_count", "ancestry_names", "cm_span"):
        require(direct[key] == flare2[key], f"M35D paired {key} differs")
    require(direct["sample_count"] == canonical["sample_count"] == 32 and
            direct["haplotype_count"] == canonical["haplotype_count"] == 2 and
            direct["marker_count"] == truth_subset["retained_marker_count"] == 42732 and
            canonical["marker_count"] == truth_subset["source_marker_count"] == 42986 and
            direct["ancestry_names"] == canonical["ancestry_names"] == ANCESTRIES,
            "M35D score axes differ")

    def delta(right: dict[str, Any], left: dict[str, Any]) -> dict[str, Any]:
        return {
            "macro_ancestry_dose_MAE": right["macro_ancestry_dose_MAE"] - left["macro_ancestry_dose_MAE"],
            "haplotype_Brier": right["haplotype_Brier"] - left["haplotype_Brier"],
            "NAM_truth_present_MAE": right["NAM_truth_present_MAE"] - left["NAM_truth_present_MAE"],
            "per_ancestry_MAE": {name: right["per_ancestry_MAE"][name] -
                                         left["per_ancestry_MAE"][name] for name in ANCESTRIES},
            "boundary": {
                tolerance: {
                    "f1": right["boundary"][tolerance]["f1"] - left["boundary"][tolerance]["f1"],
                    "false_transitions_per_cM": (
                        right["boundary"][tolerance]["false_transitions_per_cM"] -
                        left["boundary"][tolerance]["false_transitions_per_cM"]),
                    "matched": right["boundary"][tolerance]["matched"] -
                               left["boundary"][tolerance]["matched"],
                    "predicted": right["boundary"][tolerance]["predicted"] -
                                 left["boundary"][tolerance]["predicted"],
                } for tolerance in sorted(direct["boundary"], key=float)
            },
        }

    result = {
        "schema_version": "1.0.0",
        "stage": "M35D_R1_NATWGS_FINE_PAIRED_SCORE",
        "status": "PASS_M35D_R1_EXPLORATORY_PAIRED_POINT_ESTIMATE",
        "claim_level": contract["claim_level"],
        "comparison": "FLARE2_NATWGS_FINE_MINUS_FLARE_F0_SAME_69",
        "reference_sample_counts": {"AFR": 23, "EUR": 23, "NAM": 23},
        "metrics": {
            "FLARE_F0_SAME_69": direct,
            "FLARE2_NATWGS_FINE_SAME_69": flare2,
            "M34_FLARE_F0_FULL_REFERENCE_CONTEXT_ONLY": canonical,
        },
        "delta_FLARE2_minus_FLARE_F0_same_69": delta(flare2, direct),
        "cross_axis_delta_to_M34_full_reference": None,
        "cross_axis_policy": (
            "M34 full-reference F0 is context only because it has 42986 markers; "
            "the paired M35D comparison uses the exact common 42732-marker axis"
        ),
        "excluded_monomorphic_reference_marker_count": 254,
        "truth_opened_only_by_scorer": True,
        "R2_referenced": False,
        "post_hoc_seed_selection": False,
        "input_sha256": {
            "direct_metrics": sha256_file(args.direct_metrics),
            "flare2_metrics": sha256_file(args.flare2_metrics),
            "canonical_metrics": sha256_file(args.canonical_metrics),
            "inference_receipt": sha256_file(args.inference_receipt),
            "truth_subset_receipt": sha256_file(args.truth_subset_receipt),
            "truth": direct["input_sha256"]["truth"],
            "full_R1_truth": canonical["input_sha256"]["truth"],
        },
    }
    require(not args.output.exists(), "refusing to overwrite M35D paired summary")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--contract", type=Path, required=True)
    prep.add_argument("--roles", type=Path, required=True)
    prep.add_argument("--phased-scaffold-vcf", type=Path, required=True)
    prep.add_argument("--target-vcf", type=Path, required=True)
    prep.add_argument("--target-tbi", type=Path, required=True)
    prep.add_argument("--m27d-manifest", type=Path, required=True)
    prep.add_argument("--m27d-strata", type=Path, required=True)
    prep.add_argument("--m27d-training-set", type=Path, required=True)
    prep.add_argument("--m27d-related-pairs", type=Path, required=True)
    prep.add_argument("--m34-r1-donor-audit", type=Path, required=True)
    prep.add_argument("--m34-r1-mosaic-receipt", type=Path, required=True)
    prep.add_argument("--selection-seed", type=int, required=True)
    prep.add_argument("--output-prefix", type=Path, required=True)
    run = sub.add_parser("screen")
    for name in ("contract", "reference-vcf", "reference-tbi", "target-vcf", "target-tbi",
                 "sample-map", "panel-macro-map", "prepare-receipt", "genetic-map",
                 "flare-jar", "model-wrapper", "upstream-builder", "outdir"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--selection-seed", type=int, required=True)
    run.add_argument("--gmm-seed", type=int, required=True)
    run.add_argument("--granularity", choices=("fine", "coarse"), required=True)
    gate = sub.add_parser("aggregate")
    gate.add_argument("--contract", type=Path, required=True)
    gate.add_argument("--screen-dir", type=Path, action="append", required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--go-token", type=Path, required=True)
    final = sub.add_parser("final")
    for name in ("contract", "gate", "go-token", "screen-dir", "reference-vcf",
                 "target-vcf", "coarse-sample-map", "coarse-panel-macro-map",
                 "flare-jar", "outdir"):
        final.add_argument(f"--{name}", type=Path, required=True)
    summary = sub.add_parser("summarize")
    for name in ("contract", "direct-metrics", "flare2-metrics", "canonical-metrics",
                 "inference-receipt", "truth-subset-receipt", "output"):
        summary.add_argument(f"--{name}", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = {
        "prepare": prepare,
        "screen": screen,
        "aggregate": aggregate,
        "final": final_inference,
        "summarize": summarize,
    }[args.stage](args)
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
