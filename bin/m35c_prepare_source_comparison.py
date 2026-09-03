#!/usr/bin/env python3
"""Prepare matched external-NAM and NatWGS reference panels for M35C."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m35b_prepare_balanced_reference as m35b


ARMS = ("EXTERNAL_NAM", "NATWGS")
ANCESTRY_ORDER = ("AFR", "EUR", "NAM")
KINSHIP_THRESHOLD = 0.0442


class M35CPreparationError(ValueError):
    """Raised when a source-comparison invariant is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M35CPreparationError(message)


def sha256_file(path: Path) -> str:
    return m35b.sha256_file(path)


def axis_sha256(rows: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(
        "rt", encoding="utf-8"
    )


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == "1.0.0", "M35C contract schema differs")
    require(
        payload.get("experiment_id") == "M35C_NATWGS_SOURCE_SENSITIVITY_CHR22",
        "M35C contract identity differs",
    )
    require(
        payload.get("status") == "PREREGISTERED_EXPLORATORY_SOURCE_SCREEN",
        "M35C contract status differs",
    )
    require(
        payload["scope"] == {
            "chromosome": "22",
            "marker_count": 42986,
            "marker_axis_sha256": "e82ef9b853283de33f5873b2fbdebebe79291969a9fe43deb4c8d685d4a71ea0",
            "target_set": "M34_R0_VALID_32_UNCHANGED",
            "target_truth_hidden_until_cluster_gate": True,
            "claim": "whether_a_PC_Relate_filtered_NatWGS_NAM_source_changes_truth_blind_FLARE2_cluster_separation",
        },
        "M35C scope differs",
    )
    design = payload["reference_design"]
    require(design["counts_per_arm"] == {"AFR": 23, "EUR": 23, "NAM": 23},
            "M35C reference counts differ")
    require(design["arms"] == list(ARMS), "M35C arm order differs")
    require(design["selection_seeds"] == [350101, 350202, 350303],
            "M35C selection seeds differ")
    require(design["same_AFR_EUR_within_selection_seed"] is True,
            "M35C AFR/EUR matching policy differs")
    gate = payload["cluster_screen"]
    require(gate["gmm_seeds"] == [351103, 352207, 353301],
            "M35C GMM seeds differ")
    require(gate["primary_arm"] == "NATWGS" and gate["primary_granularity"] == "coarse",
            "M35C primary screen differs")
    require(gate["primary_gate"] == "all_9_NATWGS_coarse_combinations_must_pass",
            "M35C primary gate differs")
    require(gate["nam_support_minimum"] == 0.5 and
            gate["assignment_log_margin_minimum"] == 0.25,
            "M35C cluster thresholds differ")
    require(payload["relatedness_policy"]["method"] == "PC_Relate_without_KING" and
            payload["relatedness_policy"]["kinship_threshold"] == KINSHIP_THRESHOLD,
            "M35C relatedness policy differs")
    return payload


def verify_inputs(contract: dict[str, Any], paths: dict[str, Path]) -> None:
    expected = set(contract["inputs"]) - {"genetic_map"}
    require(set(paths) == expected, "M35C preparation input members differ")
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"invalid M35C input: {name}")
        require(sha256_file(path) == contract["inputs"][name]["sha256"],
                f"M35C input hash differs: {name}")


def load_related_edges(path: Path) -> list[tuple[str, str, float]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(rows and {"ID1", "ID2", "kin"}.issubset(rows[0]),
            "M35C PC-Relate table lacks required columns")
    edges = []
    for row in rows:
        kinship = float(row["kin"])
        if kinship >= KINSHIP_THRESHOLD:
            require(row["ID1"] != row["ID2"], "M35C PC-Relate edge is self-referential")
            edges.append((row["ID1"], row["ID2"], kinship))
    return edges


def component_closure(seeds: set[str], edges: list[tuple[str, str, float]]) -> set[str]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right, _kinship in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    closure = set(seeds)
    pending = list(seeds)
    while pending:
        current = pending.pop()
        for neighbour in adjacency.get(current, ()):
            if neighbour not in closure:
                closure.add(neighbour)
                pending.append(neighbour)
    return closure


def pair_rows(edges: list[tuple[str, str, float]], members: set[str]) -> list[str]:
    values = sorted(
        (min(left, right), max(left, right), kinship)
        for left, right, kinship in edges
        if left in members and right in members
    )
    return [f"{left}\t{right}\t{kinship:.15g}" for left, right, kinship in values]


def derive_natwgs_candidates(
    contract: dict[str, Any], strata_path: Path, training_path: Path,
    related_path: Path, donor_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    strata = read_tsv(strata_path)
    required = {
        "sample_id", "match_status", "population_interpretable", "Source", "Ancestry",
        "Population", "Country", "Exclude", "Maximum_unrelated_dataset",
    }
    require(strata and required.issubset(strata[0]), "M35C strata table lacks required columns")
    require(len({row["sample_id"] for row in strata}) == len(strata),
            "M35C strata sample axis is duplicated")
    training = {line.strip() for line in training_path.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    require(training, "M35C PC-Relate independent set is empty")
    donor_rows = read_tsv(donor_path)
    require(donor_rows and "donor_sample_id" in donor_rows[0],
            "M35C donor audit lacks donor IDs")
    donors = {row["donor_sample_id"] for row in donor_rows}
    edges = load_related_edges(related_path)
    donor_components = component_closure(donors, edges)

    historical = sorted(
        row["sample_id"] for row in strata
        if row["Source"] == "NatWGS" and row["Country"] == "Brazil"
        and row["Exclude"] == "FALSE" and row["Maximum_unrelated_dataset"] == "TRUE"
    )
    historical_pairs = pair_rows(edges, set(historical))
    historical_expected = contract["relatedness_policy"]["historical_23_audit"]
    require(len(historical) == historical_expected["sample_count"] and
            axis_sha256(historical) == historical_expected["sample_axis_sha256"],
            "M35C historical Brazilian-23 axis differs")
    require(len(historical_pairs) == historical_expected["internal_edge_count"] and
            axis_sha256(historical_pairs) == historical_expected["internal_edges_sha256"],
            "M35C historical Brazilian-23 PC-Relate edges differ")

    base_rows = {
        row["sample_id"]: row for row in strata
        if row["Source"] == "NatWGS" and row["Ancestry"] == "Native_American"
        and row["Exclude"] == "FALSE" and row["match_status"] == "MATCHED"
        and row["population_interpretable"] == "TRUE"
    }
    eligible = {
        sample: row for sample, row in base_rows.items()
        if sample in training and sample not in donor_components
    }
    selected_policy = contract["relatedness_policy"]["eligible_natwgs"]
    eligible_axis = sorted(eligible)
    require(len(eligible) == selected_policy["sample_count"] and
            axis_sha256(eligible_axis) == selected_policy["sample_axis_sha256"],
            "M35C eligible NatWGS axis differs")
    require(not pair_rows(edges, set(eligible)),
            "M35C eligible NatWGS candidates contain a PC-Relate edge")
    require(set(eligible).isdisjoint(donors) and set(eligible).isdisjoint(donor_components),
            "M35C eligible NatWGS candidates overlap an R0 donor component")

    direct_cross_edges = [
        (left, right) for left, right, _kinship in edges
        if (left in eligible and right in donors) or (right in eligible and left in donors)
    ]
    require(not direct_cross_edges, "M35C NatWGS candidates have a PC-Relate edge to R0 donors")
    country_counts = dict(sorted(Counter(row["Country"] for row in eligible.values()).items()))
    population_counts = dict(sorted(Counter(row["Population"] for row in eligible.values()).items()))
    require(sum(country_counts.values()) == len(eligible) and all(country_counts),
            "M35C NatWGS country metadata differs")
    return eligible, {
        "method": "PC_Relate_without_KING",
        "kinship_threshold": KINSHIP_THRESHOLD,
        "historical_brazilian_flagged": {
            "sample_count": len(historical),
            "sample_axis_sha256": axis_sha256(historical),
            "in_primary_independent_set": sum(sample in training for sample in historical),
            "internal_edge_count": len(historical_pairs),
            "internal_edges_sha256": axis_sha256(historical_pairs),
            "usable_as_23_independent_samples": False,
        },
        "eligible_natwgs": {
            "sample_count": len(eligible),
            "sample_axis_sha256": axis_sha256(eligible_axis),
            "country_counts": country_counts,
            "population_count": len(population_counts),
            "brazilian_count": country_counts.get("Brazil", 0),
            "internal_edge_count": 0,
            "direct_R0_donor_edge_count": 0,
            "R0_donor_component_overlap_count": 0,
        },
        "R0_donors": {
            "unique_count": len(donors),
            "sample_axis_sha256": axis_sha256(sorted(donors)),
        },
    }


def _hamilton_country_allocation(country_samples: dict[str, list[str]], target: int,
                                 seed: int) -> dict[str, int]:
    require(target >= len(country_samples), "M35C target cannot preserve every NatWGS country")
    require(target <= sum(map(len, country_samples.values())),
            "M35C NatWGS target exceeds candidates")
    allocation = {country: 1 for country in country_samples}
    remaining = target - len(allocation)
    capacity = {country: len(samples) - 1 for country, samples in country_samples.items()}
    total_capacity = sum(capacity.values())
    quotas = {country: remaining * capacity[country] / total_capacity
              for country in country_samples}
    floors = {country: int(quotas[country]) for country in country_samples}
    for country, count in floors.items():
        allocation[country] += count
    left = remaining - sum(floors.values())
    order = sorted(country_samples, key=lambda country: (
        -(quotas[country] - floors[country]),
        hashlib.sha256(f"M35C|{seed}|country|{country}".encode("utf-8")).hexdigest(),
    ))
    for country in order:
        if left == 0:
            break
        if allocation[country] < len(country_samples[country]):
            allocation[country] += 1
            left -= 1
    require(left == 0 and sum(allocation.values()) == target,
            "M35C country allocation did not reach its target")
    return allocation


def select_natwgs(candidates: dict[str, dict[str, str]], seed: int,
                   target: int) -> tuple[set[str], dict[str, Any]]:
    by_country: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sample, row in candidates.items():
        require(row["Country"] and row["Population"], "M35C NatWGS stratum is empty")
        require(not any(character.isspace() for character in row["Population"]),
                "M35C NatWGS population label contains whitespace")
        by_country[row["Country"]][row["Population"]].append(sample)
    country_samples = {
        country: [sample for samples in populations.values() for sample in samples]
        for country, populations in by_country.items()
    }
    allocation = _hamilton_country_allocation(country_samples, target, seed)
    chosen: list[str] = []
    population_counts: Counter[str] = Counter()
    for country in sorted(by_country):
        ranked_by_population = {
            population: sorted(samples, key=lambda sample: hashlib.sha256(
                f"M35C|{seed}|{country}|{population}|{sample}".encode("utf-8")
            ).hexdigest())
            for population, samples in by_country[country].items()
        }
        country_chosen: list[str] = []
        round_index = 0
        while len(country_chosen) < allocation[country]:
            available = [population for population, samples in ranked_by_population.items()
                         if round_index < len(samples)]
            require(available, "M35C population round-robin exhausted early")
            population_order = sorted(available, key=lambda population: hashlib.sha256(
                f"M35C|{seed}|{country}|round{round_index}|{population}".encode("utf-8")
            ).hexdigest())
            for population in population_order:
                if len(country_chosen) == allocation[country]:
                    break
                sample = ranked_by_population[population][round_index]
                country_chosen.append(sample)
                population_counts[population] += 1
            round_index += 1
        chosen.extend(country_chosen)
    require(len(chosen) == target and len(set(chosen)) == target,
            "M35C NatWGS selection size differs")
    return set(chosen), {
        "method": "country_floor_capacity_Hamilton_then_population_round_robin_with_sha256_ties",
        "selection_seed": seed,
        "selected_count": len(chosen),
        "country_allocation": dict(sorted(allocation.items())),
        "population_counts": dict(sorted(population_counts.items())),
        "selected_country_count": len({candidates[sample]["Country"] for sample in chosen}),
        "selected_population_count": len(population_counts),
        "selected_sample_axis_sha256": axis_sha256(sorted(chosen)),
    }


def safe_panel_label(value: str) -> str:
    label = re.sub(r"\s+", "_", value.strip())
    require(label and re.fullmatch(r"[^\t\r\n]+", label) is not None,
            "M35C panel label is unsafe")
    return label


def write_maps(prefix: Path, arm: str, samples: list[str],
               annotations: dict[str, tuple[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for granularity in ("coarse", "fine"):
        sample_map = prefix.with_name(f"{prefix.name}.{arm.lower()}.{granularity}.sample_panel.tsv")
        macro_map = prefix.with_name(f"{prefix.name}.{arm.lower()}.{granularity}.panel_macro.tsv")
        sample_rows: list[str] = []
        panel_to_macro: dict[str, str] = {}
        for sample in samples:
            macro, population = annotations[sample]
            panel = macro if granularity == "coarse" else safe_panel_label(population)
            previous = panel_to_macro.setdefault(panel, macro)
            require(previous == macro, "M35C panel label maps to multiple macro-ancestries")
            sample_rows.append(f"{sample}\t{panel}\n")
        sample_map.write_text("".join(sample_rows), encoding="utf-8")
        macro_map.write_text("".join(
            f"{panel}\t{macro}\n" for panel, macro in sorted(panel_to_macro.items())
        ), encoding="utf-8")
        result[granularity] = {
            "sample_map": sample_map.name,
            "sample_map_sha256": sha256_file(sample_map),
            "panel_macro_map": macro_map.name,
            "panel_macro_map_sha256": sha256_file(macro_map),
            "panel_count": len(panel_to_macro),
        }
    return result


def materialize_reference_arms(
    scaffold: Path, target_loci: list[str], output_prefix: Path,
    selected: dict[str, set[str]], annotations: dict[str, dict[str, tuple[str, str]]],
) -> dict[str, Any]:
    target_set = set(target_loci)
    require(len(target_set) == len(target_loci), "M35C target marker axis is duplicated")
    output_paths = {
        arm: output_prefix.with_name(f"{output_prefix.name}.{arm.lower()}.ref.vcf") for arm in ARMS
    }
    require(not any(path.exists() for path in output_paths.values()),
            "refusing to overwrite M35C reference VCF")
    writers = {arm: path.open("wt", encoding="utf-8") for arm, path in output_paths.items()}
    source_samples: list[str] | None = None
    selected_indices: dict[str, list[int]] = {}
    sample_axes: dict[str, list[str]] = {}
    observed_loci: list[str] = []
    gt_count = {arm: 0 for arm in ARMS}
    try:
        with open_text(scaffold) as reader:
            for line_number, line in enumerate(reader, 1):
                if line.startswith("#CHROM"):
                    fields = line.rstrip("\n").split("\t")
                    source_samples = fields[9:]
                    require(len(source_samples) == len(set(source_samples)),
                            "M35C scaffold sample axis is duplicated")
                    for arm in ARMS:
                        require(selected[arm].issubset(source_samples),
                                f"M35C {arm} sample is absent from phased scaffold")
                        selected_indices[arm] = [index for index, sample in enumerate(source_samples)
                                                 if sample in selected[arm]]
                        sample_axes[arm] = [source_samples[index] for index in selected_indices[arm]]
                        require(len(sample_axes[arm]) == 69, f"M35C {arm} sample count differs")
                        writers[arm].write("\t".join([*fields[:9], *sample_axes[arm]]) + "\n")
                    continue
                if line.startswith("#"):
                    for writer in writers.values():
                        writer.write(line)
                    continue
                if not line.strip():
                    continue
                require(source_samples is not None, "M35C scaffold record precedes sample header")
                fields = line.rstrip("\n").split("\t")
                require(len(fields) == 9 + len(source_samples),
                        f"M35C malformed scaffold row {line_number}")
                locus = "\t".join((fields[0].removeprefix("chr"), fields[1],
                                    fields[3].upper(), fields[4].upper()))
                if locus not in target_set:
                    continue
                require(not observed_loci or int(fields[1]) > int(observed_loci[-1].split("\t")[1]),
                        "M35C scaffold marker order differs")
                observed_loci.append(locus)
                format_fields = fields[8].split(":")
                require("GT" in format_fields, "M35C scaffold lacks GT")
                gt_index = format_fields.index("GT")
                for arm in ARMS:
                    sample_fields = [fields[9 + index] for index in selected_indices[arm]]
                    for value in sample_fields:
                        parts = value.split(":")
                        require(gt_index < len(parts), "M35C scaffold genotype lacks GT")
                        alleles = parts[gt_index].split("|")
                        require(len(alleles) == 2 and all(allele in {"0", "1"} for allele in alleles)
                                and "/" not in parts[gt_index],
                                "M35C reference genotype is unphased, missing or non-biallelic")
                        gt_count[arm] += 1
                    writers[arm].write("\t".join([*fields[:9], *sample_fields]) + "\n")
    finally:
        for writer in writers.values():
            writer.close()
    require(observed_loci == target_loci, "M35C scaffold and target marker axes differ")
    result: dict[str, Any] = {}
    for arm in ARMS:
        counts = Counter(annotations[arm][sample][0] for sample in sample_axes[arm])
        require(dict(counts) == {"AFR": 23, "EUR": 23, "NAM": 23},
                f"M35C {arm} ancestry balance differs")
        result[arm] = {
            "vcf": output_paths[arm].name,
            "vcf_sha256": sha256_file(output_paths[arm]),
            "sample_count": len(sample_axes[arm]),
            "sample_axis_sha256": axis_sha256(sample_axes[arm]),
            "marker_count": len(observed_loci),
            "marker_axis_sha256": axis_sha256(observed_loci),
            "validated_phased_genotypes": gt_count[arm],
            "macro_counts": dict(sorted(counts.items())),
            "maps": write_maps(output_prefix, arm, sample_axes[arm], annotations[arm]),
        }
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract)
    require(args.selection_seed in contract["reference_design"]["selection_seeds"],
            "M35C selection seed is not preregistered")
    paths = {
        "roles": args.roles,
        "phased_scaffold_vcf": args.phased_scaffold_vcf,
        "target_vcf": args.target_vcf,
        "target_tbi": args.target_tbi,
        "m27d_manifest": args.m27d_manifest,
        "m27d_strata": args.m27d_strata,
        "m27d_training_set": args.m27d_training_set,
        "m27d_related_pairs": args.m27d_related_pairs,
        "m34_donor_audit": args.m34_donor_audit,
        "m34_mosaic_receipt": args.m34_mosaic_receipt,
    }
    verify_inputs(contract, paths)

    m27d_manifest = json.loads(args.m27d_manifest.read_text(encoding="utf-8"))
    require(m27d_manifest.get("stage") == "M27D_PASS0_PCRELATE" and
            m27d_manifest.get("params", {}).get("king_executed") is False,
            "M35C source was not authenticated as PC-Relate without KING")
    require(m27d_manifest["inputs"]["m27d_sample_strata.private.tsv"] ==
            contract["inputs"]["m27d_strata"]["sha256"] and
            m27d_manifest["sha256"]["m27d_pass0_training_set.private.txt"] ==
            contract["inputs"]["m27d_training_set"]["sha256"] and
            m27d_manifest["sha256"]["m27d_pass0_related_pairs.private.tsv.gz"] ==
            contract["inputs"]["m27d_related_pairs"]["sha256"],
            "M35C M27D manifest lineage differs")
    mosaic_receipt = json.loads(args.m34_mosaic_receipt.read_text(encoding="utf-8"))
    require(mosaic_receipt.get("stage") == "M34_NAM_EXPLORATORY_MOSAICS" and
            mosaic_receipt["parameters"]["rotation"] == 0 and
            mosaic_receipt["parameters"]["target_individuals"] == 32,
            "M35C R0 mosaic receipt differs")
    require(mosaic_receipt["inputs"]["phased_vcf"]["sha256"] ==
            contract["inputs"]["phased_scaffold_vcf"]["sha256"] and
            mosaic_receipt["outputs"]["m34_donor_audit.private.tsv"]["sha256"] ==
            contract["inputs"]["m34_donor_audit"]["sha256"],
            "M35C R0 donor/scaffold lineage differs")

    natwgs, relatedness_audit = derive_natwgs_candidates(
        contract, args.m27d_strata, args.m27d_training_set,
        args.m27d_related_pairs, args.m34_donor_audit,
    )
    roles = m35b.load_ref_train(args.roles)
    external_chosen, external_selection = m35b.deterministic_subset(
        roles, args.selection_seed, contract["reference_design"]["counts_per_arm"]["AFR"]
    )
    natwgs_chosen, natwgs_selection = select_natwgs(
        natwgs, args.selection_seed, contract["reference_design"]["counts_per_arm"]["NAM"]
    )
    shared_afr_eur = {
        sample for sample in external_chosen
        if m35b.ANCESTRY_MAP[roles[sample]["ancestry"]] in {"AFR", "EUR"}
    }
    external_nam = external_chosen - shared_afr_eur
    require(len(shared_afr_eur) == 46 and len(external_nam) == 23,
            "M35C external reference selection differs")
    require(external_nam.isdisjoint(natwgs_chosen),
            "M35C external and NatWGS NAM selections overlap")
    selected = {
        "EXTERNAL_NAM": set(external_chosen),
        "NATWGS": shared_afr_eur | natwgs_chosen,
    }
    require(selected["EXTERNAL_NAM"] & selected["NATWGS"] == shared_afr_eur,
            "M35C source arms differ outside the NAM block")
    annotations: dict[str, dict[str, tuple[str, str]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for sample in shared_afr_eur:
            annotations[arm][sample] = (
                m35b.ANCESTRY_MAP[roles[sample]["ancestry"]], roles[sample]["population"]
            )
    for sample in external_nam:
        annotations["EXTERNAL_NAM"][sample] = ("NAM", roles[sample]["population"])
    for sample in natwgs_chosen:
        annotations["NATWGS"][sample] = ("NAM", natwgs[sample]["Population"])

    target = m35b.scan_target(args.target_vcf, "22")
    require(len(target["loci"]) == contract["scope"]["marker_count"] and
            target["marker_axis_sha256"] == contract["scope"]["marker_axis_sha256"],
            "M35C target marker axis differs")
    require(all(samples.isdisjoint(target["samples"]) for samples in selected.values()),
            "M35C reference arm overlaps target samples")
    references = materialize_reference_arms(
        args.phased_scaffold_vcf, target["loci"], args.output_prefix, selected, annotations
    )
    for arm in ARMS:
        selected_path = args.output_prefix.with_name(
            f"{args.output_prefix.name}.{arm.lower()}.selected_samples.txt"
        )
        selected_path.write_text("".join(f"{sample}\n" for sample in
                                         sorted(selected[arm])), encoding="utf-8")
        references[arm]["selected_samples_file"] = selected_path.name
        references[arm]["selected_samples_file_sha256"] = sha256_file(selected_path)

    receipt = {
        "schema_version": "1.0.0",
        "stage": "M35C_MATCHED_SOURCE_REFERENCE_PREPARATION",
        "status": "PASS_MATCHED_EXTERNAL_NAM_VS_NATWGS_23_23_23",
        "selection_seed": args.selection_seed,
        "claim_level": "exploratory",
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "relatedness_audit": relatedness_audit,
        "selection": {
            "external_reference": external_selection,
            "natwgs_reference": natwgs_selection,
            "shared_AFR_EUR_count": len(shared_afr_eur),
            "shared_AFR_EUR_axis_sha256": axis_sha256(sorted(shared_afr_eur)),
            "external_NAM_axis_sha256": axis_sha256(sorted(external_nam)),
            "natwgs_NAM_axis_sha256": axis_sha256(sorted(natwgs_chosen)),
            "same_AFR_EUR_within_selection_seed": True,
        },
        "target": {
            "sample_count": len(target["samples"]),
            "sample_axis_sha256": axis_sha256(target["samples"]),
            "marker_count": len(target["loci"]),
            "marker_axis_sha256": target["marker_axis_sha256"],
            "truth_opened": False,
        },
        "reference_arms": references,
        "valid_or_test_role_used_as_reference": False,
        "king_executed": False,
    }
    receipt_path = args.output_prefix.with_suffix(".prepare_receipt.json")
    require(not receipt_path.exists(), "refusing to overwrite M35C preparation receipt")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--phased-scaffold-vcf", type=Path, required=True)
    parser.add_argument("--target-vcf", type=Path, required=True)
    parser.add_argument("--target-tbi", type=Path, required=True)
    parser.add_argument("--m27d-manifest", type=Path, required=True)
    parser.add_argument("--m27d-strata", type=Path, required=True)
    parser.add_argument("--m27d-training-set", type=Path, required=True)
    parser.add_argument("--m27d-related-pairs", type=Path, required=True)
    parser.add_argument("--m34-donor-audit", type=Path, required=True)
    parser.add_argument("--m34-mosaic-receipt", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = prepare(parse_args())
    print(json.dumps({"status": result["status"], "seed": result["selection_seed"]}, sort_keys=True))
