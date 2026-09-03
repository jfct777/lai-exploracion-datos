#!/usr/bin/env python3
"""Build a deterministic 25/25/25 reference subset without touching target samples."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TextIO


ANCESTRY_MAP = {
    "African": "AFR",
    "European": "EUR",
    "Native_American": "NAM",
}
ANCESTRY_ORDER = ("AFR", "EUR", "NAM")


class BalancedReferenceError(ValueError):
    """Raised when the balanced reference contract is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BalancedReferenceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def axis_sha256(rows: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def load_ref_train(roles_path: Path) -> dict[str, dict[str, str]]:
    with roles_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"sample_id", "ancestry", "population", "canonical_population", "role"}
    require(rows and required.issubset(rows[0]), "M35B role table lacks required columns")
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["role"] != "REF_TRAIN":
            continue
        sample = row["sample_id"]
        require(sample and sample not in selected, "M35B REF_TRAIN sample is empty or duplicated")
        require(row["ancestry"] in ANCESTRY_MAP, "M35B REF_TRAIN ancestry is unsupported")
        require(row["population"] and not any(char.isspace() for char in row["population"]),
                "M35B population label is empty or unsafe")
        selected[sample] = row
    require(selected, "M35B role table has no REF_TRAIN samples")
    return selected


def _hamilton_with_population_floor(populations: dict[str, list[str]], target: int,
                                    seed: int, ancestry: str) -> dict[str, int]:
    """Allocate samples proportionally while retaining every represented population."""
    require(target <= sum(map(len, populations.values())), "M35B allocation exceeds candidates")
    require(target >= len(populations),
            "M35B population-preserving Hamilton allocation lacks enough slots")
    allocation = {population: 1 for population in populations}
    remaining = target - len(populations)
    capacities = {population: len(samples) - 1 for population, samples in populations.items()}
    capacity_total = sum(capacities.values())
    require(remaining <= capacity_total, "M35B residual Hamilton allocation exceeds capacity")
    if remaining == 0:
        return allocation
    quotas = {population: remaining * capacities[population] / capacity_total
              for population in populations}
    floors = {population: int(quotas[population]) for population in populations}
    for population, count in floors.items():
        allocation[population] += count
    left = remaining - sum(floors.values())
    ranked = sorted(
        populations,
        key=lambda population: (
            -(quotas[population] - floors[population]),
            hashlib.sha256(f"M35B|{seed}|{ancestry}|{population}".encode("utf-8")).hexdigest(),
        ),
    )
    for population in ranked:
        if left == 0:
            break
        if allocation[population] < len(populations[population]):
            allocation[population] += 1
            left -= 1
    require(left == 0 and sum(allocation.values()) == target,
            "M35B Hamilton allocation did not reach its target")
    require(all(1 <= allocation[population] <= len(samples)
                for population, samples in populations.items()),
            "M35B Hamilton allocation violates population capacity")
    return allocation


def deterministic_subset(ref_train: dict[str, dict[str, str]], seed: int,
                         per_ancestry: int) -> tuple[set[str], dict[str, object]]:
    by_ancestry: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sample, row in ref_train.items():
        by_ancestry[ANCESTRY_MAP[row["ancestry"]]][row["population"]].append(sample)
    require(set(by_ancestry) == set(ANCESTRY_ORDER), "M35B REF_TRAIN lacks a macro-ancestry")
    chosen: set[str] = set()
    rank_hashes: dict[str, str] = {}
    allocations: dict[str, dict[str, int]] = {}
    for ancestry in ANCESTRY_ORDER:
        populations = by_ancestry[ancestry]
        candidates = [sample for samples in populations.values() for sample in samples]
        require(len(candidates) >= per_ancestry,
                f"M35B has fewer than {per_ancestry} {ancestry} REF_TRAIN samples")
        allocation = _hamilton_with_population_floor(populations, per_ancestry, seed, ancestry)
        allocations[ancestry] = dict(sorted(allocation.items()))
        selected: list[str] = []
        for population, samples in sorted(populations.items()):
            ranked = sorted(
                samples,
                key=lambda sample: hashlib.sha256(
                    f"M35B|{seed}|{ancestry}|{population}|{sample}".encode("utf-8")
                ).hexdigest(),
            )
            selected.extend(ranked[:allocation[population]])
        chosen.update(selected)
        rank_hashes[ancestry] = axis_sha256(selected)
    require(len(chosen) == per_ancestry * len(ANCESTRY_ORDER),
            "M35B balanced subset size differs")
    populations = {
        ancestry: dict(sorted(Counter(
            ref_train[sample]["population"] for sample in chosen
            if ANCESTRY_MAP[ref_train[sample]["ancestry"]] == ancestry
        ).items()))
        for ancestry in ANCESTRY_ORDER
    }
    return chosen, {
        "selection_method": "population_floor_then_capacity_weighted_Hamilton_with_sha256_ties",
        "selection_seed": seed,
        "per_ancestry": per_ancestry,
        "candidate_counts": {
            name: sum(len(samples) for samples in by_ancestry[name].values())
            for name in ANCESTRY_ORDER
        },
        "selected_counts": {name: per_ancestry for name in ANCESTRY_ORDER},
        "Hamilton_population_allocation": allocations,
        "population_counts": populations,
        "within_ancestry_rank_sha256": rank_hashes,
    }


def scan_target(path: Path, chromosome: str) -> dict[str, object]:
    samples: list[str] | None = None
    loci: list[str] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                require(samples is None and len(fields) > 9, "M35B target VCF sample header differs")
                samples = fields[9:]
                require(len(samples) == len(set(samples)), "M35B target samples are duplicated")
            elif line.startswith("#") or not line.strip():
                continue
            else:
                fields = line.rstrip("\n").split("\t")
                require(samples is not None and len(fields) == 9 + len(samples),
                        "M35B target VCF row width differs")
                require(fields[0].removeprefix("chr") == chromosome.removeprefix("chr"),
                        "M35B target chromosome differs")
                loci.append("\t".join((fields[0].removeprefix("chr"), fields[1],
                                        fields[3].upper(), fields[4].upper())))
    require(samples is not None and loci, "M35B target VCF is empty")
    return {"samples": samples, "loci": loci, "marker_axis_sha256": axis_sha256(loci)}


def subset_reference(reference_path: Path, output_vcf: Path, chosen: set[str],
                     ref_train: dict[str, dict[str, str]], chromosome: str) -> dict[str, object]:
    require(not output_vcf.exists(), "refusing to overwrite M35B reference VCF")
    source_samples: list[str] | None = None
    selected_samples: list[str] | None = None
    selected_indices: list[int] | None = None
    loci: list[str] = []
    with open_text(reference_path) as reader, output_vcf.open("wt", encoding="utf-8") as writer:
        for line in reader:
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                require(source_samples is None and len(fields) > 9, "M35B reference header differs")
                source_samples = fields[9:]
                require(len(source_samples) == len(set(source_samples)),
                        "M35B reference samples are duplicated")
                require(set(source_samples) == set(ref_train),
                        "M35B reference sample axis differs from REF_TRAIN")
                selected_indices = [index for index, sample in enumerate(source_samples) if sample in chosen]
                selected_samples = [source_samples[index] for index in selected_indices]
                require(len(selected_samples) == len(chosen), "M35B selected sample axis differs")
                writer.write("\t".join([*fields[:9], *selected_samples]) + "\n")
                continue
            if line.startswith("#"):
                writer.write(line)
                continue
            if not line.strip():
                continue
            require(source_samples is not None and selected_indices is not None,
                    "M35B reference record precedes sample header")
            fields = line.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(source_samples), "M35B reference row width differs")
            require(fields[0].removeprefix("chr") == chromosome.removeprefix("chr"),
                    "M35B reference chromosome differs")
            loci.append("\t".join((fields[0].removeprefix("chr"), fields[1],
                                    fields[3].upper(), fields[4].upper())))
            writer.write("\t".join([*fields[:9], *(fields[9 + index] for index in selected_indices)]) + "\n")
    require(selected_samples is not None and loci, "M35B reference VCF is empty")
    return {
        "samples": selected_samples,
        "sample_axis_sha256": axis_sha256(selected_samples),
        "marker_count": len(loci),
        "marker_axis_sha256": axis_sha256(loci),
    }


def write_panel_maps(prefix: Path, samples: list[str], ref_train: dict[str, dict[str, str]]) -> dict[str, object]:
    outputs: dict[str, object] = {}
    for granularity in ("coarse", "fine"):
        sample_map = prefix.with_name(f"{prefix.name}.{granularity}.sample_panel.tsv")
        macro_map = prefix.with_name(f"{prefix.name}.{granularity}.panel_macro.tsv")
        sample_rows: list[str] = []
        panel_to_macro: dict[str, str] = {}
        for sample in samples:
            row = ref_train[sample]
            macro = ANCESTRY_MAP[row["ancestry"]]
            panel = macro if granularity == "coarse" else row["population"]
            previous = panel_to_macro.setdefault(panel, macro)
            require(previous == macro, "M35B fine panel maps to multiple macro-ancestries")
            sample_rows.append(f"{sample}\t{panel}\n")
        sample_map.write_text("".join(sample_rows), encoding="utf-8")
        macro_map.write_text("".join(f"{panel}\t{macro}\n"
                                     for panel, macro in sorted(panel_to_macro.items())), encoding="utf-8")
        outputs[granularity] = {
            "sample_map": sample_map.name,
            "sample_map_sha256": sha256_file(sample_map),
            "panel_macro_map": macro_map.name,
            "panel_macro_map_sha256": sha256_file(macro_map),
            "panel_count": len(panel_to_macro),
        }
    return outputs


def prepare(args: argparse.Namespace) -> dict[str, object]:
    require(args.selection_seed >= 0, "M35B selection seed must be non-negative")
    require(args.per_ancestry > 0, "M35B per-ancestry count must be positive")
    require(args.expected_markers > 0, "M35B expected marker count must be positive")
    for path in (args.roles, args.reference_vcf, args.reference_tbi, args.target_vcf, args.target_tbi):
        require(path.is_file() and not path.is_symlink(), f"invalid M35B input: {path}")
    ref_train = load_ref_train(args.roles)
    chosen, selection = deterministic_subset(ref_train, args.selection_seed, args.per_ancestry)
    target = scan_target(args.target_vcf, args.chromosome)
    prefix = args.output_prefix
    output_vcf = prefix.with_suffix(".ref.vcf")
    reference = subset_reference(args.reference_vcf, output_vcf, chosen, ref_train, args.chromosome)
    require(reference["marker_count"] == args.expected_markers,
            "M35B reference marker count differs from the frozen spine")
    require(len(target["loci"]) == args.expected_markers and
            reference["marker_axis_sha256"] == target["marker_axis_sha256"],
            "M35B reference and unchanged target marker axes differ")
    require(set(reference["samples"]).isdisjoint(target["samples"]),
            "M35B reference overlaps unchanged R0 target")
    panels = write_panel_maps(prefix, reference["samples"], ref_train)
    selected_path = prefix.with_suffix(".selected_samples.txt")
    selected_path.write_text("".join(f"{sample}\n" for sample in reference["samples"]), encoding="utf-8")
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M35B_BALANCED_REFERENCE_PREPARATION",
        "status": "PASS_BALANCED_25_25_25",
        "chromosome": args.chromosome.removeprefix("chr"),
        "selection": selection,
        "source": {
            "roles_sha256": sha256_file(args.roles),
            "reference_vcf_sha256": sha256_file(args.reference_vcf),
            "reference_tbi_sha256": sha256_file(args.reference_tbi),
            "target_vcf_sha256": sha256_file(args.target_vcf),
            "target_tbi_sha256": sha256_file(args.target_tbi),
        },
        "reference": reference,
        "unchanged_target": {
            "sample_count": len(target["samples"]),
            "sample_axis_sha256": axis_sha256(target["samples"]),
            "marker_count": len(target["loci"]),
            "marker_axis_sha256": target["marker_axis_sha256"],
        },
        "selected_samples_sha256": sha256_file(selected_path),
        "panels": panels,
        "valid_or_test_role_used_as_reference": False,
    }
    receipt_path = prefix.with_suffix(".prepare_receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--reference-vcf", type=Path, required=True)
    parser.add_argument("--reference-tbi", type=Path, required=True)
    parser.add_argument("--target-vcf", type=Path, required=True)
    parser.add_argument("--target-tbi", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--per-ancestry", type=int, required=True)
    parser.add_argument("--expected-markers", type=int, required=True)
    parser.add_argument("--chromosome", default="22")
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = prepare(parse_args())
    print(json.dumps({"status": result["status"], "selection": result["selection"]}, sort_keys=True))
