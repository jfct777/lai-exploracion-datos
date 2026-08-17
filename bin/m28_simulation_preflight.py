#!/usr/bin/env python3
"""Generate and audit a leakage-free chr22 LAI simulation preflight.

M28 does not run LAI and does not estimate an effect.  It creates independent
frequency, reference and mosaic-donor pools, constructs phased target mosaics with
exact local-ancestry truth, and checks whether the preregistered rare-allele channel
is observable before a later comparator is designed.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


@dataclass(frozen=True)
class GeneticMap:
    chrom: str
    bp: tuple[int, ...]
    cm: tuple[float, ...]

    @property
    def start_bp(self) -> int:
        return self.bp[0]

    @property
    def end_bp(self) -> int:
        return self.bp[-1]

    @property
    def length_bp(self) -> int:
        return self.end_bp - self.start_bp + 1

    @property
    def span_cm(self) -> float:
        return self.cm[-1] - self.cm[0]

    def cm_to_offset(self, value: float) -> int:
        """Interpolate cM to a zero-based, half-open sequence coordinate."""
        if not self.cm[0] <= value <= self.cm[-1]:
            raise ValueError(f"cM {value} lies outside the map")
        right = bisect.bisect_right(self.cm, value)
        if right == 0:
            return 0
        if right >= len(self.cm):
            return self.length_bp - 1
        left = right - 1
        cm0, cm1 = self.cm[left], self.cm[right]
        bp0, bp1 = self.bp[left], self.bp[right]
        if cm1 == cm0:
            genomic_bp = bp0
        else:
            fraction = (value - cm0) / (cm1 - cm0)
            genomic_bp = round(bp0 + fraction * (bp1 - bp0))
        return min(max(genomic_bp - self.start_bp, 0), self.length_bp - 1)


@dataclass(frozen=True)
class MosaicSegment:
    target_haplotype: str
    start: int
    end: int
    ancestry: str
    donor_node: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("stage") != "M28_LAI_SIMULATION_PREFLIGHT":
        raise ValueError("The preregistration is not the M28 preflight contract")
    return contract


def read_genetic_map(path: Path, contract: dict, verify_hash: bool = True) -> GeneticMap:
    region = contract["region"]
    if verify_hash and sha256(path) != region["map_sha256"]:
        raise ValueError("Genetic-map sha256 does not match the preregistration")
    chroms: list[str] = []
    bp: list[int] = []
    cm: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"Map line {line_number} does not have three fields")
            chroms.append(fields[0].removeprefix("chr"))
            bp.append(int(fields[1]))
            cm.append(float(fields[2]))
    if len(bp) < 2:
        raise ValueError("Genetic map needs at least two rows")
    if set(chroms) != {region["chrom"].removeprefix("chr")}:
        raise ValueError("Genetic map chromosome does not match the contract")
    if any(left >= right for left, right in zip(bp, bp[1:])):
        raise ValueError("Genetic-map positions are not strictly increasing")
    if any(left > right for left, right in zip(cm, cm[1:])):
        raise ValueError("Genetic-map cM values decrease")
    result = GeneticMap(region["chrom"], tuple(bp), tuple(cm))
    expected = (
        region["start_bp"], region["end_bp"], region["length_bp_inclusive"]
    )
    observed = (result.start_bp, result.end_bp, result.length_bp)
    if observed != expected or not math.isclose(result.span_cm, region["span_cm"], abs_tol=1e-9):
        raise ValueError(f"Genetic-map extent mismatch: observed {observed}")
    return result


def minor_allele_stats(genotypes: list[int] | tuple[int, ...]) -> dict:
    called = [int(value) for value in genotypes if int(value) >= 0]
    if not called:
        raise ValueError("Cannot estimate allele frequency without called haplotypes")
    if any(value not in (0, 1) for value in called):
        raise ValueError("M28 expects binary haploid genotypes")
    derived_ac = sum(called)
    an = len(called)
    ancestral_ac = an - derived_ac
    minor_code = 1 if derived_ac <= ancestral_ac else 0
    mac = min(derived_ac, ancestral_ac)
    return {
        "an": an,
        "derived_ac": derived_ac,
        "minor_code": minor_code,
        "mac": mac,
        "maf": mac / an,
    }


def rare_under_contract(stats: dict, contract: dict) -> bool:
    definition = contract["rare_definition"]
    return (
        stats["mac"] >= definition["minimum_mac"]
        and stats["maf"] < definition["maximum_maf_exclusive"]
    )


def derive_seeds(root_seed: int) -> dict[str, int]:
    import numpy as np

    names = ("ancestry", "mutation", "pool", "mosaic")
    children = np.random.SeedSequence(root_seed).spawn(len(names))
    return {
        name: int(child.generate_state(1, dtype="uint32")[0])
        for name, child in zip(names, children)
    }


def msprime_rate_map(genetic_map: GeneticMap):
    import msprime

    positions = [value - genetic_map.start_bp for value in genetic_map.bp]
    morgans = [(value - genetic_map.cm[0]) / 100.0 for value in genetic_map.cm]
    rates = [
        (m1 - m0) / (p1 - p0)
        for p0, p1, m0, m1 in zip(positions, positions[1:], morgans, morgans[1:])
    ]
    # The last map marker is the final included base. Extend its last local rate
    # by one base so the simulated half-open interval has the preregistered length.
    positions.append(genetic_map.length_bp)
    rates.append(rates[-1])
    return msprime.RateMap(position=positions, rate=rates)


def source_diploid_counts(contract: dict) -> dict[str, int]:
    freq = contract["pools"]["frequency_diploids"]
    ref = contract["pools"]["lai_reference_diploids_per_ancestry"]
    donor_haps = contract["pools"]["mosaic_donor_haplotypes_per_ancestry"]
    if donor_haps % 2:
        raise ValueError("The donor haplotype count must be even")
    return {
        ancestry: int(freq[ancestry]) + int(ref) + int(donor_haps) // 2
        for ancestry in contract["source_populations"]["labels"]
    }


def simulate_sources(genetic_map: GeneticMap, contract: dict, seeds: dict[str, int]):
    import msprime
    import stdpopsim

    if stdpopsim.__version__ != contract["software"]["stdpopsim"]:
        raise RuntimeError(f"stdpopsim version is {stdpopsim.__version__}")
    if msprime.__version__ != contract["software"]["msprime"]:
        raise RuntimeError(f"msprime version is {msprime.__version__}")
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("AmericanAdmixture_4B18")
    ancestry = msprime.sim_ancestry(
        samples=source_diploid_counts(contract),
        ploidy=2,
        demography=model.model,
        recombination_rate=msprime_rate_map(genetic_map),
        discrete_genome=True,
        model=[
            msprime.DiscreteTimeWrightFisher(duration=50),
            msprime.StandardCoalescent(),
        ],
        random_seed=seeds["ancestry"],
    )
    mutated = msprime.sim_mutations(
        ancestry,
        rate=contract["software"]["mutation_rate_per_bp_per_generation"],
        model=msprime.BinaryMutationModel(),
        random_seed=seeds["mutation"],
    )
    tables = mutated.dump_tables()
    tables.provenances.clear()
    return tables.tree_sequence()


def _population_individuals(ts, population_id: int) -> list[tuple[int, tuple[int, int]]]:
    """Return diploid source individuals and their two sample nodes."""
    result: list[tuple[int, tuple[int, int]]] = []
    sample_nodes = set(map(int, ts.samples(population=population_id)))
    for individual in ts.individuals():
        nodes = tuple(int(node) for node in individual.nodes if int(node) in sample_nodes)
        if not nodes:
            continue
        if len(nodes) != 2:
            raise ValueError(
                f"Source individual {individual.id} has {len(nodes)} sampled haplotypes; expected 2"
            )
        if any(ts.node(node).population != population_id for node in nodes):
            raise ValueError(f"Source individual {individual.id} crosses populations")
        node_times = {float(ts.node(node).time) for node in nodes}
        if node_times != {0.0}:
            raise ValueError(
                f"Source individual {individual.id} is not a present-day diploid sample"
            )
        result.append((int(individual.id), (nodes[0], nodes[1])))
    observed_nodes = [node for _, nodes in result for node in nodes]
    if set(observed_nodes) != sample_nodes or len(observed_nodes) != len(sample_nodes):
        raise ValueError("Could not map every source haplotype to one diploid individual")
    return result


def _allocate_individual_pools(
    ts, contract: dict, rng
) -> dict[str, dict[str, list[int]]]:
    """Allocate complete diploid individuals, never single homologues, to roles."""
    populations = {population.metadata["name"]: population.id for population in ts.populations()}
    pools = {name: {} for name in ("FREQ", "REF_LAI", "DONOR")}
    freq = contract["pools"]["frequency_diploids"]
    ref_diploids = int(contract["pools"]["lai_reference_diploids_per_ancestry"])
    donor_haps = int(contract["pools"]["mosaic_donor_haplotypes_per_ancestry"])
    if donor_haps % 2:
        raise ValueError("The donor haplotype count must be even")
    donor_diploids = donor_haps // 2

    for ancestry in contract["source_populations"]["labels"]:
        individuals = _population_individuals(ts, populations[ancestry])
        rng.shuffle(individuals)
        freq_count = int(freq[ancestry])
        expected = freq_count + ref_diploids + donor_diploids
        if len(individuals) != expected:
            raise ValueError(
                f"Expected {expected} {ancestry} individuals, observed {len(individuals)}"
            )
        role_individuals = {
            "FREQ": individuals[:freq_count],
            "REF_LAI": individuals[freq_count:freq_count + ref_diploids],
            "DONOR": individuals[freq_count + ref_diploids:],
        }
        for role, rows in role_individuals.items():
            pools[role][ancestry] = [node for _, nodes in rows for node in nodes]
    return pools


def _allocate_legacy_haplotype_pools(
    ts, contract: dict, rng
) -> dict[str, dict[str, list[int]]]:
    """Reproduce the version-1 technical smoke allocation exactly."""
    populations = {population.metadata["name"]: population.id for population in ts.populations()}
    pools = {name: {} for name in ("FREQ", "REF_LAI", "DONOR")}
    freq = contract["pools"]["frequency_diploids"]
    ref_haps = 2 * contract["pools"]["lai_reference_diploids_per_ancestry"]
    donor_haps = contract["pools"]["mosaic_donor_haplotypes_per_ancestry"]
    for ancestry in contract["source_populations"]["labels"]:
        nodes = list(map(int, ts.samples(population=populations[ancestry])))
        rng.shuffle(nodes)
        freq_haps = 2 * int(freq[ancestry])
        expected = freq_haps + ref_haps + donor_haps
        if len(nodes) != expected:
            raise ValueError(f"Expected {expected} {ancestry} nodes, observed {len(nodes)}")
        pools["FREQ"][ancestry] = nodes[:freq_haps]
        pools["REF_LAI"][ancestry] = nodes[freq_haps:freq_haps + ref_haps]
        pools["DONOR"][ancestry] = nodes[freq_haps + ref_haps:]
    return pools


def allocate_pools(ts, contract: dict, pool_seed: int) -> dict[str, dict[str, list[int]]]:
    import numpy as np

    rng = np.random.default_rng(pool_seed)
    allocation_unit = contract["pools"].get("allocation_unit")
    if allocation_unit is None and int(contract.get("version", 0)) == 1:
        pools = _allocate_legacy_haplotype_pools(ts, contract, rng)
    elif allocation_unit == "diploid_individual":
        pools = _allocate_individual_pools(ts, contract, rng)
    else:
        raise ValueError("Unsupported or missing pools.allocation_unit")
    flat = [node for role in pools.values() for nodes in role.values() for node in nodes]
    if len(flat) != len(set(flat)):
        raise ValueError("A source node was assigned to more than one pool")
    return pools


def audit_pool_disjunction(ts, pools: dict[str, dict[str, list[int]]]) -> dict:
    """Audit role overlap at both haplotype-node and diploid-individual levels."""
    node_roles: dict[int, str] = {}
    individual_roles: dict[int, set[str]] = {}
    role_individuals: dict[str, set[int]] = {role: set() for role in pools}
    for role, ancestry_pools in pools.items():
        for nodes in ancestry_pools.values():
            for node in nodes:
                if node in node_roles:
                    raise ValueError(f"Source node {node} appears in multiple roles")
                node_roles[node] = role
                individual = int(ts.node(node).individual)
                if individual < 0:
                    raise ValueError(f"Source node {node} has no individual")
                individual_roles.setdefault(individual, set()).add(role)
                role_individuals[role].add(individual)
    crossing = {
        individual: sorted(roles)
        for individual, roles in individual_roles.items()
        if len(roles) > 1
    }
    return {
        "source_nodes": len(node_roles),
        "source_individuals": len(individual_roles),
        "individuals_by_role": {
            role: len(individuals) for role, individuals in sorted(role_individuals.items())
        },
        "cross_role_individuals": len(crossing),
        "cross_role_examples": [
            {"individual_id": individual, "roles": roles}
            for individual, roles in sorted(crossing.items())[:10]
        ],
    }


def draw_mosaics(
    genetic_map: GeneticMap,
    donor_nodes: dict[str, list[int]],
    contract: dict,
    seed: int,
) -> list[list[MosaicSegment]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    labels = tuple(contract["source_populations"]["labels"])
    proportions = contract["source_populations"]["mixture_proportions"]
    probabilities = [proportions[label] for label in labels]
    available = {label: list(nodes) for label, nodes in donor_nodes.items()}
    for nodes in available.values():
        rng.shuffle(nodes)
    target_haps = contract["pools"]["target_haplotypes"]
    generations = contract["source_populations"]["pulse_generations"]
    mean_events = generations * genetic_map.span_cm / 100.0
    mosaics: list[list[MosaicSegment]] = []
    for hap_index in range(target_haps):
        name = f"T{hap_index // 2:03d}_h{hap_index % 2}"
        count = int(rng.poisson(mean_events))
        event_cm = sorted(rng.uniform(genetic_map.cm[0], genetic_map.cm[-1], count))
        offsets = sorted({genetic_map.cm_to_offset(float(value)) for value in event_cm})
        offsets = [value for value in offsets if 0 < value < genetic_map.length_bp]
        boundaries = [0, *offsets, genetic_map.length_bp]
        segments: list[MosaicSegment] = []
        for start, end in zip(boundaries, boundaries[1:]):
            ancestry = str(rng.choice(labels, p=probabilities))
            if not available[ancestry]:
                raise RuntimeError(f"DONOR_EXHAUSTED:{ancestry}")
            donor = int(available[ancestry].pop())
            segments.append(MosaicSegment(name, start, end, ancestry, donor))
        mosaics.append(segments)
    used = [segment.donor_node for segments in mosaics for segment in segments]
    if len(used) != len(set(used)):
        raise RuntimeError("DONOR_REUSE")
    return mosaics


def merge_truth(segments: list[MosaicSegment]) -> list[MosaicSegment]:
    merged: list[MosaicSegment] = []
    for segment in segments:
        if merged and merged[-1].ancestry == segment.ancestry:
            previous = merged[-1]
            merged[-1] = MosaicSegment(
                previous.target_haplotype,
                previous.start,
                segment.end,
                previous.ancestry,
                -1,
            )
        else:
            merged.append(
                MosaicSegment(
                    segment.target_haplotype,
                    segment.start,
                    segment.end,
                    segment.ancestry,
                    -1,
                )
            )
    return merged


def validate_segment_cover(segments: list[MosaicSegment], length: int) -> None:
    if not segments or segments[0].start != 0 or segments[-1].end != length:
        raise ValueError("Mosaic does not cover the complete interval")
    if any(left.end != right.start for left, right in zip(segments, segments[1:])):
        raise ValueError("Mosaic has a gap or overlap")
    if any(segment.start >= segment.end for segment in segments):
        raise ValueError("Mosaic contains an empty segment")


@contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                yield text


def write_segments(path: Path, mosaics: list[list[MosaicSegment]], genetic_map: GeneticMap) -> None:
    with deterministic_gzip_text(path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("target_haplotype", "chrom", "start_bp", "end_bp_exclusive", "ancestry", "donor_node"))
        for segments in mosaics:
            for segment in segments:
                writer.writerow((
                    segment.target_haplotype,
                    genetic_map.chrom,
                    genetic_map.start_bp + segment.start,
                    genetic_map.start_bp + segment.end,
                    segment.ancestry,
                    segment.donor_node,
                ))


def write_pool_manifest(path: Path, pools: dict[str, dict[str, list[int]]], ts) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("role", "ancestry", "individual_id", "node_id", "node_identity_sha256"))
        for role in sorted(pools):
            for ancestry in sorted(pools[role]):
                for node in pools[role][ancestry]:
                    individual = int(ts.node(node).individual)
                    node_identity = hashlib.sha256(f"source-node:{node}".encode()).hexdigest()
                    writer.writerow((role, ancestry, individual, node, node_identity))


def segment_at(segments: list[MosaicSegment], offset: int, pointer: int) -> tuple[MosaicSegment, int]:
    while pointer + 1 < len(segments) and offset >= segments[pointer].end:
        pointer += 1
    segment = segments[pointer]
    if not segment.start <= offset < segment.end:
        raise ValueError("Variant position is outside its mosaic segment")
    return segment, pointer


def audit_rare_exposure(ts, pools, mosaics, genetic_map, contract, outdir: Path) -> dict:
    sample_index = {int(node): index for index, node in enumerate(ts.samples())}
    freq_nodes = [node for ancestry in contract["source_populations"]["labels"] for node in pools["FREQ"][ancestry]]
    target_names = [segments[0].target_haplotype for segments in mosaics]
    pointers = [0] * len(mosaics)
    truth = [merge_truth(segments) for segments in mosaics]
    truth_pointers = [0] * len(truth)
    tract_counts = [[0] * len(segments) for segments in truth]
    mac_histogram: dict[int, int] = {}
    rare_sites = 0
    ref_supported = {ancestry: 0 for ancestry in contract["source_populations"]["labels"]}
    donor_supported = dict(ref_supported)
    matrix_path = outdir / "m28_rare_haplotypes.tsv.gz"
    catalog_path = outdir / "m28_rare_catalog.tsv.gz"
    with deterministic_gzip_text(matrix_path) as matrix_handle, deterministic_gzip_text(catalog_path) as catalog_handle:
        matrix = csv.writer(matrix_handle, delimiter="\t", lineterminator="\n")
        catalog = csv.writer(catalog_handle, delimiter="\t", lineterminator="\n")
        matrix.writerow(("chrom", "position", "minor_code", *target_names))
        catalog.writerow(("chrom", "position", "minor_code", "mac", "an", "maf", "ref_AFR", "ref_EUR", "ref_ASIA", "donor_AFR", "donor_EUR", "donor_ASIA"))
        for variant in ts.variants():
            stats = minor_allele_stats([int(variant.genotypes[sample_index[node]]) for node in freq_nodes])
            if not rare_under_contract(stats, contract):
                continue
            rare_sites += 1
            mac_histogram[stats["mac"]] = mac_histogram.get(stats["mac"], 0) + 1
            minor = stats["minor_code"]
            ref_counts = {}
            donor_counts = {}
            for ancestry in contract["source_populations"]["labels"]:
                ref_counts[ancestry] = sum(
                    int(variant.genotypes[sample_index[node]]) == minor
                    for node in pools["REF_LAI"][ancestry]
                )
                donor_counts[ancestry] = sum(
                    int(variant.genotypes[sample_index[node]]) == minor
                    for node in pools["DONOR"][ancestry]
                )
                if ref_counts[ancestry] > 0:
                    ref_supported[ancestry] += 1
                if donor_counts[ancestry] > 0:
                    donor_supported[ancestry] += 1
            offset = int(variant.site.position)
            target_values: list[int] = []
            for index, segments in enumerate(mosaics):
                event_segment, pointers[index] = segment_at(segments, offset, pointers[index])
                value = int(variant.genotypes[sample_index[event_segment.donor_node]])
                target_values.append(value)
                truth_segment, truth_pointers[index] = segment_at(truth[index], offset, truth_pointers[index])
                if value == minor and ref_counts[truth_segment.ancestry] > 0:
                    tract_counts[index][truth_pointers[index]] += 1
            genomic_position = genetic_map.start_bp + offset
            matrix.writerow((genetic_map.chrom, genomic_position, minor, *target_values))
            catalog.writerow((
                genetic_map.chrom, genomic_position, minor, stats["mac"], stats["an"],
                f"{stats['maf']:.12g}", ref_counts["AFR"], ref_counts["EUR"],
                ref_counts["ASIA"], donor_counts["AFR"], donor_counts["EUR"],
                donor_counts["ASIA"],
            ))
    by_ancestry = {ancestry: [] for ancestry in contract["source_populations"]["labels"]}
    for hap_truth, counts in zip(truth, tract_counts):
        for segment, count in zip(hap_truth, counts):
            by_ancestry[segment.ancestry].append(count)
    medians = {
        ancestry: statistics.median(values) if values else 0
        for ancestry, values in by_ancestry.items()
    }
    transitions = {
        ancestry: sum(
            1
            for segments in truth
            for left, right in zip(segments, segments[1:])
            if ancestry in (left.ancestry, right.ancestry)
        )
        for ancestry in by_ancestry
    }
    return {
        "rare_sites": rare_sites,
        "mac_histogram": {str(key): value for key, value in sorted(mac_histogram.items())},
        "ref_supported_sites": ref_supported,
        "donor_supported_sites": donor_supported,
        "median_observable_rare_copies_per_truth_tract": medians,
        "truth_transitions_involving_ancestry": transitions,
        "truth_transitions_total": sum(len(segments) - 1 for segments in truth),
        "silent_donor_switches_total": sum(
            1
            for segments in mosaics
            for left, right in zip(segments, segments[1:])
            if left.ancestry == right.ancestry
        ),
        "target_haplotypes": len(mosaics),
    }


def run(args: argparse.Namespace) -> dict:
    contract = load_contract(args.preregistration)
    if args.root_seed not in contract["seeds"]["root_seeds"]:
        raise ValueError("Root seed is not preregistered")
    genetic_map = read_genetic_map(args.genetic_map, contract)
    seeds = derive_seeds(args.root_seed)
    args.outdir.mkdir(parents=True, exist_ok=False)
    ts = simulate_sources(genetic_map, contract, seeds)
    pools = allocate_pools(ts, contract, seeds["pool"])
    disjunction = audit_pool_disjunction(ts, pools)
    allocation_unit = contract["pools"].get("allocation_unit", "legacy_haplotype_node")
    if allocation_unit == "diploid_individual" and disjunction["cross_role_individuals"]:
        raise RuntimeError("INDIVIDUAL_ROLE_OVERLAP")
    mosaics = draw_mosaics(genetic_map, pools["DONOR"], contract, seeds["mosaic"])
    for segments in mosaics:
        validate_segment_cover(segments, genetic_map.length_bp)
        validate_segment_cover(merge_truth(segments), genetic_map.length_bp)
    ts_path = args.outdir / "m28_sources.trees"
    ts.dump(ts_path)
    pool_path = args.outdir / "m28_pools.private.tsv"
    write_pool_manifest(pool_path, pools, ts)
    write_segments(args.outdir / "m28_mosaic_events.private.tsv.gz", mosaics, genetic_map)
    truth = [merge_truth(segments) for segments in mosaics]
    write_segments(args.outdir / "m28_lai_truth.tsv.gz", truth, genetic_map)
    exposure = audit_rare_exposure(ts, pools, mosaics, genetic_map, contract, args.outdir)
    gates = {
        "S0_MAP": True,
        "S1_REPRODUCIBILITY": None,
        "S2_DISJUNCTION": (
            disjunction["cross_role_individuals"] == 0
            if allocation_unit == "diploid_individual"
            else True
        ),
        "S3_PHASE_AND_TRUTH": True,
        "S4_RARENESS": set(exposure["mac_histogram"]).issubset({"2", "3", "4", "5"}),
        "S5_EXPOSURE": all(value > 0 for value in exposure["median_observable_rare_copies_per_truth_tract"].values()),
        "S6_SCOPE": True,
    }
    decision = "GO_REPRODUCIBILITY_CHECK" if all(value is not False for value in gates.values()) else "STOP_PREFLIGHT"
    report = {
        "stage": contract["stage"],
        "contract_sha256": sha256(args.preregistration),
        "map_sha256": sha256(args.genetic_map),
        "root_seed": args.root_seed,
        "derived_seeds": seeds,
        "software": contract["software"],
        "source_diploid_counts": source_diploid_counts(contract),
        "pool_allocation_unit": allocation_unit,
        "pool_disjunction": disjunction,
        "sequence_length": int(ts.sequence_length),
        "trees": ts.num_trees,
        "sites": ts.num_sites,
        "mutations": ts.num_mutations,
        "exposure": exposure,
        "gates": gates,
        "decision": decision,
        "scope": "technical_preflight_no_LAI_no_effect_estimation",
    }
    report_path = args.outdir / "m28_preflight.public.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--root-seed", required=True, type=int)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
