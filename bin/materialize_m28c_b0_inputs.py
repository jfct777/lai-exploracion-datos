#!/usr/bin/env python3
"""Materialize leakage-safe B0 reference and target VCFs for the M28C smoke.

The command deliberately has no truth input. Marker selection is already frozen by
M28B-v4; this stage only projects those sites from the authenticated tree sequence.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


@dataclass(frozen=True)
class Segment:
    start_bp: int
    end_bp: int
    ancestry: str
    donor_node: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                yield text


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("stage") != "M28C_B0_INPUT_PREFLIGHT":
        raise ValueError("Unexpected preregistration stage")
    if contract.get("scope") != "technical_smoke_only_no_LAI_no_effect_estimation":
        raise ValueError("Unexpected preregistration scope")
    return contract


def verify_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed}")
    return observed


def read_b0_markers(path: Path, expected_count: int) -> list[tuple[int, int]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"site_id", "chrom", "position"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("B0 table is missing required columns")
    markers = [(int(row["site_id"]), int(row["position"])) for row in rows]
    if len(markers) != expected_count:
        raise ValueError(f"Expected {expected_count} B0 markers, observed {len(markers)}")
    if len({site for site, _ in markers}) != len(markers):
        raise ValueError("B0 site IDs are not unique")
    if len({position for _, position in markers}) != len(markers):
        raise ValueError("B0 positions are not unique")
    if markers != sorted(markers, key=lambda row: row[1]):
        raise ValueError("B0 markers are not ordered by genomic position")
    return markers


def read_pool_manifest(
    path: Path, ancestries: list[str]
) -> tuple[dict[str, list[int]], dict[int, tuple[str, str]], dict[int, int]]:
    nodes = {ancestry: [] for ancestry in ancestries}
    roles: dict[int, tuple[str, str]] = {}
    node_individuals: dict[int, int] = {}
    individual_assignments: dict[int, tuple[str, str, list[int]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or [])
        legacy_columns = {"role", "ancestry", "node_id", "haplotype_sha256"}
        individual_columns = {
            "role", "ancestry", "individual_id", "node_id", "node_identity_sha256"
        }
        if columns not in (legacy_columns, individual_columns):
            raise ValueError("Unexpected pool-manifest columns")
        for row in reader:
            node = int(row["node_id"])
            ancestry = row["ancestry"]
            role = row["role"]
            if ancestry not in nodes:
                raise ValueError(f"Unexpected pool ancestry: {ancestry}")
            if node in roles:
                raise ValueError(f"Source node {node} appears in more than one pool")
            roles[node] = (role, ancestry)
            if columns == individual_columns:
                individual = int(row["individual_id"])
                node_individuals[node] = individual
                assigned = individual_assignments.setdefault(
                    individual, (role, ancestry, [])
                )
                if assigned[:2] != (role, ancestry):
                    raise ValueError(
                        f"Source individual {individual} crosses roles or ancestries"
                    )
                assigned[2].append(node)
            if row["role"] == "REF_LAI":
                nodes[ancestry].append(node)
    if individual_assignments:
        incomplete = {
            individual: values[2]
            for individual, values in individual_assignments.items()
            if len(values[2]) != 2 or len(set(values[2])) != 2
        }
        if incomplete:
            raise ValueError(
                "Pool manifest does not keep exactly two unique nodes per individual: "
                f"{list(incomplete)[:5]}"
            )
    return nodes, roles, node_individuals


def pair_reference_haplotypes(
    nodes: dict[str, list[int]], expected_haplotypes: int,
    node_individuals: dict[int, int] | None = None,
) -> list[tuple[str, str, int, int]]:
    pairs: list[tuple[str, str, int, int]] = []
    observed: list[int] = []
    for ancestry, ancestry_nodes in nodes.items():
        ordered = sorted(ancestry_nodes)
        if len(ordered) != expected_haplotypes or len(ordered) % 2:
            raise ValueError(
                f"Expected {expected_haplotypes} REF_LAI haplotypes for {ancestry}, "
                f"observed {len(ordered)}"
            )
        if node_individuals:
            by_individual: dict[int, list[int]] = {}
            for node in ordered:
                if node not in node_individuals:
                    raise ValueError(f"REF_LAI node {node} has no source individual")
                by_individual.setdefault(node_individuals[node], []).append(node)
            malformed = {
                individual: values
                for individual, values in by_individual.items()
                if len(values) != 2
            }
            if malformed:
                raise ValueError(
                    f"REF_LAI individuals do not have two homologues: {list(malformed)[:5]}"
                )
            ancestry_pairs = [
                tuple(sorted(values))
                for _, values in sorted(by_individual.items())
            ]
        else:
            ancestry_pairs = [
                (ordered[index], ordered[index + 1])
                for index in range(0, len(ordered), 2)
            ]
        for index, (left, right) in enumerate(ancestry_pairs):
            sample = f"REF_{ancestry}_{index:03d}"
            pairs.append((sample, ancestry, left, right))
            observed.extend((left, right))
    if len(observed) != len(set(observed)):
        raise ValueError("A REF_LAI haplotype was reused")
    return pairs


def read_mosaics(path: Path) -> dict[str, list[Segment]]:
    mosaics: dict[str, list[Segment]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = row["target_haplotype"]
            mosaics.setdefault(name, []).append(
                Segment(
                    int(row["start_bp"]),
                    int(row["end_bp_exclusive"]),
                    row["ancestry"],
                    int(row["donor_node"]),
                )
            )
    for name, segments in mosaics.items():
        if segments != sorted(segments, key=lambda segment: segment.start_bp):
            raise ValueError(f"Mosaic segments are not ordered for {name}")
        if any(left.end_bp != right.start_bp for left, right in zip(segments, segments[1:])):
            raise ValueError(f"Mosaic segments contain a gap or overlap for {name}")
        if any(segment.start_bp >= segment.end_bp for segment in segments):
            raise ValueError(f"Mosaic segment is empty for {name}")
    return mosaics


def pair_target_haplotypes(
    mosaics: dict[str, list[Segment]], expected_haplotypes: int
) -> list[tuple[str, str, str]]:
    expected = [f"T{index // 2:03d}_h{index % 2}" for index in range(expected_haplotypes)]
    if sorted(mosaics) != expected:
        missing = sorted(set(expected) - set(mosaics))
        extra = sorted(set(mosaics) - set(expected))
        raise ValueError(f"Unexpected target haplotypes; missing={missing}, extra={extra}")
    return [(f"T{index:03d}", f"T{index:03d}_h0", f"T{index:03d}_h1") for index in range(expected_haplotypes // 2)]


def segment_node_at(segments: list[Segment], position: int, pointer: int) -> tuple[int, int]:
    while pointer + 1 < len(segments) and position >= segments[pointer].end_bp:
        pointer += 1
    segment = segments[pointer]
    if not segment.start_bp <= position < segment.end_bp:
        raise ValueError(f"Position {position} is not covered by a donor segment")
    return segment.donor_node, pointer


def write_vcf_header(handle: TextIO, chromosome: str, length: int, samples: list[str]) -> None:
    handle.write("##fileformat=VCFv4.2\n")
    handle.write(f"##contig=<ID={chromosome},length={length}>\n")
    handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype">\n')
    handle.write('##INFO=<ID=TSID,Number=1,Type=Integer,Description="Authenticated tree-sequence site ID">\n')
    handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
    handle.write("\t".join(samples) + "\n")


def write_private_reference_pairs(
    path: Path, pairs: list[tuple[str, str, int, int]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("sample_id", "ancestry", "haplotype_0_node", "haplotype_1_node"))
        writer.writerows(pairs)
    os.chmod(path, 0o600)


def write_sample_map(path: Path, pairs: list[tuple[str, str, int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for sample, ancestry, _, _ in pairs:
            writer.writerow((sample, ancestry))


def audit_vcf(path: Path, expected_samples: int) -> dict:
    sample_count = None
    positions: list[int] = []
    site_ids: list[int] = []
    all_phased_binary = True
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                sample_count = len(fields[9:])
                continue
            if sample_count is None or len(fields) != 9 + sample_count:
                raise ValueError(f"Malformed VCF row in {path.name}")
            if fields[0] != "22" or fields[3:5] != ["A", "C"] or fields[8] != "GT":
                all_phased_binary = False
            positions.append(int(fields[1]))
            if not fields[2].startswith("m28s"):
                raise ValueError(f"Malformed marker ID in {path.name}")
            site_ids.append(int(fields[2][4:]))
            all_phased_binary &= all(gt in {"0|0", "0|1", "1|0", "1|1"} for gt in fields[9:])
    if sample_count != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples in {path.name}, observed {sample_count}"
        )
    return {
        "sample_count": sample_count,
        "record_count": len(positions),
        "positions": positions,
        "site_ids": site_ids,
        "ordered_unique": positions == sorted(positions) and len(positions) == len(set(positions)),
        "all_phased_binary": all_phased_binary,
    }


def materialize(args: argparse.Namespace) -> dict:
    import tskit

    contract = load_contract(args.preregistration)
    expected = contract["expected"]
    authenticated = contract["inputs"]
    input_hashes = {
        "tree_sequence": verify_hash(args.tree_sequence, authenticated["tree_sequence_sha256"], "tree sequence"),
        "pool_manifest": verify_hash(args.pool_manifest, authenticated["pool_manifest_sha256"], "pool manifest"),
        "mosaic_events": verify_hash(args.mosaic_events, authenticated["mosaic_events_sha256"], "mosaic events"),
        "b0_markers": verify_hash(args.b0_markers, authenticated["b0_markers_sha256"], "B0 markers"),
    }
    markers = read_b0_markers(args.b0_markers, int(expected["b0_markers"]))
    marker_by_site = dict(markers)
    marker_sites = set(marker_by_site)
    ancestries = list(contract["ancestries"])
    reference_nodes, pool_roles, node_individuals = read_pool_manifest(
        args.pool_manifest, ancestries
    )
    reference_pairs = pair_reference_haplotypes(
        reference_nodes,
        int(expected["reference_haplotypes_per_ancestry"]),
        node_individuals,
    )
    mosaics = read_mosaics(args.mosaic_events)
    target_pairs = pair_target_haplotypes(mosaics, int(expected["target_haplotypes"]))

    reference_flat = [node for _, _, left, right in reference_pairs for node in (left, right)]
    donor_nodes = sorted({segment.donor_node for segments in mosaics.values() for segment in segments})
    donor_occurrences = [segment.donor_node for segments in mosaics.values() for segment in segments]
    if len(donor_occurrences) != len(set(donor_occurrences)):
        raise ValueError("A DONOR node is reused across mosaic segments")
    if set(reference_flat) & set(donor_nodes):
        raise ValueError("REF_LAI and DONOR nodes overlap")
    invalid_donors = [
        segment.donor_node
        for segments in mosaics.values()
        for segment in segments
        if pool_roles.get(segment.donor_node) != ("DONOR", segment.ancestry)
    ]
    if invalid_donors:
        raise ValueError(f"Mosaic DONOR role or ancestry mismatch: {invalid_donors[:5]}")
    requested_nodes = reference_flat + donor_nodes
    if len(requested_nodes) != len(set(requested_nodes)):
        raise ValueError("Requested source nodes contain duplicates")
    node_column = {node: index for index, node in enumerate(requested_nodes)}

    args.outdir.mkdir(parents=True, exist_ok=False)
    ref_vcf = args.outdir / "m28c_b0_reference.vcf.gz"
    target_vcf = args.outdir / "m28c_b0_target.vcf.gz"
    sample_map = args.outdir / "m28c_b0_reference.sample_map.tsv"
    private_pairs = args.outdir / "m28c_b0_reference_pairs.private.tsv"
    write_sample_map(sample_map, reference_pairs)
    write_private_reference_pairs(private_pairs, reference_pairs)

    chromosome = str(contract["chromosome"])
    start_bp = int(contract["coordinate_contract"]["tree_sequence_start_bp"])
    ref_samples = [row[0] for row in reference_pairs]
    target_samples = [row[0] for row in target_pairs]
    pointers = {name: 0 for name in mosaics}
    observed_sites: list[int] = []
    ts = tskit.load(str(args.tree_sequence))
    expected_start = int(contract["coordinate_contract"]["tree_sequence_start_bp"])
    expected_end = int(contract["coordinate_contract"]["tree_sequence_end_bp_exclusive"])
    if start_bp != expected_start or start_bp + int(ts.sequence_length) != expected_end:
        raise ValueError("Tree-sequence interval does not match the coordinate contract")
    for name, segments in mosaics.items():
        if segments[0].start_bp != expected_start or segments[-1].end_bp != expected_end:
            raise ValueError(f"Mosaic does not cover the full simulated interval for {name}")
    with deterministic_gzip_text(ref_vcf) as ref_handle, deterministic_gzip_text(target_vcf) as target_handle:
        contig_length = int(contract["coordinate_contract"]["hg38_chr22_length_bp"])
        write_vcf_header(ref_handle, chromosome, contig_length, ref_samples)
        write_vcf_header(target_handle, chromosome, contig_length, target_samples)
        for variant in ts.variants(samples=requested_nodes, copy=False):
            site_id = int(variant.site.id)
            if site_id not in marker_sites:
                continue
            local_position = float(variant.site.position)
            if not local_position.is_integer():
                raise ValueError(f"Non-integer tree-sequence position at site {site_id}")
            genomic_position = start_bp + int(local_position)
            if genomic_position != marker_by_site[site_id]:
                raise ValueError(f"Position mismatch at site {site_id}")
            genotypes = variant.genotypes
            if any(int(value) not in (0, 1) for value in genotypes):
                raise ValueError(f"Non-binary or missing genotype at site {site_id}")

            ref_gt = [
                f"{int(genotypes[node_column[left]])}|{int(genotypes[node_column[right]])}"
                for _, _, left, right in reference_pairs
            ]
            target_gt: list[str] = []
            for _, left_name, right_name in target_pairs:
                left_node, pointers[left_name] = segment_node_at(
                    mosaics[left_name], genomic_position, pointers[left_name]
                )
                right_node, pointers[right_name] = segment_node_at(
                    mosaics[right_name], genomic_position, pointers[right_name]
                )
                target_gt.append(
                    f"{int(genotypes[node_column[left_node]])}|{int(genotypes[node_column[right_node]])}"
                )
            fields = [chromosome, str(genomic_position), f"m28s{site_id}", "A", "C", ".", "PASS", f"TSID={site_id}", "GT"]
            ref_handle.write("\t".join([*fields, *ref_gt]) + "\n")
            target_handle.write("\t".join([*fields, *target_gt]) + "\n")
            observed_sites.append(site_id)

    if observed_sites != [site for site, _ in markers]:
        missing = sorted(marker_sites - set(observed_sites))
        extra = sorted(set(observed_sites) - marker_sites)
        raise ValueError(f"Materialized B0 differs from frozen B0; missing={missing[:5]}, extra={extra[:5]}")

    ref_audit = audit_vcf(ref_vcf, len(reference_pairs))
    target_audit = audit_vcf(target_vcf, len(target_pairs))
    vcf_parity = (
        ref_audit["site_ids"] == target_audit["site_ids"] == observed_sites
        and ref_audit["positions"] == target_audit["positions"] == [position for _, position in markers]
        and ref_audit["all_phased_binary"]
        and target_audit["all_phased_binary"]
        and ref_audit["ordered_unique"]
        and target_audit["ordered_unique"]
    )

    output_hashes = {
        path.name: sha256(path)
        for path in (ref_vcf, target_vcf, sample_map, private_pairs)
    }
    ancestry_counts = {
        ancestry: sum(row[1] == ancestry for row in reference_pairs) for ancestry in ancestries
    }
    gates = {
        "I0_HASHES": True,
        "I1_B0_CARDINALITY": len(observed_sites) == int(expected["b0_markers"]),
        "I2_ROLE_BOUNDARY": not bool(set(reference_flat) & set(donor_nodes)) and not invalid_donors,
        "I3_REFERENCE": ancestry_counts == {
            ancestry: int(expected["reference_pseudodiploids_per_ancestry"])
            for ancestry in ancestries
        },
        "I4_TARGET": len(target_pairs) == int(expected["target_diploids"]),
        "I5_INTERNAL_VCF": vcf_parity,
        "I6_SCOPE": True,
    }
    decision = contract["decision"]["pass"] if all(gates.values()) else contract["decision"]["fail"]
    report = {
        "stage": contract["stage"],
        "scope": contract["scope"],
        "root_seed": contract["root_seed"],
        "seed_role": contract["seed_role"],
        "contract_sha256": sha256(args.preregistration),
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "counts": {
            "b0_markers": len(observed_sites),
            "reference_haplotypes": len(reference_flat),
            "reference_pseudodiploids": len(reference_pairs),
            "reference_pseudodiploids_by_ancestry": ancestry_counts,
            "target_haplotypes": len(mosaics),
            "target_diploids": len(target_pairs),
            "unique_donor_nodes_used": len(donor_nodes),
        },
        "vcf_audit": {
            "reference": {key: value for key, value in ref_audit.items() if key not in {"positions", "site_ids"}},
            "target": {key: value for key, value in target_audit.items() if key not in {"positions", "site_ids"}},
            "exact_marker_parity": vcf_parity,
        },
        "allele_encoding": contract["coordinate_contract"]["note"],
        "merged_truth_table_accessed": False,
        "generative_mosaic_events_accessed": True,
        "event_ancestry_used_for_model_or_selection": False,
        "lai_executed": False,
        "gates": gates,
        "decision": decision,
    }
    report_path = args.outdir / "m28c_b0_input_preflight.public.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(private_pairs, 0o600)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-sequence", required=True, type=Path)
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--mosaic-events", required=True, type=Path)
    parser.add_argument("--b0-markers", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    report = materialize(parse_args())
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
