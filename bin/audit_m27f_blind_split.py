#!/usr/bin/env python3
"""Freeze M27F roles from authenticated metadata and IBD, without reading a VCF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import audit_m27e_ibd_rare_transfer as m27e


ROLES = ("REF_TRAIN", "SOURCE_VALID", "SOURCE_TEST")
REMAINDER_ORDER = ("SOURCE_VALID", "SOURCE_TEST", "REF_TRAIN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibd-file", action="append", type=Path, required=True)
    parser.add_argument("--genetic-map", action="append", type=Path, required=True)
    parser.add_argument("--resolved-strata", type=Path, required=True)
    parser.add_argument("--resolved-strata-manifest", type=Path, required=True)
    parser.add_argument("--upstream-m27e-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def block_digest(members: Iterable[str]) -> str:
    payload = "\n".join(sorted(members)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_strata_sample_ids(path: Path) -> list[str]:
    """Read only the identity column needed to authenticate the complete strata table."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError("Resolved-strata table lacks sample_id")
        samples = [row["sample_id"] for row in reader]
    if not samples or len(samples) != len(set(samples)):
        raise ValueError("Resolved-strata sample identities are empty or duplicated")
    return samples


def validate_upstream_manifest(
    path: Path,
    preregistration: dict[str, object],
    inputs: Iterable[Path],
) -> dict[str, object]:
    """Authenticate the exact M27E IBD, map and strata bytes reused by M27F."""
    contract = preregistration["upstream_contract"]
    observed_manifest_hash = m27e.sha256_file(path)
    if observed_manifest_hash != contract["m27e_manifest_sha256"]:
        raise ValueError("M27E manifest SHA-256 differs from the preregistration")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != contract["m27e_manifest_stage"]:
        raise ValueError("Unexpected M27E manifest stage")
    if manifest.get("git_commit") != contract["m27e_generator_commit"]:
        raise ValueError("Unexpected M27E generator commit")
    expected_hashes = manifest.get("inputs", {})
    verified = {}
    paths = list(inputs)
    if len({item.name for item in paths}) != len(paths):
        raise ValueError("M27F inputs must have unambiguous basenames")
    for item in paths:
        expected = expected_hashes.get(item.name)
        observed = m27e.sha256_file(item)
        if expected is None or observed != expected:
            raise ValueError(f"Input is not authenticated by M27E: {item.name}")
        verified[item.name] = observed
    return {
        "m27e_manifest_sha256": observed_manifest_hash,
        "m27e_generator_commit": manifest["git_commit"],
        "n_authenticated_inputs": len(verified),
        "authenticated_input_set_sha256": hashlib.sha256(
            "".join(f"{name}\t{verified[name]}\n" for name in sorted(verified)).encode("utf-8")
        ).hexdigest(),
    }


def balanced_quotas(n_units: int, roles: tuple[str, ...] = ROLES) -> dict[str, int]:
    """Maximize the weakest role; allocate remainders to validation, test, then reference."""
    if n_units < 0 or set(roles) != set(REMAINDER_ORDER):
        raise ValueError("Invalid role contract")
    base, remaining = divmod(n_units, len(roles))
    quotas = {role: base for role in roles}
    for role in REMAINDER_ORDER[:remaining]:
        quotas[role] += 1
    return quotas


def assignment_totals(
    units: list[tuple[str, int, int]], assignment: dict[str, str], roles: tuple[str, ...]
) -> dict[str, dict[str, int]]:
    totals = {role: {"units": 0, "samples": 0, "populations": 0} for role in roles}
    for digest, samples, populations in units:
        role = assignment[digest]
        totals[role]["units"] += 1
        totals[role]["samples"] += samples
        totals[role]["populations"] += populations
    return totals


def exact_assignment_score(
    units: list[tuple[str, int, int]], assignment: dict[str, str], roles: tuple[str, ...]
) -> tuple[object, ...]:
    """Score one complete assignment using only frozen unit, population and sample counts."""
    totals = assignment_totals(units, assignment, roles)
    unit_counts = [totals[role]["units"] for role in roles]
    population_counts = [totals[role]["populations"] for role in roles]
    sample_counts = [totals[role]["samples"] for role in roles]
    sample_total = sum(sample_counts)
    assignment_text = "\n".join(f"{digest}\t{assignment[digest]}" for digest, *_ in sorted(units))
    return (
        max(unit_counts) - min(unit_counts),
        -min(unit_counts),
        totals["REF_TRAIN"]["units"],
        abs(totals["SOURCE_VALID"]["units"] - totals["SOURCE_TEST"]["units"]),
        max(population_counts) - min(population_counts),
        max(sample_counts) - min(sample_counts),
        sum((len(roles) * count - sample_total) ** 2 for count in sample_counts),
        hashlib.sha256((assignment_text + "\n").encode("utf-8")).hexdigest(),
    )


def exhaustive_assignment(
    units: list[tuple[str, int, int]], roles: tuple[str, ...] = ROLES
) -> tuple[dict[str, str], dict[str, object]]:
    """Enumerate every assignment for a small stratum and select the frozen optimum."""
    ordered = sorted(units)
    if len({digest for digest, _size, _populations in ordered}) != len(ordered):
        raise ValueError("Atomic-unit digests are not unique")
    best_assignment: dict[str, str] | None = None
    best_score: tuple[object, ...] | None = None
    evaluated = 0
    for role_vector in itertools.product(roles, repeat=len(ordered)):
        candidate = {unit[0]: role for unit, role in zip(ordered, role_vector)}
        score = exact_assignment_score(ordered, candidate, roles)
        evaluated += 1
        if best_score is None or score < best_score:
            best_assignment = candidate
            best_score = score
    if best_assignment is None or best_score is None:
        raise ValueError("No role assignment was evaluated")
    totals = assignment_totals(ordered, best_assignment, roles)
    return best_assignment, {
        "method": "exhaustive",
        "n_assignments_evaluated": evaluated,
        "observed": totals,
        "objective_without_private_tiebreak": list(best_score[:-1]),
        "selected_assignment_sha256": best_score[-1],
    }


def deterministic_assignment(
    units: list[tuple[str, int, int]],
    roles: tuple[str, ...] = ROLES,
) -> tuple[dict[str, str], dict[str, object]]:
    """Assign a large stratum to balanced quotas using only population and sample counts."""
    if len({digest for digest, _size, _populations in units}) != len(units):
        raise ValueError("Atomic-unit digests are not unique")
    quotas = balanced_quotas(len(units), roles)
    target_samples = sum(size for _digest, size, _populations in units) / len(roles)
    target_populations = sum(populations for _digest, _size, populations in units) / len(roles)
    totals = {role: {"units": 0, "samples": 0, "populations": 0} for role in roles}
    assignment: dict[str, str] = {}
    ordered = sorted(units, key=lambda row: (-row[2], -row[1], row[0]))
    for digest, size, populations in ordered:
        candidates = [role for role in roles if totals[role]["units"] < quotas[role]]
        if not candidates:
            raise ValueError("No role capacity remains for an atomic unit")

        def score(role: str) -> tuple[float, float, int]:
            population_fill = (
                (totals[role]["populations"] + populations) / target_populations
                if target_populations else float("inf")
            )
            sample_fill = (
                (totals[role]["samples"] + size) / target_samples
                if target_samples else float("inf")
            )
            return population_fill, sample_fill, roles.index(role)

        role = min(candidates, key=score)
        assignment[digest] = role
        totals[role]["units"] += 1
        totals[role]["samples"] += size
        totals[role]["populations"] += populations
    if any(totals[role]["units"] != quotas[role] for role in roles):
        raise ValueError("Final atomic-unit counts differ from balanced quotas")
    return assignment, {"method": "balanced_quota_greedy", "quotas": quotas, "observed": totals}


def run(args: argparse.Namespace) -> dict[str, object]:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27F_BLIND_ROLE_SPLIT" or prereg.get("version") != 3:
        raise ValueError("Invalid M27F preregistration")
    upstream = prereg["upstream_contract"]
    ibd_files = m27e.indexed_by_chromosome(args.ibd_file, "IBD")
    maps = m27e.indexed_by_chromosome(args.genetic_map, "genetic map")
    panel_ids = read_strata_sample_ids(args.resolved_strata)
    if len(panel_ids) != int(upstream["expected_panel_samples"]):
        raise ValueError("Resolved strata does not contain the expected panel size")
    authenticated = validate_upstream_manifest(
        args.upstream_m27e_manifest,
        prereg,
        [*ibd_files.values(), *maps.values(), args.resolved_strata, args.resolved_strata_manifest],
    )
    metadata, strata_receipt = m27e.read_resolved_strata(
        args.resolved_strata,
        args.resolved_strata_manifest,
        panel_ids,
        upstream,
    )
    pairs, endpoints, ibd_receipt = m27e.read_ibd(
        ibd_files,
        {sample: sample for sample in panel_ids},
        {"reported_segment_min_lod": 3.0, "reported_segment_min_cm": 2.0},
    )
    observed_ids = ibd_receipt.pop("observed_ids")
    genome_cm, per_chromosome_cm = m27e.autosomal_span_cm(endpoints, maps)
    policy = prereg["block_policy"]
    roots, block_receipt = m27e.build_blocks(
        panel_ids,
        metadata,
        pairs,
        genome_cm,
        float(policy["max_segment_floor_cm"]),
        float(policy["kinship_floor"]),
        bool(policy["union_canonical_population"]),
    )

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for sample in panel_ids:
        members_by_root[roots[sample]].append(sample)
    digest_by_root = {root: block_digest(members) for root, members in members_by_root.items()}
    populations_by_root = {
        root: {
            m27e.population_stratum(metadata[sample])
            for sample in members
            if metadata[sample]["_population_interpretable"] == "True"
        }
        for root, members in members_by_root.items()
    }
    ancestries_by_root = {
        root: {
            metadata[sample]["Ancestry"]
            for sample in members
            if metadata[sample]["_population_interpretable"] == "True"
        }
        for root, members in members_by_root.items()
    }

    discovery_core = {
        sample
        for sample in panel_ids
        if metadata[sample]["_population_interpretable"] == "True"
        and metadata[sample]["Source"] == prereg["discovery"]["source"]
        and metadata[sample]["Ancestry"] == prereg["discovery"]["ancestry"]
    }
    discovery_roots = {roots[sample] for sample in discovery_core}
    role_by_root: dict[str, str] = {root: "DISCOVERY" for root in discovery_roots}
    target_ancestries = tuple(prereg["target_ancestries"])
    mixed_ancestry_roots = {
        root for root, ancestries in ancestries_by_root.items() if len(ancestries) > 1
    }
    exact_limit = int(prereg["assignment_algorithm"]["small_stratum_exact_limit"])
    allocation_by_ancestry: dict[str, dict[str, object]] = {}
    order_invariant = True
    for ancestry in target_ancestries:
        candidate_roots = [
            root
            for root, ancestries in ancestries_by_root.items()
            if ancestries == {ancestry}
            and root not in discovery_roots
            and root not in mixed_ancestry_roots
        ]
        units = [
            (digest_by_root[root], len(members_by_root[root]), len(populations_by_root[root]))
            for root in candidate_roots
        ]
        allocator = exhaustive_assignment if len(units) <= exact_limit else deterministic_assignment
        assignment, receipt = allocator(units)
        reversed_assignment, _ = allocator(list(reversed(units)))
        order_invariant = order_invariant and assignment == reversed_assignment
        digest_to_root = {digest_by_root[root]: root for root in candidate_roots}
        for digest, role in assignment.items():
            role_by_root[digest_to_root[digest]] = role
        allocation_by_ancestry[ancestry] = receipt

    rows = []
    for sample in panel_ids:
        root = roots[sample]
        interpretable = metadata[sample]["_population_interpretable"] == "True"
        if root in role_by_root:
            role = role_by_root[root]
        else:
            role = "EXCLUDED"
        if sample in discovery_core:
            reason = "DISCOVERY_CORE"
        elif root in discovery_roots:
            reason = "DISCOVERY_POPULATION_OR_IBD_CLOSURE"
        elif not interpretable:
            reason = "UNRESOLVED_METADATA"
        elif root in mixed_ancestry_roots:
            reason = "MIXED_ANCESTRY_ATOMIC_UNIT"
        elif metadata[sample]["Ancestry"] not in target_ancestries:
            reason = "ANCESTRY_OUTSIDE_AFR_EUR_NAM"
        else:
            reason = ""
        rows.append(
            {
                "sample_id": sample,
                "source": metadata[sample]["Source"],
                "ancestry": metadata[sample]["Ancestry"] or "UNRESOLVED",
                "population": metadata[sample]["Population"],
                "canonical_population": (
                    m27e.population_stratum(metadata[sample]) if interpretable else ""
                ),
                "atomic_unit_id": digest_by_root[root],
                "role": role,
                "exclusion_reason": reason,
            }
        )

    roles_by_block: dict[str, set[str]] = defaultdict(set)
    roles_by_population: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        roles_by_block[row["atomic_unit_id"]].add(row["role"])
        if row["canonical_population"]:
            roles_by_population[row["canonical_population"]].add(row["role"])
    no_block_crossing = all(len(values) == 1 for values in roles_by_block.values())
    no_population_crossing = all(len(values) == 1 for values in roles_by_population.values())

    counts: dict[str, dict[str, dict[str, int]]] = {}
    for ancestry in target_ancestries:
        counts[ancestry] = {}
        for role in ("DISCOVERY",) + ROLES:
            selected = [row for row in rows if row["ancestry"] == ancestry and row["role"] == role]
            counts[ancestry][role] = {
                "n_samples": len(selected),
                "n_atomic_units": len({row["atomic_unit_id"] for row in selected}),
                "n_populations": len({row["canonical_population"] for row in selected}),
            }

    minimum_units = int(prereg["assignment_algorithm"]["minimum_units_per_role_and_ancestry"])
    minimum_populations = int(
        prereg["assignment_algorithm"]["minimum_populations_per_role_and_ancestry"]
    )
    f0 = (
        len(panel_ids) == int(upstream["expected_panel_samples"])
        and observed_ids == set(panel_ids)
        and strata_receipt["n_population_interpretable"]
        == int(upstream["expected_population_interpretable_samples"])
        and strata_receipt["n_population_unresolved"]
        == int(upstream["expected_population_unresolved_samples"])
    )
    f1 = (
        len(discovery_core) == int(upstream["expected_discovery_samples"])
        and all(role_by_root.get(roots[sample]) == "DISCOVERY" for sample in discovery_core)
        and all(role_by_root[root] == "DISCOVERY" for root in discovery_roots)
    )
    f2 = no_block_crossing and no_population_crossing and not mixed_ancestry_roots
    f3 = all(
        counts[ancestry][role]["n_atomic_units"] >= minimum_units
        and counts[ancestry][role]["n_populations"] >= minimum_populations
        for ancestry in target_ancestries
        for role in ROLES
    )
    f4 = True
    f5 = all(
        max(counts[ancestry][role]["n_atomic_units"] for role in ROLES)
        - min(counts[ancestry][role]["n_atomic_units"] for role in ROLES)
        <= 1
        and (
            allocation_by_ancestry[ancestry]["method"] != "exhaustive"
            or allocation_by_ancestry[ancestry]["n_assignments_evaluated"]
            == len(ROLES) ** sum(
                counts[ancestry][role]["n_atomic_units"] for role in ROLES
            )
        )
        for ancestry in target_ancestries
    )
    f6 = order_invariant
    gates = {"F0": f0, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F6": f6}
    if not f0:
        decision = "STOP_INPUT_OR_IDENTITY_CONTRACT"
    elif not f1 or not f2:
        decision = "STOP_ROLE_LEAKAGE"
    elif not f3:
        decision = "STOP_SPLIT_NOT_IDENTIFIABLE"
    elif not f5 or not f6:
        decision = "STOP_ALLOCATION_CONTRACT"
    else:
        decision = "GO_REF_EXTRACTION_ONLY"

    args.outdir.mkdir(parents=True, exist_ok=True)
    private_path = args.outdir / "m27f_split.private.tsv"
    with private_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    private_sha256 = m27e.sha256_file(private_path)
    public = {
        "stage": prereg["stage"],
        "decision": decision,
        "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "counts_by_ancestry_and_role": counts,
        "allocation_by_ancestry": allocation_by_ancestry,
        "n_panel_samples": len(panel_ids),
        "n_discovery_core_samples": len(discovery_core),
        "n_discovery_quarantine_samples": sum(
            row["role"] == "DISCOVERY" and row["sample_id"] not in discovery_core for row in rows
        ),
        "n_discovery_atomic_units": len(discovery_roots),
        "n_all_atomic_units": len(members_by_root),
        "n_mixed_ancestry_atomic_units": len(mixed_ancestry_roots),
        "no_block_crossing": no_block_crossing,
        "no_population_crossing": no_population_crossing,
        "order_invariant_assignment": order_invariant,
        "vcf_inputs_declared": False,
        "genotypes_read": False,
        "rare_support_used_for_assignment": False,
        "source_test_genotypes_opened": False,
        "source_test_sealed": True,
        "private_manifest_sha256": private_sha256,
        "sample_ids_emitted_publicly": False,
        "population_labels_emitted_publicly": False,
        "upstream_authentication": authenticated,
        "strata_receipt": strata_receipt,
        "ibd_receipt": {
            "n_segments": ibd_receipt["n_segments"],
            "n_unique_pairs": ibd_receipt["n_unique_pairs"],
            "n_observed_samples": ibd_receipt["n_observed_samples"],
            "n_duplicate_segment_keys": ibd_receipt["n_duplicate_segment_keys"],
        },
        "observed_autosomal_map_span_cm": genome_cm,
        "per_chromosome_observed_map_span_cm": per_chromosome_cm,
        "block_receipt": block_receipt,
        "interpretation": (
            "This receipt freezes role membership only. It does not measure rare support, power, "
            "LAI accuracy, or justify simulation, model training, or opening SOURCE_TEST."
        ),
    }
    (args.outdir / "m27f_split.public.json").write_text(
        json.dumps(public, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return public


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
