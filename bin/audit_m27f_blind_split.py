#!/usr/bin/env python3
"""Build the M27F role split from metadata and frozen IBD only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import audit_m27e_ibd_rare_transfer as m27e


ROLES = ("REF_TRAIN", "SOURCE_VALID", "SOURCE_TEST")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibd-file", action="append", type=Path, required=True)
    parser.add_argument("--genetic-map", action="append", type=Path, required=True)
    parser.add_argument("--panel-vcf", type=Path, required=True)
    parser.add_argument("--discovery-vcf", type=Path, required=True)
    parser.add_argument("--resolved-strata", type=Path, required=True)
    parser.add_argument("--resolved-strata-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def block_digest(members: list[str]) -> str:
    payload = "\n".join(sorted(members)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def deterministic_assignment(
    blocks: list[tuple[str, int]], roles: tuple[str, ...] = ROLES
) -> dict[str, str]:
    """Balance block counts first and sample counts second, without randomness."""
    totals = {role: [0, 0] for role in roles}
    assignment: dict[str, str] = {}
    for digest, size in sorted(blocks, key=lambda row: (-row[1], row[0])):
        role = min(roles, key=lambda value: (totals[value][0], totals[value][1], roles.index(value)))
        assignment[digest] = role
        totals[role][0] += 1
        totals[role][1] += size
    return assignment


def run(args: argparse.Namespace) -> dict[str, object]:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27F_BLIND_ROLE_SPLIT" or prereg.get("version") != 1:
        raise ValueError("Invalid M27F preregistration")

    panel_ids = m27e.read_vcf_samples(args.panel_vcf)
    discovery_ids = m27e.read_vcf_samples(args.discovery_vcf)
    upstream = prereg["upstream_contract"]
    metadata, strata_receipt = m27e.read_resolved_strata(
        args.resolved_strata,
        args.resolved_strata_manifest,
        panel_ids,
        {
            "resolved_strata_sha256": upstream["resolved_strata_sha256"],
            "resolved_strata_manifest_sha256": upstream["resolved_strata_manifest_sha256"],
            "expected_population_interpretable_samples": upstream["expected_population_interpretable_samples"],
            "expected_population_unresolved_samples": upstream["expected_population_unresolved_samples"],
        },
    )
    ibd_files = m27e.indexed_by_chromosome(args.ibd_file, "IBD")
    maps = m27e.indexed_by_chromosome(args.genetic_map, "genetic map")
    pairs, endpoints, ibd_receipt = m27e.read_ibd(
        ibd_files,
        {sample: sample for sample in panel_ids},
        {"reported_segment_min_lod": 3.0, "reported_segment_min_cm": 2.0},
    )
    genome_cm, _ = m27e.autosomal_span_cm(endpoints, maps)
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
    discovery = set(discovery_ids)
    expected_discovery = {
        sample for sample in panel_ids
        if metadata[sample]["Source"] == prereg["discovery"]["source"]
        and metadata[sample]["Ancestry"] == prereg["discovery"]["ancestry"]
    }
    discovery_roots = {roots[sample] for sample in discovery}

    role_by_root: dict[str, str] = {root: "DISCOVERY" for root in discovery_roots}
    target_ancestries = tuple(prereg["target_ancestries"])
    mixed_target_blocks = sum(
        len({metadata[sample]["Ancestry"] for sample in members}.intersection(target_ancestries)) > 1
        for members in members_by_root.values()
    )
    for ancestry in target_ancestries:
        candidates: list[tuple[str, int]] = []
        roots_for_ancestry = set()
        for root, members in members_by_root.items():
            ancestries = {metadata[sample]["Ancestry"] for sample in members}
            if ancestry not in ancestries or root in discovery_roots:
                continue
            roots_for_ancestry.add(root)
            candidates.append((digest_by_root[root], len(members)))
        assignment = deterministic_assignment(candidates)
        digest_to_root = {digest_by_root[root]: root for root in roots_for_ancestry}
        for digest, role in assignment.items():
            root = digest_to_root[digest]
            if root in role_by_root and role_by_root[root] != role:
                raise ValueError("Cross-ancestry block received conflicting roles")
            role_by_root[root] = role

    rows = []
    for sample in panel_ids:
        root = roots[sample]
        ancestry = metadata[sample]["Ancestry"]
        if root in role_by_root:
            role = role_by_root[root]
        elif metadata[sample]["_population_interpretable"] != "True":
            role = "OUT_OF_SCOPE_METADATA"
        else:
            role = "OUT_OF_SCOPE_ANCESTRY"
        population_digest = hashlib.sha256(
            m27e.population_stratum(metadata[sample]).encode()
        ).hexdigest()
        rows.append({
            "sample_id": sample,
            "block_digest": digest_by_root[root],
            "population_digest": population_digest,
            "role": role,
            "ancestry": ancestry or "UNRESOLVED",
        })

    roles_by_block: dict[str, set[str]] = defaultdict(set)
    roles_by_population: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        roles_by_block[row["block_digest"]].add(row["role"])
        roles_by_population[row["population_digest"]].add(row["role"])
    no_block_crossing = all(len(values) == 1 for values in roles_by_block.values())
    no_population_crossing = all(len(values) == 1 for values in roles_by_population.values())

    counts = {}
    for ancestry in target_ancestries:
        counts[ancestry] = {}
        for role in ("DISCOVERY",) + ROLES:
            selected = [row for row in rows if row["ancestry"] == ancestry and row["role"] == role]
            counts[ancestry][role] = {
                "n_samples": len(selected),
                "n_blocks": len({row["block_digest"] for row in selected}),
                "n_populations": len({row["population_digest"] for row in selected}),
            }

    f0 = (
        len(panel_ids) == int(upstream["expected_panel_samples"])
        and len(discovery_ids) == int(upstream["expected_discovery_samples"])
        and set(discovery_ids) <= set(panel_ids)
    )
    f1 = discovery == expected_discovery and all(
        row["role"] == "DISCOVERY" for row in rows if row["sample_id"] in discovery
    )
    f2 = no_block_crossing and no_population_crossing and mixed_target_blocks == 0
    f3 = all(counts[ancestry][role]["n_blocks"] >= 2 for ancestry in target_ancestries for role in ROLES)
    f4 = True
    gates = {"F0": f0, "F1": f1, "F2": f2, "F3": f3, "F4": f4}
    if not f0:
        decision = "STOP_INPUT_OR_IDENTITY_CONTRACT"
    elif not f1 or not f2:
        decision = "STOP_ROLE_LEAKAGE"
    elif not f3:
        decision = "STOP_SPLIT_NOT_IDENTIFIABLE"
    else:
        decision = "GO_OPEN_ROLE_EXTRACTION_ONLY"

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
        "n_panel_samples": len(panel_ids),
        "n_discovery_samples": len(discovery),
        "n_discovery_blocks": len(discovery_roots),
        "n_all_blocks": len(members_by_root),
        "n_mixed_target_ancestry_blocks": mixed_target_blocks,
        "no_block_crossing": no_block_crossing,
        "no_population_crossing": no_population_crossing,
        "genotypes_parsed": False,
        "vcf_content_used_for_assignment": "header_sample_ids_only",
        "rare_support_used_for_assignment": False,
        "source_test_genotypes_opened": False,
        "private_manifest_sha256": private_sha256,
        "sample_ids_emitted_publicly": False,
        "population_labels_emitted_publicly": False,
        "strata_receipt": strata_receipt,
        "ibd_receipt": {"n_unique_pairs": ibd_receipt["n_unique_pairs"]},
        "block_receipt": block_receipt,
    }
    (args.outdir / "m27f_split.public.json").write_text(
        json.dumps(public, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return public


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
