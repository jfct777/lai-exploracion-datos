#!/usr/bin/env python3
"""Deterministic kinship-graph operations shared by every M27D stage.

Two M27D stages need the same primitive: turn a table of kinship pairs into a set of
individuals with no relationship left between them.  The first is the ``training.set``
that PC-Relate fits on; the second is the final candidate list.  They run in different
processes and on different graphs, so the algorithm lives here once instead of being
written twice.

The independent set is *maximal by inclusion*: no further individual can be added
without creating an edge.  It is deliberately not the largest possible set.  Maximum
independent set is NP-hard, and, more importantly, searching for the largest answer
would make the retained sample count a function of the search rather than of the data.

Ordering decides which member of a related group survives, so it is fixed in advance:
higher call rate first, ties broken by a stable hash of the identifier.  Neither the
ancestry proportions, the country, the population label nor the historical
``Maximum_unrelated_dataset`` flags take part.  Those describe the result; letting them
choose it would smuggle the expected answer into the selection.

Running the same construction with the hash order reversed gives an equally valid
answer.  Reporting both is the control that shows how much of the final count is a
property of the data and how much is a property of the tie-break.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


PAIR_COLUMNS = ("ID1", "ID2", "kin")


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def stable_hash(identifier: str) -> str:
    """Identifier-derived order that does not depend on locale or insertion order."""
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()


def read_pairs(path: Path, threshold: float) -> set[tuple[str, str]]:
    """Read kinship pairs at or above ``threshold`` as unordered edges."""
    edges: set[tuple[str, str]] = set()
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [column for column in PAIR_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path.name} is missing kinship columns: {', '.join(missing)}")
        for record in reader:
            try:
                kinship = float(record["kin"])
            except (TypeError, ValueError):
                continue
            if kinship < threshold:
                continue
            left, right = record["ID1"], record["ID2"]
            if left == right:
                continue
            edges.add((left, right) if left <= right else (right, left))
    return edges


def adjacency(edges: set[tuple[str, str]]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)
    return graph


def _hash_key(identifier: str, descending: bool) -> tuple[int, ...]:
    """Hash order as a comparable key, optionally inverted.

    Only the hash half of the key flips for the alternative-order control.  Negating
    the whole key would also invert the call-rate preference, which is not the factor
    being varied.
    """
    digest = stable_hash(identifier)
    sign = -1 if descending else 1
    return tuple(sign * int(character, 16) for character in digest)


def maximal_independent_set(
    nodes: list[str],
    edges: set[tuple[str, str]],
    call_rate: dict[str, float],
    descending_hash: bool = False,
) -> list[str]:
    """Greedy maximal-by-inclusion independent set under the preregistered order."""
    graph = adjacency(edges)
    ordered = sorted(
        nodes,
        key=lambda node: (-call_rate.get(node, 0.0), _hash_key(node, descending_hash)),
    )
    selected: list[str] = []
    blocked: set[str] = set()
    for node in ordered:
        if node in blocked:
            continue
        selected.append(node)
        blocked.update(graph.get(node, ()))
        blocked.add(node)
    return sorted(selected, key=stable_hash)


def read_call_rates(path: Path) -> dict[str, float]:
    call_rate: dict[str, float] = {}
    with open_text(path) as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            try:
                call_rate[record["sample_id"]] = float(record["call_rate"])
            except (KeyError, TypeError, ValueError):
                continue
    return call_rate


def read_sample_universe(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_independent(selected: list[str], edges: set[tuple[str, str]]) -> int:
    chosen = set(selected)
    return sum(1 for left, right in edges if left in chosen and right in chosen)


def resolve_threshold(args: argparse.Namespace) -> float:
    """The preregistration is the only source of truth for the primary threshold.

    Passing it as a plain number as well would create two places to change it, and the
    two stages that use it run in different processes, so a drift between them would not
    show up until the candidate list disagreed with the training set.
    """
    if args.preregistration is not None:
        contract = json.loads(args.preregistration.read_text(encoding="utf-8"))
        if contract.get("pcrelate", {}).get("king_allowed"):
            raise SystemExit("M27D forbids KING")
        threshold = float(contract["pcrelate"]["primary_phi_threshold"])
        if args.threshold is not None and abs(args.threshold - threshold) > 1e-12:
            raise SystemExit(
                f"Threshold {args.threshold} disagrees with the preregistered {threshold}"
            )
        return threshold
    if args.threshold is None:
        raise SystemExit("Provide --preregistration, or --threshold for standalone use")
    return args.threshold


def run(args: argparse.Namespace) -> dict[str, object]:
    args.threshold = resolve_threshold(args)
    nodes = read_sample_universe(args.samples)
    if len(set(nodes)) != len(nodes):
        raise SystemExit("Sample universe contains duplicate identifiers")
    call_rate = read_call_rates(args.call_rates)
    edges = read_pairs(args.pairs, args.threshold)
    node_set = set(nodes)
    external = {node for edge in edges for node in edge} - node_set
    if external:
        raise SystemExit(
            f"{len(external)} identifiers appear in the kinship pairs but not in the sample universe"
        )

    primary = maximal_independent_set(nodes, edges, call_rate)
    alternate = maximal_independent_set(nodes, edges, call_rate, descending_hash=True)
    internal_edges = verify_independent(primary, edges)
    if internal_edges:
        raise SystemExit(f"Independent set retained {internal_edges} internal kinship edges")

    args.out_set.parent.mkdir(parents=True, exist_ok=True)
    args.out_set.write_text("\n".join(primary) + "\n", encoding="utf-8")
    if args.out_alternate_set is not None:
        args.out_alternate_set.write_text("\n".join(alternate) + "\n", encoding="utf-8")

    graph = adjacency(edges)
    summary: dict[str, object] = {
        "stage": args.stage,
        "phi_threshold": args.threshold,
        "n_samples": len(nodes),
        "n_related_edges": len(edges),
        "n_samples_with_at_least_one_edge": len(graph),
        "n_independent_primary_order": len(primary),
        "n_independent_alternate_order": len(alternate),
        "n_shared_by_both_orders": len(set(primary) & set(alternate)),
        "order_sensitivity_count_delta": len(primary) - len(alternate),
        "tie_break_order": ["higher_call_rate", "stable_identifier_hash"],
        "selection_used_ancestry_or_population": False,
        "selection_used_historical_unrelated_flags": False,
        "internal_edges_in_primary_set": internal_edges,
        "sample_ids_emitted_in_public_summary": False,
    }
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--call-rates", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--stage", default="M27D_INDEPENDENT_SET")
    parser.add_argument("--out-set", type=Path, required=True)
    parser.add_argument("--out-alternate-set", type=Path, default=None)
    parser.add_argument("--out-summary", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
