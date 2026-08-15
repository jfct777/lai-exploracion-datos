#!/usr/bin/env python3
"""Resolve M27D panel samples to metadata without publishing sample identifiers.

The private TSV is a workflow input for deterministic stratified sampling, for the
non-excluded sample universe and for the audit of candidate strata.  The public JSON
contains aggregate counts only.  Identity resolution reuses the alias rules already
validated by M27B and adds a general, order-independent policy for alias collisions.

Resolution policy, applied in this order to the rows reachable through any alias:

1. Exactly one reachable row wins outright (``DIRECT_UNIQUE``).
2. Otherwise rows flagged ``Exclude`` are dropped, but only while that leaves at
   least one row: exclusion is a preference, not a hard filter, so a collision made
   entirely of excluded rows still reaches the checks below.
3. Rows without genotypes are dropped unconditionally.  A row that cannot contribute
   genotypes cannot be the panel member, and a missing ``N_genotypes`` counts as no
   genotypes rather than as a permissive default.
4. Rows whose ``IID`` matches the panel identifier directly are preferred over rows
   reachable only through an alias column, again only while that leaves a row.
5. The sample resolves only if exactly one row survives.  Anything else fails closed.

The policy never breaks a tie by row order, by expected population or by any
historical flag.  Samples with no reachable row keep their genotypes but lose every
population annotation; they are marked ``UNMATCHED`` and are not interpretable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from audit_rare_scaffold_bridge import (
    ALIAS_COLUMNS,
    alias_variants,
    canonical_sample_id,
    panel_sample_id,
    read_vcf_samples,
)


REPORT_COLUMNS = ("Source", "Ancestry", "Population", "Country")
AUDIT_COLUMNS = ("Exclude", "N_genotypes", "Maximum_unrelated_dataset")

DIRECT_UNIQUE = "DIRECT_UNIQUE"
RESOLVED_ACTIVE_GENOTYPED = "RESOLVED_ACTIVE_GENOTYPED"
RESOLVED_ACTIVE_GENOTYPED_IID = "RESOLVED_ACTIVE_GENOTYPED_IID"
UNMATCHED = "UNMATCHED"
AMBIGUOUS_FAIL_CLOSED = "AMBIGUOUS_FAIL_CLOSED"

RESOLVED_METHODS = frozenset(
    {DIRECT_UNIQUE, RESOLVED_ACTIVE_GENOTYPED, RESOLVED_ACTIVE_GENOTYPED_IID}
)

TRUE_TOKENS = frozenset({"TRUE", "T", "1", "YES", "Y"})


def dedoubled_panel_id(value: str) -> str:
    """Collapse a PLINK double identifier of any width, not only ``X_X``.

    PLINK writes ``FID_IID`` into the VCF sample column, and in this panel both halves
    are the same person.  When the identifier itself contains an underscore the doubled
    form has four fields, ``A_B_A_B``, and collapsing only the two-field case leaves it
    unmatched against a metadata table that stores ``A_B``.

    The shared M27B helper handles the two-field case only.  It is deliberately left
    alone: its published artifacts hash against those exact bytes.  M27D normalises here
    instead, which is why its orphan count is lower than the one M27B reported.
    """
    fields = value.split("_")
    if len(fields) >= 2 and len(fields) % 2 == 0:
        half = len(fields) // 2
        if fields[:half] == fields[half:]:
            return "_".join(fields[:half])
    return value


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving one panel sample against the metadata table."""

    sample_id: str
    method: str
    n_candidate_rows: int
    row: dict[str, str] | None

    @property
    def resolved(self) -> bool:
        return self.method in RESOLVED_METHODS

    @property
    def match_status(self) -> str:
        if self.resolved:
            return "MATCHED"
        if self.method == UNMATCHED:
            return "UNMATCHED"
        return "AMBIGUOUS"


def is_excluded(value: str) -> bool:
    """A row counts as excluded only on an explicit true token."""
    return str(value).strip().upper() in TRUE_TOKENS


def has_genotypes(value: str) -> bool:
    """Missing or unparseable genotype counts are treated as no genotypes."""
    try:
        return float(str(value).strip()) > 0
    except (TypeError, ValueError):
        return False


def build_row_index(rows: list[dict[str, str]]) -> dict[str, set[int]]:
    index: dict[str, set[int]] = defaultdict(set)
    for row_number, row in enumerate(rows):
        for column in ALIAS_COLUMNS:
            for alias in alias_variants(row.get(column, "")):
                index[alias].add(row_number)
    return index


def resolve_sample(
    sample: str, rows: list[dict[str, str]], index: dict[str, set[int]]
) -> Resolution:
    direct = canonical_sample_id(dedoubled_panel_id(panel_sample_id(sample)))
    candidates: set[int] = set()
    for alias in alias_variants(sample) | alias_variants(direct) | {direct}:
        candidates.update(index.get(alias, set()))

    if not candidates:
        return Resolution(sample, UNMATCHED, 0, None)
    if len(candidates) == 1:
        return Resolution(sample, DIRECT_UNIQUE, 1, rows[next(iter(candidates))])

    n_candidates = len(candidates)
    active = {i for i in candidates if not is_excluded(rows[i].get("Exclude", ""))}
    pool = active or candidates
    pool = {i for i in pool if has_genotypes(rows[i].get("N_genotypes", ""))}
    if not pool:
        return Resolution(sample, AMBIGUOUS_FAIL_CLOSED, n_candidates, None)

    direct_hits = {
        i
        for i in pool
        if canonical_sample_id(str(rows[i].get("IID", "")).strip()) == direct
    }
    method = RESOLVED_ACTIVE_GENOTYPED_IID if direct_hits else RESOLVED_ACTIVE_GENOTYPED
    pool = direct_hits or pool
    if len(pool) != 1:
        return Resolution(sample, AMBIGUOUS_FAIL_CLOSED, n_candidates, None)
    return Resolution(sample, method, n_candidates, rows[next(iter(pool))])


def resolve_rows(samples: list[str], rows: list[dict[str, str]]) -> list[Resolution]:
    index = build_row_index(rows)
    return [resolve_sample(sample, rows, index) for sample in samples]


def suppressed_counts(
    resolved: list[Resolution], minimum: int
) -> list[dict[str, object]]:
    """Aggregate strata counts, hiding any stratum smaller than ``minimum``.

    Only interpretable samples are counted: a sample without metadata has no
    population to report, and inventing one from its identifier is exactly the
    inference this stage refuses to make.
    """
    counts: Counter[tuple[str, str, str]] = Counter()
    for resolution in resolved:
        if not resolution.resolved or resolution.row is None:
            continue
        row = resolution.row
        counts[(row.get("Source", ""), row.get("Ancestry", ""), row.get("Population", ""))] += 1
    visible: list[dict[str, object]] = []
    suppressed = 0
    for (source, ancestry, population), count in sorted(counts.items()):
        if count < minimum:
            suppressed += count
            continue
        visible.append(
            {"source": source, "ancestry": ancestry, "population": population, "n": count}
        )
    if suppressed:
        visible.append(
            {
                "source": "SUPPRESSED",
                "ancestry": "SUPPRESSED",
                "population": "SUPPRESSED_LT_N",
                "n": suppressed,
            }
        )
    return visible


def run(args: argparse.Namespace) -> dict[str, object]:
    samples = read_vcf_samples(args.panel_vcf)
    with args.metadata.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    resolved = resolve_rows(samples, rows)

    args.private_out.parent.mkdir(parents=True, exist_ok=True)
    with args.private_out.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sample_id",
            "match_status",
            "resolution_method",
            "n_candidate_rows",
            "population_interpretable",
            *REPORT_COLUMNS,
            *AUDIT_COLUMNS,
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for resolution in resolved:
            row = resolution.row
            writer.writerow(
                {
                    "sample_id": resolution.sample_id,
                    "match_status": resolution.match_status,
                    "resolution_method": resolution.method,
                    "n_candidate_rows": resolution.n_candidate_rows,
                    "population_interpretable": "TRUE" if resolution.resolved else "FALSE",
                    **{
                        column: (row.get(column, "") if row is not None else "")
                        for column in (*REPORT_COLUMNS, *AUDIT_COLUMNS)
                    },
                }
            )

    method_counts = Counter(resolution.method for resolution in resolved)
    status_counts = Counter(resolution.match_status for resolution in resolved)
    unresolved = method_counts[AMBIGUOUS_FAIL_CLOSED]
    summary: dict[str, object] = {
        "stage": "M27D_SAMPLE_STRATA_RESOLUTION",
        "n_panel_samples": len(samples),
        "n_metadata_rows": len(rows),
        "n_matched": status_counts["MATCHED"],
        "n_ambiguous": status_counts["AMBIGUOUS"],
        "n_unmatched": status_counts["UNMATCHED"],
        "resolution_methods": {
            method: method_counts[method]
            for method in (
                DIRECT_UNIQUE,
                RESOLVED_ACTIVE_GENOTYPED,
                RESOLVED_ACTIVE_GENOTYPED_IID,
                UNMATCHED,
                AMBIGUOUS_FAIL_CLOSED,
            )
        },
        "n_population_interpretable": sum(1 for r in resolved if r.resolved),
        "n_population_not_interpretable": sum(1 for r in resolved if not r.resolved),
        "resolution_used_row_order": False,
        "resolution_used_expected_population": False,
        "strata_counts_suppressed": suppressed_counts(resolved, args.suppress_below),
        "sample_ids_emitted_in_public_summary": False,
        "private_table_contains_sample_ids": True,
    }
    args.summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if unresolved:
        raise SystemExit(
            f"M27D strata resolution failed closed on {unresolved} alias collisions "
            "that the deterministic policy could not reduce to a single row"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-vcf", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--suppress-below", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
