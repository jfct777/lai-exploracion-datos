#!/usr/bin/env python3
"""Resolve M27D panel samples to metadata without publishing sample identifiers.

The private TSV is a workflow input for deterministic, stratified resource
sampling.  The public JSON contains aggregate counts only.  Identity resolution
reuses the alias rules already validated by M27B.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_rare_scaffold_bridge import (
    ALIAS_COLUMNS,
    alias_variants,
    canonical_sample_id,
    panel_sample_id,
    read_vcf_samples,
)


REPORT_COLUMNS = ("Source", "Ancestry", "Population", "Country")


def build_row_index(rows: list[dict[str, str]]) -> dict[str, set[int]]:
    index: dict[str, set[int]] = defaultdict(set)
    for row_number, row in enumerate(rows):
        for column in ALIAS_COLUMNS:
            for alias in alias_variants(row.get(column, "")):
                index[alias].add(row_number)
    return index


def resolve_rows(
    samples: list[str], rows: list[dict[str, str]]
) -> list[tuple[str, str, dict[str, str] | None]]:
    index = build_row_index(rows)
    resolved: list[tuple[str, str, dict[str, str] | None]] = []
    for sample in samples:
        direct = canonical_sample_id(panel_sample_id(sample))
        candidates: set[int] = set()
        for alias in alias_variants(sample) | alias_variants(direct):
            candidates.update(index.get(alias, set()))
        if len(candidates) == 1:
            row = rows[next(iter(candidates))]
            resolved.append((sample, "MATCHED", row))
        elif candidates:
            resolved.append((sample, "AMBIGUOUS", None))
        else:
            resolved.append((sample, "UNMATCHED", None))
    return resolved


def suppressed_counts(
    resolved: list[tuple[str, str, dict[str, str] | None]], minimum: int
) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for _sample, status, row in resolved:
        if status != "MATCHED" or row is None:
            continue
        counts[(row.get("Source", ""), row.get("Ancestry", ""), row.get("Population", ""))] += 1
    visible: list[dict[str, object]] = []
    suppressed = 0
    for (source, ancestry, population), count in sorted(counts.items()):
        if count < minimum:
            suppressed += count
            continue
        visible.append(
            {
                "source": source,
                "ancestry": ancestry,
                "population": population,
                "n": count,
            }
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
        fieldnames = ["sample_id", "match_status", *REPORT_COLUMNS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for sample, status, row in resolved:
            writer.writerow(
                {
                    "sample_id": sample,
                    "match_status": status,
                    **{
                        column: row.get(column, "") if row is not None else ""
                        for column in REPORT_COLUMNS
                    },
                }
            )

    status_counts = Counter(status for _sample, status, _row in resolved)
    summary: dict[str, object] = {
        "stage": "M27D_SAMPLE_STRATA_RESOLUTION",
        "n_panel_samples": len(samples),
        "n_metadata_rows": len(rows),
        "n_matched": status_counts["MATCHED"],
        "n_ambiguous": status_counts["AMBIGUOUS"],
        "n_unmatched": status_counts["UNMATCHED"],
        "strata_counts_suppressed": suppressed_counts(resolved, args.suppress_below),
        "sample_ids_emitted_in_public_summary": False,
        "private_table_contains_sample_ids": True,
    }
    args.summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
