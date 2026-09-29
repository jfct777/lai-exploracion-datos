#!/usr/bin/env python3
"""Describe existing PC-Relate on authenticated M16.5 edges; no new fitting.

All fractions are conditional on retained graph edges, not all possible pairs
of people. A coefficient below the operational threshold is not independence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import m165_sweep_figures as saved


DEFAULT_THRESHOLD = 0.0221
EDGE_CLASSES = ("within_assigned", "between_assigned", "with_unassigned")


def stat_signature(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def stream_kinship(path, sample_index, target_edges, expected_sha256):
    """Read/hash the complete plain TSV once; retain only target unordered pairs.

    None means a retained source row has non-finite/missing kinship. An absent
    key means no source row was found. Duplicate retained pairs always fail,
    including equal values, reversed endpoints, and non-finite first values.
    """
    saved.require(re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256 or ""),
                  "Expected PC-Relate SHA256 must contain 64 hexadecimal characters")
    path = Path(path)
    before = stat_signature(path.stat())
    digest = hashlib.sha256()
    values = {}
    audit = dict(n_source_rows=0, n_rows_not_target=0, n_source_bytes=0)

    with path.open("rb") as handle:
        def decoded_lines():
            for raw in handle:
                digest.update(raw)
                audit["n_source_bytes"] += len(raw)
                try:
                    yield raw.decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError("Invalid UTF-8 in PC-Relate source") from None

        reader = csv.reader(decoded_lines(), delimiter="\t", strict=True)
        try:
            header = next(reader, [])
            if header:
                header[0] = header[0].lstrip("\ufeff")
            saved.require(len(header) == len(set(header)) and {"ID1", "ID2", "kin"} <= set(header),
                          "PC-Relate requires unique ID1, ID2 and kin columns")
            a_column, b_column, kin_column = (header.index(name) for name in ("ID1", "ID2", "kin"))
            for row in reader:
                if not row:
                    continue
                audit["n_source_rows"] += 1
                saved.require(len(row) == len(header), "Malformed PC-Relate row width")
                saved.require(row[a_column] and row[b_column], "Empty PC-Relate endpoint")
                a, b = sample_index.get(row[a_column]), sample_index.get(row[b_column])
                if a is None or b is None:
                    audit["n_rows_not_target"] += 1
                    continue
                pair = (min(a, b), max(a, b))
                if pair not in target_edges:
                    audit["n_rows_not_target"] += 1
                    continue
                saved.require(pair not in values, "Duplicate retained unordered PC-Relate pair")
                raw_value = row[kin_column].strip()
                if raw_value.lower() in {"", ".", "na", "nan", "null"}:
                    value = None
                else:
                    try:
                        value = float(raw_value)
                    except ValueError:
                        raise ValueError("Invalid retained PC-Relate kinship value") from None
                    if not math.isfinite(value):
                        value = None
                values[pair] = value
        except csv.Error:
            raise ValueError("Invalid PC-Relate TSV syntax") from None

    observed = digest.hexdigest()
    saved.require(observed == expected_sha256.lower(), "PC-Relate full-file SHA256 mismatch")
    saved.require(stat_signature(path.stat()) == before, "PC-Relate source changed while streaming")
    audit.update(sha256=observed, n_retained_union_edges=len(target_edges),
                 n_retained_source_pairs=len(values),
                 n_retained_finite=sum(value is not None for value in values.values()),
                 n_retained_missing_nonfinite=sum(value is None for value in values.values()),
                 n_retained_missing_absent=len(target_edges)-len(values))
    return values, audit


def kinship_state(pair, values, threshold):
    if pair not in values:
        return "missing_absent"
    if values[pair] is None:
        return "missing_nonfinite"
    return "kin_ge_threshold" if values[pair] >= threshold else "kin_lt_threshold"


def count_metrics(counts):
    total = sum(counts.values())
    observed = counts["kin_ge_threshold"] + counts["kin_lt_threshold"]
    return dict(n_edges=total, n_observed_kinship=observed,
                n_kin_ge_threshold=counts["kin_ge_threshold"],
                n_kin_lt_threshold=counts["kin_lt_threshold"],
                n_missing_absent=counts["missing_absent"],
                n_missing_nonfinite=counts["missing_nonfinite"], n_missing_kinship=total-observed,
                fraction_kin_ge_all_edges=counts["kin_ge_threshold"]/total if total else None,
                fraction_kin_ge_observed_edges=counts["kin_ge_threshold"]/observed if observed else None)


def summarize(data, values, threshold):
    """Count disjoint edge classes, never treating label -1 as a community."""
    graphs, partitions = [], []
    for graph in data["graphs"]:
        states = [(a, b, kinship_state((a, b), values, threshold)) for a, b in graph["edges"]]
        total = count_metrics(Counter(state for _, _, state in states))
        base = dict(config_id=graph["config_id"], length_bp=graph["length_bp"],
                    min_edge_bp=graph["threshold_bp"], gap_bp=50000, min_shared=20,
                    min_max_segment_bp=0, kinship_threshold=threshold,
                    n_cohort=graph["n_cohort"], n_active=graph["n_active"],
                    n_isolated=graph["n_cohort"]-graph["n_active"])
        graphs.append(dict(**base, **total))
        for gamma, labels in sorted(graph["labels"].items()):
            counts = {name: Counter() for name in EDGE_CLASSES}
            for a, b, state in states:
                if labels[a] < 0 or labels[b] < 0:
                    name = "with_unassigned"
                else:
                    name = "within_assigned" if labels[a] == labels[b] else "between_assigned"
                counts[name][state] += 1
            metrics = {name: count_metrics(counter) for name, counter in counts.items()}
            for key in ("n_edges", "n_observed_kinship", "n_kin_ge_threshold", "n_kin_lt_threshold",
                        "n_missing_absent", "n_missing_nonfinite", "n_missing_kinship"):
                saved.require(sum(metrics[name][key] for name in EDGE_CLASSES) == total[key],
                              "Partition edge classes do not sum to graph totals")
            sizes = Counter(label for label in labels if label >= 0)
            partitions.append(dict(**base, resolution=gamma, n_assigned=sum(sizes.values()),
                                   n_communities=len(sizes), **{f"all_{k}": v for k, v in total.items()},
                                   **{f"{name}_{key}": value for name, row in metrics.items()
                                      for key, value in row.items()}))
    return graphs, partitions


def write_table(path, rows):
    saved.require(rows, "Cannot write an empty kinship summary")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: "NA" if value is None else value for key, value in row.items()} for row in rows)


def run(results_dir, pcrelate_file, expected_sha256, output_dir, threshold=DEFAULT_THRESHOLD):
    root, output = Path(results_dir), Path(output_dir)
    saved.require(not output.exists(), "Output directory exists; no overwrite")
    saved.require(not output.resolve().is_relative_to(root.resolve()), "Output must be outside input results")
    saved.require(math.isfinite(threshold) and 0 <= threshold <= .5, "Invalid kinship threshold")
    data = saved.load_results(root)
    first = data["graphs"][0]["config_id"]
    samples = [row["sample_id"] for row in saved.table(root / first / "graph_nodes.tsv")]
    sample_index = {sample: index for index, sample in enumerate(samples)}
    saved.require(len(sample_index) == data["n_cohort"], "Invalid authenticated cohort")
    target_edges = {edge for graph in data["graphs"] for edge in graph["edges"]}
    values, audit = stream_kinship(pcrelate_file, sample_index, target_edges, expected_sha256)
    graphs, partitions = summarize(data, values, threshold)
    saved.require(len(graphs) == 6 and len(partitions) == 42, "Incomplete six-graph/seven-resolution summary")
    for name, expected in data["input_sha256"].items():
        saved.require(saved.sha256(root / name) == expected, "Saved graph source changed during summary")
    output.mkdir(parents=True, exist_ok=False)
    write_table(output / "graph_kinship_summary.tsv", graphs)
    write_table(output / "partition_kinship_summary.tsv", partitions)
    result = dict(status="COMPLETE_DESCRIPTIVE_EDGE_KINSHIP", n_graphs=6, n_partitions=42,
                  parameters=dict(threshold=threshold, expected_sha256=expected_sha256.lower()),
                  pcrelate_audit=audit, input_sha256=data["input_sha256"],
                  source_sha256=data["source_sha256"], source_core_sha256=data["core_sha256"],
                  script_sha256=saved.sha256(__file__), validator_sha256=saved.sha256(saved.__file__),
                  no_reclustering=True, no_new_kinship=True, no_pvalues=True,
                  contains_individual_identifiers=False,
                  semantics=dict(
                      unit="unordered retained graph edge, not all possible person pairs",
                      within_assigned="both endpoints assigned to the same reported community",
                      between_assigned="both endpoints assigned to different reported communities",
                      with_unassigned="at least one endpoint has label -1; disjoint from within/between",
                      missing="absent source pair and non-finite source coefficient remain separate; neither is below threshold",
                      fractions="kin>=threshold/all retained edges and kin>=threshold/finite observed edges; zero denominator is NA",
                      gamma="graph-level kinship is constant across resolutions; 42 partitions are not independent replicates",
                      inference="operational cutoff, not certified families, independence, population validation or ancestry",
                      scope="PC-Relate is pre-existing and includes chr22; descriptive comparison, not independent validation",
                      privacy="aggregate-only output; keep private, small graph/partition groups may remain sensitive"),
                  outputs_sha256={path.name: saved.sha256(path) for path in sorted(output.iterdir())})
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--pcrelate-file", required=True, help="Existing plain UTF-8 PC-Relate TSV")
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output-dir", required=True)
    result = run(**vars(parser.parse_args(argv)))
    print(json.dumps({key: result[key] for key in ("status", "n_graphs", "n_partitions")}))


if __name__ == "__main__":
    main()
