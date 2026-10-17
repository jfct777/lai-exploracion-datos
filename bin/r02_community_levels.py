#!/usr/bin/env python3
"""Describe ancestry and kinship support of authenticated, fixed communities.

This module does not recluster, train, estimate ancestry, or select a biological
winner. Person-weighted descriptions and component-balanced sensitivities have
different estimands. All configurations remain visible, including degenerate
partitions and missing observations. A shared-support analysis makes exclusions
explicit rather than assigning unknown people to a fabricated community.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys


SCHEMA = "r02_community_levels_v1"
STATUS = "COMPLETE_DESCRIPTIVE_LEVEL_COMPARISON_NOT_POPULATION_VALIDATION"
PHIS = (.0221, .0442)
ANCESTRY = ("Autosomes_African_anc", "Autosomes_European_anc",
            "Autosomes_Indigenous_anc", "Autosomes_EastAsian_anc")
SUPPORTS = ("assigned_complete", "common_all_partitions")
LIMITATIONS = [
    "Fixed transductive partitions; no independent population validation or new clustering.",
    "Ancestry values are pre-existing estimates, not truth; their estimation uncertainty is unavailable.",
    "Eta squared is observed ancestry association, not prediction accuracy, fine structure, or rare increment.",
    "Adjusted eta squared is a degrees-of-freedom description, not a hypothesis test or biological optimum.",
    "Component balancing changes the estimand toward a typical observed component; it does not remove all kinship.",
    "PC-Relate components are transitive, threshold-dependent connections, not certified pedigrees or independent units.",
    "Shared-support association describes the intersection of assigned people, not automatically the full cohort.",
    "Seed ARI measures algorithmic stability on active nodes, not biological replication.",
    "Collection, region, ancestry and technical batch can be confounded; no batch adjustment is claimed.",
    "DP, GQ, local callability, haplotype-resolved rare ancestry and founder-demographic evidence are absent here.",
    "No p-values, confidence intervals, universal ranking, or winner are computed after examining 57 partitions.",
    "Individual and small-cell outputs are private; successful execution does not authorize public distribution.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eta_squared(vectors, labels, weights=None):
    """Weighted within/total sums of squares, retaining all four Q components.

    Coordinates are ancestry fractions. No log ratio or zero replacement is
    applied. The joint quantity is Euclidean variation of that declared vector,
    not four independent tests. Degrees-of-freedom adjustment is only reported
    for person weighting. One-person and constant supports remain unevaluable.
    """
    n = len(vectors)
    require(len(labels) == n, "Ancestry/label lengths differ")
    require(all(type(label) is int and label >= 0 for label in labels), "Assigned integer labels required")
    k = len(set(labels))
    row = dict(n_support=n, n_communities_support=k, sum_weights=0.,
               sse_within=None, sst_total=None, eta2=None, eta2_adjusted=None,
               eta_status="NO_ELIGIBLE_PEOPLE", adjustment_status="NOT_EVALUABLE")
    if not n:
        return row
    dimensions = len(vectors[0])
    require(dimensions > 0 and all(len(v) == dimensions for v in vectors), "Unequal ancestry dimensions")
    require(all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1
                for v in vectors for x in v), "Finite ancestry fractions required")
    weighted = weights is not None
    w = [1.] * n if weights is None else list(weights)
    require(len(w) == n and all(math.isfinite(x) and x > 0 for x in w), "Positive finite weights required")
    total = math.fsum(w)
    row["sum_weights"] = total
    if n < 2:
        row["eta_status"] = "INSUFFICIENT_PEOPLE"
        return row
    mean = [math.fsum(v[d] * weight for v, weight in zip(vectors, w)) / total for d in range(dimensions)]
    grouped = {}
    for i, label in enumerate(labels):
        grouped.setdefault(label, []).append(i)
    centers = {label: [math.fsum(vectors[i][d] * w[i] for i in indices) /
                       math.fsum(w[i] for i in indices) for d in range(dimensions)]
               for label, indices in grouped.items()}
    sst = math.fsum(weight * math.fsum((v[d] - mean[d]) ** 2 for d in range(dimensions))
                    for v, weight in zip(vectors, w))
    sse = math.fsum(weight * math.fsum((v[d] - centers[label][d]) ** 2 for d in range(dimensions))
                    for v, label, weight in zip(vectors, labels, w))
    row.update(sse_within=sse, sst_total=sst)
    # This floor only guards roundoff for constant fraction vectors, not biology.
    if sst <= 1e-24 * total:
        row["eta_status"] = "CONSTANT_ANCESTRY"
        return row
    value = 1. - sse / sst
    require(-1e-12 <= value <= 1 + 1e-12, "Within variance exceeds total variance")
    row.update(eta2=min(1., max(0., value)),
               eta_status="ONE_COMMUNITY_NO_DISCRETE_CONTRAST" if k == 1 else "DESCRIPTIVE_ASSOCIATION")
    if weighted:
        row["adjustment_status"] = "NOT_DEFINED_FOR_COMPONENT_BALANCING"
    elif n <= k:
        row["adjustment_status"] = "NO_RESIDUAL_DEGREES_OF_FREEDOM"
    else:
        row.update(eta2_adjusted=1. - (1. - row["eta2"]) * (n - 1) / (n - k),
                   adjustment_status="DESCRIPTIVE_DF_ADJUSTMENT")
    return row


def component_metrics(indices, components):
    """Concentration among observed components; missing IDs never form a group."""
    counts = Counter(components[i] for i in indices if components[i] is not None)
    known = sum(counts.values())
    maximum = max(counts.values()) if known else None
    return dict(n_component_known=known, n_component_missing=len(indices) - known,
                n_components=len(counts), maximum_component_n=maximum,
                maximum_component_fraction_all=maximum / len(indices) if known else None,
                maximum_component_fraction_known=maximum / known if known else None,
                effective_components=known ** 2 / sum(v ** 2 for v in counts.values()) if known else None,
                component_status="NO_KNOWN_COMPONENTS" if not known else
                    "DESCRIPTIVE_PARTIAL_COMPONENTS" if known < len(indices) else "DESCRIPTIVE_COMPONENTS")


def complete_indices(metadata):
    return [i for i, row in enumerate(metadata) if all(row[field] is not None for field in ANCESTRY)]


def shared_support(partitions, metadata):
    require(partitions, "No partitions")
    n = len(metadata)
    require(all(len(p["labels"]) == n for p in partitions), "Partition/cohort lengths differ")
    return [i for i in complete_indices(metadata) if all(p["labels"][i] >= 0 for p in partitions)]


def support_metrics(partition, metadata, indices, components=None):
    """Balance components only among the eligible people of this exact support."""
    require(len(set(indices)) == len(indices), "Duplicated support person")
    require(all(0 <= i < len(metadata) and partition["labels"][i] >= 0 and
                all(metadata[i][f] is not None for f in ANCESTRY) for i in indices), "Invalid eligible support")
    eligible = list(indices) if components is None else [i for i in indices if components[i] is not None]
    counts = None if components is None else Counter(components[i] for i in eligible)
    weights = None if counts is None else [1. / counts[components[i]] for i in eligible]
    result = eta_squared([[metadata[i][f] for f in ANCESTRY] for i in eligible],
                         [partition["labels"][i] for i in eligible], weights)
    for field, short in zip(ANCESTRY, ("AFR", "EUR", "NAM", "EAS")):
        per_field = eta_squared([[metadata[i][field]] for i in eligible],
                               [partition["labels"][i] for i in eligible], weights)
        for key in ("eta2", "eta2_adjusted", "sst_total", "sse_within", "eta_status"):
            result[key + "_" + short] = per_field[key]
    result.update(n_support_before_component_filter=len(indices),
                  n_excluded_unknown_component=len(indices) - len(eligible),
                  n_balanced_components=len(counts) if counts is not None else None)
    return result


def load_bundle(contract_path, expected_sha256, max_memory_mb, expected_samples, expected_partitions):
    """Reuse the established authenticated readers, without opening edge matrices."""
    import r02_community_evidence as base
    path = Path(contract_path).resolve()
    inputs = base.Inputs(path.parent, max_memory_mb)
    inputs.add(dict(path=str(path), sha256=expected_sha256))
    contract = base.strict_json(path)
    require(set(contract) == {"schema", "expected_samples", "samples", "study_contract", "metadata", "graphs", "comparisons"}
            and contract["schema"] == base.SCHEMA, "Unexpected community input contract")
    require(contract["expected_samples"] == expected_samples, "Expected cohort count differs")
    samples = base.evidence.read_samples(inputs.add(contract["samples"]), expected_samples)
    study_path = inputs.add(contract["study_contract"])
    sm = base.strict_json(study_path)
    require(sm["thresholds"] == list(PHIS), "Both original PC-Relate cutoffs are required")
    inputs.adjacent(study_path, sm, "persons.private.tsv")
    study = base.biological.read_study(study_path, samples, sm["kinship_audit"]["sha256"], list(PHIS))
    metadata, audit = base.read_annotations(inputs.add(contract["metadata"]["file"]), samples, contract["metadata"])
    require(tuple(base.ANCESTRY) == ANCESTRY, "Four-component ancestry semantics changed")
    missing = {v.strip().casefold() for v in base.metadata_base.DEFAULTS["missing_tokens"]}
    for person, annotations in zip(study["people"], metadata):
        for field in set(base.CATEGORICAL) & set(study["metadata_fields"]):
            value = person[field].strip()
            require(annotations[field] == (None if value.casefold() in missing else value), "Study metadata projection differs")
    graphs = [base.load_graph(spec, inputs, samples) for spec in contract["graphs"]]
    require(graphs and len({g["id"] for g in graphs}) == len(graphs) and
            len({str(g["manifest_path"]) for g in graphs}) == len(graphs), "Duplicate or empty graph ledger")
    reused = [r for graph in graphs for r in base.reuse_kinship(graph, inputs, sm["kinship_audit"]["sha256"], PHIS)]
    partitions = []
    for graph in graphs:
        manifest = graph["manifest"]
        settings = manifest["parameters"] if graph["family"] == "H" else manifest["settings"]
        n_seeds = settings.get("n_seeds")
        require(type(n_seeds) is int and n_seeds >= 2, "At least two authenticated stability seeds required")
        name = "leiden_ari_by_resolution.tsv"
        require(name in manifest["outputs_sha256"], "Saved seed stability table absent")
        stability = {}
        for row in base.kinship_base.read_table(inputs.adjacent(graph["manifest_path"], manifest, name)):
            gamma = float(row["resolution"])
            require(gamma in graph["labels"] and gamma not in stability, "Unknown or duplicate stability resolution")
            values = {key: float(row[key]) for key in ("median_ari", "q25_ari", "q75_ari", "min_ari", "max_ari")}
            require(all(math.isfinite(v) and -1 <= v <= 1 for v in values.values()), "Invalid saved ARI")
            require(values["min_ari"] <= values["q25_ari"] <= values["median_ari"] <= values["q75_ari"] <= values["max_ari"],
                    "Invalid saved ARI quantile order")
            require(int(row["n_nodes"]) == sum(graph["active"]), "Stability node denominator differs")
            require(int(row["n_pairs"]) == n_seeds * (n_seeds - 1) // 2,
                    "Stability seed-pair denominator differs")
            stability[gamma] = dict(**values, ari_n_nodes=int(row["n_nodes"]), ari_n_seed_pairs=int(row["n_pairs"]))
        require(set(stability) == set(graph["labels"]), "Incomplete stability ledger")
        for gamma, labels in sorted(graph["labels"].items()):
            partitions.append(dict(partition_id=f"partition_{len(partitions):03d}", graph_id=graph["id"],
                family=graph["family"], resolution=gamma, labels=labels, active=graph["active"],
                stability=stability[gamma], graph_manifest_sha256=inputs.records[str(graph["manifest_path"])]["sha256"]))
    require(len(partitions) == expected_partitions, "Expected partition count differs")
    inputs.recheck()
    return dict(inputs=inputs, samples=samples, metadata=metadata, metadata_audit=audit, study=study,
                partitions=partitions, reused_kinship=reused, categories=list(base.CATEGORICAL), base=base)


def config_rows(bundle, common):
    metadata, partitions = bundle["metadata"], bundle["partitions"]
    complete = set(complete_indices(metadata))
    for partition in partitions:
        labels = partition["labels"]
        n = len(labels)
        assigned = [i for i, label in enumerate(labels) if label >= 0]
        for support, indices in ((SUPPORTS[0], [i for i in assigned if i in complete]), (SUPPORTS[1], common)):
            for phi in (None, *PHIS):
                components = None if phi is None else bundle["study"]["assignments"][phi]
                row = dict(partition_id=partition["partition_id"], graph_id=partition["graph_id"], family=partition["family"],
                    resolution=partition["resolution"], n_cohort=n, n_assigned=len(assigned), n_unassigned=n-len(assigned),
                    n_isolated=sum(not v for v in partition["active"]), n_communities=len(set(labels)-{-1}),
                    fraction_assigned=len(assigned)/n, n_complete_ancestry_cohort=len(complete),
                    n_assigned_incomplete_ancestry=sum(i not in complete for i in assigned),
                    support=support, weighting="person" if phi is None else "component_balanced",
                    kinship_threshold=phi, **support_metrics(partition, metadata, indices, components),
                    **partition["stability"])
                row["fraction_support_cohort"] = row["n_support"] / n
                yield row


def community_rows(bundle):
    metadata = bundle["metadata"]
    for p in bundle["partitions"]:
        for scope, label, indices in bundle["base"].groups_for(p["labels"], p["active"]):
            if scope not in {"community", "unassigned"}:
                continue
            complete = [i for i in indices if all(metadata[i][field] is not None for field in ANCESTRY)]
            ancestry = {}
            for field in ANCESTRY:
                values = [metadata[i][field] for i in complete]
                ancestry[field + "_mean"] = math.fsum(values)/len(values) if values else None
                ancestry[field + "_min"] = min(values) if values else None
                ancestry[field + "_max"] = max(values) if values else None
            for phi in PHIS:
                yield dict(partition_id=p["partition_id"], graph_id=p["graph_id"], family=p["family"],
                    resolution=p["resolution"], scope=scope, community_local=label if scope == "community" else -1,
                    n_cohort=len(metadata), n_samples=len(indices), n_complete_ancestry=len(complete),
                    n_incomplete_ancestry=len(indices)-len(complete), fraction_cohort=len(indices)/len(metadata),
                    kinship_threshold=phi, **component_metrics(indices, bundle["study"]["assignments"][phi]), **ancestry)


def individual_rows(bundle, common):
    common_set = set(common)
    for i, (sample, metadata) in enumerate(zip(bundle["samples"], bundle["metadata"])):
        yield dict(sample_id=sample, common_support=i in common_set,
                   complete_ancestry=all(metadata[f] is not None for f in ANCESTRY),
                   **{f: metadata[f] for f in (*ANCESTRY, *bundle["categories"])},
                   **{f"component_phi_{phi:g}": bundle["study"]["assignments"][phi][i] for phi in PHIS},
                   **{p["partition_id"]: p["labels"][i] for p in bundle["partitions"]})


def write_tsv(path, rows, check):
    count, writer = 0, None
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        for row in rows:
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t", lineterminator="\n")
                writer.writeheader()
            writer.writerow({key: "NA" if value is None else value for key, value in row.items()})
            count += 1
            if count % 1000 == 0:
                check()
    require(count > 0, "Empty output table")
    return count


def save_json(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run(inputs_path, expected_sha256, output_dir, max_memory_mb, expected_samples, expected_partitions, min_free_mib):
    require(type(expected_samples) is int and expected_samples > 1, "Expected sample count must exceed one")
    require(type(expected_partitions) is int and expected_partitions > 0, "Positive expected partition count required")
    require(type(min_free_mib) is int and min_free_mib >= 0, "Nonnegative disk reserve required")
    output = Path(output_dir).absolute()
    require(not output.exists() and not output.is_symlink(), "Output exists; overwrite forbidden")
    require(output.parent.is_dir(), "Output parent must already exist")
    require(shutil.disk_usage(output.parent).free >= min_free_mib*1024**2, "Disk reserve unavailable")
    bundle = load_bundle(inputs_path, expected_sha256, max_memory_mb, expected_samples, expected_partitions)
    inputs = bundle["inputs"]
    require(all(not Path(path).is_relative_to(output) for path in inputs.records), "Output contains an input")
    common = shared_support(bundle["partitions"], bundle["metadata"])
    output.mkdir(mode=0o700)
    def check():
        inputs.check()
        require(shutil.disk_usage(output).free >= min_free_mib*1024**2, "Disk reserve exhausted")
    counts = {"config_metrics.tsv": write_tsv(output/"config_metrics.tsv", config_rows(bundle, common), check),
              "community_metrics.tsv": write_tsv(output/"community_metrics.tsv", community_rows(bundle), check),
              "individuals.private.tsv": write_tsv(output/"individuals.private.tsv", individual_rows(bundle, common), check)}
    require(counts["config_metrics.tsv"] == expected_partitions * len(SUPPORTS) * (1 + len(PHIS)),
            "Incomplete configuration/support/weighting grid")
    require(counts["individuals.private.tsv"] == expected_samples, "Individual output cohort incomplete")
    expected_community_rows = len(PHIS) * sum(len(set(p["labels"]) - {-1}) + 1 for p in bundle["partitions"])
    require(counts["community_metrics.tsv"] == expected_community_rows, "Community grid incomplete")
    save_json(output/"kinship_reused.json", bundle["reused_kinship"])
    ledger = [{key: p[key] for key in ("partition_id", "graph_id", "family", "resolution", "graph_manifest_sha256")}
              for p in bundle["partitions"]]
    summary = dict(schema=SCHEMA, status=STATUS, n_cohort=len(bundle["samples"]), n_partitions=len(ledger),
        n_graphs=len({p["graph_id"] for p in ledger}), n_common_assigned_complete=len(common),
        common_support_status="DESCRIPTIVE_INTERSECTION" if len(common) >= 2 else "INSUFFICIENT_SHARED_SUPPORT",
        n_complete_ancestry_cohort=len(complete_indices(bundle["metadata"])), ancestry_fields=list(ANCESTRY),
        kinship_thresholds=list(PHIS), partition_ledger=ledger, table_rows=counts, no_winner_selected=True,
        no_pvalues=True, no_confidence_intervals=True, no_population_validation=True,
        level_status={"K01": "DESCRIPTIVE_ANCESTRY_ASSOCIATION", "K05": "DESCRIPTIVE_DEPENDENCE_CONCENTRATION",
                      "K02": "NOT_IDENTIFIED", "K03": "NOT_IDENTIFIED", "K04": "NOT_IDENTIFIED",
                      "K06": "NOT_IDENTIFIED", "K07": "NOT_ESTIMATED"},
        definitions={"eta2": "1 - weighted within-community Q sum of squares / weighted total Q sum of squares",
            "eta2_adjusted": "1-(1-eta2)*(n-1)/(n-K); person weights only; retain negative values",
            "Q": "Four ancestry fractions in fixed AFR, EUR, NAM, EAS order; complete vectors only",
            "component_weight": "1 / eligible members of this component in this exact support",
            "effective_components": "known_people^2 / sum(component_size^2), concentration diversity not independence",
            "shared_support": "People assigned in every frozen partition and complete for all four ancestry fractions",
            "stability": "Reused ARI among seed pairs on active nodes; no new seed runs",
            "ancestry_variance_roundoff_floor": "SST <= 1e-24 * sum_weights is numerical zero, not a biological threshold"},
        limitations=LIMITATIONS)
    save_json(output/"summary.json", summary)
    inputs.recheck()
    check()
    # Include every project module loaded by authenticated readers, not just this wrapper.
    bin_dir = Path(__file__).resolve().parent
    sources = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            source = Path(filename).resolve()
            if source.parent == bin_dir and source.suffix == ".py":
                sources.add(source)
    manifest = dict(schema=SCHEMA, status=STATUS, n_cohort=expected_samples, n_partitions=expected_partitions,
        sample_ids_sha256=bundle["base"].evidence.sample_hash(bundle["samples"]), chromosomes=bundle["base"].AUTOSOMES,
        input_files=inputs.records, metadata_audit=bundle["metadata_audit"], study_provenance=bundle["study"]["provenance"],
        source_sha256={str(p): sha256(p) for p in sorted(sources)},
        parameters=dict(max_memory_mb=max_memory_mb, min_free_mib=min_free_mib,
                        expected_samples=expected_samples, expected_partitions=expected_partitions),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, table_rows=counts,
        no_reclustering=True, no_new_kinship=True, no_new_target=True, no_winner_selected=True,
        no_pvalues=True, no_confidence_intervals=True, no_population_validation=True,
        contains_individual_identifiers=True, public_distribution_allowed=False, limitations=LIMITATIONS,
        outputs_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    save_json(output/"manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--expected-inputs-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-memory-mb", required=True, type=int)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--expected-partitions", required=True, type=int)
    parser.add_argument("--min-free-mib", required=True, type=int)
    args = parser.parse_args()
    os.umask(0o077)
    result = run(args.inputs, args.expected_inputs_sha256, args.output_dir, args.max_memory_mb,
                 args.expected_samples, args.expected_partitions, args.min_free_mib)
    print(json.dumps({key: result[key] for key in ("status", "n_cohort", "n_partitions")}))


if __name__ == "__main__":
    main()
