#!/usr/bin/env python3
"""Private, descriptive integration of saved R02 communities; no new clustering.

The input contract selects immutable graph manifests and explicit partition
comparisons. Only authenticated small tables are read, never edge matrices or
VCFs. Existing kinship summaries are reused, not re-estimated. Ancestry units
must be declared; fractions are not inferred from the size of the values.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import re
import resource
import sys

import m165_metadata_summary as metadata_base
import r02_biological_evaluation as biological
import r02_genomic_pair_evidence as evidence
import r02_study_design as study_base
import r02_weighted_kinship as kinship_base


SCHEMA = "r02_community_evidence_inputs_v1"
STATUS = "COMPLETE_DESCRIPTIVE_COMMUNITY_EVIDENCE_NOT_VALIDATED"
ANCESTRY = metadata_base.ANCESTRY
CATEGORICAL = (*metadata_base.CATEGORICAL, "Source", "Origin")
AUTOSOMES = [str(c) for c in range(1, 23)]
require = evidence.require
sha256 = evidence.sha256


def strict_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


class Inputs:
    """Conservative small-table admission and Linux RSS guard; not a hard cgroup."""
    def __init__(self, root, max_memory_mb):
        require(type(max_memory_mb) is int and max_memory_mb > 0, "Positive memory limit required")
        self.root, self.limit = Path(root), max_memory_mb * 1024**2
        self.records = {}
        self.check()

    def check(self):
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= self.limit,
                "Observed peak RSS exceeds max-memory-mb")

    def add(self, record):
        require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "Expected path and SHA256")
        expected = record["sha256"]
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid SHA256")
        path = Path(record["path"])
        path = (self.root / path if not path.is_absolute() else path).resolve(strict=True)
        require(path.is_file(), "Input must be a file")
        size = path.stat().st_size
        prior = self.records.get(str(path))
        require(prior is None or prior == dict(sha256=expected, bytes=size), "Conflicting input identities")
        # Python dict/string tables can greatly exceed their serialized size.
        sizes = sum(item["bytes"] for item in self.records.values()) + (0 if prior else size)
        require(32 * sizes <= self.limit, "Small-table memory admission exceeds max-memory-mb")
        require(sha256(path) == expected, "Input SHA256 mismatch: " + path.name)
        self.records[str(path)] = dict(sha256=expected, bytes=size)
        self.check()
        return path

    def adjacent(self, manifest_path, manifest, name):
        require(Path(name).name == name and name not in {"", ".", ".."}, "Unsafe adjacent filename")
        expected = manifest.get("outputs_sha256", {}).get(name)
        return self.add(dict(path=str(manifest_path.parent / name), sha256=expected))

    def recheck(self):
        for name, record in self.records.items():
            require(Path(name).stat().st_size == record["bytes"] and sha256(name) == record["sha256"],
                    "Input changed during integration")
        self.check()


def positive_integer(value, name):
    require(type(value) is int and value > 0, "Positive integer required: " + name)
    return value


def read_annotations(path, samples, specification):
    """Reuse the study's exact-ID/conflicting-duplicate parser; retain whitelist only."""
    require(set(specification) == {"file", "sample_column", "ancestry_units", "ancestry_sum_tolerance"},
            "Unexpected metadata specification")
    unit = specification["ancestry_units"]
    require(unit in {"fraction", "percent"}, "Explicit fraction or percent ancestry unit required")
    tolerance = specification["ancestry_sum_tolerance"]
    require(type(tolerance) in (int, float) and math.isfinite(tolerance) and 0 <= tolerance <= .01,
            "Ancestry rounding tolerance must be in fraction units, between 0 and .01")
    raw, coverage, audit = study_base.read_metadata(path, samples, specification["sample_column"],
                                                   [*CATEGORICAL, *ANCESTRY])
    require(all(sample in raw for sample in samples), "Metadata does not cover cohort")
    require(all(item["status"] != "COLUMN_ABSENT" for item in coverage), "Missing authorized metadata column")
    missing = {value.strip().casefold() for value in metadata_base.DEFAULTS["missing_tokens"]}
    scale = 1. if unit == "fraction" else 100.
    aligned = []
    for sample in samples:
        row, annotations = raw[sample], {}
        for column in CATEGORICAL:
            value = row[column].strip()
            require(value != metadata_base.MISSING, "Reserved missing-category label")
            annotations[column] = None if value.casefold() in missing else value
        for column in ANCESTRY:
            value = row[column].strip()
            if value.casefold() in missing:
                annotations[column] = None
            else:
                number = float(value)
                require(math.isfinite(number) and 0 <= number <= scale, "Ancestry outside declared unit range")
                annotations[column] = number / scale
        observed = [annotations[column] for column in ANCESTRY if annotations[column] is not None]
        require(sum(observed) <= 1 + tolerance, "Observed ancestry components exceed one")
        if len(observed) == len(ANCESTRY):
            require(abs(sum(observed) - 1) <= tolerance, "Four ancestry components do not sum to one")
        aligned.append(annotations)
    coverage = [dict(field=field, n_measured=sum(row[field] is not None for row in aligned),
                     n_people=len(samples), status="AVAILABLE") for field in (*CATEGORICAL, *ANCESTRY)]
    audit.update(input_units=unit, output_units="fraction", scale_divisor=scale,
                 ancestry_sum_tolerance=tolerance, ancestry_columns=list(ANCESTRY),
                 categorical_columns=list(CATEGORICAL), coverage=coverage,
                 no_clinical_columns_exported=True, no_ancestry_estimation=True)
    return aligned, audit


def load_graph(spec, inputs, samples):
    require(set(spec) == {"id", "family", "manifest", "kinship_manifests"}, "Unexpected graph specification")
    graph_id, family = spec["id"], spec["family"]
    require(isinstance(graph_id, str) and re.fullmatch(r"[A-Za-z0-9_.:/-]+", graph_id), "Invalid graph ID")
    require(family in {"H", "R", "C", "R_plus_C"}, "Unknown graph family")
    path = inputs.add(spec["manifest"])
    manifest = strict_json(path)
    require(manifest.get("status") in {"COMPLETE_DESCRIPTIVE", "COMPLETE_NO_POSITIVE_EDGES"}
            and manifest.get("chromosomes") == AUTOSOMES and manifest.get("n_cohort") == len(samples),
            "Expected completed 22-autosome graph for this cohort")
    if family == "H":
        graph_key = manifest["configuration"]["config_id"]
        require(manifest.get("source_sha256", {}).get("sample_ids") == evidence.sample_hash(samples),
                "H source cohort hash differs")
        require(manifest.get("no_founder_classification") is True, "H scope is not descriptive")
        settings, node_name = manifest["parameters"], "graph_nodes.tsv"
    else:
        require(manifest.get("graph") == family and manifest.get("weight_units") == "dimensionless",
                "Weighted graph family/units differ")
        graph_key, settings, node_name = family, manifest["settings"], "nodes.private.tsv"
    require(settings.get("expected_samples") == len(samples), "Graph settings cohort differs")
    gammas = settings.get("resolutions", [])
    require(isinstance(gammas, list) and gammas and len(set(gammas)) == len(gammas)
            and all(type(g) in (int, float) and math.isfinite(g) and g > 0 for g in gammas),
            "Invalid resolution ledger")
    minimum = positive_integer(settings.get("min_community_size"), "min_community_size")
    nodes_path = inputs.adjacent(path, manifest, node_name)
    assignment_path = inputs.adjacent(path, manifest, "leiden_assignments.tsv")
    nodes, assignments = kinship_base.read_table(nodes_path), kinship_base.read_table(assignment_path)
    require([r["sample_id"] for r in nodes] == samples and [r["sample_id"] for r in assignments] == samples,
            "Graph node/assignment cohort or order differs")
    require(set(assignments[0]) == {"sample_id", *(f"community_res_{g:g}" for g in gammas)},
            "Assignment columns differ from resolution ledger")
    degrees = [int(row["degree"]) for row in nodes]
    require(all(value >= 0 for value in degrees) and sum(degrees) == 2 * manifest["n_edges"], "Invalid graph degrees")
    active = [degree > 0 for degree in degrees]
    require(sum(active) == manifest["n_active"], "Graph active denominator differs")
    if family == "H":
        require([int(r["node_id"]) for r in nodes] == list(range(len(samples))), "Invalid H node order")
    else:
        require([r["active"] for r in nodes] == [str(v) for v in active], "Weighted active flags differ")
    labels = {}
    for gamma in gammas:
        values = [int(row[f"community_res_{gamma:g}"]) for row in assignments]
        require(all(v >= -1 for v in values) and all(v == -1 for v, a in zip(values, active) if not a),
                "Invalid labels or assigned isolated node")
        require(all(n >= minimum for n in Counter(v for v in values if v >= 0).values()),
                "Community smaller than declared minimum")
        labels[float(gamma)] = values
    return dict(id=graph_id, family=family, key=graph_key, manifest_path=path, manifest=manifest,
                labels=labels, active=active, kinship_specs=spec["kinship_manifests"])


def bound_hash(mapping, suffix, expected):
    matches = [v for k, v in mapping.items() if k == suffix or k.endswith("/" + suffix)]
    require(matches == [expected], "Kinship summary is not bound uniquely to graph artifact: " + suffix)


def reuse_kinship(graph, inputs, pcrelate_sha, thresholds):
    """Import native diagnostic rows unchanged, with graph IDs and source receipts."""
    specs = graph["kinship_specs"]
    require(isinstance(specs, list) and specs, "Completed kinship manifests required")
    result, cells, graph_cells = [], set(), set()
    for spec in specs:
        path = inputs.add(spec)
        manifest = strict_json(path)
        expected_status = "COMPLETE_DESCRIPTIVE_EDGE_KINSHIP" if graph["family"] == "H" else "COMPLETE_DESCRIPTIVE_WEIGHTED_EDGE_KINSHIP"
        require(manifest.get("status") == expected_status and manifest.get("pcrelate_audit", {}).get("sha256") == pcrelate_sha,
                "Kinship status/source mismatch")
        require(all(manifest.get(key) is True for key in ("no_reclustering", "no_new_kinship", "no_pvalues")),
                "Unexpected kinship semantics")
        prefix = graph["manifest_path"].parent.name
        mapping = manifest.get("input_sha256", {})
        for filename in ("manifest.json", "leiden_assignments.tsv"):
            expected = inputs.records[str(graph["manifest_path"] if filename == "manifest.json" else graph["manifest_path"].parent / filename)]["sha256"]
            bound_hash(mapping, prefix + "/" + filename, expected)
        for filename, scope in (("graph_kinship_summary.tsv", "graph"), ("partition_kinship_summary.tsv", "partition")):
            table_path = inputs.adjacent(path, manifest, filename)
            key = "config_id" if graph["family"] == "H" else "graph"
            for row in kinship_base.read_table(table_path):
                if row[key] != graph["key"]:
                    continue
                phi = float(row["kinship_threshold"])
                require(phi in thresholds and int(row["n_cohort"]) == len(graph["active"])
                        and int(row["n_active"]) == sum(graph["active"]), "Kinship cohort/threshold differs")
                if scope == "partition":
                    gamma = float(row["resolution"])
                    require(gamma in graph["labels"], "Unknown kinship resolution")
                    labels = graph["labels"][gamma]
                    require(int(row["n_assigned"]) == sum(v >= 0 for v in labels)
                            and int(row["n_communities"]) == len(set(labels) - {-1}), "Kinship partition counts differ")
                    cell = (phi, gamma)
                    require(cell not in cells, "Duplicate kinship partition cell")
                    cells.add(cell)
                else:
                    require(phi not in graph_cells, "Duplicate kinship graph cell")
                    graph_cells.add(phi)
                result.append(dict(graph_id=graph["id"], family=graph["family"], scope=scope,
                                   source_manifest_sha256=spec["sha256"], native_fields=row))
    require(graph_cells == set(thresholds) and cells == set(itertools.product(thresholds, graph["labels"])),
            "Incomplete reused kinship grid")
    return result


def groups_for(labels, active):
    groups = [("cohort", "", list(range(len(labels)))),
              ("assigned", "", [i for i, v in enumerate(labels) if v >= 0]),
              ("unassigned", "", [i for i, v in enumerate(labels) if v == -1]),
              ("active_unassigned", "", [i for i, v in enumerate(labels) if v == -1 and active[i]]),
              ("isolated", "", [i for i, a in enumerate(active) if not a])]
    by_label = {}
    for i, label in enumerate(labels):
        if label >= 0:
            by_label.setdefault(label, []).append(i)
    return groups + [("community", label, indices) for label, indices in sorted(by_label.items())]


def describe(graphs, study, metadata):
    for graph in graphs:
        n = len(graph["active"])
        for gamma, labels in sorted(graph["labels"].items()):
            assigned = sum(label >= 0 for label in labels)
            for scope, label, indices in groups_for(labels, graph["active"]):
                base = dict(graph_id=graph["id"], family=graph["family"], resolution=gamma, scope=scope,
                            community_local=label, n_samples=len(indices), n_cohort=n, n_assigned=assigned,
                            fraction_cohort=len(indices)/n,
                            fraction_assigned=len(indices)/assigned if scope == "community" and assigned else None)
                yield "scopes", base
                for phi, components in study["assignments"].items():
                    counts = Counter(components[i] for i in indices if components[i] is not None)
                    known, largest = sum(counts.values()), max(counts.values(), default=None)
                    yield "components", dict(**base, kinship_threshold=phi, n_components=len(counts),
                        n_component_known=known, n_component_missing=len(indices)-known,
                        max_component_n=largest,
                        max_observed_component_fraction_all=largest/len(indices) if known else None,
                        max_component_fraction_known=largest/known if known else None,
                        component_status="DESCRIPTIVE" if known == len(indices) and known else
                            "DESCRIPTIVE_PARTIAL" if known else "NO_EVALUABLE")
                for field in CATEGORICAL:
                    counts = Counter(metadata[i][field] for i in indices)
                    missing = counts.get(None, 0); observed = len(indices)-missing
                    for value in sorted(set(counts)-{None}) + [None]:
                        count = counts.get(value, 0)
                        yield "categorical", dict(**base, field=field,
                            category=metadata_base.MISSING if value is None else value, is_missing=value is None,
                            n=count, n_observed=observed, n_missing=missing,
                            fraction_all=count/len(indices) if indices else None,
                            fraction_observed=count/observed if value is not None and observed else None)
                complete = [i for i in indices if all(metadata[i][f] is not None for f in ANCESTRY)]
                # Complete-vector means are separately available; per-field denominators may differ.
                for observation_scope, eligible in (("field_observed", indices), ("complete_vector", complete)):
                    for field in ANCESTRY:
                        values = [metadata[i][field] for i in eligible if metadata[i][field] is not None]
                        mean = sum(values)/len(values) if values else None
                        std = math.sqrt(sum((v-mean)**2 for v in values)/len(values)) if values else None
                        field_missing = sum(metadata[i][field] is None for i in indices)
                        yield "ancestry", dict(**base, field=field, observation_scope=observation_scope,
                            n_observed=len(values), n_unavailable_for_scope=len(indices)-len(values),
                            field_n_missing=field_missing,
                            n_excluded_incomplete_vector=len(indices)-len(complete)-field_missing
                                if observation_scope == "complete_vector" else 0,
                            n_complete_vector=len(complete),
                            mean_fraction=mean, std_fraction=std, std_ddof=0, unit="fraction")


def compare_partitions(specifications, graphs):
    from sklearn.metrics import adjusted_rand_score
    lookup, seen = {g["id"]: g for g in graphs}, set()
    for item in specifications:
        require(set(item) == {"left", "right", "left_resolution", "right_resolution"}, "Unexpected comparison specification")
        left, right = lookup[item["left"]], lookup[item["right"]]
        gl, gr = float(item["left_resolution"]), float(item["right_resolution"])
        require(gl in left["labels"] and gr in right["labels"], "Comparison resolution absent")
        key = tuple(sorted(((left["id"], gl), (right["id"], gr))))
        require(key[0] != key[1] and key not in seen, "Self or duplicate partition comparison")
        seen.add(key)
        a, b = left["labels"][gl], right["labels"][gr]
        common = [i for i, (x, y) in enumerate(zip(a, b)) if x >= 0 and y >= 0]
        n = len(a)
        base = dict(left_graph=left["id"], left_resolution=gl, right_graph=right["id"], right_resolution=gr,
                    n_cohort=n, n_both_assigned=len(common), fraction_cohort_both_assigned=len(common)/n,
                    n_left_only_assigned=sum(x >= 0 and y < 0 for x, y in zip(a, b)),
                    n_right_only_assigned=sum(x < 0 and y >= 0 for x, y in zip(a, b)),
                    n_neither_assigned=sum(x < 0 and y < 0 for x, y in zip(a, b)))
        na, nb = len({a[i] for i in common}), len({b[i] for i in common})
        usable = len(common) >= 2 and na >= 2 and nb >= 2
        yield "comparisons", dict(**base, left_n_communities_common=na, right_n_communities_common=nb,
            ari=float(adjusted_rand_score([a[i] for i in common], [b[i] for i in common])) if usable else None,
            status="DESCRIPTIVE_COMMON_ASSIGNED" if usable else "NO_EVALUABLE_DEGENERATE_OR_INSUFFICIENT")
        # -1 remains a reporting row/column in the full-cohort table, never an ARI community.
        counts = Counter(zip(a, b))
        rows, columns = Counter(a), Counter(b)
        for (x, y), count in sorted(counts.items()):
            yield "crosstab", dict(**base, left_community_local=x, right_community_local=y,
                either_unassigned=x < 0 or y < 0, n=count, fraction_cohort=count/n,
                fraction_left_scope=count/rows[x], fraction_right_scope=count/columns[y])


def write_rows(output, rows, budget):
    """Stream each table; no all-comparisons table is accumulated in memory."""
    from contextlib import ExitStack
    counts, writers = Counter(), {}
    with ExitStack() as stack:
        for name, row in rows:
            if name not in writers:
                handle = stack.enter_context((output/(name+".tsv")).open("x", newline="", encoding="utf-8"))
                writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t", lineterminator="\n")
                writer.writeheader(); writers[name] = writer
            writers[name].writerow({k: "NA" if v is None else v for k, v in row.items()})
            counts[name] += 1
            if sum(counts.values()) % 1000 == 0:
                budget.check()
    return dict(counts)


def run(contract_path, expected_contract_sha256, output_dir, max_memory_mb):
    contract_path = Path(contract_path).resolve()
    output = Path(output_dir).absolute()
    require(not output.exists() and not output.is_symlink(), "Output already exists; no overwrite")
    inputs = Inputs(contract_path.parent, max_memory_mb)
    inputs.add(dict(path=str(contract_path), sha256=expected_contract_sha256))
    contract = strict_json(contract_path)
    require(set(contract) == {"schema", "expected_samples", "samples", "study_contract", "metadata", "graphs", "comparisons"}
            and contract["schema"] == SCHEMA, "Unexpected input contract schema/keys")
    n = positive_integer(contract["expected_samples"], "expected_samples")
    samples = evidence.read_samples(inputs.add(contract["samples"]), n)
    study_path = inputs.add(contract["study_contract"])
    study_manifest = strict_json(study_path)
    thresholds = study_manifest["thresholds"]
    require(thresholds == [.0221, .0442], "Preserve both declared PC-Relate cutoffs")
    kinship_sha = study_manifest["kinship_audit"]["sha256"]
    inputs.adjacent(study_path, study_manifest, "persons.private.tsv")
    study = biological.read_study(study_path, samples, kinship_sha, thresholds)
    metadata_path = inputs.add(contract["metadata"]["file"])
    metadata, metadata_audit = read_annotations(metadata_path, samples, contract["metadata"])
    missing = {value.strip().casefold() for value in metadata_base.DEFAULTS["missing_tokens"]}
    shared = [field for field in CATEGORICAL if field in study["metadata_fields"]]
    for person, annotations in zip(study["people"], metadata):
        for field in shared:
            value = person[field].strip()
            require(annotations[field] == (None if value.casefold() in missing else value),
                    "Original metadata differs from authenticated study projection: " + field)
    metadata_audit["study_projection_fields_verified"] = shared
    require(isinstance(contract["graphs"], list) and contract["graphs"], "Explicit graph ledger required")
    graphs = [load_graph(spec, inputs, samples) for spec in contract["graphs"]]
    require(len({g["id"] for g in graphs}) == len(graphs), "Duplicate graph ID")
    require(len({str(g["manifest_path"]) for g in graphs}) == len(graphs), "Same graph submitted twice")
    require(isinstance(contract["comparisons"], list), "Explicit comparison list required")
    reused = [row for graph in graphs for row in reuse_kinship(graph, inputs, kinship_sha, thresholds)]
    # Validate comparison ledger before creating outputs, without retaining cross-tabs.
    for _ in compare_partitions(contract["comparisons"], graphs):
        inputs.check()
    require(all(not Path(p).is_relative_to(output) for p in inputs.records), "Output would contain an input")
    inputs.recheck()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    counts = write_rows(output, itertools.chain(describe(graphs, study, metadata),
                       compare_partitions(contract["comparisons"], graphs)), inputs)
    with (output/"kinship_reused.json").open("x") as handle:
        json.dump(reused, handle, indent=2, allow_nan=False); handle.write("\n")
    inputs.recheck()
    source_paths = [Path(__file__), Path(metadata_base.__file__), Path(study_base.__file__),
                    Path(biological.__file__), Path(kinship_base.__file__), Path(evidence.__file__),
                    Path(biological.diagnostics.__file__), Path(metadata_base.saved.__file__),
                    Path(kinship_base.historical.__file__)]
    manifest = dict(schema="r02_community_evidence_v1", status=STATUS, n_cohort=n,
        sample_ids_sha256=evidence.sample_hash(samples), chromosomes=AUTOSOMES,
        n_graphs=len(graphs), n_partitions=sum(len(g["labels"]) for g in graphs), table_rows=counts,
        graph_ledger=[dict(graph_id=g["id"], family=g["family"], source_key=g["key"],
            manifest_sha256=inputs.records[str(g["manifest_path"])]["sha256"], resolutions=list(g["labels"])) for g in graphs],
        metadata_audit=metadata_audit, study_provenance=study["provenance"],
        input_files=inputs.records, source_sha256={p.name: sha256(p) for p in source_paths},
        package_versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
        memory=dict(max_memory_mb=max_memory_mb, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    admission="32 times cumulative serialized input bytes; RSS checked; external hard limit still required"),
        no_reclustering=True, no_new_kinship=True, no_new_target=True, no_pvalues=True,
        no_population_validation=True, incremental_rare_utility="NOT_ESTIMATED",
        contains_individual_identifiers=False, public_distribution_allowed=False,
        limitations=["Descriptive transductive partitions; no independent evaluation or incremental rare-effect attribution",
            "Ancestry fields are pre-existing estimates with undocumented uncertainty, not new estimates or truth",
            "PC-Relate components are cutoff-dependent connections, not certified families or independent replicas",
            "Category association does not distinguish collection, region, ancestry and absent sequencing-batch metadata",
            "DP, GQ, sequencing batch and local callability are not supplied; no adjustment for them is claimed",
            "ARI excludes anyone unassigned in either partition; no common assigned support is imputed",
            "Local community IDs and Leiden resolutions are not shared biological identities/scales",
            "H, R, C and R_plus_C weights are not compared or pooled; kinship summaries retain their native fields",
            "Per-field ancestry means can use different people; complete-vector rows are reported separately",
            "Private aggregate cells may identify small groups; absence of sample IDs does not authorize publication"],
        outputs_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    with (output/"manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False); handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--expected-inputs-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-memory-mb", required=True, type=int)
    args = parser.parse_args()
    import os
    os.umask(0o077)
    result = run(args.inputs, args.expected_inputs_sha256, args.output_dir, args.max_memory_mb)
    print(json.dumps({key: result[key] for key in ("status", "n_cohort", "n_graphs", "n_partitions")}))


if __name__ == "__main__":
    main()
