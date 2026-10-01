#!/usr/bin/env python3
"""Describe pre-existing PC-Relate on authenticated R/C/R+C graph edges.

No kinship is estimated, nobody is excluded, and communities are not refitted.
Dense numeric arrays bound memory; there is no Python tuple for every pair.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy import sparse

import m165_graph_kinship as historical
import r02_genomic_pair_evidence as evidence


AUTOSOMES = [str(c) for c in range(1, 23)]
EDGE_CLASSES = historical.EDGE_CLASSES
ABSENT, NONFINITE, FINITE = 0, 1, 2
DEFAULT_THRESHOLDS = (0.0221, 0.0442)


def read_table(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        evidence.require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
                         "Missing or duplicated table columns")
        rows = list(reader)
    evidence.require(all(None not in row and None not in row.values() for row in rows), "Malformed table row")
    return rows


def authenticate(results_dir, samples):
    root = Path(results_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    evidence.require(manifest["status"] == "COMPLETE_DESCRIPTIVE_WEIGHTED_COMMUNITIES"
                     and manifest["chromosomes"] == AUTOSOMES and manifest["n_cohort"] == len(samples),
                     "Expected completed 22-autosome weighted communities for this cohort")
    records = manifest["graphs"]
    names = [r["graph"] for r in records]
    evidence.require(names and len(names) == len(set(names)) and set(names) <= {"R", "C", "R_plus_C"},
                     "Invalid weighted graph identities")
    inputs = {str(manifest_path): evidence.sha256(manifest_path)}
    target = np.zeros((len(samples), len(samples)), dtype=bool)
    graphs = []
    for record in records:
        name = record["graph"]
        directory = root / name
        graph_manifest = directory / "manifest.json"
        evidence.require(json.loads(graph_manifest.read_text()) == record, "Graph manifest differs from campaign manifest")
        inputs[str(graph_manifest)] = evidence.sha256(graph_manifest)
        hashes = record["outputs_sha256"]
        evidence.require({"weights.npz", "nodes.private.tsv", "leiden_assignments.tsv"} <= set(hashes),
                         "Missing authenticated weights, nodes or assignments")
        for filename, expected in hashes.items():
            evidence.require(Path(filename).name == filename, "Graph output must have a simple filename")
            path = directory / filename
            evidence.require(evidence.sha256(path) == expected, "Weighted graph output hash mismatch")
            inputs[str(path)] = expected
        nodes = read_table(directory / "nodes.private.tsv")
        assignments = read_table(directory / "leiden_assignments.tsv")
        evidence.require([r["sample_id"] for r in nodes] == samples
                         and [r["sample_id"] for r in assignments] == samples,
                         "Weighted nodes or assignments differ from fixed sample order")
        matrix = sparse.load_npz(directory / "weights.npz").tocsr()
        evidence.require(matrix.has_canonical_format and matrix.shape == target.shape
                         and np.isfinite(matrix.data).all() and np.all(matrix.data > 0)
                         and np.all(matrix.diagonal() == 0), "Invalid weighted adjacency matrix")
        difference = matrix - matrix.T
        evidence.require(not difference.nnz or np.max(np.abs(difference.data)) <= 1e-12,
                         "Asymmetric weighted adjacency matrix")
        degree = np.diff(matrix.indptr)
        weighted_degree = np.asarray(matrix.sum(axis=1)).ravel()
        active = degree > 0
        evidence.require(record["chromosomes"] == AUTOSOMES and record["n_cohort"] == len(samples)
                         and record["n_edges"] * 2 == matrix.nnz and record["n_active"] == int(active.sum())
                         and record["weight_units"] == "dimensionless", "Graph counts or units disagree with weights")
        evidence.require(np.array_equal([int(r["degree"]) for r in nodes], degree)
                         and np.allclose([float(r["weighted_degree"]) for r in nodes], weighted_degree,
                                         rtol=1e-10, atol=1e-12)
                         and [r["active"] for r in nodes] == [str(bool(a)) for a in active],
                         "Node measurements disagree with weights")
        settings = record["settings"]
        evidence.require(settings == manifest["parameters"] and settings["resolutions"] == [0.5, 1.0, 2.0],
                         "Weighted community configurations disagree")
        labels = {}
        for gamma in settings["resolutions"]:
            column = f"community_res_{gamma:g}"
            array = np.asarray([int(r[column]) for r in assignments], dtype=np.int64)
            evidence.require(np.all(array >= -1) and np.all(array[~active] == -1),
                             "Invalid assignment or isolated person assigned to a community")
            _, sizes = np.unique(array[array >= 0], return_counts=True)
            evidence.require(np.all(sizes >= settings["min_community_size"]), "Community smaller than declared minimum")
            labels[float(gamma)] = array
        upper = sparse.triu(matrix, k=1, format="coo")
        target[upper.row, upper.col] = True
        target[upper.col, upper.row] = True
        graphs.append(dict(name=name, directory=directory, labels=labels, n_active=int(active.sum()),
                           n_cohort=len(samples), weights_sha256=hashes["weights.npz"]))
    return graphs, target, inputs, manifest


def stream_kinship_arrays(path, sample_index, target, expected_sha256):
    """Same full-file/duplicate semantics as historical.stream_kinship, numeric storage."""
    evidence.require(re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256 or ""), "Expected PC-Relate SHA256 must contain 64 hexadecimal characters")
    n = len(sample_index)
    evidence.require(set(sample_index.values()) == set(range(n)) and target.shape == (n, n)
                     and target.dtype == bool and np.array_equal(target, target.T)
                     and not np.any(target.diagonal()), "Invalid sample mapping or target edge matrix")
    states = np.zeros((n, n), dtype=np.uint8)
    kinship = np.full((n, n), np.nan, dtype=np.float64)
    path = Path(path)
    before = historical.stat_signature(path.stat())
    digest = hashlib.sha256()
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
            evidence.require(len(header) == len(set(header)) and {"ID1", "ID2", "kin"} <= set(header),
                             "PC-Relate requires unique ID1, ID2 and kin columns")
            ia, ib, ik = (header.index(c) for c in ("ID1", "ID2", "kin"))
            for row in reader:
                if not row:
                    continue
                audit["n_source_rows"] += 1
                evidence.require(len(row) == len(header), "Malformed PC-Relate row width")
                evidence.require(row[ia] and row[ib], "Empty PC-Relate endpoint")
                a, b = sample_index.get(row[ia]), sample_index.get(row[ib])
                if a is None or b is None or not target[a, b]:
                    audit["n_rows_not_target"] += 1
                    continue
                evidence.require(states[a, b] == ABSENT, "Duplicate retained unordered PC-Relate pair")
                raw_value = row[ik].strip()
                if raw_value.lower() in {"", ".", "na", "nan", "null"}:
                    value = math.nan
                else:
                    try:
                        value = float(raw_value)
                    except ValueError:
                        raise ValueError("Invalid retained PC-Relate kinship value") from None
                state = FINITE if math.isfinite(value) else NONFINITE
                states[a, b] = states[b, a] = state
                if state == FINITE:
                    kinship[a, b] = kinship[b, a] = value
        except csv.Error:
            raise ValueError("Invalid PC-Relate TSV syntax") from None
    evidence.require(digest.hexdigest() == expected_sha256.lower(), "PC-Relate full-file SHA256 mismatch")
    evidence.require(historical.stat_signature(path.stat()) == before, "PC-Relate source changed while streaming")
    upper = np.triu(target, 1)
    audit.update(sha256=digest.hexdigest(), n_retained_union_edges=int(upper.sum()),
                 n_retained_source_pairs=int(np.count_nonzero(upper & (states != ABSENT))),
                 n_retained_finite=int(np.count_nonzero(upper & (states == FINITE))),
                 n_retained_missing_nonfinite=int(np.count_nonzero(upper & (states == NONFINITE))),
                 n_retained_missing_absent=int(np.count_nonzero(upper & (states == ABSENT))))
    return kinship, states, audit


def metrics(weights, kinship, states, threshold):
    evidence.require(weights.shape == kinship.shape == states.shape and weights.ndim == 1,
                     "Mismatched edge vectors")
    evidence.require(np.isfinite(weights).all() and np.all(weights > 0)
                     and np.all(np.isin(states, [ABSENT, NONFINITE, FINITE]))
                     and np.isfinite(kinship[states == FINITE]).all(), "Invalid edge measurements")
    finite = states == FINITE
    selected = finite & (kinship >= threshold)
    masks = dict(observed_kinship=finite, kin_ge_threshold=selected,
                 kin_lt_threshold=finite & ~selected, missing_absent=states == ABSENT,
                 missing_nonfinite=states == NONFINITE, missing_kinship=~finite)
    result = dict(n_edges=len(weights), sum_weight=float(weights.sum(dtype=np.float64)))
    for name, mask in masks.items():
        result["n_" + name] = int(mask.sum())
        result["weight_" + name] = float(weights[mask].sum(dtype=np.float64))
    for denominator, count, weight in (("all_edges", result["n_edges"], result["sum_weight"]),
                                       ("observed_edges", result["n_observed_kinship"], result["weight_observed_kinship"])):
        result["fraction_n_kin_ge_" + denominator] = result["n_kin_ge_threshold"] / count if count else None
        result["fraction_weight_kin_ge_" + denominator] = result["weight_kin_ge_threshold"] / weight if weight else None
    return result


def summarize(graphs, kinship, states, thresholds):
    graph_rows, partition_rows = [], []
    for graph in graphs:
        upper = sparse.triu(sparse.load_npz(graph["directory"] / "weights.npz"), k=1, format="coo")
        a, b, weight = upper.row, upper.col, upper.data
        values, state = kinship[a, b], states[a, b]
        for threshold in thresholds:
            base = dict(graph=graph["name"], kinship_threshold=threshold, n_cohort=graph["n_cohort"],
                        n_active=graph["n_active"], n_isolated=graph["n_cohort"] - graph["n_active"])
            total = metrics(weight, values, state, threshold)
            graph_rows.append(dict(**base, **total))
            for gamma, labels in sorted(graph["labels"].items()):
                assigned = (labels[a] >= 0) & (labels[b] >= 0)
                within = assigned & (labels[a] == labels[b])
                masks = dict(within_assigned=within, between_assigned=assigned & ~within,
                             with_unassigned=~assigned)
                parts = {name: metrics(weight[mask], values[mask], state[mask], threshold)
                         for name, mask in masks.items()}
                for key, expected in total.items():
                    if key.startswith("fraction_"):
                        continue
                    observed = sum(part[key] for part in parts.values())
                    evidence.require(observed == expected if key.startswith("n_") else math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12),
                                     "Partition edge classes do not sum to graph totals")
                partition_rows.append(dict(**base, resolution=gamma,
                    n_assigned=int(np.count_nonzero(labels >= 0)), n_communities=len(np.unique(labels[labels >= 0])),
                    **{"all_" + key: value for key, value in total.items()},
                    **{name + "_" + key: value for name, part in parts.items() for key, value in part.items()}))
    return graph_rows, partition_rows


def run(results_dir, samples, expected_samples, pcrelate_file, expected_sha256, output_dir,
        thresholds=DEFAULT_THRESHOLDS):
    root, output = Path(results_dir), Path(output_dir)
    evidence.require(not output.exists(), "Output directory exists; no overwrite")
    evidence.require(not output.resolve().is_relative_to(root.resolve()), "Output must be outside input results")
    thresholds = tuple(thresholds)
    evidence.require(thresholds and len(set(thresholds)) == len(thresholds)
                     and all(math.isfinite(t) and 0 <= t <= .5 for t in thresholds), "Invalid kinship thresholds")
    samples_path = Path(samples)
    ids = evidence.read_samples(samples_path, expected_samples)
    graphs, target, inputs, source_manifest = authenticate(root, ids)
    inputs[str(samples_path)] = evidence.sha256(samples_path)
    kinship, states, audit = stream_kinship_arrays(pcrelate_file, {s: i for i, s in enumerate(ids)}, target, expected_sha256)
    graph_rows, partition_rows = summarize(graphs, kinship, states, thresholds)
    for filename, digest in inputs.items():
        evidence.require(evidence.sha256(filename) == digest, "Weighted graph source changed during summary")
    output.mkdir(parents=True, exist_ok=False)
    historical.write_table(output / "graph_kinship_summary.tsv", graph_rows)
    historical.write_table(output / "partition_kinship_summary.tsv", partition_rows)
    result = dict(status="COMPLETE_DESCRIPTIVE_WEIGHTED_EDGE_KINSHIP", chromosomes=AUTOSOMES,
        n_graphs=len(graphs), n_thresholds=len(thresholds), n_graph_rows=len(graph_rows), n_partition_rows=len(partition_rows),
        n_cohort=len(ids), sample_ids_sha256=evidence.sample_hash(ids), parameters=dict(thresholds=thresholds),
        pcrelate_audit=audit, input_sha256=inputs, source_graph_parameters=source_manifest["parameters"],
        source_sha256={Path(p).name: evidence.sha256(p) for p in (__file__, historical.__file__, evidence.__file__)},
        no_reclustering=True, no_new_kinship=True, no_pvalues=True, contains_individual_identifiers=False,
        public_distribution_allowed=False,
        semantics=dict(unit="unordered retained positive-weight graph edge, counted once; not all possible pairs",
            weight="original dimensionless R, C or normalized R_plus_C weight; never base pairs or kinship",
            within_assigned="both endpoints assigned to the same reported community",
            between_assigned="both endpoints assigned to different reported communities",
            with_unassigned="at least one endpoint has label -1; never a community formed by unassigned people",
            missing="absent source row and non-finite kinship are distinct; neither means below threshold",
            fractions="separate counts and weight; >=threshold divided by all retained or only finite-kinship edges; empty denominator NA",
            cutoff="operational historical thresholds; neither family certification nor proof of independence below threshold",
            dependence="edges reuse people and chromosomes; resolutions/seeds are not biological replicates",
            validation="pre-existing PC-Relate includes these people and autosomes; descriptive reuse, not independent population validation",
            privacy="aggregate-only tables remain private; small communities can expose relationships"),
        outputs_sha256={p.name: evidence.sha256(p) for p in output.iterdir() if p.is_file()})
    with (output / "manifest.json").open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results-dir", "samples", "pcrelate-file", "expected-sha256", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-samples", type=int, default=2619)
    parser.add_argument("--thresholds", default="0.0221,0.0442")
    args = vars(parser.parse_args(argv))
    args["thresholds"] = tuple(float(t) for t in args["thresholds"].split(","))
    result = run(**args)
    print(json.dumps({k: result[k] for k in ("status", "n_graphs", "n_thresholds", "n_partition_rows")}))


if __name__ == "__main__":
    main()
