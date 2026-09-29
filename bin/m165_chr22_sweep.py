#!/usr/bin/env python3
"""Private, descriptive chr22 graph sweep over saved M14 pair summaries.

Preparation streams the saved table once and authenticates its compressed bytes.
Each graph task reuses M16.5's graph/Leiden implementation, never NMF or founder
classification. Seeds measure algorithmic variation, not biological replication.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path


PAIR_FIELDS = ("sample_a", "sample_b", "n_segments", "total_shared_bp",
               "n_shared_variants_total", "max_segment_bp")
GRAPH_FIELDS = ("length_bp", "gap_bp", "min_shared", "min_edge_bp",
                "min_max_segment_bp")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def samples_from(path, expected):
    samples = Path(path).read_text(encoding="utf-8").splitlines()
    require(len(samples) == expected and len(set(samples)) == len(samples),
            "Sample count/uniqueness differs from declared cohort")
    require(all(s and s.strip() == s and not any(c.isspace() for c in s)
                for s in samples), "Invalid sample identifier")
    return samples


def validate_settings(settings):
    required = {"configurations", "expected_samples", "resolutions", "n_seeds",
                "seed", "min_community_size", "consensus_resolution"}
    require(set(settings) == required, "Settings keys differ from sweep contract")
    for key in ("expected_samples", "n_seeds", "min_community_size"):
        require(type(settings[key]) is int and settings[key] > 0, f"Invalid {key}")
    require(type(settings["seed"]) is int and settings["seed"] >= 0, "Invalid seed")
    resolutions = settings["resolutions"]
    require(isinstance(resolutions, list) and resolutions and
            len(set(resolutions)) == len(resolutions) and
            all(isinstance(r, (int, float)) and math.isfinite(r) and r > 0 for r in resolutions),
            "Resolutions must be unique positive finite values")
    require(settings["consensus_resolution"] in resolutions,
            "Consensus resolution must be in the declared grid")
    plans = []
    for item in settings["configurations"]:
        require(set(item) == set(GRAPH_FIELDS), "Graph configuration keys differ")
        require(all(type(item[k]) is int for k in GRAPH_FIELDS), "Graph parameters must be integers")
        require(all(item[k] > 0 for k in GRAPH_FIELDS if k != "min_max_segment_bp")
                and item["min_max_segment_bp"] == 0,
                "This pair-summary sweep requires positive thresholds and U=0")
        effective = max(item["min_shared"], math.ceil((item["length_bp"] - 1) / item["gap_bp"]) + 1)
        source = f'L{item["length_bp"]}_G{item["gap_bp"]}_N{effective}'
        config_id = f'{source}_T{item["min_edge_bp"]}_U0'
        plans.append(dict(item, min_shared_effective=effective,
                          source_config_id=source, config_id=config_id))
    require(plans and len({p["config_id"] for p in plans}) == len(plans),
            "Graph configurations must be nonempty and unique")
    return plans


class HashingReader:
    """Hash compressed bytes while gzip performs its single forward read."""
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        block = self.handle.read(size)
        self.digest.update(block)
        return block


def write_pairs(path, rows):
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS, delimiter="\t",
                                        lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def prepare(pair_summary, configuration_summary, sample_ids, output, settings):
    plans = validate_settings(settings)
    output = Path(output)
    require(not output.exists(), "Preparation destination already exists")
    samples = samples_from(sample_ids, settings["expected_samples"])
    sample_set = set(samples)
    source_ids = {p["source_config_id"] for p in plans}
    with Path(configuration_summary).open(encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle, delimiter="\t"))
    targets = {}
    for source in source_ids:
        matched = [r for r in summaries if r["config_id"] == source]
        require(len(matched) == 1, f"Expected one measured source configuration: {source}")
        targets[source] = matched[0]
    retained = {source: [] for source in source_ids}
    seen = {source: set() for source in source_ids}
    scanned = 0
    with Path(pair_summary).open("rb") as raw:
        hashed = HashingReader(raw)
        with gzip.GzipFile(fileobj=hashed, mode="rb") as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                require({"config_id", *PAIR_FIELDS}.issubset(reader.fieldnames or []),
                        "Missing saved pair-summary columns")
                for row in reader:
                    scanned += 1
                    source = row["config_id"]
                    if source not in source_ids:
                        continue
                    a, b = row["sample_a"], row["sample_b"]
                    require(a != b and {a, b} <= sample_set, "Pair outside cohort or self-pair")
                    key = tuple(sorted((a, b)))
                    require(key not in seen[source], "Duplicate or reversed pair in saved source")
                    seen[source].add(key)
                    values = {k: int(row[k]) for k in PAIR_FIELDS[2:]}
                    require(all(v > 0 for v in values.values()), "Nonpositive pair summary value")
                    require(values["max_segment_bp"] <= values["total_shared_bp"],
                            "Longest segment exceeds total pair length")
                    retained[source].append(dict(sample_a=key[0], sample_b=key[1], **values))
        source_sha = hashed.digest.hexdigest()
    totals = {}
    for source, rows in retained.items():
        actual = {"n_pairs": len(rows)}
        for key in ("n_segments", "total_shared_bp", "n_shared_variants_total"):
            actual[key] = sum(r[key] for r in rows)
        require(all(actual[k] == int(targets[source][k]) for k in actual),
                f"Saved pair rows fail source aggregate parity: {source}")
        totals[source] = actual
        rows.sort(key=lambda r: (r["sample_a"], r["sample_b"]))
    input_hashes = {"pair_configuration_summary": source_sha,
                    "configuration_summary": sha256(configuration_summary),
                    "sample_ids": sha256(sample_ids)}
    output.mkdir(parents=True, exist_ok=False)
    for plan in plans:
        folder = output / plan["config_id"]
        folder.mkdir()
        write_pairs(folder / "pairs.tsv.gz", retained[plan["source_config_id"]])
        with (folder / "samples.txt").open("x", encoding="utf-8") as handle:
            handle.write("\n".join(samples) + "\n")
        payload = {"schema_version": 1, "status": "PREPARED_AGGREGATE_PARITY",
                   "chromosome": "22", "configuration": plan,
                   "settings": {k: v for k, v in settings.items() if k != "configurations"},
                   "source_totals": totals[plan["source_config_id"]],
                   "source_sha256": input_hashes,
                   "prepared_sha256": {name: sha256(folder / name) for name in ("pairs.tsv.gz", "samples.txt")},
                   "contains_individual_identifiers": True, "public_distribution_allowed": False}
        write_json(folder / "input_manifest.json", payload)
    receipt = {"status": "PREPARED", "n_configurations": len(plans),
               "n_source_configurations": len(source_ids), "rows_scanned": scanned,
               "source_sha256": input_hashes, "source_totals": totals,
               "adapter_sha256": sha256(__file__)}
    write_json(output / "preparation.json", receipt)
    return receipt


def run(configuration_dir, output, core_script):
    import importlib.util
    import sys
    import numpy as np
    import pandas as pd

    configuration_dir, output = Path(configuration_dir), Path(output)
    require(not output.exists(), "Graph destination already exists")
    manifest = json.loads((configuration_dir / "input_manifest.json").read_text())
    require(manifest["status"] == "PREPARED_AGGREGATE_PARITY" and manifest["chromosome"] == "22",
            "Wrong prepared-input contract")
    for name, expected in manifest["prepared_sha256"].items():
        require(name in ("pairs.tsv.gz", "samples.txt") and
                sha256(configuration_dir / name) == expected, "Prepared input hash mismatch")
    require(set(manifest["prepared_sha256"]) == {"pairs.tsv.gz", "samples.txt"}, "Incomplete prepared hashes")
    settings, plan = manifest["settings"], manifest["configuration"]
    canonical = validate_settings(dict(settings, configurations=[{k: plan[k] for k in GRAPH_FIELDS}]))[0]
    require(plan == canonical, "Prepared configuration identity differs from its parameters")
    samples = samples_from(configuration_dir / "samples.txt", settings["expected_samples"])
    pairs = pd.read_csv(configuration_dir / "pairs.tsv.gz", sep="\t",
                        dtype={"sample_a": str, "sample_b": str})
    # The saved direct-segment Jaccard is not a statistic; this column is unused
    # by the fixed log1p transform but required by the shared core's schema.
    pairs["mean_jaccard"] = np.nan
    spec = importlib.util.spec_from_file_location("m165_shared_core", core_script)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    weighted = core.aggregate_pair_weights(pairs, None, "log1p", min_max_segment_bp=0)
    matrix, kept = core.build_sparse_matrix(weighted, samples, plan["min_edge_bp"])
    graph = core.sparse_to_igraph(matrix, samples)
    output.mkdir(parents=True, exist_ok=False)
    kept_pairs = weighted.loc[kept].copy()
    core.save_graph(output, matrix, graph, samples, kept_pairs, plan["min_edge_bp"], "log1p", 0)
    kept_pairs.to_csv(output / "retained_pairs.private.tsv.gz", sep="\t", index=False,
                      compression={"method": "gzip", "mtime": 0})
    if graph.ecount():
        assignments, qualities, consensus, _, memberships = core.run_leiden_multiresolution(
            graph, settings["resolutions"], settings["n_seeds"], settings["min_community_size"],
            settings["seed"], settings["consensus_resolution"])
        require({"rb_quality", "is_representative"}.issubset(qualities.columns),
                "Shared M16.5 core must contain the corrected RB-quality ranking")
        ari = core.compute_ari_multi_seed(memberships)
    else:
        assignments = pd.DataFrame({f"community_res_{r:g}": [-1] * len(samples) for r in settings["resolutions"]})
        qualities = pd.DataFrame(columns=["resolution", "seed", "modularity", "rb_quality", "is_representative"])
        consensus, memberships, ari = None, {}, pd.DataFrame()
    core.save_leiden(output, samples, assignments, qualities, consensus,
                     settings["consensus_resolution"], ari_df=ari)
    active = np.asarray(graph.degree()) > 0
    rows = []
    for resolution in settings["resolutions"]:
        labels = assignments[f"community_res_{resolution:g}"].to_numpy()
        require(np.all(labels[~active] == -1), "Isolated nodes received a community")
        assigned = labels >= 0
        _, sizes = np.unique(labels[assigned], return_counts=True)
        rows.append({"config_id": plan["config_id"], "resolution": resolution,
                     "n_cohort": len(samples), "n_active": int(active.sum()),
                     "n_isolated": int((~active).sum()), "n_assigned": int(assigned.sum()),
                     "n_active_unassigned": int((active & ~assigned).sum()),
                     "n_communities": len(sizes), "largest_community": int(sizes.max()) if sizes.size else 0})
    pd.DataFrame(rows).to_csv(output / "resolution_summary.tsv", sep="\t", index=False)
    with gzip.open(output / "seed_memberships.private.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("resolution", "seed", "sample_id", "community_raw", "active"))
        for resolution, vectors in memberships.items():
            seeds = qualities.loc[qualities.resolution == resolution, "seed"].tolist()
            require(len(seeds) == len(vectors), "Seed metadata does not match memberships")
            for seed, vector in zip(seeds, vectors):
                writer.writerows((resolution, seed, sample, int(label), int(is_active))
                                 for sample, label, is_active in zip(samples, vector, active))
    versions = {}
    for package in ("numpy", "pandas", "scipy", "igraph", "leidenalg", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    result = {"status": "COMPLETE_DESCRIPTIVE" if graph.ecount() else "COMPLETE_NO_EDGES",
              "configuration": plan, "parameters": settings, "chromosome": "22",
              "n_cohort": len(samples), "n_active": int(active.sum()), "n_edges": graph.ecount(),
              "input_manifest_sha256": sha256(configuration_dir / "input_manifest.json"),
              "source_sha256": manifest["source_sha256"], "adapter_sha256": sha256(__file__),
              "core_sha256": sha256(core_script), "package_versions": versions,
              "contains_individual_identifiers": True, "public_distribution_allowed": False,
              "scope": "Unphased rare co-sharing, chr22, transductive and descriptive; not validated populations, IBD, LAI or supervised targets",
              "no_nmf": True, "no_founder_classification": True, "no_confirmatory_pvalues": True,
              "quality_semantics": "rb_quality selects representative within fixed graph and resolution only; scores do not select resolution",
              "consensus_semantics": "coassignment frequencies across algorithmic seeds, not ancestry probabilities",
              "outputs_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    preparation = sub.add_parser("prepare")
    for name in ("pair-summary", "configuration-summary", "sample-ids", "output", "settings-base64"):
        preparation.add_argument("--" + name, required=True)
    execution = sub.add_parser("run")
    for name in ("configuration-dir", "output", "core-script"):
        execution.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        settings = json.loads(base64.b64decode(args.settings_base64, validate=True))
        result = prepare(args.pair_summary, args.configuration_summary, args.sample_ids, args.output, settings)
    else:
        result = run(args.configuration_dir, args.output, args.core_script)
    print(json.dumps({k: result[k] for k in ("status", "n_configurations", "n_cohort", "n_edges") if k in result}))


if __name__ == "__main__":
    main()
