#!/usr/bin/env python3
"""Stream descriptive M14 configuration diagnostics; never select a winner.

Consumes saved pair/configuration summaries, not genotypes. Aggregate input must
be sorted by configuration and canonical lexical pair (the autosomal writer's
order). Single-chromosome input uses the original M14 sample-axis pair groups.
These contracts detect duplicates without storing a matrix for each setting.
One cohort-wide PC-Relate mapping is shared across all configurations; degree
accumulators scale as configurations * thresholds * people, not people squared.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re

import m165_chr22_sweep as sweep
import m165_graph_kinship as kinship


STATUS = "BIOLOGICAL_DESCRIPTIVE_NOT_VALIDATED"
DEFAULT_THRESHOLDS = (0.0221, 0.0442)
AUTOSOMES = [str(c) for c in range(1, 23)]
COUNTS = ("n_pairs", "n_segments", "total_shared_bp", "n_shared_variants_total")
CONFIG_FIELDS = ("config_id", "max_gap_bp", "min_length_bp", "min_shared_effective", *COUNTS)
PAIR_FIELDS = ("config_id", *sweep.PAIR_FIELDS)
KIN_STATES = ("kin_ge_threshold", "kin_lt_threshold", "missing_absent", "missing_nonfinite")
require = sweep.require


def integer(value, name, minimum=0):
    require(isinstance(value, str) and re.fullmatch(r"[0-9]+", value) is not None,
            f"Invalid integer field: {name}")
    result = int(value)
    require(result >= minimum, f"Integer below minimum: {name}")
    return result


def validate_hash(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
            f"Expected SHA256 is invalid: {name}")
    return value.lower()


def table_reader(handle, required):
    reader = csv.DictReader(handle, delimiter="\t", strict=True)
    names = reader.fieldnames or []
    require(names and len(names) == len(set(names)) and set(required) <= set(names),
            "Missing or duplicated table columns")
    for row in reader:
        require(None not in row and None not in row.values(), "Malformed table row width")
        yield row


def configurations(path, expected):
    require(type(expected) is int and expected > 0, "Invalid expected configuration count")
    result = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in table_reader(handle, CONFIG_FIELDS):
            name = row["config_id"]
            require(name not in result, "Duplicate configuration in ledger")
            values = {key: integer(row[key], key, 1 if key in CONFIG_FIELDS[1:4] else 0)
                      for key in CONFIG_FIELDS[1:]}
            gap, length, count = (values[k] for k in CONFIG_FIELDS[1:4])
            require(name == f"L{length}_G{gap}_N{count}", "Configuration ID/numeric values disagree")
            require(count >= (length - 2 + gap) // gap + 1,
                    "Effective configuration violates necessary segment geometry")
            require((values["n_pairs"] == 0) == (values["n_segments"] == 0)
                    and values["n_segments"] >= values["n_pairs"], "Invalid ledger pair/segment counts")
            if not values["n_pairs"]:
                require(not any(values[k] for k in COUNTS), "Zero-edge ledger has nonzero totals")
            result[name] = dict(config_id=name, **values)
    require(len(result) == expected, "Declared configuration ledger is incomplete")
    return result


class CohortPairs:
    """Lazy all-cohort unordered pairs for the existing PC-Relate reader."""
    def __init__(self, n):
        self.n = n

    def __contains__(self, pair):
        a, b = pair
        return 0 <= a < b < self.n

    def __len__(self):
        return self.n * (self.n - 1) // 2


class PairOrder:
    def __init__(self, aggregated):
        self.aggregated = aggregated
        self.previous = None
        self.configs_in_pair = set()

    def check(self, config, a, b, sample_index):
        if self.aggregated:
            left, right = sorted((a, b))
            key = (config, left, right)
            require(self.previous is None or key > self.previous,
                    "Duplicate/reversed pair or unsorted configuration/pair groups")
            self.previous = key
        else:
            key = tuple(sorted((sample_index[a], sample_index[b])))
            require(self.previous is None or key >= self.previous,
                    "Unsorted or repeated sample-axis pair group")
            if key != self.previous:
                self.configs_in_pair.clear()
            require(config not in self.configs_in_pair, "Duplicate or reversed pair/configuration")
            self.configs_in_pair.add(config)
            self.previous = key


class Accumulator:
    def __init__(self, n, thresholds):
        self.n_pairs = self.n_segments = self.bp = self.variants = self.longest = 0
        self.weight = 0.0
        self.degree = [0] * n
        self.weighted_degree = [0.0] * n
        self.bp_degree = [0] * n
        self.kin = {threshold: {state: [0, 0, 0.0] for state in KIN_STATES}
                    for threshold in thresholds}

    def add(self, a, b, n, bp, variants, longest, pair, kin_values):
        weight = math.log1p(bp)
        self.n_pairs += 1
        self.n_segments += n
        self.bp += bp
        self.variants += variants
        self.longest = max(self.longest, longest)
        self.weight += weight
        for sample in (a, b):
            self.degree[sample] += 1
            self.weighted_degree[sample] += weight
            self.bp_degree[sample] += bp
        for threshold, classes in self.kin.items():
            item = classes[kinship.kinship_state(pair, kin_values, threshold)]
            item[0] += 1
            item[1] += bp
            item[2] += weight

    def rows(self, config, edge_threshold, n_people):
        active = sum(value > 0 for value in self.degree)
        maximum = max(self.weighted_degree, default=0.0)
        base = dict(config_id=config["config_id"], max_gap_bp=config["max_gap_bp"],
                    min_length_bp=config["min_length_bp"],
                    min_shared_effective=config["min_shared_effective"],
                    min_edge_bp=edge_threshold, n_people=n_people,
                    n_possible_pairs=n_people * (n_people - 1) // 2,
                    n_active=active, n_isolated=n_people-active,
                    fraction_people_active=active/n_people, n_pairs=self.n_pairs,
                    fraction_possible_pairs_retained=ratio(self.n_pairs, n_people*(n_people-1)//2),
                    n_segments=self.n_segments, total_shared_bp=self.bp,
                    n_shared_variants_total=self.variants, max_segment_bp=self.longest or None,
                    mean_segment_bp=ratio(self.bp, self.n_segments),
                    mean_pair_shared_bp=ratio(self.bp, self.n_pairs), sum_weight=self.weight,
                    max_degree=max(self.degree, default=0), max_weighted_degree=maximum,
                    max_weighted_degree_share=ratio(maximum, 2*self.weight),
                    max_incident_weight_fraction=ratio(maximum, self.weight),
                    max_incident_bp_fraction=ratio(max(self.bp_degree, default=0), self.bp),
                    status=STATUS, local_ibd_status="NOT_EVALUATED", quality_status="NOT_EVALUATED")
        for threshold, classes in sorted(self.kin.items()):
            row = dict(base, kinship_threshold=threshold)
            for state, (count, bp, weight) in classes.items():
                row.update({"n_"+state: count, "bp_"+state: bp, "weight_"+state: weight})
            for prefix in ("n", "bp", "weight"):
                observed = row[prefix+"_kin_ge_threshold"] + row[prefix+"_kin_lt_threshold"]
                missing = row[prefix+"_missing_absent"] + row[prefix+"_missing_nonfinite"]
                row[prefix+"_observed_kinship"] = observed
                row[prefix+"_missing_kinship"] = missing
                row["fraction_"+prefix+"_kin_ge_all_pairs"] = ratio(row[prefix+"_kin_ge_threshold"], observed+missing)
                row["fraction_"+prefix+"_kin_ge_observed_pairs"] = ratio(row[prefix+"_kin_ge_threshold"], observed)
                row["fraction_"+prefix+"_missing_all_pairs"] = ratio(missing, observed+missing)
            yield row


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def authenticate(receipt_path, aggregated, pair_path, config_path, sample_path,
                 samples, configs, explicit_hashes):
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    order_hash = hashlib.sha256(("\n".join(samples)+"\n").encode()).hexdigest()
    if aggregated:
        require(receipt.get("status") == "COMPLETE_AUTOSOMAL_AGGREGATION"
                and receipt.get("chromosomes") == AUTOSOMES
                and receipt.get("n_samples") == len(samples), "Autosomal aggregation scope/cohort mismatch")
        require(receipt.get("sample_ids_sha256") == sweep.sha256(sample_path),
                "Aggregation sample-ID file hash mismatch")
        require(receipt.get("selected_source_configurations") == sorted(configs),
                "Aggregation does not contain the complete declared configuration ledger")
        expected = {"pair_summary": receipt.get("outputs_sha256", {}).get("pair_configuration_summary.tsv.gz"),
                    "configuration_summary": receipt.get("outputs_sha256", {}).get("configuration_summary.tsv")}
        chromosomes = AUTOSOMES
    else:
        chrom = str(receipt.get("chrom", "")).removeprefix("chr")
        require(receipt.get("status") == "COMPLETE_EXPLORATORY_NOT_VALIDATED"
                and chrom in AUTOSOMES and receipt.get("n_samples") == len(samples)
                and receipt.get("carrier_allele_mode") == "source_minor"
                and receipt.get("selected_samples_order_sha256") == order_hash,
                "Single-chromosome source scope, allele or sample-order mismatch")
        require(receipt.get("n_effective_configurations") == len(configs),
                "Single-chromosome effective configuration count mismatch")
        expected = {key: explicit_hashes.get(key) for key in ("pair_summary", "configuration_summary")}
        chromosomes = [chrom]
    contract = receipt.get("source_rare_contract", {})
    require(contract.get("contract") == "minor_v1", "Missing source_minor rare contract")
    for key, value in expected.items():
        expected[key] = validate_hash(value, key)
        if explicit_hashes.get(key) is not None:
            require(expected[key] == validate_hash(explicit_hashes[key], key),
                    "Explicit input hash conflicts with source receipt")
    require(sweep.sha256(config_path) == expected["configuration_summary"],
            "Configuration-summary SHA256 mismatch")
    return receipt, chromosomes, expected


def run(pair_summary, configuration_summary, sample_ids, expected_samples,
        pcrelate_file, expected_pcrelate_sha256, output_dir, aggregation_receipt=None,
        source_receipt=None, expected_configurations=74, thresholds=DEFAULT_THRESHOLDS,
        edge_thresholds_bp=(0,), pair_sha256=None, configuration_sha256=None,
        receipt_sha256=None, sample_ids_sha256=None):
    output = Path(output_dir)
    require(not output.exists(), "Output directory exists; no overwrite")
    require(bool(aggregation_receipt) != bool(source_receipt), "Provide exactly one source/aggregation receipt")
    require(type(expected_samples) is int and expected_samples > 0, "Invalid expected sample count")
    thresholds, edges = tuple(thresholds), tuple(edge_thresholds_bp)
    require(thresholds and len(set(thresholds)) == len(thresholds)
            and all(math.isfinite(t) and 0 <= t <= .5 for t in thresholds), "Invalid kinship thresholds")
    require(edges and len(set(edges)) == len(edges)
            and all(type(t) is int and t >= 0 for t in edges) and 0 in edges,
            "Edge thresholds require unique nonnegative integers including baseline zero")
    thresholds, edges = tuple(sorted(thresholds)), tuple(sorted(edges))
    pair_path, config_path, sample_path = map(Path, (pair_summary, configuration_summary, sample_ids))
    receipt_path = Path(aggregation_receipt or source_receipt)
    paths = [pair_path, config_path, sample_path, receipt_path, Path(pcrelate_file)]
    signatures = {str(path): kinship.stat_signature(path.stat()) for path in paths}
    samples = sweep.samples_from(sample_path, expected_samples)
    observed_sample_hash, observed_receipt_hash = sweep.sha256(sample_path), sweep.sha256(receipt_path)
    for actual, expected, name in ((observed_sample_hash, sample_ids_sha256, "sample IDs"),
                                   (observed_receipt_hash, receipt_sha256, "source receipt")):
        if expected is not None:
            require(actual == validate_hash(expected, name), f"{name} SHA256 mismatch")
    configs = configurations(config_path, expected_configurations)
    receipt, chromosomes, hashes = authenticate(receipt_path, bool(aggregation_receipt), pair_path,
        config_path, sample_path, samples, configs,
        {"pair_summary": pair_sha256, "configuration_summary": configuration_sha256})
    sample_index = {sample: i for i, sample in enumerate(samples)}
    values, kin_audit = kinship.stream_kinship(pcrelate_file, sample_index, CohortPairs(len(samples)),
                                             expected_pcrelate_sha256)
    accumulators = {(config, threshold): Accumulator(len(samples), thresholds)
                    for config in configs for threshold in edges}
    order = PairOrder(bool(aggregation_receipt))
    scanned = 0
    with pair_path.open("rb") as raw:
        hashed = sweep.HashingReader(raw)
        with gzip.GzipFile(fileobj=hashed, mode="rb") as zipped, io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
            for row in table_reader(handle, PAIR_FIELDS):
                config, a, b = (row[key] for key in ("config_id", "sample_a", "sample_b"))
                require(config in configs, "Undeclared configuration in pair table")
                require(a != b and a in sample_index and b in sample_index, "Self-pair or ID outside declared cohort")
                order.check(config, a, b, sample_index)
                n, bp, variants, longest = (integer(row[key], key, 1) for key in (*COUNTS[1:], "max_segment_bp"))
                plan = configs[config]
                require(longest <= bp <= n*longest and bp >= n*plan["min_length_bp"]
                        and longest >= plan["min_length_bp"] and variants >= n*plan["min_shared_effective"],
                        "Pair length/support values violate declared configuration")
                ia, ib = sample_index[a], sample_index[b]
                pair = (min(ia, ib), max(ia, ib))
                for threshold in edges:
                    if bp >= threshold:
                        accumulators[config, threshold].add(ia, ib, n, bp, variants, longest, pair, values)
                scanned += 1
        require(hashed.digest.hexdigest() == hashes["pair_summary"], "Pair-summary compressed-byte SHA256 mismatch")
    ledger, rows = [], []
    for config in sorted(configs):
        plan, base = configs[config], accumulators[config, 0]
        observed = dict(n_pairs=base.n_pairs, n_segments=base.n_segments,
                        total_shared_bp=base.bp, n_shared_variants_total=base.variants)
        require(all(observed[key] == plan[key] for key in COUNTS), "Configuration ledger/pair-table totals disagree")
        ledger.append(dict(plan, counts_verified=True, status=STATUS))
        for threshold in edges:
            rows.extend(accumulators[config, threshold].rows(plan, threshold, len(samples)))
    for path in paths:
        require(kinship.stat_signature(path.stat()) == signatures[str(path)], "Source changed during diagnostics")
    output.mkdir(parents=True, exist_ok=False)
    kinship.write_table(output / "configuration_ledger.tsv", ledger)
    kinship.write_table(output / "configuration_diagnostics.tsv", rows)
    result = dict(schema_version=1, status=STATUS, chromosomes=chromosomes, n_people=len(samples),
        n_configurations=len(configs), n_rows_scanned=scanned, n_diagnostic_rows=len(rows),
        configuration_ids=sorted(configs), parameters=dict(kinship_thresholds=thresholds, edge_thresholds_bp=edges),
        pcrelate_audit=kin_audit, source_rare_contract=receipt["source_rare_contract"],
        input_sha256={str(pair_path): hashes["pair_summary"], str(config_path): hashes["configuration_summary"],
                      str(sample_path): observed_sample_hash, str(receipt_path): observed_receipt_hash,
                      str(Path(pcrelate_file)): kin_audit["sha256"]},
        source_sha256={Path(p).name: sweep.sha256(p) for p in
                      (__file__, sweep.__file__, kinship.__file__, kinship.saved.__file__)},
        evidence_gaps=dict(local_ibd="NOT_EVALUATED", quality_and_batch="NOT_EVALUATED",
                           independent_population_validation="NOT_EVALUATED",
                           held_out_prediction="NOT_EVALUATED", uncertainty_intervals="NOT_EVALUATED"),
        no_pvalues=True, no_winner_selected=True, no_reclustering=True, no_new_kinship=True,
        contains_individual_identifiers=False, public_distribution_allowed=False,
        semantics=dict(weight="ln(1 + pair total_shared_bp / 1 bp); historical H weight, not IBD",
            length="summed candidate-chain bp, not union coverage or haplotype-copy IBD",
            isolated="no retained candidate-chain edge at this threshold; not absence of relationship",
            zero_pairs="no qualifying chain reported, not proof of measured genetic dissimilarity",
            opportunity="all listed people retained as denominator; joint genomic callability is not evaluated",
            max_weighted_degree_share="max incident H weight / (2 * sum unordered-edge H weight)",
            max_incident_weight_fraction="max incident H weight / sum unordered-edge H weight",
            kinship="absent and nonfinite separated; below cutoff is not proof of independence",
            fractions="conditional on retained edges; all-edge and finite-kinship denominators both reported",
            null_denominator="NA in tables / null in JSON, never a fabricated zero fraction",
            sampling="full fixed cohort; no holdout or family-independent validation claim",
            thresholds="operational sensitivity values; not optimal or certified family boundaries",
            streaming="sorted-group contract; one shared cohort kinship map and O(configurations*thresholds*people) accumulators"),
        outputs_sha256={p.name: sweep.sha256(p) for p in output.iterdir() if p.is_file()})
    sweep.write_json(output / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pair-summary", "configuration-summary", "sample-ids", "pcrelate-file", "output-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--expected-pcrelate-sha256", "--pcrelate-sha256", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--expected-configurations", type=int, default=74)
    receipt = parser.add_mutually_exclusive_group(required=True)
    receipt.add_argument("--aggregation-receipt")
    receipt.add_argument("--source-receipt")
    for name in ("pair-sha256", "configuration-sha256", "receipt-sha256", "sample-ids-sha256"):
        parser.add_argument("--"+name)
    parser.add_argument("--thresholds", default="0.0221,0.0442",
                        help="Historical operational PC-Relate sensitivities, not independent-family truth")
    parser.add_argument("--edge-thresholds-bp", default="0",
                        help="Explicit accumulated-chain bp thresholds including 0; no optimality claim")
    args = vars(parser.parse_args(argv))
    args["thresholds"] = tuple(float(value) for value in args["thresholds"].split(","))
    args["edge_thresholds_bp"] = tuple(int(value) for value in args["edge_thresholds_bp"].split(","))
    result = run(**args)
    print(json.dumps({key: result[key] for key in ("status", "n_configurations", "n_diagnostic_rows")}))


if __name__ == "__main__":
    main()
