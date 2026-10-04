#!/usr/bin/env python3
"""Authenticated, descriptive integration of M14.2 configurations.

This is not a population classifier, a selector, or an independent biological
validation. Counts condition on M14-selected intervals. Pair-sites and pair-bp
are descriptive denominators, never independent observations for inference.

Every segment bundle is an immutable M14.2 manifest with three adjacent tables.
SQLite validates one chromosome's chains and configuration links at a time. A
configuration has one G; its maximal chains must be disjoint within pair/chrom.
Rejecting overlapping chains is intentional: I/U cannot be de-duplicated from
interval sums alone. Different configurations are never pooled together.
Validated, sorted scratch streams are merged by pair across chromosomes before
any accumulated-bp threshold is applied. No chromosome-local T selection occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import gzip
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile

import r02_m14_configuration_diagnostics as diagnostics


STATUS = "BIOLOGICAL_DESCRIPTIVE_NOT_VALIDATED"
SCHEMA = "r02_biological_evaluation_v1"
FILES = ("segment_evidence.tsv.gz", "segment_configuration_links.tsv.gz", "configurations.tsv")
IBD_TERRITORY = "ibd_pair_territory.tsv.gz"
CONFIG_FIELDS = ("config_id", "max_gap_bp", "min_length_bp", "min_shared_effective")
SEGMENT_FIELDS = ("chain_id", "chrom", "sample_a", "sample_b", "max_gap_bp", "start_pos",
                  "end_pos", "length_bp", "n_shared_variants", "I", "U", "Q", "J",
                  "rare_status", "rare_catalog_sites", "rare_missing_gt_count", "O", "Q_C",
                  "common_status", "length_cm", "map_status", "callable_bp", "ibd_union_bp",
                  "ibd_fraction", "ibd_status")
METRIC_FIELDS = ("config_id", "min_edge_bp", "metric", "numerator", "denominator", "value",
                 "status", "reason", "unit", "n_segments_evaluable", "n_segments_total", "scope")
CORE_CRITERIA = ("rare_allele", "rare_gt", "rare_missingness", "coordinate_system", "quality_filters")
DEFAULT_EDGES = (0, 250000, 500000, 750000, 1000000)
MISSING = {"", ".", "na", "nan", "null", "none", "unknown", "unassigned"}
COMPONENT_FIELDS = ("config_id", "min_edge_bp", "kinship_threshold", "n_components_total",
    "n_dependence_components_represented", "n_components_in_cross_edges", "n_within_component_pairs",
    "n_between_component_pairs", "n_unassigned_component_pairs", "within_component_weight",
    "between_component_weight", "unassigned_component_weight", "max_component_incident_weight",
    "total_edge_weight", "max_component_incident_weight_fraction", "status", "interpretation")
require = diagnostics.require
sha256 = diagnostics.sweep.sha256


def read_table(path, required):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        yield from diagnostics.table_reader(handle, required)


def number(value, name, integer=False, optional=False):
    if optional and value == "NA":
        return None
    if integer:
        return diagnostics.integer(value, name)
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid numeric field: {name}") from None
    require(math.isfinite(result) and result >= 0, f"Nonfinite/negative field: {name}")
    return result


def state(value):
    require(value == "OK" or (value.startswith("NO_EVALUABLE:") and value.split(":", 1)[1]),
            "Evidence status must be OK or NO_EVALUABLE:reason")
    return value


def check_ratio(raw, numerator, denominator, label):
    value = number(raw, label, optional=True)
    if numerator is None or denominator is None or denominator == 0:
        require(value is None, f"Undefined {label} must be NA")
    else:
        require(value is not None and math.isclose(value, numerator / denominator,
                rel_tol=1e-7, abs_tol=1e-10), f"Inconsistent {label}")


def parse_segment(row, chrom, sample_index):
    """Validate counts before any aggregation; NA is not a biological zero."""
    a, b = sample_index.get(row["sample_a"]), sample_index.get(row["sample_b"])
    require(a is not None and b is not None and a < b,
            "Segment endpoints must be distinct and canonical in the analytical sample order")
    require(row["chrom"].removeprefix("chr") == chrom, "Segment chromosome differs from manifest")
    require(row["chain_id"].startswith("chain_"), "Invalid chain ID prefix")
    diagnostics.validate_hash(row["chain_id"][6:], "chain_id")
    values = dict(chain_id=row["chain_id"], chrom=int(chrom), a=a, b=b)
    for field in ("max_gap_bp", "start_pos", "end_pos", "length_bp", "n_shared_variants"):
        values[field] = diagnostics.integer(row[field], field, 1)
    require(values["end_pos"] - values["start_pos"] + 1 == values["length_bp"],
            "Segment length/coordinates disagree")
    require(values["length_bp"] <= (values["n_shared_variants"]-1)*values["max_gap_bp"]+1,
            "Segment violates maximum-gap geometry")
    for field in ("I", "U", "Q", "rare_catalog_sites", "rare_missing_gt_count"):
        values[field] = number(row[field], field, integer=True)
    i, u, q, sites, missing = (values[k] for k in
                             ("I", "U", "Q", "rare_catalog_sites", "rare_missing_gt_count"))
    require(0 <= i <= u <= q <= sites, "Invalid rare I/U/Q/catalogue counts")
    require(i == values["n_shared_variants"], "Rare intersection differs from M14 shared variants")
    require(sites-q <= missing <= 2*(sites-q), "GT missingness and joint-call counts disagree")
    values["rare_status"] = state(row["rare_status"])
    require((row["rare_status"] == "OK") == (u > 0), "Rare status/denominator disagree")
    check_ratio(row["J"], i, u, "J")
    for field in ("O", "Q_C", "callable_bp", "ibd_union_bp"):
        values[field] = number(row[field], field, integer=True, optional=True)
    values["length_cm"] = number(row["length_cm"], "length_cm", optional=True)
    for field in ("common_status", "map_status", "ibd_status"):
        values[field] = state(row[field])
    o, qc = values["O"], values["Q_C"]
    require((o is None) == (qc is None), "Partial common count fields")
    if qc is not None:
        require(o <= qc, "Opposite homozygotes exceed called common sites")
    require((row["common_status"] == "OK") == (qc is not None and qc > 0),
            "Common status/denominator disagree")
    require((row["map_status"] == "OK") == (values["length_cm"] is not None),
            "Map status/length disagree")
    callable_bp, ibd = values["callable_bp"], values["ibd_union_bp"]
    if callable_bp is not None:
        require(callable_bp <= values["length_bp"], "Callable territory exceeds interval")
    if ibd is not None:
        require(callable_bp is not None and ibd <= callable_bp, "IBD union exceeds callable territory")
    require((row["ibd_status"] == "OK") ==
            (ibd is not None and callable_bp is not None and callable_bp > 0),
            "IBD status/denominator disagree")
    check_ratio(row["ibd_fraction"], ibd, callable_bp, "ibd_fraction")
    for field in ("max_shared_gap_bp", "ibd_overlap_pieces", "ibd_left_uncovered_bp", "ibd_right_uncovered_bp"):
        values[field] = number(row.get(field, "NA"), field, integer=True, optional=True)
    if values["max_shared_gap_bp"] is not None:
        require(values["max_shared_gap_bp"] <= values["max_gap_bp"], "Observed shared-site gap exceeds G")
    for field in ("ibd_left_uncovered_bp", "ibd_right_uncovered_bp"):
        if values[field] is not None:
            require(ibd is not None and ibd > 0 and values[field] < values["length_bp"], "Invalid IBD terminal uncovered length")
    return values


def configuration_ledger(path, expected):
    return configuration_records(read_table(path, CONFIG_FIELDS), expected)


def configuration_records(rows, expected):
    """One validation contract for TSV ledgers and authenticated checkpoints."""
    result = {}
    for row in rows:
        name = row["config_id"]
        require(name not in result, "Duplicate configuration")
        values = {key: diagnostics.integer(row[key], key, 1) for key in CONFIG_FIELDS[1:]}
        length, gap, count = (values[k] for k in ("min_length_bp", "max_gap_bp", "min_shared_effective"))
        require(name == f"L{length}_G{gap}_N{count}", "Configuration ID/numeric values disagree")
        require(count >= (length-2+gap)//gap+1, "Configuration violates effective geometry")
        result[name] = dict(config_id=name, **values)
    require(type(expected) is int and expected > 0 and len(result) == expected,
            "Incomplete configuration ledger")
    return result


def metric(config, edge, name, numerator, denominator, unit, observed, total,
           reason="", scope="selected_disjoint_M14_intervals"):
    if numerator is None or denominator is None or denominator == 0:
        value, status = None, "NO_EVALUABLE"
        reason = reason or "zero_denominator"
    else:
        value = numerator / denominator
        status = "OBSERVED_ZERO" if numerator == 0 else "OBSERVED"
        if observed < total:
            status += "_PARTIAL"
        reason = reason or ("partial_evidence" if observed < total else "")
    return dict(config_id=config, min_edge_bp=edge, metric=name, numerator=numerator,
                denominator=denominator, value=value, status=status, reason=reason, unit=unit,
                n_segments_evaluable=observed, n_segments_total=total, scope=scope)


class LocalTotals:
    def __init__(self):
        self.n = 0
        self.totals = Counter()
        self.observed = Counter()
        self.missing = {channel: Counter() for channel in ("rare", "common", "map", "ibd")}
        self.chromosomes = {}
        self.extra_observed = Counter()

    def add(self, row):
        self.n += 1
        for field in ("length_bp", "I", "U", "Q", "rare_catalog_sites", "rare_missing_gt_count"):
            self.totals[field] += row[field]
        chrom = self.chromosomes.setdefault(row["chrom"], Counter())
        chrom.update(n_segments=1, pair_bp=row["length_bp"], I=row["I"], U=row["U"], Q=row["Q"])
        for channel, fields in (("rare", ()), ("common", ("O", "Q_C")),
                                ("map", ("length_cm",)), ("ibd", ("callable_bp", "ibd_union_bp"))):
            if row[channel+"_status"] == "OK":
                self.observed[channel] += 1
                for field in fields:
                    self.totals[field] += row[field]
            else:
                self.missing[channel][row[channel+"_status"].split(":", 1)[1]] += 1
        for field in ("max_shared_gap_bp", "ibd_overlap_pieces", "ibd_left_uncovered_bp", "ibd_right_uncovered_bp"):
            if row[field] is not None:
                self.totals[field] += row[field]
                self.extra_observed[field] += 1

    def merge(self, other):
        """Add disjoint, validated pair contributions, preserving unknown reasons."""
        self.n += other.n
        self.totals.update(other.totals)
        self.observed.update(other.observed)
        self.extra_observed.update(other.extra_observed)
        for channel in self.missing:
            self.missing[channel].update(other.missing[channel])
        for chrom, counts in other.chromosomes.items():
            self.chromosomes.setdefault(chrom, Counter()).update(counts)

    def rows(self, config, edge, kin, ibd_territory):
        n, totals = self.n, self.totals
        specs = (("rare_local_J", "I", "U", "rare", "joint_called_pair_carrier_sites"),
                 ("rare_joint_call_fraction", "Q", "rare_catalog_sites", "rare", "pair_catalogue_sites"),
                 ("rare_missing_GT_fraction", "rare_missing_gt_count", None, "rare", "person_catalogue_calls"),
                 ("common_opposite_homozygote_fraction", "O", "Q_C", "common", "joint_called_pair_common_sites"),
                 ("ibd_union_coverage_fraction", "ibd_union_bp", "callable_bp", "ibd", "callable_pair_bp"),
                 ("mean_genetic_length_cm", "length_cm", None, "map", "cm_per_interval"))
        for name, numerator, denominator, channel, unit in specs:
            observed = self.observed[channel]
            den = totals[denominator] if denominator else (2*totals["rare_catalog_sites"]
                       if channel == "rare" else observed)
            reason = json.dumps(dict(sorted(self.missing[channel].items())), sort_keys=True) if self.missing[channel] else ""
            yield metric(config, edge, name, totals[numerator] if observed else None,
                         den if observed else None, unit, observed, n,
                         reason or ("no_selected_intervals" if not n else ""))
        for channel in ("rare", "common", "map", "ibd"):
            yield metric(config, edge, channel+"_evaluable_interval_fraction", self.observed[channel],
                         n, "intervals", n, n)
        yield metric(config, edge, "mean_segment_length_bp", totals["length_bp"], n,
                     "bp_per_interval", n, n)
        yield metric(config, edge, "people_represented_fraction", kin["n_active"], kin["n_people"],
                     "people", n, n, scope="complete_analytical_cohort")
        yield metric(config, edge, "possible_pairs_retained_fraction", kin["n_pairs"], kin["n_possible_pairs"],
                     "unordered_pairs_not_independent", n, n, scope="complete_analytical_cohort")
        yield metric(config, edge, "shared_site_density_per_bp", totals["I"], totals["length_bp"],
                     "shared_pair_sites_per_selected_pair_bp", n, n)
        for field in ("max_shared_gap_bp", "ibd_overlap_pieces", "ibd_left_uncovered_bp", "ibd_right_uncovered_bp"):
            observed = self.extra_observed[field]
            yield metric(config, edge, "mean_"+field, totals[field] if observed else None,
                         observed, "pieces_per_interval" if field == "ibd_overlap_pieces" else "bp_per_interval",
                         observed, n, "" if observed else "annotation_unavailable_or_no_overlap",
                         scope="within_M14_interval_not_true_ancestral_boundary_error")
        if ibd_territory["n_evaluable_pair_chromosomes"]:
            denominator = ibd_territory["ibd_union_bp"]
            overlap = totals["ibd_union_bp"]
            require(overlap <= denominator, "M14 recovery exceeds global IBD union")
            for name, numerator in (("ibd_recall_all_evaluable_pairs", overlap),
                                    ("ibd_fraction_outside_M14", denominator-overlap)):
                yield metric(config, edge, name, numerator, denominator, "joint_callable_IBD_pair_bp",
                             self.observed["ibd"], n,
                             "zero_IBD_territory" if denominator == 0 else
                             "fixed_global_callable_pair_universe_not_only_M14_pairs",
                             scope="all_pair_chromosomes_with_authenticated_joint_callability")
        else:
            for name in ("ibd_recall_all_evaluable_pairs", "ibd_fraction_outside_M14"):
                yield metric(config, edge, name, None, None, "joint_callable_IBD_pair_bp", 0, n,
                             "global_IBD_input_absent_or_no_joint_callable_pair_territory")
        for name, reason in (
            ("quality_batch_confounding", "no_authenticated_person_level_quality_batch_contrast"),
            ("independent_population_validation", "no_independent_population_outcome"),
            ("incremental_rare_utility", "no_matched_B0_B1_B2_heldout_predictions"),
            ("biological_negative_result", "effect_precision_and_identification_not_established")):
            yield metric(config, edge, name, None, None, "not_estimated", 0, n, reason)


def setup_database(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-32768")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("""CREATE TABLE chains (
      chain_id TEXT PRIMARY KEY, chrom INTEGER, a INTEGER, b INTEGER, max_gap_bp INTEGER,
      start_pos INTEGER, end_pos INTEGER, length_bp INTEGER, n_shared_variants INTEGER,
      I INTEGER, U INTEGER, Q INTEGER, rare_catalog_sites INTEGER, rare_missing_gt_count INTEGER,
      O INTEGER, Q_C INTEGER, length_cm REAL, callable_bp INTEGER, ibd_union_bp INTEGER,
      rare_status TEXT, common_status TEXT, map_status TEXT, ibd_status TEXT,
      max_shared_gap_bp INTEGER, ibd_overlap_pieces INTEGER, ibd_left_uncovered_bp INTEGER, ibd_right_uncovered_bp INTEGER,
      UNIQUE(chrom,a,b,max_gap_bp,start_pos,end_pos))""")
    db.execute("CREATE TABLE links (chain_id TEXT REFERENCES chains(chain_id), config_id TEXT, source_chrom INTEGER, PRIMARY KEY(chain_id,config_id))")
    db.execute("CREATE INDEX config_links ON links(config_id,source_chrom,chain_id)")
    db.execute("CREATE INDEX chain_geometry ON chains(chrom,max_gap_bp,length_bp,n_shared_variants)")
    db.execute("CREATE TABLE ibd_territory (chrom INTEGER, a INTEGER, b INTEGER, callable_bp INTEGER, ibd_union_bp INTEGER, status TEXT, PRIMARY KEY(chrom,a,b))")
    return db


def resources(db_path, max_database_mb, min_free_disk_mb):
    # Include rollback/WAL sidecars, not just the main database. Compressed
    # scratch streams are covered by the free-space reserve, not this DB cap.
    database_bytes = sum(path.stat().st_size for path in
        (db_path, Path(str(db_path)+"-journal"), Path(str(db_path)+"-wal"), Path(str(db_path)+"-shm"))
        if path.exists())
    require(database_bytes <= max_database_mb*1024**2, "SQLite database resource limit exceeded")
    require(shutil.disk_usage(db_path.parent).free >= min_free_disk_mb*1024**2,
            "Insufficient scratch free-space reserve")
    return database_bytes


def load_bundle(db, manifest_path, sample_path, sample_index, expected_configs, registry,
                db_path, max_database_mb, min_free_disk_mb, resource_check=None):
    check_resources = resource_check or (lambda: resources(db_path, max_database_mb, min_free_disk_mb))
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest_digest = sha256(path)
    require(manifest.get("schema") == "r02_segment_evidence_v1" and
            manifest.get("status") == "COMPLETE_DESCRIPTIVE_NOT_VALIDATED", "Incomplete/unsupported M14.2 manifest")
    chrom = str(manifest.get("chrom", "")).removeprefix("chr")
    require(chrom in diagnostics.AUTOSOMES and chrom not in registry["chromosomes"],
            "Duplicate or unsupported chromosome bundle")
    require(manifest.get("n_samples") == len(sample_index) and
            manifest.get("sample_ids_sha256") == sha256(sample_path), "M14.2 analytical cohort/order mismatch")
    require(manifest.get("coordinate_system") == "1-based-inclusive", "Unsupported coordinate convention")
    require(isinstance(manifest.get("genome_build"), str) and manifest["genome_build"], "Genome build is missing")
    require(isinstance(manifest.get("source_rare_contract"), dict) and manifest["source_rare_contract"],
            "Source rare-allele contract missing")
    rare_contract = manifest["source_rare_contract"]
    require(rare_contract.get("dnabr_original_alleles") == "v1" and
            rare_contract.get("dnabr_rare_contract") == "minor_v1", "Unsupported rare source allele convention")
    diagnostics.validate_hash(rare_contract.get("dnabr_rare_cohort_sha256"), "rare source cohort")
    criteria = manifest.get("evidence_criteria", {})
    require(all(key in criteria and criteria[key] is not None for key in CORE_CRITERIA),
            "Evidence criteria are incomplete")
    core = {key: criteria[key] for key in CORE_CRITERIA}
    identity = dict(genome_build=manifest["genome_build"], source_rare_contract=manifest["source_rare_contract"],
                    evidence_criteria=core)
    if registry["identity"] is not None:
        require(registry["identity"] == identity, "Incompatible chromosome universe/allele/evidence criteria")
    else:
        registry["identity"] = identity
    for channel in ("common_criteria", "ibd_quantity", "ibd_criteria", "ibd_caller", "ibd_absence_semantics"):
        value = criteria.get(channel)
        if value is not None:
            if channel in registry["optional_criteria"]:
                require(registry["optional_criteria"][channel] == value, "Incompatible optional evidence criteria")
            registry["optional_criteria"][channel] = value
    signatures = {}
    bundle_files = FILES + ((IBD_TERRITORY,) if IBD_TERRITORY in manifest.get("outputs_sha256", {}) else ())
    require((criteria.get("ibd_quantity") is not None) == (IBD_TERRITORY in bundle_files),
            "IBD evidence requires an authenticated complete pair-territory table")
    if IBD_TERRITORY in bundle_files:
        ibd_criteria = criteria.get("ibd_criteria", {})
        caller = ibd_criteria.get("caller", {})
        require(caller.get("name") and caller.get("version") and
                ibd_criteria.get("absence_means_no_called_ibd_within_callable") is True,
                "IBD caller/version and absence semantics must be declared")
    for name in bundle_files:
        source = path.parent/name
        expected = diagnostics.validate_hash(manifest.get("outputs_sha256", {}).get(name), name)
        signatures[source] = diagnostics.kinship.stat_signature(source.stat())
        require(sha256(source) == expected, "M14.2 output SHA256 mismatch: "+name)
    configs = configuration_ledger(path.parent/FILES[2], expected_configs)
    if registry["configs"] is None:
        registry["configs"] = configs
    else:
        require(registry["configs"] == configs, "Configuration ledger differs across chromosomes")
    try:
        if IBD_TERRITORY in bundle_files:
            require(criteria.get("ibd_quantity") == "any_copy_ibd_territory", "Unsupported global IBD quantity")
            for row in read_table(path.parent/IBD_TERRITORY, ("chrom", "sample_a", "sample_b", "callable_bp", "ibd_union_bp", "ibd_status")):
                a, b = sample_index.get(row["sample_a"]), sample_index.get(row["sample_b"])
                require(row["chrom"].removeprefix("chr") == chrom and a is not None and b is not None and a < b,
                        "Global IBD pair/chromosome is not canonical analytical territory")
                callable_bp = number(row["callable_bp"], "global callable_bp", integer=True, optional=True)
                ibd_bp = number(row["ibd_union_bp"], "global ibd_union_bp", integer=True, optional=True)
                ibd_state = state(row["ibd_status"])
                require((ibd_bp is None) == (callable_bp is None), "Partial global IBD denominator")
                if ibd_bp is not None:
                    require(ibd_bp <= callable_bp, "Global IBD union exceeds callable territory")
                require((ibd_state == "OK") == (callable_bp is not None and callable_bp > 0),
                        "Global IBD status/denominator disagree")
                db.execute("INSERT INTO ibd_territory VALUES (?,?,?,?,?,?)", (int(chrom), a, b, callable_bp, ibd_bp, ibd_state))
            registry["ibd_chromosomes"].append(chrom)
        count = 0
        for row in read_table(path.parent/FILES[0], SEGMENT_FIELDS):
            values = parse_segment(row, chrom, sample_index)
            if values["common_status"] == "OK":
                require(criteria.get("common_criteria") is not None, "Measured commons lack declared criteria")
            if values["ibd_status"] == "OK":
                require(criteria.get("ibd_quantity") == "any_copy_ibd_territory", "Unsupported IBD quantity")
            columns = tuple(values)
            db.execute("INSERT INTO chains ("+",".join(columns)+") VALUES ("+",".join("?" for _ in columns)+")",
                       tuple(values.values()))
            count += 1
            if count % 4096 == 0:
                db.commit()
                check_resources()
        link_count = 0
        for row in read_table(path.parent/FILES[1], ("chain_id", "config_id")):
            require(row["config_id"] in configs, "Link references unknown configuration")
            db.execute("INSERT INTO links VALUES (?,?,?)", (row["chain_id"], row["config_id"], int(chrom)))
            link_count += 1
            if link_count % 4096 == 0:
                db.commit()
                check_resources()
        # Check completeness as well as validity: silently dropped links distort denominators.
        for config in configs.values():
            invalid = db.execute("SELECT count(*) FROM links JOIN chains USING(chain_id) WHERE source_chrom=? AND config_id=? AND (chrom!=? OR max_gap_bp!=? OR length_bp<? OR n_shared_variants<?)",
                (int(chrom), config["config_id"], int(chrom), config["max_gap_bp"], config["min_length_bp"], config["min_shared_effective"])).fetchone()[0]
            require(invalid == 0, "Invalid chain/configuration membership or wrong chromosome")
        for config in configs.values():
            expected = db.execute("SELECT count(*) FROM chains WHERE chrom=? AND max_gap_bp=? AND length_bp>=? AND n_shared_variants>=?",
                (int(chrom), config["max_gap_bp"], config["min_length_bp"], config["min_shared_effective"])).fetchone()[0]
            observed = db.execute("SELECT count(*) FROM links JOIN chains USING(chain_id) WHERE chrom=? AND config_id=?",
                                  (int(chrom), config["config_id"])).fetchone()[0]
            require(expected == observed, "Incomplete chain/configuration memberships")
    except sqlite3.IntegrityError as error:
        raise ValueError("Duplicate chain/geometry/membership or unknown/wrong-chromosome link") from error
    for source, signature in signatures.items():
        require(diagnostics.kinship.stat_signature(source.stat()) == signature, "M14.2 source changed during read")
    db.commit()
    check_resources()
    registry["chromosomes"].append(chrom)
    require(sha256(path) == manifest_digest, "M14.2 manifest changed during read")
    registry["sources"].append(dict(path=str(path.resolve()), sha256=manifest_digest, chrom=chrom,
                                    outputs_sha256=manifest["outputs_sha256"], n_chains=count, n_links=link_count))


def write_table(path, rows, fields=None):
    iterator = iter(rows)
    first = next(iterator, None)
    require(first is not None or fields, "No table schema")
    fields = fields or tuple(first)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in (() if first is None else (first,)):
            writer.writerow({key: "NA" if value is None else value for key, value in row.items()})
        for row in iterator:
            writer.writerow({key: "NA" if value is None else value for key, value in row.items()})


def read_study(path, samples, pcrelate_sha, thresholds):
    """Reuse authenticated dependence assignments; never create or revise roles."""
    if path is None:
        return None
    path = Path(path)
    digest = sha256(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    semantic_hash = hashlib.sha256(("\n".join(samples)+"\n").encode()).hexdigest()
    require(record.get("schema") == "r02_study_design_v1" and
            record.get("status") == "COMPLETE_DESCRIPTIVE_AUDIT" and
            record.get("n_samples") == len(samples) and
            record.get("sample_ids_sha256") == semantic_hash,
            "Study contract schema/status/analytical cohort mismatch")
    require(record.get("kinship_audit", {}).get("sha256") == pcrelate_sha.lower() and
            set(record.get("thresholds", [])) == set(thresholds), "Study and evaluator kinship criteria differ")
    people_path = path.parent/"persons.private.tsv"
    expected = diagnostics.validate_hash(record.get("outputs_sha256", {}).get(people_path.name), "persons.private.tsv")
    signature = diagnostics.kinship.stat_signature(people_path.stat())
    require(sha256(people_path) == expected, "Study persons SHA256 mismatch")
    columns = [f"component_phi_{phi:g}" for phi in thresholds]
    people = list(read_table(people_path, ("sample_id", "role", *columns)))
    require([row["sample_id"] for row in people] == samples, "Study persons order/coverage differs")
    require(all(row["role"] in {"FIT", "SELECT", "EVALUATE", "EXCLUDE", "UNASSIGNED"} for row in people),
            "Study contains invalid roles")
    metadata_fields = [key for key in people[0] if key not in {"sample_id", "role", *columns}]
    nominal_columns = record.get("nominal_columns", [])
    require(isinstance(nominal_columns, list) and len(nominal_columns) == len(set(nominal_columns)) and
            all(field in metadata_fields for field in nominal_columns), "Invalid explicitly declared nominal metadata columns")
    assignments = {phi: [None if row[column].strip().lower() in MISSING else row[column]
                        for row in people] for phi, column in zip(thresholds, columns)}
    require(sha256(path) == digest and diagnostics.kinship.stat_signature(people_path.stat()) == signature,
            "Study inputs changed during read")
    return dict(assignments=assignments, people=people, metadata_fields=metadata_fields, nominal_columns=nominal_columns,
                provenance=dict(path=str(path.resolve()), sha256=digest,
                    persons_sha256=expected, status=record["status"], roles_supplied=record.get("roles_supplied"),
                    usage="authenticated_dependence_components_and_metadata_coverage_not_training_eligibility"))


class ComponentTotals:
    """Streaming equivalent of the original per-configuration pair query."""
    def __init__(self, thresholds, study):
        self.study = study
        self.data = {phi: (set(), set(), Counter(), Counter(), Counter()) for phi in thresholds}

    def add(self, a, b, bp):
        if self.study is None:
            return
        weight = math.log1p(bp)
        for phi, (represented, crossed, incident, counts, weights) in self.data.items():
            mapping = self.study["assignments"][phi]
            ca, cb = mapping[a], mapping[b]
            endpoints = {value for value in (ca, cb) if value is not None}
            represented.update(endpoints)
            for component in endpoints:  # Internal edges touch one component once.
                incident[component] += weight
            category = "unassigned" if ca is None or cb is None else "within" if ca == cb else "between"
            counts[category] += 1
            weights[category] += weight
            if category == "between":
                crossed.update(endpoints)

    def rows(self, name, edge):
        rows = []
        for phi, (represented, crossed, incident, counts, weights) in self.data.items():
            row = {field: None for field in COMPONENT_FIELDS}
            row.update(config_id=name, min_edge_bp=edge, kinship_threshold=phi,
                       status="NO_EVALUABLE" if self.study is None else "DESCRIPTIVE",
                       interpretation="study_contract_absent" if self.study is None else
                       "Transitive observed-kinship components; missing kinship is unresolved, cross-component is not independence")
            if self.study is not None:
                mapping = self.study["assignments"][phi]
                if any(value is None for value in mapping):
                    row["status"] = "DESCRIPTIVE_PARTIAL_UNASSIGNED_COMPONENTS"
                total = sum(weights.values())
                maximum = max(incident.values(), default=0.0)
                row.update(n_components_total=len({value for value in mapping if value is not None}),
                    n_dependence_components_represented=len(represented), n_components_in_cross_edges=len(crossed),
                    n_within_component_pairs=counts["within"], n_between_component_pairs=counts["between"],
                    n_unassigned_component_pairs=counts["unassigned"], within_component_weight=weights["within"],
                    between_component_weight=weights["between"], unassigned_component_weight=weights["unassigned"],
                    max_component_incident_weight=maximum if represented else None, total_edge_weight=total,
                    max_component_incident_weight_fraction=min(1.0, maximum/total) if total and represented else None)
            rows.append(row)
        return rows

def metadata_summary(name, edge, degree, study):
    if study is None:
        return [dict(config_id=name, min_edge_bp=edge, field="not_supplied", stratum="all_people",
            n_measured=None, n_people=len(degree), fraction_measured=None, status="NO_EVALUABLE")]
    result = []
    for represented in (True, False):
        indices = [i for i, count in enumerate(degree) if bool(count) == represented]
        for field in ["role", *study["metadata_fields"]]:
            known = sum(study["people"][i][field].strip().lower() not in MISSING for i in indices)
            result.append(dict(config_id=name, min_edge_bp=edge, field=field,
                stratum="represented" if represented else "excluded_no_retained_edges",
                n_measured=known, n_people=len(indices), fraction_measured=known/len(indices) if indices else None,
                status="DESCRIPTIVE_COVERAGE_ONLY" if indices else "NO_EVALUABLE"))
    return result


def nominal_summary(name, edge, degree, study):
    """Private aggregate composition, only for declared nominal metadata fields.

    Do not categorize arbitrary numeric/continuous fields or report person IDs.
    Small cells remain potentially identifying: this table is private even
    though it contains counts rather than individual records.
    """
    if study is None:
        return []
    result = []
    for represented in (True, False):
        indices = [i for i, count in enumerate(degree) if bool(count) == represented]
        for field in study["nominal_columns"]:
            counts = Counter("UNKNOWN" if study["people"][i][field].strip().lower() in MISSING
                             else study["people"][i][field] for i in indices)
            for category, count in sorted(counts.items()):
                result.append(dict(config_id=name, min_edge_bp=edge, field=field, category=category,
                    stratum="represented" if represented else "excluded_no_retained_edges",
                    n_people_category=count, n_people=len(indices), fraction_people=count/len(indices),
                    status="PRIVATE_DESCRIPTIVE_NOMINAL_COMPOSITION_NOT_POPULATION_TRUTH"))
    return result


def validate_intervals(db, configs):
    """Keep interval-union and IBD checks before reducing a chromosome."""
    for name, config in sorted(configs.items()):
        # Order validation makes sums valid unions within pair/chrom/config, not across pairs.
        previous, previous_end = None, None
        for row in db.execute("SELECT chains.* FROM links JOIN chains USING(chain_id) WHERE config_id=? ORDER BY a,b,chrom,start_pos,end_pos", (name,)):
            key = row["a"], row["b"], row["chrom"]
            require(key != previous or row["start_pos"] > previous_end,
                    "Overlapping intervals within configuration/pair/chromosome cannot be summed")
            previous, previous_end = key, row["end_pos"]
        # Each configuration's IBD overlaps must fit its authenticated pair/chrom union.
        for row in db.execute("SELECT chains.chrom,chains.a,chains.b,sum(chains.ibd_union_bp) AS overlap,ibd_territory.status AS global_status,ibd_territory.ibd_union_bp AS global_ibd FROM links JOIN chains USING(chain_id) LEFT JOIN ibd_territory ON chains.chrom=ibd_territory.chrom AND chains.a=ibd_territory.a AND chains.b=ibd_territory.b WHERE config_id=? AND ibd_status='OK' GROUP BY chains.chrom,chains.a,chains.b", (name,)):
            require(row["global_status"] is not None, "Local IBD evidence lacks matching global pair territory")
            require(row["global_status"] == "OK" and row["overlap"] <= row["global_ibd"],
                    "Within-configuration IBD overlaps exceed authenticated pair territory")


def export_chromosome(db, directory, chrom, check_resources):
    """Compact sorted scratch files; membership was exhaustively authenticated.

    Geometry therefore reconstructs exactly the supplied memberships, with no
    links retained between chromosomes. The separate sorted ID stream preserves
    the previous database's cross-chromosome duplicate-chain rejection.
    """
    segments, ids = directory/f"chr{chrom}.segments.jsonl.gz", directory/f"chr{chrom}.ids.gz"
    with gzip.open(ids, "xt", encoding="utf-8") as handle:
        for count, row in enumerate(db.execute("SELECT chain_id FROM chains ORDER BY chain_id"), 1):
            handle.write(row[0]+"\n")
            if count % 4096 == 0:
                check_resources()
    with gzip.open(segments, "xt", encoding="utf-8") as handle:
        columns = [row[1] for row in db.execute("PRAGMA table_info(chains)") if row[1] != "chain_id"]
        handle.write(json.dumps(columns)+"\n")
        query = "SELECT "+",".join(columns)+" FROM chains ORDER BY a,b,chrom,start_pos,end_pos,max_gap_bp"
        for count, row in enumerate(db.execute(query), 1):
            handle.write(json.dumps(tuple(row), separators=(",", ":"), allow_nan=False)+"\n")
            if count % 4096 == 0:
                check_resources()
    check_resources()
    return segments, ids


def unique_chain_ids(paths):
    with ExitStack() as stack:
        streams = [stack.enter_context(gzip.open(path, "rt", encoding="utf-8")) for path in paths]
        previous = None
        for identifier in heapq.merge(*streams):
            require(identifier != previous, "Duplicate chain ID across chromosome bundles")
            previous = identifier


def stream_segments(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        fields = json.loads(next(handle))
        for line in handle:
            yield dict(zip(fields, json.loads(line)))


def aggregate_pairs(paths, configs, n_people, values, thresholds, edges, study, check_resources):
    """Close one pair across all chromosomes, THEN apply each global T.

    Memory does not grow with the number of intervals or pair/config rows.
    Persistent accumulators grow with configurations × thresholds × people;
    temporary pair evidence with configurations × chromosomes and status keys.
    """
    aggregates = {(name, edge): (diagnostics.Accumulator(n_people, thresholds),
                  LocalTotals(), ComponentTotals(thresholds, study))
                  for name in configs for edge in edges}
    by_gap = {}
    for name, config in configs.items():
        by_gap.setdefault(config["max_gap_bp"], []).append((name, config))
    key = lambda row: (row["a"], row["b"], row["chrom"], row["start_pos"], row["end_pos"], row["max_gap_bp"])
    streams = [stream_segments(path) for path in paths]
    try:
        merged = heapq.merge(*streams, key=key)
        for count, ((a, b), rows) in enumerate(itertools.groupby(merged, lambda row: (row["a"], row["b"])), 1):
            pair_totals, longest = {}, Counter()
            for row in rows:
                for name, config in by_gap.get(row["max_gap_bp"], ()):
                    if row["length_bp"] >= config["min_length_bp"] and row["n_shared_variants"] >= config["min_shared_effective"]:
                        if name not in pair_totals:
                            pair_totals[name] = LocalTotals()
                        pair_totals[name].add(row)
                        longest[name] = max(longest[name], row["length_bp"])
            for name, totals in pair_totals.items():
                bp = totals.totals["length_bp"]
                for edge in edges:
                    if bp < edge:
                        continue
                    accumulator, global_totals, components = aggregates[name, edge]
                    accumulator.add(a, b, totals.n, bp, totals.totals["I"], longest[name], (a, b), values)
                    global_totals.merge(totals)
                    components.add(a, b, bp)
            if count % 4096 == 0:
                check_resources()
    finally:
        for stream in streams:
            stream.close()
    check_resources()
    return aggregates


def summarize(aggregates, configs, n_people, edges, ibd_territory, study=None):
    metrics, kin_rows, chrom_rows, cards, component_rows, metadata_rows, nominal_rows = [], [], [], [], [], [], []
    for name, config in sorted(configs.items()):
        for edge in edges:
            accumulator, totals, components = aggregates[name, edge]
            require(totals.n == accumulator.n_segments and totals.totals["length_bp"] == accumulator.bp,
                    "Local evidence and pair diagnostics totals disagree")
            rows = list(accumulator.rows(config, edge, n_people))
            for row in rows:
                row.update(local_ibd_status="DESCRIPTIVE_PARTIAL" if totals.observed["ibd"] else "NO_EVALUABLE",
                           quality_status="GT_MISSINGNESS_ONLY_NOT_BATCH_CONTROL")
            kin_rows.extend(rows)
            local_rows = list(totals.rows(name, edge, rows[0], ibd_territory))
            metrics.extend(local_rows)
            component_result = components.rows(name, edge)
            component_rows.extend(component_result)
            metadata_rows.extend(metadata_summary(name, edge, accumulator.degree, study))
            nominal_rows.extend(nominal_summary(name, edge, accumulator.degree, study))
            for chrom, counts in sorted(totals.chromosomes.items()):
                chrom_rows.append(dict(config_id=name, min_edge_bp=edge, chrom=chrom,
                    **counts, total_pair_bp=accumulator.bp,
                    fraction_pair_bp=counts["pair_bp"]/accumulator.bp,
                    status=STATUS, scope="selected_intervals_not_genome_callability"))
            cards.append(dict(config_id=name, min_edge_bp=edge, status=STATUS,
                question="What support, measurement and complementary evidence describe this M14 configuration?",
                answer=dict(n_people=n_people, n_active=rows[0]["n_active"], n_pairs=accumulator.n_pairs,
                    n_segments=totals.n, summed_disjoint_pair_bp=accumulator.bp,
                    measurements={r["metric"]: {k: r[k] for k in ("numerator", "denominator", "value", "status", "reason")} for r in local_rows}),
                missing={channel: dict(counts) for channel, counts in totals.missing.items() if counts},
                dependence_components=component_result,
                global_IBD_denominators=ibd_territory,
                comparison_scope="Same cohort, chromosome bundles, filters and thresholds; each geometry conditions on its own selected territory. Direct I/U comparisons on common territory are not estimated.",
                decision_status="DESCRIPTIVE_ONLY_NO_SELECTION",
                uncertainty_status="NO_EVALUABLE_NO_DEPENDENCE_RESAMPLING_DESIGN",
                limitations=["M14-selected local J is not independent validation",
                    "Pair-bp and pair-sites are not independent biological replicates",
                    "IBD concordance is inferred complementary evidence, not population truth",
                    "Observed zero is descriptive, not a precise negative biological result",
                    "No matched common-territory contrast, biological winner or population declaration"]))
    return metrics, kin_rows, chrom_rows, cards, component_rows, metadata_rows, nominal_rows


def prepare_chromosome(db_path, directory, path, sample_ids, sample_index,
                       expected_configurations, registry, ibd_territory,
                       max_database_mb, min_free_disk_mb, check_resources):
    """Validate one immutable bundle and export the existing sorted streams."""
    db = setup_database(db_path)
    try:
        check_resources()
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        db.execute("PRAGMA max_page_count="+str(max(1, int(max_database_mb*1024**2)//page_size)))
        load_bundle(db, path, sample_ids, sample_index, expected_configurations, registry,
                    db_path, max_database_mb, min_free_disk_mb, check_resources)
        validate_intervals(db, registry["configs"])
        global_row = db.execute("SELECT count(*) AS n,sum(callable_bp) AS callable_bp,sum(ibd_union_bp) AS ibd_union_bp FROM ibd_territory WHERE status='OK'").fetchone()
        ibd_territory["n_evaluable_pair_chromosomes"] += global_row["n"]
        for field in ("callable_bp", "ibd_union_bp"):
            ibd_territory[field] += global_row[field] or 0
        segment_stream, id_stream = export_chromosome(db, Path(directory),
            registry["chromosomes"][-1], check_resources)
    finally:
        db.close()
    # Only this process's newly created, disposable scratch database.
    db_path.unlink()
    return segment_stream, id_stream


def finish_prepared(segments, identifiers, registry, ibd_territory, samples,
                    sample_ids, pcrelate_file, expected_pcrelate_sha256, output_dir,
                    study, thresholds, edge_thresholds_bp, usage, max_database_mb,
                    min_free_disk_mb, check_resources, write_manifest=True):
    """Shared global reduction; accumulated-bp filtering remains inside aggregate_pairs."""
    output = Path(output_dir)
    require(not output.exists(), "Output already exists; no overwrite")
    sample_index = {sample: i for i, sample in enumerate(samples)}
    sample_sha = sha256(sample_ids)
    unique_chain_ids(identifiers)
    kin_values, kin_audit = diagnostics.kinship.stream_kinship(pcrelate_file, sample_index,
        diagnostics.CohortPairs(len(samples)), expected_pcrelate_sha256)
    aggregates = aggregate_pairs(segments, registry["configs"], len(samples), kin_values,
        thresholds, sorted(edge_thresholds_bp), study, check_resources)
    metrics, kin_rows, chrom_rows, cards, component_rows, metadata_rows, nominal_rows = summarize(
        aggregates, registry["configs"], len(samples), sorted(edge_thresholds_bp), ibd_territory, study)
    check_resources()
    require(sha256(sample_ids) == sample_sha, "Sample-ID file changed during evaluation")
    output.mkdir(parents=True)
    write_table(output/"configuration_metrics.tsv", metrics, METRIC_FIELDS)
    write_table(output/"configuration_kinship.tsv", kin_rows)
    chrom_fields = ("config_id", "min_edge_bp", "chrom", "n_segments", "pair_bp", "I", "U", "Q",
                    "total_pair_bp", "fraction_pair_bp", "status", "scope")
    write_table(output/"configuration_chromosomes.tsv", chrom_rows, chrom_fields)
    write_table(output/"configuration_components.tsv", component_rows, COMPONENT_FIELDS)
    write_table(output/"configuration_metadata_coverage.tsv", metadata_rows)
    write_table(output/"configuration_metadata_categories.private.tsv", nominal_rows,
                ("config_id", "min_edge_bp", "field", "category", "stratum", "n_people_category", "n_people", "fraction_people", "status"))
    diagnostics.sweep.write_json(output/"configuration_cards.json", dict(schema=SCHEMA, status=STATUS, cards=cards))
    result = dict(schema=SCHEMA, status=STATUS, sample_ids_sha256=sample_sha, n_samples=len(samples),
        chromosomes=sorted(registry["chromosomes"], key=int), n_configurations=len(registry["configs"]),
        n_cards=len(cards), n_metrics=len(metrics), **registry["identity"],
        optional_evidence_criteria=registry["optional_criteria"], input_manifests=registry["sources"],
        ibd_territory_chromosomes=sorted(registry["ibd_chromosomes"], key=int),
        pcrelate_audit=kin_audit, study_contract=study["provenance"] if study else None,
        parameters=dict(kinship_thresholds=list(thresholds), edge_thresholds_bp=sorted(edge_thresholds_bp),
            max_database_mb=max_database_mb, min_free_disk_mb=min_free_disk_mb),
        resources=dict(sqlite_bytes=usage["peak_observed_sqlite_bytes"], **usage, cache_limit_mib=32,
            scalability="one-chromosome SQLite validation; compressed pair-sorted streams; merge across all supplied chromosomes before T; cohort PC-Relate mapping and configuration/threshold/person accumulators in memory; full-scale resources unmeasured",
            scratch_limits="max_database_mb caps main SQLite pages and checks main/sidecars; compressed streams use min_free_disk_mb reserve, not a fixed byte cap; SQLite transient sorting files and process RSS require executor-level limits",
            restart="single process; no internal resume; completed Nextflow task caching is external"),
        no_pvalues=True, no_winner_selected=True, no_population_labels=True,
        contains_individual_identifiers=False, inference_status="NO_EVALUABLE_DESCRIPTIVE_ONLY",
        privacy=dict(private_outputs=["configuration_metadata_categories.private.tsv"],
            reason="Nominal metadata counts can contain identifying small cells; controlled-access output, not public graphics/chat"),
        missing_for_inference=["predeclared_question_effect_and_precision", "authenticated_dependency_units_and_roles",
            "prior_exposure_and_leakage_audit", "measurement_batch_and_independent_outcomes",
            "matched_pair_common_territory_contrast", "dependence_aware_uncertainty_design"],
        semantics=dict(local_J="sum_I/sum_U over disjoint M14-selected intervals within configuration; not genome-wide J",
            overlap="within-configuration pair/chrom overlaps rejected; never pool configurations",
            ibd="sum territorial-union overlap / sum jointly callable pair-bp where IBD is evaluable; not IBD-copy length",
            kinship="historical sensitivities; below-cut does not certify independence",
            quality="incomplete diploid GT only; no new DP/GQ filters or batch adjustment",
            observed_zero="positive denominator and zero numerator; NOT a validated negative biological result"),
        outputs_sha256={p.name: sha256(p) for p in sorted(output.iterdir())},
        implementation_sha256=sha256(__file__),
        code_sha256={Path(module.__file__).name: sha256(module.__file__) for module in
                     (diagnostics, diagnostics.sweep, diagnostics.kinship, diagnostics.kinship.saved)})
    if write_manifest:
        diagnostics.sweep.write_json(output/"manifest.json", result)
    return result


def run(segment_manifests, sample_ids, expected_samples, pcrelate_file, expected_pcrelate_sha256,
        output_dir, expected_configurations=74, study_contract=None, thresholds=diagnostics.DEFAULT_THRESHOLDS,
        edge_thresholds_bp=DEFAULT_EDGES, scratch_dir=None, max_database_mb=8192, min_free_disk_mb=1024):
    output = Path(output_dir)
    require(not output.exists(), "Output already exists; no overwrite")
    require(segment_manifests, "At least one segment manifest is required")
    require(thresholds and len(set(thresholds)) == len(thresholds) and
            all(math.isfinite(t) and 0 <= t <= .5 for t in thresholds), "Invalid kinship thresholds")
    require(edge_thresholds_bp and 0 in edge_thresholds_bp and len(set(edge_thresholds_bp)) == len(edge_thresholds_bp)
            and all(type(t) is int and t >= 0 for t in edge_thresholds_bp), "Invalid accumulated-bp thresholds; baseline zero required")
    require(math.isfinite(max_database_mb) and math.isfinite(min_free_disk_mb) and
            max_database_mb > 0 and min_free_disk_mb >= 0, "Invalid scratch resource limits")
    samples = diagnostics.sweep.samples_from(sample_ids, expected_samples)
    require(len(samples) >= 2, "Pair evaluation needs at least two analytical samples")
    sample_index = {sample: i for i, sample in enumerate(samples)}
    sample_sha = sha256(sample_ids)
    study = read_study(study_contract, samples, expected_pcrelate_sha256, thresholds)
    registry = dict(identity=None, configs=None, optional_criteria={}, chromosomes=[], ibd_chromosomes=[], sources=[])
    ibd_territory = dict(n_evaluable_pair_chromosomes=0, callable_bp=0, ibd_union_bp=0)
    usage = dict(peak_observed_sqlite_bytes=0, peak_observed_scratch_bytes=0)
    with tempfile.TemporaryDirectory(prefix="r02-biological-evaluation-", dir=scratch_dir) as temporary:
        db_path = Path(temporary)/"evidence.sqlite"
        def check_resources():
            size = resources(db_path, max_database_mb, min_free_disk_mb)
            usage["peak_observed_sqlite_bytes"] = max(usage["peak_observed_sqlite_bytes"], size)
            scratch_bytes = sum(p.stat().st_size for p in Path(temporary).iterdir() if p.is_file())
            usage["peak_observed_scratch_bytes"] = max(usage["peak_observed_scratch_bytes"], scratch_bytes)
        check_resources()
        segments, identifiers = [], []
        for path in segment_manifests:
            segment_stream, id_stream = prepare_chromosome(db_path, Path(temporary),
                path, sample_ids, sample_index, expected_configurations, registry,
                ibd_territory, max_database_mb, min_free_disk_mb, check_resources)
            segments.append(segment_stream)
            identifiers.append(id_stream)
        require(sha256(sample_ids) == sample_sha, "Sample-ID file changed during preparation")
        return finish_prepared(segments, identifiers, registry, ibd_territory, samples,
            sample_ids, pcrelate_file, expected_pcrelate_sha256, output, study, thresholds,
            edge_thresholds_bp, usage, max_database_mb, min_free_disk_mb, check_resources)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-manifest", dest="segment_manifests", action="append", required=True)
    for name in ("sample-ids", "pcrelate-file", "expected-pcrelate-sha256", "output-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--expected-configurations", type=int, default=74)
    parser.add_argument("--study-contract")
    parser.add_argument("--scratch-dir")
    parser.add_argument("--max-database-mb", type=int, default=8192)
    parser.add_argument("--min-free-disk-mb", type=int, default=1024)
    parser.add_argument("--thresholds", default="0.0221,0.0442")
    parser.add_argument("--edge-thresholds-bp", default="0,250000,500000,750000,1000000")
    args = vars(parser.parse_args(argv))
    args["thresholds"] = tuple(float(value) for value in args["thresholds"].split(","))
    args["edge_thresholds_bp"] = tuple(int(value) for value in args["edge_thresholds_bp"].split(","))
    result = run(**args)
    print(json.dumps({key: result[key] for key in ("status", "n_configurations", "n_cards")}))


if __name__ == "__main__":
    main()
