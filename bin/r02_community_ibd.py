#!/usr/bin/env python3
"""Describe reported IBD within and between fixed, authenticated communities.

Two restartable stages preserve chromosome-specific coordinates. The first
unions calls by pair and by person; the second sums disjoint chromosomes and
describes all saved partitions. No new IBD, ancestry, phase, pedigree or IBD1/2
classification is inferred. No-reported-call is a reporting state, not a
biological negative. Pair counts are descriptive, never independent replicates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import itertools
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import tempfile
import time

from r02_community_levels import ANCESTRY, require, sha256

SCHEMA = "r02_community_ibd_inputs_v1"
PART_SCHEMA = "r02_community_ibd_chromosome_v1"
RESULT_SCHEMA = "r02_community_ibd_summary_v1"
RELATIONS = ("ALL", "SAME_COMPONENT", "DIFFERENT_COMPONENT", "UNKNOWN_COMPONENT")
GEOMETRY = ("reported_union_bp", "hap_pair_union_bp_sum", "one_hap_edge_bp",
            "two_disjoint_hap_edges_bp", "two_shared_homologue_edges_bp",
            "three_hap_edges_bp", "four_hap_edges_bp", "source_record_bp_sum",
            "source_records", "source_record_cm_sum")
PAIR_FIELDS = ("chromosome", "sample_a", "sample_b", *GEOMETRY)
PERSON_FIELDS = ("chromosome", "sample_id", "reported_any_partner_union_bp",
                 "reported_pair_union_bp_sum", "reported_partners")
LIMITATIONS = [
    "Reported-call geometry only; no reported call is not biological absence.",
    "No certified IBD detection territory, genotype callability or sequencing batch adjustment.",
    "Haplotype-pair edges are retained; observed edge patterns do not certify IBD1/IBD2.",
    "Original record cM sums are audit quantities, not union cM or ages.",
    "Fixed transductive communities; no independent population validation or new clustering.",
    "Component strata are transitive PC-Relate connections, not certified pedigrees or independent units.",
    "Global ancestry summaries describe confounding; they do not adjust or match ancestry.",
    "No p-values, confidence intervals, founder inference, fine-structure claim or rare increment.",
    "Local ancestry annotations, when inventoried, do not phase unphased rare alleles.",
    "Private pair/person outputs and small cells; no public distribution authorized.",
]


def strict_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    with Path(path).open() as handle:
        return json.load(handle, object_pairs_hook=unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def save_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def table(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "Invalid table header")
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), "Ragged table")
            yield row


def write_table(path, rows, fields=None):
    opener = gzip.open if str(path).endswith(".gz") else open
    count = 0
    with opener(path, "xt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t") if fields else None
        if writer:
            writer.writeheader()
        for row in rows:
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
                writer.writeheader()
            writer.writerow({k: "NA" if v is None else v for k, v in row.items()})
            count += 1
        require(writer is not None, "Cannot write an untyped empty table")
    return count


class Guard:
    """Bound RSS, scratch and SQLite size; stage completion never overrides guards."""
    def __init__(self, scratch, max_memory_mb, max_db_mb, min_free_mb):
        require(type(max_memory_mb) is int and max_memory_mb > 0 and
                type(max_db_mb) is int and max_db_mb > 0 and
                type(min_free_mb) is int and min_free_mb >= 0, "Explicit valid resource limits required")
        self.scratch = Path(scratch).resolve(strict=True)
        require(self.scratch.is_dir(), "Scratch directory missing")
        self.memory, self.database, self.reserve = max_memory_mb, max_db_mb, min_free_mb
        self.check()

    def check(self):
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 <= self.memory, "RSS limit exceeded")
        require(shutil.disk_usage(self.scratch).free >= self.reserve * 2**20, "Scratch reserve exhausted")

    def connect(self, path):
        db = sqlite3.connect(path)
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-16384")
        size = db.execute("PRAGMA page_size").fetchone()[0]
        db.execute(f"PRAGMA max_page_count={max(1, self.database * 2**20 // size)}")
        return db


class Ledger:
    def __init__(self, root):
        self.root, self.records = Path(root), {}

    def add(self, record, root=None):
        require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "Expected path/SHA256 record")
        p = Path(record["path"])
        p = ((self.root if root is None else Path(root)) / p).resolve(strict=True)
        require(p.is_file() and sha256(p) == record["sha256"], "Input hash mismatch: " + p.name)
        entry = dict(sha256=record["sha256"], bytes=p.stat().st_size)
        require(str(p) not in self.records or self.records[str(p)] == entry, "Conflicting input identity")
        self.records[str(p)] = entry
        return p

    def adjacent(self, manifest_path, manifest, name):
        require(Path(name).name == name, "Unsafe adjacent name")
        return self.add(dict(path=name, sha256=manifest["outputs_sha256"][name]), manifest_path.parent)

    def recheck(self):
        for p, entry in self.records.items():
            require(Path(p).stat().st_size == entry["bytes"] and sha256(p) == entry["sha256"], "Input changed")


def load_contract(path, expected_sha):
    path = Path(path).resolve(strict=True)
    ledger = Ledger(path.parent)
    ledger.add(dict(path=str(path), sha256=expected_sha))
    contract = strict_json(path)
    require(set(contract) == {"schema", "genome_build", "expected_samples", "expected_partitions",
            "chromosomes", "community_evidence", "ibd", "local_ancestry_support", "scientific_scope"}, "Unexpected contract fields")
    require(contract["schema"] == SCHEMA and contract["genome_build"], "Invalid schema/build")
    require(contract["scientific_scope"] == "REPORTED_IBD_DESCRIPTIVE_COMMUNITY_ASSOCIATION", "Unsupported scientific claim")
    chromosomes = contract["chromosomes"]
    require(isinstance(chromosomes, list) and chromosomes and len(set(chromosomes)) == len(chromosomes)
            and all(c in {str(i) for i in range(1, 23)} for c in chromosomes), "Explicit unique autosomes required")
    require([s["chromosome"] for s in contract["ibd"]] == chromosomes, "IBD chromosome inventory differs")
    require(type(contract["expected_samples"]) is int and contract["expected_samples"] >= 2 and
            type(contract["expected_partitions"]) is int and contract["expected_partitions"] > 0, "Invalid expected dimensions")
    return contract, ledger


def load_people(contract, ledger):
    path = ledger.add(contract["community_evidence"])
    manifest = strict_json(path)
    require(manifest["schema"] == "r02_community_levels_v1" and
            manifest["status"] == "COMPLETE_DESCRIPTIVE_LEVEL_COMPARISON_NOT_POPULATION_VALIDATION", "Unexpected community evidence")
    require(manifest["n_cohort"] == contract["expected_samples"] and
            manifest["n_partitions"] == contract["expected_partitions"], "Community dimensions differ")
    summary = strict_json(ledger.adjacent(path, manifest, "summary.json"))
    rows = list(table(ledger.adjacent(path, manifest, "individuals.private.tsv")))
    samples = [r["sample_id"] for r in rows]
    require(len(samples) == contract["expected_samples"] and len(set(samples)) == len(samples), "Cohort incomplete/duplicated")
    import hashlib
    sample_hash = hashlib.sha256(("\n".join(samples) + "\n").encode()).hexdigest()
    require(sample_hash == manifest["sample_ids_sha256"], "Ordered cohort hash differs")
    partitions, phis = summary["partition_ledger"], summary["kinship_thresholds"]
    require(len(partitions) == contract["expected_partitions"] and
            len({p["partition_id"] for p in partitions}) == len(partitions), "Partition ledger incomplete/duplicated")
    require(phis and len(set(phis)) == len(phis) and all(math.isfinite(p) and 0 < p < 1 for p in phis), "Invalid component thresholds")
    ancestry, components, labels = [], {}, {}
    for r in rows:
        values = [None if r[f] == "NA" else float(r[f]) for f in ANCESTRY]
        require(all(v is None or math.isfinite(v) and 0 <= v <= 1 for v in values), "Invalid ancestry fraction")
        ancestry.append(None if any(v is None for v in values) else values)
    for phi in phis:
        field = f"component_phi_{phi:g}"
        components[phi] = [None if r[field] == "NA" else r[field] for r in rows]
    for p in partitions:
        label = [int(r[p["partition_id"]]) for r in rows]
        require(all(v >= -1 for v in label), "Invalid community label")
        labels[p["partition_id"]] = label
    return dict(samples=samples, order={s: i for i, s in enumerate(samples)}, sample_hash=sample_hash,
                ancestry=ancestry, components=components, partitions=partitions, labels=labels, phis=phis)


def geometry(records):
    """Inclusive interval sweep retaining all four haplotype-pair edge identities.

    Multiplicity of overlapping records for one edge never increases territory.
    Two disjoint edges are a reported pattern, not a pedigree IBD2 inference.
    Returns merged any-copy intervals for the independent per-person union.
    """
    require(records, "Empty call group")
    events = defaultdict(lambda: [0, 0, 0, 0])
    result = dict.fromkeys(GEOMETRY, 0)
    for lo, hi, h1, h2, lod, cm in records:
        require(type(lo) is int and type(hi) is int and 1 <= lo <= hi and h1 in (1, 2) and h2 in (1, 2), "Invalid call geometry")
        require(math.isfinite(lod) and math.isfinite(cm) and cm >= 0, "Invalid native score")
        edge = (h1 - 1) * 2 + h2 - 1
        events[lo][edge] += 1
        events[hi + 1][edge] -= 1
        result["source_records"] += 1
        result["source_record_bp_sum"] += hi - lo + 1
        result["source_record_cm_sum"] += cm
    counts, prior, merged = [0] * 4, None, []
    for pos in sorted(events):
        if prior is not None and pos > prior:
            active = tuple(i for i, n in enumerate(counts) if n > 0)
            size = pos - prior
            if active:
                result["reported_union_bp"] += size
                result["hap_pair_union_bp_sum"] += len(active) * size
                key = {1: "one_hap_edge_bp", 3: "three_hap_edges_bp", 4: "four_hap_edges_bp"}.get(len(active))
                if len(active) == 2:
                    key = "two_disjoint_hap_edges_bp" if active in ((0, 3), (1, 2)) else "two_shared_homologue_edges_bp"
                result[key] += size
                if merged and merged[-1][1] == prior - 1:
                    merged[-1] = (merged[-1][0], pos - 1)
                else:
                    merged.append((prior, pos - 1))
        counts = [a + b for a, b in zip(counts, events[pos])]
        require(all(n >= 0 for n in counts), "Unbalanced interval sweep")
        prior = pos
    require(counts == [0] * 4, "Unclosed interval sweep")
    return result, merged


def union_size(ordered):
    """Union of sorted inclusive intervals without retaining the full collection."""
    total, left, right, prior = 0, None, None, None
    for lo, hi in ordered:
        require(1 <= lo <= hi and (prior is None or lo >= prior), "Intervals unordered or invalid")
        prior = lo
        if left is None:
            left, right = lo, hi
        elif lo <= right + 1:
            right = max(right, hi)
        else:
            total += right - left + 1
            left, right = lo, hi
    return total + (right - left + 1 if left is not None else 0)


def finish_manifest(output, ledger, schema, status, **values):
    ledger.recheck()
    result = dict(schema=schema, status=status, input_files=ledger.records,
                  source_sha256={Path(__file__).name: sha256(__file__), "r02_community_levels.py": sha256(Path(__file__).with_name("r02_community_levels.py"))},
                  contains_individual_identifiers=True, public_distribution_allowed=False,
                  callability="UNKNOWN", ibd1_ibd2="NOT_INFERRED", limitations=LIMITATIONS,
                  **values, outputs_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    save_json(output / "manifest.json", result)
    return result


def chromosome_stage(path, expected_sha, chrom, output, guard, max_pair_records):
    start = time.monotonic()
    contract, ledger = load_contract(path, expected_sha)
    require(chrom in contract["chromosomes"] and type(max_pair_records) is int and max_pair_records > 0, "Invalid chromosome/record bound")
    people = load_people(contract, ledger)
    spec = next(s for s in contract["ibd"] if s["chromosome"] == chrom)
    require(set(spec) == {"chromosome", "manifest", "contract"}, "Unexpected IBD source fields")
    mp, cp = ledger.add(spec["manifest"]), ledger.add(spec["contract"])
    manifest, source = strict_json(mp), strict_json(cp)
    require(manifest["status"] == "PREPARED_REPORTED_GEOMETRY_NOT_BIOLOGICAL_VALIDATION" and
            manifest["chromosome"] == chrom and manifest["analytical_samples"] == len(people["samples"]), "IBD preparation differs")
    require(source["schema"] == "r02_segment_ibd_v2" and source["genome_build"] == contract["genome_build"] and
            source["chromosome"] == chrom and source["coordinate_system"] == "1-based-inclusive" and
            source["semantics"] == "ANY_COPY_REPORTED" and source["evidence_scope"] == "reported_call_geometry_only" and
            source["callset_scope"] == "complete_reported_calls_for_declared_chromosome_and_samples" and
            source["absence_semantics"] == "no_reported_call_not_biological_absence" and
            source["analytical_samples_sha256"] == people["sample_hash"], "IBD support/semantics differ")
    require(cp.parent == mp.parent and sha256(cp) == manifest["outputs_sha256"][cp.name], "IBD contract/manifest mismatch")
    calls = ledger.add(source["intervals"], cp.parent)
    require(sha256(calls) == manifest["outputs_sha256"][calls.name], "IBD calls/manifest mismatch")
    output = Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    totals = Counter()
    with tempfile.TemporaryDirectory(prefix="community-ibd-", dir=guard.scratch) as temp:
        db = guard.connect(str(Path(temp) / "person.sqlite"))
        try:
            db.execute("CREATE TABLE territory(person INTEGER,partner INTEGER,lo INTEGER,hi INTEGER)")
            order = people["order"]
            previous = None
            def read_calls():
                nonlocal previous
                for row in table(calls):
                    require(row["chrom"] == chrom and row["ibd_state"] == "ANY_COPY_REPORTED", "Unexpected call schema")
                    require(row["sample_a"] in order and row["sample_b"] in order, "Foreign call sample")
                    a, b = order[row["sample_a"]], order[row["sample_b"]]
                    lo, hi, h1, h2 = (int(row[k]) for k in ("start_pos", "end_pos", "hap_a", "hap_b"))
                    require(a < b, "Prepared pair not in authenticated cohort order")
                    key = a, b, lo, hi
                    require(previous is None or key >= previous, "Prepared calls not sorted")
                    previous = key
                    yield a, b, (lo, hi, h1, h2, float(row["lod"]), float(row["length_cm"]))
            def pair_rows():
                for (a, b), grouped in itertools.groupby(read_calls(), key=lambda r: r[:2]):
                    records = []
                    for _, _, call in grouped:
                        require(len(records) < max_pair_records, "Pair record bound exceeded")
                        records.append(call)
                    stats, merged = geometry(records)
                    for lo, hi in merged:
                        db.executemany("INSERT INTO territory VALUES(?,?,?,?)", ((a, b, lo, hi), (b, a, lo, hi)))
                    totals.update(stats)
                    totals["pairs_with_reported_call"] += 1
                    if totals["pairs_with_reported_call"] % 1000 == 0:
                        guard.check()
                    yield dict(chromosome=chrom, sample_a=people["samples"][a], sample_b=people["samples"][b], **stats)
            write_table(output / "pairs.private.tsv.gz", pair_rows(), PAIR_FIELDS)
            db.commit()
            db.execute("CREATE INDEX territory_order ON territory(person,lo,hi)")
            def person_rows():
                for i, sample in enumerate(people["samples"]):
                    union = union_size(db.execute("SELECT lo,hi FROM territory WHERE person=? ORDER BY lo,hi", (i,)))
                    pair_sum, partners = db.execute("SELECT COALESCE(SUM(hi-lo+1),0),COUNT(DISTINCT partner) FROM territory WHERE person=?", (i,)).fetchone()
                    yield dict(chromosome=chrom, sample_id=sample, reported_any_partner_union_bp=union,
                               reported_pair_union_bp_sum=pair_sum, reported_partners=partners)
                    if i % 100 == 0:
                        guard.check()
            write_table(output / "persons.private.tsv.gz", person_rows(), PERSON_FIELDS)
        finally:
            db.close()
    require(totals["source_records"] == manifest["analytical_records"] and
            totals["pairs_with_reported_call"] == manifest["pairs_with_reported_ibd"] and
            totals["reported_union_bp"] == manifest["pair_bp_reported_union"], "Union/counts differ from source preparation")
    guard.check()
    return finish_manifest(output, ledger, PART_SCHEMA, "COMPLETE_REPORTED_IBD_CHROMOSOME_DESCRIPTION",
        chromosome=chrom, genome_build=contract["genome_build"], source_contract_sha256=expected_sha,
        community_evidence_sha256=contract["community_evidence"]["sha256"], samples_sha256=people["sample_hash"],
        n_samples=len(people["samples"]), totals=dict(totals), seconds=time.monotonic()-start)


def moment(indices, ancestry):
    indices = list(indices)
    q = [ancestry[i] for i in indices if ancestry[i] is not None]
    return (len(indices), len(q), [math.fsum(v[d] for v in q) for d in range(4)],
            math.fsum(math.fsum(x*x for x in v) for v in q))


def pair_moments(a, b=None):
    """Exact possible-pair counts and Q distance sums without enumerating pairs."""
    n, nq, sums, sq = a
    if b is None:
        return [n*(n-1)//2, nq*(nq-1)//2,
                max(0., nq*sq-math.fsum(v*v for v in sums))]
    m, mq, msums, msq = b
    return [n*m, nq*mq, max(0., mq*sq+nq*msq-2*math.fsum(x*y for x, y in zip(sums, msums)))]


def subtract(a, b):
    result = [x-y for x, y in zip(a, b)]
    require(result[0] >= 0 and result[1] >= 0 and result[2] >= -1e-7, "Invalid combinatorial subtraction")
    result[2] = max(0., result[2])
    return result


def relation_denominators(left, right, ancestry, components):
    """Partition possible pairs into known same/different and unknown components."""
    left, right = list(left), None if right is None else list(right)
    require(len(set(left)) == len(left) and (right is None or len(set(right)) == len(right) and not set(left)&set(right)), "Overlapping/duplicated scope")
    def buckets(indices):
        result = defaultdict(list)
        for i in indices:
            if components[i] is not None:
                result[components[i]].append(i)
        return result
    lc, rc = buckets(left), None if right is None else buckets(right)
    known_left = [i for v in lc.values() for i in v]
    all_pairs = pair_moments(moment(left, ancestry), None if right is None else moment(right, ancestry))
    known = pair_moments(moment(known_left, ancestry), None if rc is None else moment([i for v in rc.values() for i in v], ancestry))
    same = [0, 0, 0.]
    for key, group in lc.items():
        values = pair_moments(moment(group, ancestry), None if rc is None else moment(rc.get(key, []), ancestry))
        same = [a+b for a, b in zip(same, values)]
    return dict(ALL=all_pairs, SAME_COMPONENT=same, DIFFERENT_COMPONENT=subtract(known, same), UNKNOWN_COMPONENT=subtract(all_pairs, known))


def scopes(labels):
    """Separate global disjoint categories from overlapping per-community views."""
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    assigned = [i for label, ids in groups.items() if label >= 0 for i in ids]
    unassigned = groups.get(-1, [])
    yield "all_pairs", -1, list(range(len(labels))), None
    yield "both_unassigned", -1, unassigned, None
    yield "one_unassigned", -1, assigned, unassigned
    # Global within/between are assembled from the local views without duplicate pairs.
    for label in sorted(k for k in groups if k >= 0):
        yield "community_within", label, groups[label], None
        yield "community_to_other_assigned", label, groups[label], [i for i in assigned if labels[i] != label]


def possible_support(people):
    result = {}
    for partition in people["partitions"]:
        pid = partition["partition_id"]
        labels = people["labels"][pid]
        for phi in people["phis"]:
            for scope, label, left, right in scopes(labels):
                values = relation_denominators(left, right, people["ancestry"], people["components"][phi])
                for relation, counts in values.items():
                    result[pid, scope, label, phi, relation] = counts
            for scope in ("within_assigned", "between_assigned"):
                source_scope = "community_within" if scope == "within_assigned" else "community_to_other_assigned"
                for relation in RELATIONS:
                    values = [v for (p, s, _, f, r), v in result.items() if p == pid and s == source_scope and f == phi and r == relation]
                    total = [math.fsum(v[d] for v in values) for d in range(3)]
                    if scope == "between_assigned":
                        total = [v/2 for v in total]
                    total[:2] = [int(v) for v in total[:2]]
                    result[pid, scope, -1, phi, relation] = total
    return result


def observed_scopes(a, b):
    result = [("all_pairs", -1)]
    if a < 0 and b < 0:
        return result + [("both_unassigned", -1)]
    if a < 0 or b < 0:
        return result + [("one_unassigned", -1)]
    if a == b:
        return result + [("within_assigned", -1), ("community_within", a)]
    return result + [("between_assigned", -1), ("community_to_other_assigned", a), ("community_to_other_assigned", b)]


def describe_pairs_reference(pair_rows, people, possible, guard):
    observed = {}
    degree = [0] * len(people["samples"])
    for row_number, (a, b, metrics, n_chromosomes) in enumerate(pair_rows):
        degree[a] += 1
        degree[b] += 1
        qa, qb = people["ancestry"][a], people["ancestry"][b]
        distance = None if qa is None or qb is None else math.fsum((x-y)**2 for x, y in zip(qa, qb))
        for p in people["partitions"]:
            pid = p["partition_id"]
            labels = people["labels"][pid]
            for phi in people["phis"]:
                ca, cb = people["components"][phi][a], people["components"][phi][b]
                relation = "UNKNOWN_COMPONENT" if ca is None or cb is None else "SAME_COMPONENT" if ca == cb else "DIFFERENT_COMPONENT"
                for scope, label in observed_scopes(labels[a], labels[b]):
                    for rel in ("ALL", relation):
                        key = pid, scope, label, phi, rel
                        require(key in possible, "Observed pair outside declared support")
                        if key not in observed:
                            observed[key] = dict(n_reported_pairs=0, n_reported_chromosome_pairs=0, person_bits=0,
                                n_reported_pairs_complete_Q=0, Q_distance_sum=0., Q_distance_bp_sum=0., complete_Q_reported_bp=0,
                                **dict.fromkeys(GEOMETRY, 0))
                        value = observed[key]
                        value["n_reported_pairs"] += 1
                        value["n_reported_chromosome_pairs"] += n_chromosomes
                        value["person_bits"] |= (1 << a) | (1 << b)
                        for name, amount in zip(GEOMETRY, metrics):
                            value[name] += amount
                        if distance is not None:
                            value["n_reported_pairs_complete_Q"] += 1
                            value["Q_distance_sum"] += distance
                            value["Q_distance_bp_sum"] += distance * metrics[0]
                            value["complete_Q_reported_bp"] += metrics[0]
        if row_number % 1000 == 0:
            guard.check()
    return observed, degree


def describe_pairs(pair_rows, people, possible, guard, block_size):
    """Bounded vector reductions, equivalent to the small scalar reference.

    Pair identity is sparse. The only dense presence matrix is
    community-scope by person, not person by person or locus by person.
    """
    import numpy as np
    require(type(block_size) is int and block_size > 0, "Positive pair block size required")
    keys = list(possible)
    lookup = {key: i for i, key in enumerate(keys)}
    n_people = len(people["samples"])
    require(len(keys)*n_people < guard.memory*2**20*.4, "Presence workspace exceeds memory admission")
    fields = [*GEOMETRY, "n_reported_pairs", "n_reported_chromosome_pairs", "n_reported_pairs_complete_Q",
              "Q_distance_sum", "Q_distance_bp_sum", "complete_Q_reported_bp"]
    accum = np.zeros((len(keys), len(fields)), dtype=np.float64)
    present = np.zeros((len(keys), n_people), dtype=np.bool_)
    degree = np.zeros(n_people, dtype=np.int64)
    q = np.array([[np.nan]*4 if v is None else v for v in people["ancestry"]], dtype=np.float64)
    component_codes = {}
    for phi in people["phis"]:
        codes = {v: i for i, v in enumerate(sorted({v for v in people["components"][phi] if v is not None}))}
        component_codes[phi] = np.array([-1 if v is None else codes[v] for v in people["components"][phi]], dtype=np.int64)
    labels = {k: np.asarray(v, dtype=np.int64) for k, v in people["labels"].items()}
    it = iter(pair_rows)
    while True:
        batch = list(itertools.islice(it, block_size))
        if not batch:
            break
        a = np.array([r[0] for r in batch], dtype=np.int64)
        b = np.array([r[1] for r in batch], dtype=np.int64)
        values = np.asarray([r[2] for r in batch], dtype=np.float64)
        require(np.all(np.isfinite(values)) and np.all(values >= 0) and np.all(values[:, :-1] < 2**53), "Geometry not exactly representable")
        chromosomes = np.asarray([r[3] for r in batch], dtype=np.float64)
        np.add.at(degree, a, 1)
        np.add.at(degree, b, 1)
        distances = np.sum((q[a]-q[b])**2, axis=1)
        complete = np.isfinite(distances)
        distances = np.where(complete, distances, 0.)
        payload = np.column_stack((values, np.ones(len(batch)), chromosomes, complete.astype(float),
                                   distances, distances*values[:, 0], np.where(complete, values[:, 0], 0.)))
        indexes = np.arange(len(batch))
        for p in people["partitions"]:
            pid = p["partition_id"]
            la, lb = labels[pid][a], labels[pid][b]
            for phi in people["phis"]:
                ca, cb = component_codes[phi][a], component_codes[phi][b]
                relation = np.where((ca < 0)|(cb < 0), 3, np.where(ca == cb, 1, 2))
                all_ids, positions = [], []
                def add(scope, pos, local_labels=None):
                    if len(pos) == 0:
                        return
                    if local_labels is None:
                        ids = np.array([lookup[pid, scope, -1, phi, r] for r in RELATIONS], dtype=np.int64)
                        all_ids.extend((np.full(len(pos), ids[0], dtype=np.int64), ids[relation[pos]]))
                    else:
                        observed_labels = np.unique(local_labels)
                        table = np.full((int(observed_labels.max())+1, 4), -1, dtype=np.int64)
                        for label in observed_labels:
                            table[label] = [lookup[pid, scope, int(label), phi, r] for r in RELATIONS]
                        all_ids.extend((table[local_labels, 0], table[local_labels, relation[pos]]))
                    positions.extend((pos, pos))
                add("all_pairs", indexes)
                within = indexes[(la >= 0)&(la == lb)]
                between = indexes[(la >= 0)&(lb >= 0)&(la != lb)]
                add("within_assigned", within)
                add("between_assigned", between)
                add("one_unassigned", indexes[(la < 0)^(lb < 0)])
                add("both_unassigned", indexes[(la < 0)&(lb < 0)])
                add("community_within", within, la[within])
                add("community_to_other_assigned", between, la[between])
                add("community_to_other_assigned", between, lb[between])
                ids, pos = np.concatenate(all_ids), np.concatenate(positions)
                unique, inverse = np.unique(ids, return_inverse=True)
                for d in range(len(fields)):
                    accum[unique, d] += np.bincount(inverse, weights=payload[pos, d], minlength=len(unique))
                present[ids, a[pos]] = True
                present[ids, b[pos]] = True
        guard.check()
    observed = {}
    int_fields = set(GEOMETRY[:-1]) | {"n_reported_pairs", "n_reported_chromosome_pairs", "n_reported_pairs_complete_Q", "complete_Q_reported_bp"}
    for i, key in enumerate(keys):
        if accum[i, len(GEOMETRY)] == 0:
            continue
        entry = {}
        for field, value in zip(fields, accum[i]):
            if field in int_fields:
                require(value < 2**53 and value.is_integer(), "Integer aggregation loses precision")
                value = int(value)
            entry[field] = value
        entry["n_people_with_reported_calls"] = int(np.count_nonzero(present[i]))
        observed[key] = entry
    return observed, degree.tolist()


def result_rows(people, possible, observed):
    partition_lookup = {p["partition_id"]: p for p in people["partitions"]}
    for key, denom in sorted(possible.items()):
        pid, scope, label, phi, relation = key
        record = observed.get(key, {})
        n = record.get("n_reported_pairs", 0)
        require(n <= denom[0], "More reported than possible pairs")
        nq = record.get("n_reported_pairs_complete_Q", 0)
        bp = record.get("reported_union_bp", 0)
        p = partition_lookup[pid]
        yield dict(partition_id=pid, graph_id=p["graph_id"], family=p["family"], resolution=p["resolution"],
            scope=scope, community_local=label, kinship_threshold=phi, component_relation=relation,
            n_possible_cohort_pairs=denom[0], n_reported_pairs=n, n_no_reported_call_pairs=denom[0]-n,
            no_call_semantics="NO_REPORTED_CALL_NOT_BIOLOGICAL_ABSENCE",
            reported_pair_fraction=n/denom[0] if denom[0] else None,
            n_people_with_reported_calls=record.get("n_people_with_reported_calls", record.get("person_bits", 0).bit_count()),
            n_reported_chromosome_pairs=record.get("n_reported_chromosome_pairs", 0),
            **{field: record.get(field, 0) for field in GEOMETRY},
            mean_reported_union_bp_given_call=bp/n if n else None,
            reported_union_bp_per_possible_pair=bp/denom[0] if denom[0] else None,
            n_possible_pairs_complete_Q=denom[1], mean_Q_distance_possible_pairs=denom[2]/denom[1] if denom[1] else None,
            n_reported_pairs_complete_Q=nq, mean_Q_distance_reported_pairs=record.get("Q_distance_sum", 0)/nq if nq else None,
            mean_Q_distance_reported_bp_weighted=record.get("Q_distance_bp_sum", 0)/record["complete_Q_reported_bp"] if record.get("complete_Q_reported_bp") else None,
            support_status="NO_POSSIBLE_PAIRS" if not denom[0] else "NO_REPORTED_CALLS" if not n else "DESCRIPTIVE_REPORTED_CALLS")


def write_interpretation(output, people, possible, observed):
    """Readable within/between comparisons, without ranking a biological winner."""
    indexed = {(r['partition_id'],r['scope'],r['kinship_threshold'],r['component_relation']):r
               for r in result_rows(people,possible,observed) if r['scope'] in ('within_assigned','between_assigned')}
    contrasts=[]
    for p in people['partitions']:
        for phi in people['phis']:
            for relation in RELATIONS:
                a=indexed[p['partition_id'],'within_assigned',phi,relation]
                b=indexed[p['partition_id'],'between_assigned',phi,relation]
                row={k:p[k] for k in ('partition_id','graph_id','family','resolution')}
                row.update(kinship_threshold=phi,component_relation=relation)
                for name,source in (('within',a),('between',b)):
                    for field in ('n_possible_cohort_pairs','n_reported_pairs','reported_pair_fraction',
                                  'reported_union_bp','mean_reported_union_bp_given_call',
                                  'reported_union_bp_per_possible_pair','mean_Q_distance_possible_pairs',
                                  'mean_Q_distance_reported_pairs','n_people_with_reported_calls'):
                        row[name+'_'+field]=source[field]
                row['interpretation']='DESCRIPTIVE_REPORTED_CALL_COMPARISON_NOT_ANCESTRY_ADJUSTED'
                contrasts.append(row)
    write_table(output/'within_between_comparisons.tsv',contrasts)
    def shown(value,scale=1):
        return 'NA' if value is None else f'{value/scale:.4g}'
    with (output/'INFORME.md').open('x') as out:
        out.write('# IBD informado dentro y entre comunidades: Ronda 2\n\n')
        out.write(f'Se reutilizan {len(people["partitions"])} particiones guardadas y {len(people["samples"])} personas. '
                  'Una partición es la asignación completa de personas a comunidades producida por una configuración de M16.5. '
                  'No se volvió a inferir IBD, ancestría, fase ni comunidades.\n\n')
        out.write('## Qué pregunta responde\n\n'
                  'IBD significa ADN informado por el programa fuente como heredado de un antepasado compartido. '
                  'Comparamos cuánto territorio de esas llamadas se encuentra entre personas de la misma comunidad y entre comunidades distintas. '
                  'Un resultado dentro de una comunidad puede deberse a familias, ancestría amplia, demografía, calidad o varias causas. '
                  'Por eso se muestran también las parejas de componentes de parentesco distintos y las diferencias ancestrales de las parejas comparadas. '
                  'La integración aporta evidencia genealógica complementaria; no es una prueba independiente que certifique poblaciones.\n\n'
                  '## Cómo se cuentan las longitudes\n\n'
                  'La longitud principal es la unión de intervalos por pareja y cromosoma, en pares de bases (bp). '
                  'Por ejemplo, 1–5 y 3–7 cubren 7 bp, no 10; los extremos son inclusivos. '
                  'Se suman después los cromosomas porque representan territorios diferentes. '
                  'El territorio por persona se vuelve a unir entre todos sus compañeros: no es la suma de sus parejas. '
                  'Las cuatro parejas de copias cromosómicas se conservan como patrones separados. '
                  'Dos pares de copias informados no se renombran automáticamente IBD2, ni un único par se certifica como IBD1. '
                  'La suma original de centimorgan (cM), unidad relacionada con recombinación, se conserva sólo como auditoría de registros; no es longitud genética de la unión.\n\n'
                  '## Denominadores y parentesco\n\n'
                  'Las parejas posibles cuentan personas de la cohorte autenticada en cada comparación. '
                  'La fracción informada divide las parejas con alguna llamada por esas parejas posibles; no es prevalencia biológica, precisión ni sensibilidad. '
                  'Una pareja sin llamada se conserva como NO_REPORTED_CALL, no como ausencia demostrada de IBD. '
                  'La longitud media condicionada a llamada sólo usa parejas con llamadas. '
                  'La carga informada por pareja posible divide el territorio informado por todas las parejas posibles y conserva su carácter de medida de reporte. '
                  'No conocemos el territorio donde el detector habría podido detectar todos los segmentos.\n\n'
                  'Un componente PC-Relate reúne personas conectadas directa o indirectamente al coeficiente φ indicado. '
                  'No es un pedigrí certificado; componentes distintos no garantizan independencia. '
                  'Las parejas comparten personas, por lo que no se producen intervalos de confianza ni valores p tratando cada pareja como réplica independiente.\n\n'
                  '## Comparación de las 57 particiones, sin seleccionar ganador\n\n'
                  'La tabla siguiente muestra parejas de componentes distintos, separando ambos umbrales. '
                  'Las columnas de longitud son megabases-pareja informadas por pareja posible, no fracción del genoma. '
                  'Qd es la distancia euclídea al cuadrado entre las cuatro proporciones ancestrales globales, promediada sobre todas las parejas posibles del mismo soporte. '
                  'Diferencias de Qd entre dentro y fuera revelan composición ancestral distinta: la tabla no ajusta esa confusión. '
                  'Todas las relaciones, incluidas las de componente igual o desconocido, permanecen en los archivos TSV.\n\n'
                  '| Configuración | γ | φ | Parejas posibles dentro/fuera | Con llamada dentro/fuera | Mb informadas/pareja dentro/fuera | Qd dentro/fuera |\n'
                  '|---|---:|---:|---:|---:|---:|---:|\n')
        for r in contrasts:
            if r['component_relation']!='DIFFERENT_COMPONENT':
                continue
            out.write(f'| {r["graph_id"]} | {r["resolution"]} | {r["kinship_threshold"]} | '
                      f'{r["within_n_possible_cohort_pairs"]}/{r["between_n_possible_cohort_pairs"]} | '
                      f'{r["within_n_reported_pairs"]}/{r["between_n_reported_pairs"]} | '
                      f'{shown(r["within_reported_union_bp_per_possible_pair"],1e6)}/{shown(r["between_reported_union_bp_per_possible_pair"],1e6)} | '
                      f'{shown(r["within_mean_Q_distance_possible_pairs"])}/{shown(r["between_mean_Q_distance_possible_pairs"])} |\n')
        out.write('\n## Qué queda respondido y qué no\n\n'
                  'K05 queda ampliado con una comparación completa de IBD informado por comunidad, soporte y estrato de componentes. '
                  'No queda certificado el parentesco reciente de todas las parejas. '
                  'K01 recibe un diagnóstico de composición ancestral, no un ajuste causal. '
                  'K02 (estructura fina), K03 (grupos discretos frente a continuidad), K04 (atribución dentro de ancestrías), '
                  'K06 (fundador/endogamia) y K07 (aporte raro incremental) no se cierran con estas tablas. '
                  'El inventario conserva los productos IBD×Gnomix existentes sin repetirlos: no asigna raras heterocigotas sin fase. '
                  'Los análisis ROH anteriores no se trasladan automáticamente a estas nuevas comunidades y no se repiten aquí.\n\n'
                  'γ controla la resolución del algoritmo, no una escala demográfica universal. '
                  'Que una configuración tenga más longitud dentro que fuera no basta para elegirla como óptima: cambian cobertura, tamaños, '
                  'composición ancestral y oportunidades de detección. '
                  'Para afirmar estructura fina se requiere el contraste retenido y los controles definidos en el plan, además de esta evidencia complementaria.\n\n'
                  '## Reproducibilidad y privacidad\n\n'
                  'Los manifiestos registran los archivos de entrada, sus huellas SHA256, código, unidades y límites. '
                  'Los parciales por cromosoma permiten reanudar sin repetir los terminados. '
                  'Las tablas por persona, pareja y celdas pequeñas son privadas; no publicarlas en un repositorio abierto.\n')


def aggregate_stage(path, expected_sha, parts, output, guard, block_size):
    start = time.monotonic()
    contract, ledger = load_contract(path, expected_sha)
    people = load_people(contract, ledger)
    require(len(parts) == len(contract["chromosomes"]), "Incomplete chromosome parts")
    output = Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    ancestry_inventory = []
    require(isinstance(contract["local_ancestry_support"], list), "Explicit local ancestry inventory required")
    for spec in contract["local_ancestry_support"]:
        p = ledger.add(spec["manifest"])
        m = strict_json(p)
        require(m["schema"] == "r02_ibd_local_ancestry_v1" and m["chromosome"] == spec["chromosome"] and
                m["status"] == "ANNOTATED_REPORTED_IBD_WITH_EXISTING_GNOMIX_NOT_VALIDATION", "Unexpected local ancestry support")
        native_spec = next(s for s in contract["ibd"] if s["chromosome"] == spec["chromosome"])
        native_manifest = strict_json(ledger.add(native_spec["manifest"]))
        require(m["genome_build"] == contract["genome_build"] and m["analytical_samples"] == len(people["samples"]) and
                all(m["input_sha256"][f] == native_manifest["input_sha256"][f] for f in ("native", "samples", "crosswalk")),
                "Local ancestry and native IBD support identities differ")
        ancestry_inventory.append(dict(chromosome=spec["chromosome"], manifest_sha256=sha256(p),
            phase_accuracy=m.get("phase_accuracy", "UNVERIFIED"), used_for_pair_statistics=False,
            purpose="EXISTING_SUPPORT_INVENTORY_NOT_RARE_ALLELE_ASSIGNMENT"))
    n_samples = len(people["samples"])
    person_union, person_pair, partner_chr = [0]*n_samples, [0]*n_samples, [0]*n_samples
    part_totals, seen = [], set()
    with tempfile.TemporaryDirectory(prefix="community-ibd-autosomes-", dir=guard.scratch) as temp:
        db = guard.connect(str(Path(temp) / "pairs.sqlite"))
        try:
            cols = ",".join(name + (" REAL" if name.endswith("cm_sum") else " INTEGER") for name in GEOMETRY)
            db.execute(f"CREATE TABLE pairs(a INTEGER,b INTEGER,chrom TEXT,{cols},PRIMARY KEY(a,b,chrom))")
            placeholders = ",".join("?" for _ in range(3+len(GEOMETRY)))
            for part in parts:
                mp = Path(part).resolve(strict=True)
                m = strict_json(mp)
                require(m["schema"] == PART_SCHEMA and m["status"] == "COMPLETE_REPORTED_IBD_CHROMOSOME_DESCRIPTION" and
                        m["source_contract_sha256"] == expected_sha and m["samples_sha256"] == people["sample_hash"] and
                        m["community_evidence_sha256"] == contract["community_evidence"]["sha256"] and
                        m["genome_build"] == contract["genome_build"], "Part lineage mismatch")
                chrom = m["chromosome"]
                require(chrom in contract["chromosomes"] and chrom not in seen, "Duplicate/unknown part chromosome")
                seen.add(chrom)
                ledger.add(dict(path=str(mp), sha256=sha256(mp)))
                calls = ledger.adjacent(mp, m, "pairs.private.tsv.gz")
                observed = Counter()
                for i, row in enumerate(table(calls)):
                    require(row["chromosome"] == chrom, "Pair chromosome differs")
                    a, b = people["order"][row["sample_a"]], people["order"][row["sample_b"]]
                    require(a < b, "Pair order differs")
                    values = [float(row[f]) if f.endswith("cm_sum") else int(row[f]) for f in GEOMETRY]
                    require(all(math.isfinite(v) and v >= 0 for v in values) and values[0] > 0, "Invalid pair geometry")
                    db.execute(f"INSERT INTO pairs VALUES({placeholders})", (a, b, chrom, *values))
                    observed.update(dict(zip(GEOMETRY, values)))
                    observed["pairs_with_reported_call"] += 1
                    if i % 10000 == 0:
                        guard.check()
                for f in (*GEOMETRY[:-1], "pairs_with_reported_call"):
                    require(observed[f] == m["totals"][f], "Part count mismatch")
                persons = ledger.adjacent(mp, m, "persons.private.tsv.gz")
                person_count = 0
                for i, row in enumerate(table(persons)):
                    require(i < n_samples and row["sample_id"] == people["samples"][i] and row["chromosome"] == chrom, "Person territory order differs")
                    values = [int(row[f]) for f in PERSON_FIELDS[2:]]
                    require(0 <= values[0] <= values[1] and 0 <= values[2] < n_samples, "Invalid person territory")
                    person_union[i] += values[0]
                    person_pair[i] += values[1]
                    partner_chr[i] += values[2]
                    person_count += 1
                require(person_count == n_samples, "Person territory incomplete")
                part_totals.append(dict(chromosome=chrom, **m["totals"]))
                db.commit()
            require(seen == set(contract["chromosomes"]), "Autosome set incomplete")
            possible = possible_support(people)
            guard.check()
            expressions = ",".join(f"SUM({f})" for f in GEOMETRY)
            cursor = db.execute(f"SELECT a,b,{expressions},COUNT(*) FROM pairs GROUP BY a,b ORDER BY a,b")
            def unique_pairs():
                for r in cursor:
                    yield r[0], r[1], r[2:-1], r[-1]
            observed, degree = describe_pairs(unique_pairs(), people, possible, guard, block_size)
            n_unique_pairs = db.execute("SELECT COUNT(*) FROM (SELECT a,b FROM pairs GROUP BY a,b)").fetchone()[0]
        finally:
            db.close()
    count = write_table(output / "community_ibd_metrics.private.tsv.gz", result_rows(people, possible, observed))
    require(count == len(possible), "Incomplete community comparison")
    def person_rows():
        for i, sample in enumerate(people["samples"]):
            yield dict(sample_id=sample, reported_any_partner_union_bp=person_union[i],
                reported_pair_union_bp_sum=person_pair[i], reported_unique_partners=degree[i],
                reported_partner_chromosome_count=partner_chr[i],
                reporting_state="REPORTED_CALL" if degree[i] else "NO_REPORTED_CALL_NOT_BIOLOGICAL_ABSENCE")
    write_table(output / "individual_territory.private.tsv.gz", person_rows())
    write_interpretation(output, people, possible, observed)
    summary = dict(status="COMPLETE_REPORTED_IBD_COMPARISON_NOT_COMPLETE_BIOLOGICAL_VALIDATION",
        n_samples=n_samples, n_partitions=len(people["partitions"]), chromosomes=contract["chromosomes"],
        n_unique_reported_pairs=n_unique_pairs, table_rows=count, chromosome_totals=part_totals,
        local_ancestry_support=ancestry_inventory, no_winner_selected=True, no_pvalues=True, no_confidence_intervals=True,
        criteria={"K01": "GLOBAL_ANCESTRY_COMPOSITION_DIAGNOSTIC_NOT_ADJUSTMENT", "K05": "REPORTED_IBD_WITHIN_BETWEEN_COMMUNITIES_DESCRIBED",
                  "K02": "NOT_IDENTIFIED", "K03": "NOT_EVALUATED", "K04": "RARE_ALLELE_PHASE_AND_DETECTABILITY_NOT_RESOLVED",
                  "K06": "NO_NEW_DEMOGRAPHIC_COMPARISON_HISTORICAL_ROH_NOT_TRANSFERRED", "K07": "NOT_ESTIMATED"},
        definitions={"reported_union_bp": "Inclusive physical union per pair per chromosome, summed over disjoint chromosomes",
            "hap_pair_union_bp_sum": "Sum of four separate haplotype-edge unions; not diploid IBD dosage",
            "person_union": "Union across all reported partners within chromosome, then sum chromosomes; each person's position counted once",
            "reported_pair_fraction": "Fraction of cohort pairs represented in a complete reported-call source, not sensitivity or biological IBD prevalence",
            "mean_given_call": "Mean reported union only among pairs with at least one reported call",
            "per_possible_pair": "Reported physical burden per cohort pair; reporting zeros, not biological absence or callable-territory rate",
            "no_reported_call": "No record in authenticated complete callset, with unknown detection probability",
            "Q_distance": "Squared Euclidean distance of four global ancestry fractions; descriptive confounding diagnostic, not matching",
            "community_to_other_assigned": "Incident-pair view duplicated across the two endpoint communities; never sum these rows as unique pairs",
            "within_between": "Global within/between/one-unassigned/both-unassigned scopes partition unordered cohort pairs",
            "component_relation": "Same/different/unknown PC-Relate connected component at the recorded threshold; not pedigree truth"},
        limitations=LIMITATIONS)
    save_json(output / "summary.json", summary)
    guard.check()
    return finish_manifest(output, ledger, RESULT_SCHEMA, summary["status"], n_samples=n_samples,
        n_partitions=len(people["partitions"]), chromosomes=contract["chromosomes"], samples_sha256=people["sample_hash"],
        source_contract_sha256=expected_sha, table_rows=count, n_unique_reported_pairs=n_unique_pairs,
        seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("chromosome", "aggregate"))
    for name in ("contract", "output-dir", "scratch-dir"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--contract-sha256", required=True)
    for name in ("max-memory-mb", "max-db-mb", "min-free-mb"):
        parser.add_argument("--"+name, required=True, type=int)
    parser.add_argument("--chromosome")
    parser.add_argument("--max-pair-records", type=int)
    parser.add_argument("--pair-block-size", type=int)
    parser.add_argument("--parts", nargs="+", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    guard = Guard(args.scratch_dir, args.max_memory_mb, args.max_db_mb, args.min_free_mb)
    if args.stage == "chromosome":
        require(args.chromosome and args.max_pair_records and not args.parts and args.pair_block_size is None, "Chromosome arguments required")
        result = chromosome_stage(args.contract, args.contract_sha256, args.chromosome,
                                  args.output_dir, guard, args.max_pair_records)
    else:
        require(args.parts and args.pair_block_size and not args.chromosome and args.max_pair_records is None, "Aggregate parts required")
        result = aggregate_stage(args.contract, args.contract_sha256, args.parts, args.output_dir, guard, args.pair_block_size)
    print(json.dumps({k: result[k] for k in ("status", "seconds")}))


if __name__ == "__main__":
    main()
