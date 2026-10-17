#!/usr/bin/env python3
"""Build target-excluded H affinities from authenticated rare-event archives.

H is a family of maximal chains of shared source-minor rare variants.  These
chains are unphased co-carriage summaries and are never labelled IBD.  The
chromosome stage streams each archived ``rare.*.npz`` once, evaluates every
requested catalogue geometry together, and preserves exact missingness.  The
reduce stage adds disjoint chromosome/interval products into FIT--FIT and
query--FIT views; it never constructs query--query edges or reads outcomes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import resource
import shutil
import tarfile

import numpy as np

import r02_m14_chains as m14
import r02_structure_h_core as core


REQUEST_SCHEMA = "r02_structure_h_affinity_request_v1"
CHROM_SCHEMA = "r02_structure_h_chromosome_v1"
RESULT_SCHEMA = "r02_structure_family_affinity_inputs_v1"
STAGE = "COMPLETE_H_AFFINITIES_NO_COMMUNITIES_OR_SELECTION"
CODE_FILES = ("r02_structure_h_affinities.py", "r02_structure_h_core.py",
              "r02_m14_chains.py", "rare_allele_sharing_painter.py")
HASH_RE = core.re.compile(r"^[0-9a-f]{64}$")
SCOPES = {"BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path_or_bytes):
    h = hashlib.sha256()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        h.update(path_or_bytes)
    else:
        with Path(path_or_bytes).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 2**20), b""):
                h.update(block)
    return h.hexdigest()


def code_hashes():
    root = Path(__file__).resolve().parent
    return {name: sha(root / name) for name in CODE_FILES}


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def resident_bytes():
    """Current resident set on Linux; ru_maxrss is a conservative fallback."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def require_memory(limit):
    require(resident_bytes() <= limit,
            "H resident-memory envelope exceeded before further pair-state allocation")


class OutputBudget:
    """Cumulative output and free-space guard checked around every large write."""
    def __init__(self, root, max_bytes, min_free_bytes):
        self.root = Path(root); self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes; self.used = 0

    def before(self, upper_bound):
        require(type(upper_bound) is int and upper_bound >= 0 and
                self.used + upper_bound <= self.max_bytes and
                shutil.disk_usage(self.root).free - upper_bound >= self.min_free_bytes,
                "H cumulative output/free-space envelope would be exceeded")

    def add(self, path):
        size = Path(path).stat().st_size; self.used += size
        require(self.used <= self.max_bytes and
                shutil.disk_usage(self.root).free >= self.min_free_bytes,
                "H cumulative output/free-space envelope exceeded")
        return size


def npz_upper_bound(arrays):
    """Conservative deflate/container bound from uncompressed array bytes."""
    raw = sum(np.asarray(value).nbytes for value in arrays.values())
    return raw + max(65536, raw // 100)


def pinned(desc, max_bytes):
    require(isinstance(desc, dict) and set(desc) in ({"path", "sha256"},
            {"path", "sha256", "bytes"}) and
            isinstance(desc["path"], str) and Path(desc["path"]).is_absolute() and
            HASH_RE.fullmatch(desc["sha256"]) is not None,
            "Exact absolute path/SHA256 pin required")
    path = Path(desc["path"])
    declared_bytes = desc.get("bytes")
    require((declared_bytes is None or
             (type(declared_bytes) is int and declared_bytes > 0 and
              path.is_file() and path.stat().st_size == declared_bytes)) and
            path.is_file() and path.stat().st_size <= max_bytes and
            sha(path) == desc["sha256"], "Pinned source changed or exceeds byte bound")
    return path


def read_table(path, fields):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(tuple(reader.fieldnames or ()) == tuple(fields),
                f"{Path(path).name} columns differ")
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows),
            f"Malformed {Path(path).name}")
    return rows


def validate_request(path, digest):
    require(HASH_RE.fullmatch(digest or "") is not None and sha(path) == digest,
            "H request SHA256 differs")
    value = load_json(path)
    expected = {"schema", "evidence_scope", "scientific_spec_sha256",
                "implementation_sha256", "context", "catalogue", "sources",
                "geometries", "territories", "pair_policy", "transformation",
                "resources", "preadmission"}
    require(set(value) == expected and value["schema"] == REQUEST_SCHEMA and
            value["implementation_sha256"] == code_hashes() and
            value["evidence_scope"] in {"DNABR_R02", "SYNTHETIC_INSTRUMENT_ONLY"} and
            HASH_RE.fullmatch(value["scientific_spec_sha256"] or "") is not None,
            "Exact admitted H request/code/evidence scope required")
    limits = value["resources"]
    limit_fields = {"max_archive_bytes", "max_metadata_bytes", "max_chunk_bytes",
                    "max_candidate_sites", "max_pair_site_updates",
                    "max_active_pair_states", "max_output_bytes",
                    "max_metric_store_bytes", "max_memory_bytes", "min_free_bytes"}
    require(set(limits) == limit_fields and all(type(limits[k]) is int and limits[k] > 0
            for k in limit_fields), "Explicit positive H resource limits required")
    context_fields = {"materialize", "persons", "blocks"}
    require(set(value["context"]) == context_fields,
            "Exact H context pins required")
    materialize_path = pinned(value["context"]["materialize"], limits["max_metadata_bytes"])
    persons_path = pinned(value["context"]["persons"], limits["max_metadata_bytes"])
    blocks_path = pinned(value["context"]["blocks"], limits["max_metadata_bytes"])
    materialize = load_json(materialize_path)
    require(materialize.get("schema") == "r02_structure_materialize_v1" and
            materialize.get("roles_state") == "EXPLICITLY_FROZEN" and
            materialize.get("rare_allele_convention") == "SOURCE_MINOR_M02_1" and
            materialize.get("exclusion_scope") == "BLOCK_AND_LD_MARGIN" and
            materialize.get("additional_exclusion_scopes") == ["WHOLE_CHROMOSOME"],
            "Frozen source-minor context and both exclusion scopes required")
    persons = read_table(persons_path, ("sample_id", "component", "role"))
    require(persons and len({row["sample_id"] for row in persons}) == len(persons) and
            all(row["role"] in {"FIT", "SELECT", "EVALUATE"} for row in persons),
            "Unique persons with frozen roles required")
    roles = {name: [i for i, row in enumerate(persons) if row["role"] == name]
             for name in ("FIT", "SELECT", "EVALUATE")}
    if value["evidence_scope"] == "DNABR_R02":
        require(len(persons) == 2619 and {k: len(v) for k, v in roles.items()} ==
                {"FIT": 1460, "SELECT": 476, "EVALUATE": 683},
                "R02 H cohort/roles differ")
    blocks = read_table(blocks_path, ("block_id", "chromosome", "start_bp", "end_bp",
                                     "excluded_start_bp", "excluded_end_bp"))
    block_map = {}
    for row in blocks:
        require(row["block_id"] not in block_map and row["chromosome"] in materialize["sources"],
                "Unique H block on a source chromosome required")
        parsed = dict(row)
        for field in ("start_bp", "end_bp", "excluded_start_bp", "excluded_end_bp"):
            parsed[field] = int(parsed[field])
        length = int(materialize["sources"][row["chromosome"]]["length_bp"])
        require(1 <= parsed["start_bp"] <= parsed["end_bp"] <= length and
                1 <= parsed["excluded_start_bp"] <= parsed["excluded_end_bp"] <= length and
                parsed["excluded_start_bp"] <= parsed["start_bp"] <= parsed["end_bp"] <= parsed["excluded_end_bp"],
                "Target/exclusion block geometry differs")
        block_map[row["block_id"]] = parsed
    catalogue_path = pinned(value["catalogue"], limits["max_metadata_bytes"])
    catalogue = load_json(catalogue_path)
    require(catalogue.get("schema") == "m165_family_catalogue_v1" and
            catalogue.get("geometry_count") == 74 and len(catalogue.get("geometries", [])) == 74,
            "Exact 74-geometry H catalogue required")
    admitted = {row["config_id"]: core.catalogue_geometry(row)
                for row in catalogue["geometries"]}
    require(len(admitted) == 74, "Duplicate H catalogue geometry")
    geometries = []
    for row in value["geometries"]:
        require(set(row) == {"geometry", "justification"},
                "Geometry and its justification required")
        geometry = core.validate_geometry(row["geometry"])
        core.validate_justification(row["justification"], geometry["config_id"])
        require(admitted.get(geometry["config_id"]) == geometry,
                "Requested H geometry is absent or differs from the 74-geometry catalogue")
        geometries.append(geometry)
    require(geometries and len({g["config_id"] for g in geometries}) == len(geometries),
            "Non-empty unique justified H geometry request required")
    policy = value["pair_policy"]
    require(set(policy) == {"fit_component_policy", "query_roles", "query_query",
                            "justification"} and
            policy["fit_component_policy"] in {"INCLUDE", "EXCLUDE"} and
            isinstance(policy["query_roles"], list) and policy["query_roles"] and
            len(policy["query_roles"]) == len(set(policy["query_roles"])) and
            set(policy["query_roles"]) <= {"SELECT", "EVALUATE"} and
            policy["query_query"] == "EXCLUDE",
            "Explicit FIT/query pair policy required; query--query is forbidden")
    core.validate_justification(policy["justification"], "pair policy")
    transform = value["transformation"]
    require(set(transform) == {"name", "scale_bp", "justification"} and
            transform["name"] == "LOG1P_TOTAL_SHARED_BP" and
            type(transform["scale_bp"]) is int and transform["scale_bp"] > 0,
            "Explicit historical log1p H transform and positive bp scale required")
    core.validate_justification(transform["justification"], "H transform")
    territories = []
    for row in value["territories"]:
        require(set(row) == {"scope", "block_id", "justification"} and
                row["scope"] in SCOPES and row["block_id"] in block_map and
                isinstance(row["justification"], str) and row["justification"].strip(),
                "Explicit known justified H territory required")
        territories.append({"scope": row["scope"], "block_id": row["block_id"]})
    require(territories and len({(x["scope"], x["block_id"]) for x in territories}) == len(territories),
            "Non-empty unique H territories required")
    pre = value["preadmission"]
    require(set(pre) == {"status", "finite_hypothesis_count", "view_count",
                         "evaluate_opened", "claim"} and
            pre["status"] == "PRE_ADMITTED_FINITE_H_FAMILY" and
            pre["finite_hypothesis_count"] == len(geometries) and
            pre["view_count"] == len(territories) and
            pre["evaluate_opened"] is False and
            pre["claim"] == "TECHNICAL_MATERIALIZATION_NOT_BIOLOGICAL_RESULT",
            "Exact finite-family H preadmission required")
    require(set(value["sources"]) == set(materialize["sources"]),
            "Every autosomal rare archive must be pinned")
    for chrom, source in value["sources"].items():
        require(set(source) == {"archive", "manifest_snapshot"},
                "Exact archive/snapshot pins required")
        snapshot_path = pinned(source["manifest_snapshot"], limits["max_metadata_bytes"])
        snapshot = load_json(snapshot_path)
        require(snapshot.get("schema") == "r02_genomic_event_chromosome_v1" and
                snapshot.get("stage") == "COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING" and
                snapshot.get("chromosome") == chrom and snapshot.get("n_people") == len(persons) and
                snapshot.get("sample_ids_sha256") == sample_hash(persons) and
                snapshot.get("missing_policy") == "OBSERVED_MASK_AND_UINT8_255_SENTINEL" and
                snapshot.get("rare_dosage_feature_index") == 0 and
                snapshot.get("training_allowed") is False,
                "Rare event snapshot chromosome/cohort/minor/missing contract differs")
        require(source["archive"]["bytes"] <= limits["max_archive_bytes"] and
                HASH_RE.fullmatch(source["archive"]["sha256"] or "") is not None and
                Path(source["archive"]["path"]).is_absolute(),
                "Explicit bounded rare archive pin required")
    require(code_hashes() == value["implementation_sha256"] and sha(path) == digest,
            "H request/code changed during validation")
    return value, materialize, persons, roles, block_map, geometries, territories


def sample_hash(persons):
    return hashlib.sha256("".join(row["sample_id"] + "\n" for row in persons).encode()).hexdigest()


def interval_sets(chrom, length, block_map, territories):
    result = {"FULL": [[1, length]]}
    for item in territories:
        if item["scope"] != "BLOCK_AND_LD_MARGIN":
            continue
        block = block_map[item["block_id"]]
        if block["chromosome"] != chrom:
            continue
        intervals = []
        if block["excluded_start_bp"] > 1:
            intervals.append([1, block["excluded_start_bp"] - 1])
        if block["excluded_end_bp"] < length:
            intervals.append([block["excluded_end_bp"] + 1, length])
        result[item["block_id"]] = core.validate_allowed_intervals(intervals, length)
    return result


def interval_index(intervals, pos):
    for index, (start, end) in enumerate(intervals):
        if start <= pos <= end:
            return index
    return None


class MetricStore:
    """Disk-backed dense accumulators; exported as sparse positive arrays."""
    DTYPES = {"total_shared_bp": np.int64, "n_chains": np.int32,
              "n_shared_sites": np.int32, "max_chain_bp": np.int32}

    def __init__(self, directory, interval_id, geometries, capacities, max_bytes):
        self.directory, self.interval_id = Path(directory), interval_id
        self.configs = [g["config_id"] for g in geometries]
        self.config_index = {name: i for i, name in enumerate(self.configs)}
        self.arrays = {}; estimated = 0
        for kind, capacity in capacities.items():
            for metric, dtype in self.DTYPES.items():
                estimated += len(self.configs) * capacity * np.dtype(dtype).itemsize
                require(estimated <= max_bytes, "H disk-backed metric envelope exceeded")
                path = self.directory/f".{interval_id}.{kind}.{metric}.mmap"
                self.arrays[(kind, metric)] = np.memmap(
                    path, mode="w+", dtype=dtype, shape=(len(self.configs), capacity))
        self.estimated_bytes = estimated

    def add(self, config_id, kind, pair, length, count):
        row = self.config_index[config_id]
        total = self.arrays[(kind, "total_shared_bp")]
        chains = self.arrays[(kind, "n_chains")]
        shared = self.arrays[(kind, "n_shared_sites")]
        maximum = self.arrays[(kind, "max_chain_bp")]
        require(total[row, pair] <= np.iinfo(np.int64).max - length and
                chains[row, pair] < np.iinfo(np.int32).max - 1 and
                shared[row, pair] <= np.iinfo(np.int32).max - count,
                "H chain metric integer capacity exceeded")
        total[row, pair] += length; chains[row, pair] += 1; shared[row, pair] += count
        if length > maximum[row, pair]: maximum[row, pair] = length

    def export(self, output, budget):
        result = {}
        prefixes = {"FIT_FIT": "fit", "QUERY_FIT": "query_fit"}
        for row, config in enumerate(self.configs):
            arrays = {}
            for kind, prefix in prefixes.items():
                total = np.asarray(self.arrays[(kind, "total_shared_bp")][row])
                indices = np.flatnonzero(total > 0).astype(np.int64)
                arrays[prefix + "_pair_index"] = indices
                for metric in self.DTYPES:
                    arrays[prefix + "_" + metric] = np.asarray(
                        self.arrays[(kind, metric)][row, indices]).copy()
            name = f"{self.interval_id}.{config}.private.npz"
            budget.before(npz_upper_bound(arrays))
            np.savez_compressed(Path(output)/name, **arrays)
            budget.add(Path(output)/name)
            result[config] = {"path": name, "sha256": sha(Path(output)/name),
                              "positive_fit_pairs": len(arrays["fit_pair_index"]),
                              "positive_query_fit_pairs": len(arrays["query_fit_pair_index"])}
        for array in self.arrays.values(): array.flush()
        for array in self.arrays.values():
            path = Path(array.filename); del array
            path.unlink()
        self.arrays.clear()
        return result


class ChainSet:
    """Sparse M14 states writing additive metrics into one bounded store."""
    def __init__(self, geometries, max_states, store):
        self.by_gap = defaultdict(list)
        for geometry in geometries:
            self.by_gap[geometry["max_gap_bp"]].append(geometry)
        self.states = {(gap, kind): {} for gap in self.by_gap for kind in ("FIT_FIT", "QUERY_FIT")}
        self.store = store
        self.max_states = max_states
        self.active_states = 0

    def _record(self, gap, kind, pair, chain):
        _start, _end, length, count = chain
        for geometry in self.by_gap[gap]:
            if length < geometry["min_length_bp"] or count < geometry["min_shared_effective"]:
                continue
            self.store.add(geometry["config_id"], kind, pair, length, count)

    def push(self, kind, pair, pos):
        for gap in self.by_gap:
            state = self.states[(gap, kind)]
            new_pair = pair not in state
            closed = m14.update_streaming_pair_chain(state, pair, pos, gap)
            if new_pair:
                self.active_states += 1
            if closed is not None:
                self._record(gap, kind, pair, closed)
        require(self.active_states <= self.max_states,
                "Active M14 pair-state bound exceeded")

    def boundary(self):
        for (gap, kind), states in self.states.items():
            for pair, chain in m14.finish_streaming_pair_chains(states):
                self._record(gap, kind, pair, chain)
        self.active_states = 0


class IntervalAccumulator:
    def __init__(self, interval_id, intervals, geometries, n_people, n_fit, n_query,
                 max_states, store_directory, max_store_bytes):
        self.id, self.intervals = interval_id, intervals
        self.store = MetricStore(store_directory, interval_id, geometries,
            {"FIT_FIT": core.triangle_size(n_fit), "QUERY_FIT": n_query*n_fit}, max_store_bytes)
        self.chain = ChainSet(geometries, max_states, self.store)
        self.candidate_sites = 0
        self.missing_people = np.zeros(n_people, dtype=np.int64)
        self.fit_both_missing = np.zeros(core.triangle_size(n_fit), dtype=np.int64)
        self.query_fit_both_missing = np.zeros(n_query * n_fit, dtype=np.int64)
        self.region = None

    def transition(self, pos):
        region = interval_index(self.intervals, pos)
        if self.region is not None and region != self.region:
            self.chain.boundary()
        self.region = region
        return region is not None


def validate_chunk(data, expected_people, descriptor):
    expected = {"dosage", "observed", "position_bp", "position_cm", "ref", "alt",
                "counted_allele", "allele_count", "allele_number", "frequency"}
    require(set(data.files) == expected, "Rare event chunk arrays differ")
    nsites = descriptor["n_sites"]
    require(data["dosage"].shape == data["observed"].shape == (expected_people, nsites) and
            data["dosage"].dtype == np.uint8 and data["observed"].dtype == np.bool_ and
            data["position_bp"].shape == (nsites,) and
            np.all(np.diff(data["position_bp"]) > 0) and
            int(data["position_bp"][0]) == descriptor["start_bp"] and
            int(data["position_bp"][-1]) == descriptor["end_bp"] and
            np.array_equal(data["dosage"] == 255, ~data["observed"]) and
            np.all(data["dosage"][data["observed"]] <= 2) and
            np.all(np.isin(data["counted_allele"], (0, 1))),
            "Rare source-minor dosage/mask/position contract differs")


def archive_members(archive, snapshot, max_chunk_bytes):
    expected = snapshot["channels"]["rare"]
    with tarfile.open(archive, "r:gz") as tar:
        members = defaultdict(list)
        for member in tar.getmembers():
            if member.isfile():
                members[Path(member.name).name].append(member)
        require(set(row["path"] for row in expected) <= set(members),
                "Rare archive lacks snapshot chunks")
        for row in expected:
            matches = members[row["path"]]
            require(len(matches) == 1 and matches[0].size == row["bytes"] and
                    matches[0].size <= max_chunk_bytes,
                    "Ambiguous or oversized rare archive member")
            stream = tar.extractfile(matches[0])
            require(stream is not None, "Cannot read rare archive member")
            payload = stream.read(max_chunk_bytes + 1)
            require(len(payload) == row["bytes"] and sha(payload) == row["sha256"],
                    "Rare archive member changed")
            yield row, payload


def update_site(acc, pos, dosage, observed, fit_rank, query_rank, fit_components,
                component_policy, limits, counters):
    if not acc.transition(pos):
        return
    acc.candidate_sites += 1
    missing = np.flatnonzero(~observed).tolist()
    if missing:
        acc.missing_people[missing] += 1
        missing_fit = sorted(fit_rank[i] for i in missing if i in fit_rank)
        missing_query = sorted(query_rank[i] for i in missing if i in query_rank)
        for left, right in itertools.combinations(missing_fit, 2):
            if component_policy == "INCLUDE" or fit_components[left] != fit_components[right]:
                acc.fit_both_missing[core.triangle_index(left, right, len(fit_rank))] += 1
        for query in missing_query:
            base = query * len(fit_rank)
            for fit in missing_fit:
                acc.query_fit_both_missing[base + fit] += 1
    carriers = np.flatnonzero(observed & (dosage > 0)).tolist()
    carrier_fit = sorted(fit_rank[i] for i in carriers if i in fit_rank)
    carrier_query = sorted(query_rank[i] for i in carriers if i in query_rank)
    for left, right in itertools.combinations(carrier_fit, 2):
        if component_policy == "EXCLUDE" and fit_components[left] == fit_components[right]:
            continue
        acc.chain.push("FIT_FIT", core.triangle_index(left, right, len(fit_rank)), pos)
        counters[0] += 1
        if counters[0] >= counters[1]:
            require_memory(limits["max_memory_bytes"]); counters[1] += 4096
    for query in carrier_query:
        base = query * len(fit_rank)
        for fit in carrier_fit:
            acc.chain.push("QUERY_FIT", base + fit, pos)
            counters[0] += 1
            if counters[0] >= counters[1]:
                require_memory(limits["max_memory_bytes"]); counters[1] += 4096
    require(counters[0] <= limits["max_pair_site_updates"],
            "H pair-site update bound exceeded")


def write_chromosome(request_path, request_sha256, chromosome, output):
    request, materialize, persons, roles, blocks, geometries, territories = validate_request(request_path, request_sha256)
    require(chromosome in request["sources"], "Requested H chromosome absent")
    limits = request["resources"]; source = request["sources"][chromosome]
    archive = pinned(source["archive"], limits["max_archive_bytes"])
    snapshot_path = pinned(source["manifest_snapshot"], limits["max_metadata_bytes"])
    snapshot = load_json(snapshot_path); length = int(materialize["sources"][chromosome]["length_bp"])
    sets = interval_sets(chromosome, length, blocks, territories)
    fit = roles["FIT"]
    query = sorted(i for role in request["pair_policy"]["query_roles"] for i in roles[role])
    fit_rank = {person: i for i, person in enumerate(fit)}
    query_rank = {person: i for i, person in enumerate(query)}
    fit_components = [persons[i]["component"] for i in fit]
    output = Path(output)
    require(not output.exists() and output.parent.is_dir() and
            shutil.disk_usage(output.parent).free >= limits["min_free_bytes"],
            "New H chromosome output and disk reserve required")
    output.mkdir(mode=0o700)
    budget = OutputBudget(output, limits["max_output_bytes"], limits["min_free_bytes"])
    per_set_states = max(1, limits["max_active_pair_states"] // max(1, len(sets)))
    per_set_store = max(1, limits["max_metric_store_bytes"] // max(1, len(sets)))
    accumulators = {key: IntervalAccumulator(key, intervals, geometries, len(persons),
                    len(fit), len(query), per_set_states, output, per_set_store)
                    for key, intervals in sets.items()}
    counters = [0, 4096]; previous = 0; chunks = sites = input_bytes = 0
    for descriptor, payload in archive_members(archive, snapshot, limits["max_chunk_bytes"]):
        require_memory(limits["max_memory_bytes"])
        input_bytes += len(payload); chunks += 1
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            validate_chunk(data, len(persons), descriptor)
            for column, value in enumerate(data["position_bp"]):
                pos = int(value)
                require(pos > previous, "Rare archive chunks are not globally sorted")
                previous = pos; sites += 1
                require(sites <= limits["max_candidate_sites"], "H candidate-site bound exceeded")
                for accumulator in accumulators.values():
                    update_site(accumulator, pos, data["dosage"][:, column],
                                data["observed"][:, column], fit_rank, query_rank,
                                fit_components, request["pair_policy"]["fit_component_policy"],
                                limits, counters)
    require(sites == snapshot["audit"]["rare"]["retained_sites"],
            "Rare archive site total differs from snapshot")
    support = {}
    for interval_id, accumulator in accumulators.items():
        accumulator.chain.boundary()
        name = f"support.{interval_id}.private.npz"
        support_arrays = {"candidate_sites": np.int64(accumulator.candidate_sites),
            "missing_people": accumulator.missing_people,
            "fit_both_missing": accumulator.fit_both_missing,
            "query_fit_both_missing": accumulator.query_fit_both_missing}
        budget.before(npz_upper_bound(support_arrays))
        np.savez_compressed(output / name, **support_arrays); budget.add(output/name)
        support[interval_id] = {"path": name, "sha256": sha(output/name),
                                "intervals": accumulator.intervals,
                                "candidate_sites": accumulator.candidate_sites,
                                "geometries": accumulator.store.export(output, budget),
                                "metric_store_work_bytes": accumulator.store.estimated_bytes}
    require(sum(p.stat().st_size for p in output.iterdir()) <= limits["max_output_bytes"],
            "H chromosome output bound exceeded")
    result = {"schema": CHROM_SCHEMA, "stage": "CHROMOSOME_H_CHAINS_NO_GRAPH",
        "chromosome": chromosome, "request_sha256": request_sha256,
        "implementation_sha256": code_hashes(), "archive_sha256": source["archive"]["sha256"],
        "manifest_snapshot_sha256": source["manifest_snapshot"]["sha256"],
        "rare_chunks": chunks, "rare_sites": sites, "rare_chunk_bytes": input_bytes,
        "pair_site_updates": counters[0], "peak_rss_bytes": int(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss) * 1024, "interval_products": support,
        "fit_indices": fit, "query_indices": query, "query_roles": request["pair_policy"]["query_roles"],
        "outputs_sha256": {p.name: sha(p) for p in output.iterdir()},
        "source_minor_preserved": True, "missingness_preserved": True,
        "query_query_computed": False, "outcomes_read": False, "biological_validation": False}
    budget.before(1048576); write_json(output / "manifest.json", result); budget.add(output/"manifest.json")
    require(sha(request_path) == request_sha256 and sha(archive) == source["archive"]["sha256"],
            "H request/archive changed during chromosome scan")
    return result


def load_chromosome(directory, request, chromosome):
    root = Path(directory); manifest = load_json(root / "manifest.json")
    require(manifest["schema"] == CHROM_SCHEMA and manifest["stage"] == "CHROMOSOME_H_CHAINS_NO_GRAPH" and
            manifest["chromosome"] == chromosome and manifest["request_sha256"] == request["request_sha256"] and
            manifest["implementation_sha256"] == code_hashes() and
            manifest["query_query_computed"] is False and manifest["outcomes_read"] is False,
            "H chromosome product identity differs")
    for name, digest in manifest["outputs_sha256"].items():
        require((root/name).is_file() and sha(root/name) == digest,
                "H chromosome output changed")
    return root, manifest


def add_support(total, path):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {"candidate_sites", "missing_people", "fit_both_missing",
                                    "query_fit_both_missing"}, "H support arrays differ")
        arrays = {name: data[name].astype(np.int64, copy=True) for name in data.files}
    if total is None:
        return arrays
    for name in total:
        total[name] += arrays[name]
    return total


def pair_rosters(persons, fit, query, policy):
    fit_components = [persons[i]["component"] for i in fit]
    for left in range(len(fit)):
        for right in range(left + 1, len(fit)):
            if policy == "EXCLUDE" and fit_components[left] == fit_components[right]:
                continue
            yield "FIT_FIT", core.triangle_index(left, right, len(fit)), fit[left], fit[right], left, right
    for qrank, qperson in enumerate(query):
        for frank, fperson in enumerate(fit):
            yield "QUERY_FIT", qrank * len(fit) + frank, qperson, fperson, qrank, frank


class AuthenticatedHCompactView:
    """Lazy H provider for the shared affinity graph/projection consumer.

    Positive chain summaries are sparse.  A missing sparse row is a measured
    zero only when the independently stored joint-callable denominator is
    positive; otherwise the affinity is unknown.  This is the sole expansion
    point for the implicit 2.75M-pair roster.
    """
    def __init__(self, root, manifest, manifest_sha256, *, evidence_scope,
                 max_input_bytes, max_metadata_bytes, max_memory_bytes):
        self.root = Path(root).resolve(); self.manifest_path = Path(manifest).resolve()
        require(self.root.is_dir() and self.manifest_path.is_file() and
                self.manifest_path.is_relative_to(self.root) and
                sha(self.manifest_path) == manifest_sha256,
                "Pinned H compact manifest inside root required")
        self.upstream = metadata = load_json(self.manifest_path)
        require(metadata["schema"] == RESULT_SCHEMA and metadata["stage"] == STAGE and
                metadata["provider"] == "H_COMPACT_V1" and metadata["family"] == "H" and
                metadata["missing_policy"] == "UNKNOWN_NOT_ZERO" and
                metadata["pair_policy"]["query_query"] == "EXCLUDE" and
                metadata["outcomes_read"] is False and metadata["query_query_computed"] is False and
                evidence_scope in {"DNABR_R02", "SYNTHETIC_INSTRUMENT_ONLY"},
                "Reviewed compact H semantics required")
        paths = {}
        for key in ("persons", "support", "compact_affinities", "exclusion"):
            desc = metadata[key]
            require(set(desc) >= {"path", "sha256"} and HASH_RE.fullmatch(desc["sha256"]) is not None,
                    "Exact compact H artifact pin required")
            path = (self.root/desc["path"]).resolve()
            require(path.is_relative_to(self.root) and path.is_file() and sha(path) == desc["sha256"],
                    "Compact H artifact changed or escaped root")
            paths[key] = path
        require(self.manifest_path.stat().st_size + paths["persons"].stat().st_size +
                paths["exclusion"].stat().st_size <= max_metadata_bytes and
                sum(p.stat().st_size for p in paths.values()) <= max_input_bytes,
                "Compact H input/metadata byte bound exceeded")
        self.paths = paths; self.pins = {key: metadata[key]["sha256"] for key in paths}
        self.persons = read_table(paths["persons"], ("sample_id", "component", "role"))
        self.roles = {role: [i for i, row in enumerate(self.persons) if row["role"] == role]
                      for role in ("FIT", "SELECT", "EVALUATE")}
        require(sha(paths["persons"]) == metadata["persons_content_sha256"] and
                {k: len(v) for k, v in self.roles.items()} == metadata["role_counts"],
                "Compact H roster/roles changed")
        self.fit_indices = self.roles["FIT"]
        self.query_indices = sorted(i for role in metadata["pair_policy"]["query_roles"]
                                    for i in self.roles[role])
        with np.load(paths["support"], allow_pickle=False) as data:
            require(set(data.files) == {"candidate_sites", "missing_people", "fit_both_missing",
                                        "query_fit_both_missing"}, "Compact H support arrays differ")
            self.support_arrays = {name: data[name].astype(np.int64, copy=True) for name in data.files}
        with np.load(paths["compact_affinities"], allow_pickle=False) as data:
            expected = {prefix + suffix for prefix in ("fit", "query_fit") for suffix in
                        ("_pair_index", "_total_shared_bp", "_n_chains",
                         "_n_shared_sites", "_max_chain_bp")}
            require(set(data.files) == expected, "Compact H geometry arrays differ")
            self.affinity_arrays = {name: data[name].astype(np.int64, copy=True) for name in data.files}
        decoded = sum(a.nbytes for a in self.support_arrays.values()) + \
                  sum(a.nbytes for a in self.affinity_arrays.values())
        require(3*decoded <= max_memory_bytes, "Compact H decode workspace bound exceeded")
        nfit, nquery = len(self.fit_indices), len(self.query_indices)
        require(self.support_arrays["missing_people"].shape == (len(self.persons),) and
                self.support_arrays["fit_both_missing"].shape == (core.triangle_size(nfit),) and
                self.support_arrays["query_fit_both_missing"].shape == (nquery*nfit,) and
                self.support_arrays["candidate_sites"].shape == () and
                int(self.support_arrays["candidate_sites"]) > 0,
                "Compact H support shape/candidate count differs")
        for prefix, capacity in (("fit", core.triangle_size(nfit)), ("query_fit", nquery*nfit)):
            indices = self.affinity_arrays[prefix + "_pair_index"]
            require(indices.ndim == 1 and np.all(indices >= 0) and np.all(indices < capacity) and
                    np.all(np.diff(indices) > 0), "Compact H pair indices are not unique/sorted/in range")
            for suffix in ("_total_shared_bp", "_n_chains", "_n_shared_sites", "_max_chain_bp"):
                values = self.affinity_arrays[prefix + suffix]
                require(values.shape == indices.shape and np.all(values > 0),
                        "Sparse H chain metrics must be positive and aligned")
        self.source_sha256 = manifest_sha256
        self.verify()

    def _metrics(self, prefix, pair):
        indices = self.affinity_arrays[prefix + "_pair_index"]
        at = int(np.searchsorted(indices, pair))
        if at == len(indices) or int(indices[at]) != pair:
            return (0, 0, 0, 0)
        return tuple(int(self.affinity_arrays[prefix + suffix][at]) for suffix in
                     ("_total_shared_bp", "_n_chains", "_n_shared_sites", "_max_chain_bp"))

    def _joint(self, left_person, right_person, both):
        support = self.support_arrays; candidate = int(support["candidate_sites"])
        value = candidate - int(support["missing_people"][left_person]) - \
                int(support["missing_people"][right_person]) + int(both)
        require(0 <= value <= candidate, "Compact H joint-callable denominator invalid")
        return value

    def fit_edges(self):
        policy = self.upstream["pair_policy"]["fit_component_policy"]
        components = [self.persons[i]["component"] for i in self.fit_indices]
        for left in range(len(self.fit_indices)):
            for right in range(left + 1, len(self.fit_indices)):
                if policy == "EXCLUDE" and components[left] == components[right]:
                    continue
                pair = core.triangle_index(left, right, len(self.fit_indices))
                joint = self._joint(self.fit_indices[left], self.fit_indices[right],
                                    self.support_arrays["fit_both_missing"][pair])
                if joint == 0:
                    continue
                total_bp = self._metrics("fit", pair)[0]
                yield self.fit_indices[left], self.fit_indices[right], \
                      core.log1p_total_bp(total_bp, self.upstream["transformation"]["scale_bp"])
        self.verify()

    def query_rows(self, *, roles=("SELECT", "EVALUATE")):
        require(isinstance(roles, tuple) and roles and set(roles) <= set(self.upstream["pair_policy"]["query_roles"]),
                "Explicit admitted H query roles required")
        fit = self.fit_indices; candidate = int(self.support_arrays["candidate_sites"])
        for qrank, person in enumerate(self.query_indices):
            observations = {}; observed = masked = total_bp = joint_sum = chain_sum = 0
            for frank, reference in enumerate(fit):
                pair = qrank*len(fit) + frank
                joint = self._joint(person, reference,
                                    self.support_arrays["query_fit_both_missing"][pair])
                joint_sum += joint
                if joint == 0:
                    observations[reference] = None; masked += 1; continue
                metrics = self._metrics("query_fit", pair); total_bp += metrics[0]; chain_sum += metrics[1]
                observations[reference] = core.log1p_total_bp(
                    metrics[0], self.upstream["transformation"]["scale_bp"])
                observed += 1
            if self.persons[person]["role"] in roles:
                yield person, observations, {"observed_fit_count": observed,
                    "total_fit_count": len(fit), "masked_fit_count": masked,
                    "total_shared_bp_sum": total_bp, "chain_count_sum": chain_sum,
                    "joint_callable_sites_sum": joint_sum, "candidate_sites": candidate,
                    "denominator_scope": "SUM_OVER_QUERY_TO_FIT_EDGES_NOT_UNIQUE_LOCI",
                    "target_outcomes_read": False}
        self.verify()

    def graph_semantics(self):
        return {key: self.upstream[key] for key in
                ("family", "measure", "units", "transformation", "range",
                 "missing_policy", "pair_policy", "territory", "geometry", "normalization")}

    def verify(self):
        require(sha(self.manifest_path) == self.source_sha256 and
                all(sha(self.paths[key]) == digest for key, digest in self.pins.items()),
                "Compact H inputs changed after admission")


def effective_territories(territories, blocks):
    """Deduplicate identical whole-chromosome inputs while retaining 88 aliases."""
    canonical, aliases = {}, []
    for item in territories:
        block = blocks[item["block_id"]]
        key = (item["scope"], item["block_id"] if item["scope"] == "BLOCK_AND_LD_MARGIN"
               else block["chromosome"])
        canonical.setdefault(key, {"scope": item["scope"], "block_id": item["block_id"],
                                   "chromosome": block["chromosome"]})
        aliases.append({"scope": item["scope"], "block_id": item["block_id"],
                        "canonical_key": "__".join(key)})
    return canonical, aliases


def compact_arrays(summaries):
    arrays = {}
    for kind, prefix in (("FIT_FIT", "fit"), ("QUERY_FIT", "query_fit")):
        keys = sorted(summaries[kind])
        arrays[prefix + "_pair_index"] = np.asarray(keys, dtype=np.int64)
        values = np.asarray([summaries[kind][key] for key in keys], dtype=np.int64)
        if not keys:
            values = np.empty((0, 4), dtype=np.int64)
        arrays[prefix + "_total_shared_bp"] = values[:, 0]
        arrays[prefix + "_n_chains"] = values[:, 1]
        arrays[prefix + "_n_shared_sites"] = values[:, 2]
        arrays[prefix + "_max_chain_bp"] = values[:, 3]
    return arrays


def reduce_output_upper_bound(products, value, roles, blocks, geometries, territories):
    """Fail-early bound using sparse counts, pair capacities and raw NPZ bytes."""
    canonical, _aliases = effective_territories(territories, blocks)
    nfit = len(roles["FIT"])
    nquery = sum(len(roles[role]) for role in value["pair_policy"]["query_roles"])
    fit_capacity = core.triangle_size(nfit); query_capacity = nquery * nfit
    support_raw = 8 * (1 + len(sum(roles.values(), [])) + fit_capacity + query_capacity)
    total = Path(value["context"]["persons"]["path"]).stat().st_size + 16 * 2**20
    for _key, territory in canonical.items():
        chosen = {chrom: (territory["block_id"] if chrom == territory["chromosome"] and
                  territory["scope"] == "BLOCK_AND_LD_MARGIN" else
                  None if chrom == territory["chromosome"] else "FULL")
                  for chrom in value["sources"]}
        total += support_raw + max(65536, support_raw // 100) + 65536
        for geometry in geometries:
            fit_positive = query_positive = 0
            for chrom, interval_id in chosen.items():
                if interval_id is None:
                    continue
                _root, manifest = products[chrom]
                desc = manifest["interval_products"][interval_id]["geometries"][geometry["config_id"]]
                fit_positive += desc["positive_fit_pairs"]
                query_positive += desc["positive_query_fit_pairs"]
            pairs = min(fit_capacity, fit_positive) + min(query_capacity, query_positive)
            raw = pairs * 40
            total += raw + max(65536, raw // 100) + 65536
    return total


def reduce_views(request_path, request_sha256, chromosome_dirs, output):
    value, materialize, persons, roles, blocks, geometries, territories = validate_request(request_path, request_sha256)
    request = dict(value, request_sha256=request_sha256)
    require(set(chromosome_dirs) == set(value["sources"]), "All and only requested H chromosomes required")
    products = {chrom: load_chromosome(path, request, chrom) for chrom, path in chromosome_dirs.items()}
    limits = value["resources"]; fit = roles["FIT"]
    query = sorted(i for role in value["pair_policy"]["query_roles"] for i in roles[role])
    output = Path(output)
    canonical, aliases = effective_territories(territories, blocks)
    estimated_output = reduce_output_upper_bound(
        products, value, roles, blocks, geometries, territories)
    require(not output.exists() and output.parent.is_dir() and
            estimated_output <= limits["max_output_bytes"] and
            shutil.disk_usage(output.parent).free - estimated_output >= limits["min_free_bytes"],
            "New H view output and disk reserve required")
    output.mkdir(mode=0o700)
    budget = OutputBudget(output, limits["max_output_bytes"], limits["min_free_bytes"])
    budget.before(Path(value["context"]["persons"]["path"]).stat().st_size)
    shutil.copyfile(value["context"]["persons"]["path"], output/"persons.tsv")
    budget.add(output/"persons.tsv")
    registry = []
    for key, territory in canonical.items():
        target_chrom = territory["chromosome"]
        chosen = {chrom: (territory["block_id"] if chrom == target_chrom and
                  territory["scope"] == "BLOCK_AND_LD_MARGIN" else
                  None if chrom == target_chrom else "FULL") for chrom in value["sources"]}
        canonical_id = "__".join(key); view = output/canonical_id; view.mkdir(mode=0o700)
        support = None; included = []; selected_products = []
        for chrom in sorted(value["sources"], key=int):
            interval_id = chosen[chrom]
            if interval_id is None:
                continue
            root, manifest = products[chrom]
            item = manifest["interval_products"].get(interval_id)
            require(item is not None, "Required target-excluded H interval product missing")
            support = add_support(support, root/item["path"])
            included.append({"chromosome": chrom, "interval_id": interval_id,
                             "intervals": item["intervals"], "support_sha256": item["sha256"]})
            selected_products.append((root, item))
        require(support is not None and int(support["candidate_sites"]) > 0,
                "H view has no candidate rare sites")
        support_path = view/"support.private.npz"
        budget.before(npz_upper_bound(support)); np.savez_compressed(support_path, **support)
        budget.add(support_path)
        block = blocks[territory["block_id"]]
        exclusion = {"scope": territory["scope"], "canonical_id": canonical_id,
                     "representative_block_id": territory["block_id"],
                     "target_chromosome": target_chrom,
                     "representative_target": [target_chrom, block["start_bp"], block["end_bp"]],
                     "representative_excluded": [target_chrom, block["excluded_start_bp"], block["excluded_end_bp"]],
                     "included": included}
        budget.before(len(json.dumps(exclusion).encode()) + 1)
        write_json(view/"exclusion.json", exclusion); budget.add(view/"exclusion.json")
        for geometry in geometries:
            config = geometry["config_id"]; geometry_path = view/(config + ".private.npz")
            summaries = {"FIT_FIT": {}, "QUERY_FIT": {}}
            for root, item in selected_products:
                product = item["geometries"].get(config)
                require(product is not None and sha(root/product["path"]) == product["sha256"],
                        "Required H chromosome geometry product missing or changed")
                with np.load(root/product["path"], allow_pickle=False) as data:
                    expected = {prefix + suffix for prefix in ("fit", "query_fit") for suffix in
                                ("_pair_index", "_total_shared_bp", "_n_chains",
                                 "_n_shared_sites", "_max_chain_bp")}
                    require(set(data.files) == expected, "H chromosome compact geometry arrays differ")
                    for kind, prefix in (("FIT_FIT", "fit"), ("QUERY_FIT", "query_fit")):
                        indices = data[prefix + "_pair_index"]
                        metrics = np.column_stack([data[prefix + suffix] for suffix in
                            ("_total_shared_bp", "_n_chains", "_n_shared_sites", "_max_chain_bp")])
                        require(indices.ndim == 1 and metrics.shape == (len(indices), 4) and
                                np.all(np.diff(indices) > 0) and np.all(metrics > 0),
                                "H chromosome compact pair metrics invalid")
                        for pair, values in zip(indices.tolist(), metrics.tolist()):
                            current = summaries[kind].get(pair)
                            if current is None:
                                summaries[kind][pair] = values
                            else:
                                current[0] += values[0]; current[1] += values[1]; current[2] += values[2]
                                current[3] = max(current[3], values[3])
            arrays = compact_arrays(summaries)
            budget.before(npz_upper_bound(arrays)); np.savez_compressed(geometry_path, **arrays)
            budget.add(geometry_path)
            metadata = {"schema": RESULT_SCHEMA, "stage": STAGE, "provider": "H_COMPACT_V1",
                "family": "H", "measure": "M14_SHARED_SOURCE_MINOR_RARE_CHAIN_TOTAL_BP",
                "units": "LOG1P_TOTAL_SHARED_BP_OVER_DECLARED_BP_SCALE",
                "transformation": value["transformation"], "range": {"minimum": 0.0, "maximum": None},
                "missing_policy": "UNKNOWN_NOT_ZERO",
                "pair_policy": {**value["pair_policy"], "fit_fit_only": True,
                                "query_fit_only": True, "query_query": "EXCLUDE"},
                "territory": exclusion, "geometry": geometry,
                "role_counts": {k: len(v) for k, v in roles.items()},
                "persons_content_sha256": sha(output/"persons.tsv"),
                "normalization": {"affinity_formula": "log1p(total_shared_bp / scale_bp)",
                    "support_numerator": "total_shared_bp", "support_denominator": "joint_callable_rare_sites",
                    "zero_chain_with_positive_joint_callable": "OBSERVED_ZERO",
                    "zero_joint_callable": "NA"},
                "sources": {"request_sha256": request_sha256,
                    "catalogue_sha256": value["catalogue"]["sha256"],
                    "source_minor_contract": materialize["source_rare_rule"],
                    "chromosome_products": {chrom: sha(root/"manifest.json") for chrom, (root, _m) in products.items()}},
                "capacity": {"candidate_rare_sites": int(support["candidate_sites"]),
                    "implicit_fit_pair_capacity": core.triangle_size(len(fit)),
                    "implicit_query_fit_pair_capacity": len(query)*len(fit),
                    "stored_positive_fit_pairs": len(arrays["fit_pair_index"]),
                    "stored_positive_query_fit_pairs": len(arrays["query_fit_pair_index"]),
                    "resources": limits},
                "support": {"path": str(support_path.relative_to(output)), "sha256": sha(support_path),
                    "semantics": "candidate sites, per-person missing counts and pairwise both-missing counts"},
                "compact_affinities": {"path": str(geometry_path.relative_to(output)),
                    "sha256": sha(geometry_path), "zero_storage": "IMPLICIT_ONLY_WHEN_JOINT_CALLABLE_POSITIVE"},
                "persons": {"path": "persons.tsv", "sha256": sha(output/"persons.tsv")},
                "exclusion": {"path": str((view/"exclusion.json").relative_to(output)),
                              "sha256": sha(view/"exclusion.json")},
                "same_component_fit_edges": value["pair_policy"]["fit_component_policy"],
                "outcomes_read": False, "query_query_computed": False,
                "evaluate_opened": False, "biological_validation": False}
            manifest_path = view/(config + ".affinity_inputs.json")
            budget.before(len(json.dumps(metadata).encode()) + 1)
            write_json(manifest_path, metadata); budget.add(manifest_path)
            registry.append({"canonical_id": canonical_id, "config_id": config,
                             "manifest": str(manifest_path.relative_to(output)),
                             "manifest_sha256": sha(manifest_path)})
        require(sum(p.stat().st_size for p in view.iterdir()) <= limits["max_output_bytes"],
                "One canonical H territory exceeds output bound")
    result = {"schema": "r02_structure_h_affinity_registry_v1", "stage": STAGE,
              "request_sha256": request_sha256, "implementation_sha256": code_hashes(),
              "canonical_views": registry, "aliases": aliases,
              "n_requested_views": len(territories)*len(geometries),
              "n_canonical_territories": len(canonical), "n_geometries": len(geometries),
              "n_territories": len(territories), "persons_sha256": sha(output/"persons.tsv"),
              "outcomes_read": False,
              "query_query_computed": False, "biological_validation": False}
    budget.before(len(json.dumps(result).encode()) + 1)
    write_json(output/"registry.json", result); budget.add(output/"registry.json")
    require(sha(request_path) == request_sha256, "H request changed during reduction")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("chromosome", "reduce"), required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--chromosome")
    parser.add_argument("--chromosome-products", type=Path,
                        help="JSON chromosome-to-directory map for reduce")
    parser.add_argument("--chromosome-dir", action="append", default=[],
                        help="One chromosome product directory; repeat for reduce")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); os.umask(0o077)
    if args.stage == "chromosome":
        require(args.chromosome is not None and args.chromosome_products is None and not args.chromosome_dir,
                "Chromosome stage requires only --chromosome")
        result = write_chromosome(args.request, args.request_sha256, args.chromosome, args.output_dir)
    else:
        require(args.chromosome is None and bool(args.chromosome_products) != bool(args.chromosome_dir),
                "Reduce stage requires either --chromosome-products or repeated --chromosome-dir")
        directories = load_json(args.chromosome_products) if args.chromosome_products else {}
        for directory in args.chromosome_dir:
            manifest = load_json(Path(directory)/"manifest.json")
            chrom = manifest.get("chromosome")
            require(isinstance(chrom, str) and chrom not in directories,
                    "Unique chromosome product directory required")
            directories[chrom] = directory
        result = reduce_views(args.request, args.request_sha256,
                              directories, args.output_dir)
    print(json.dumps({"schema": result["schema"], "stage": result["stage"]}))


if __name__ == "__main__":
    main()
