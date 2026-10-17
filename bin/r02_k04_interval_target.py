#!/usr/bin/env python3
"""Authenticate and materialize one K04 chromosome/ancestry interval target.

This adapter exposes producer counts as solver-ready vectors for FIT or SELECT.
It validates the producer's dense all-person matrices, including entries whose
persons have role EVALUATE, but never materializes or scores EVALUATE response
pairs.  It does not define a predictor schema.  Pair rows are strict unordered
person-index pairs (i < j), in lexicographic order, within one frozen role.
Zero-opportunity pairs are absent, never encoded as a zero response.  The target
chromosome is excluded from the declared predictor universe; a later feature
adapter must authenticate that declaration against the actual predictor bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

import r02_k04_common_target as producer
import r02_k04_identified_comparison as identified


SCHEMA = "r02_k04_interval_target_input_v1"
OUTPUT_SCHEMA = "r02_k04_interval_target_v1"
OPERATION = "MATERIALIZE_SOLVER_TARGET_ONLY"
SCOPE = "CONDITIONAL_GNOMIX_COMMON_DISCORDANCE_OUTER_INTERVAL"
ROLES = {"FIT", "SELECT"}
ROLE_COUNTS = {"FIT": 1460, "SELECT": 476, "EVALUATE": 683}
ARRAY_NAMES = tuple(
    [f"{kind}_{ancestry}" for ancestry in identified.ANCESTRIES
     for kind in ("lower_count", "upper_count", "opportunities")]
    + ["person_joint_loci", "person_observed_loci"]
)
OUTPUT_FILES = ("person_i.npy", "person_j.npy", "lower.npy", "upper.npy",
                "opportunities.npy")
require = producer.require


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_hash(samples):
    return hashlib.sha256(("\n".join(samples) + "\n").encode()).hexdigest()


def _autosome(value):
    text = str(value).removeprefix("chr")
    require(text.isdigit() and 1 <= int(text) <= 22, "Expected autosome 1..22")
    return str(int(text))


def _exact_keys(value, expected, message):
    require(isinstance(value, dict) and set(value) == set(expected), message)


def _read_persons(path, expected_n, expected_roles, expected_sample_hash):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    require(rows and len(rows) == expected_n, "Person count differs")
    require(set(rows[0]) == {"person_index", "sample_id", "role"},
            "Person columns differ")
    require([row["person_index"] for row in rows] == list(map(str, range(len(rows)))),
            "Person index must be contiguous and ordered")
    samples = [row["sample_id"] for row in rows]
    require(len(set(samples)) == len(samples) and
            all(value and not any(character.isspace() for character in value) for value in samples),
            "Sample IDs are empty or duplicated")
    require(sample_hash(samples) == expected_sample_hash, "Sample order SHA256 differs")
    roles = [row["role"] for row in rows]
    require(dict(Counter(roles)) == expected_roles and set(roles) == set(ROLE_COUNTS),
            "Frozen FIT/SELECT/EVALUATE roles differ")
    return rows


def _zip_inventory(path, max_uncompressed_bytes):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(len(names) == len(set(names)) and
                set(names) == {name + ".npy" for name in ARRAY_NAMES},
                "Counts NPZ member inventory differs")
        require(all(not entry.is_dir() and not (entry.flag_bits & 1) for entry in entries),
                "Encrypted or directory NPZ member unsupported")
        total = sum(entry.file_size for entry in entries)
        require(total <= max_uncompressed_bytes, "Counts NPZ admission bound exceeded")
    return total


def _validate_matrix(value, n, name):
    require(value.dtype == np.uint64 and value.shape == (n, n),
            f"{name} must be an exact uint64 person matrix")
    require(np.array_equal(value, value.T) and not np.any(np.diag(value)),
            f"{name} must be symmetric with a zero diagonal")
    require(int(value.max(initial=0)) < 2**53,
            f"{name} exceeds exact float64 integer range")


class CommonIntervalTarget:
    """Authenticated dense producer target with bounded pair-vector iteration."""

    def __init__(self, request):
        _exact_keys(request, {"schema", "operation", "scientific_scope", "target",
            "ancestry", "role", "expected_n_people", "expected_role_counts",
            "sample_order_sha256", "predictor_chromosomes", "resources"},
            "Interval-target request fields differ")
        require(request["schema"] == SCHEMA and request["operation"] == OPERATION and
                request["scientific_scope"] == SCOPE,
                "Unsupported target operation or scientific scope")
        require(request["ancestry"] in identified.ANCESTRIES, "Unsupported ancestry")
        require(request["role"] in ROLES, "Only FIT or SELECT target access is allowed")
        require(type(request["expected_n_people"]) is int and
                request["expected_n_people"] == sum(ROLE_COUNTS.values()) and
                request["expected_role_counts"] == ROLE_COUNTS,
                "Human cohort/role specification differs")
        require(isinstance(request["sample_order_sha256"], str) and
                len(request["sample_order_sha256"]) == 64 and
                all(character in "0123456789abcdef" for character in request["sample_order_sha256"]),
                "Explicit sample-order SHA256 required")
        _exact_keys(request["target"], {"directory", "manifest_sha256", "chromosome"},
                    "Target descriptor fields differ")
        _exact_keys(request["resources"], {"max_npz_uncompressed_bytes", "max_memory_bytes",
                                          "max_output_bytes", "block_pairs"},
                    "Resource fields differ")
        resources = request["resources"]
        require(all(type(resources[k]) is int and resources[k] > 0 for k in resources),
                "Positive integer resource bounds required")
        require(resources["block_pairs"] <= 10_000_000, "Pair block bound is excessive")
        n_people = request["expected_n_people"]
        n_role = ROLE_COUNTS[request["role"]]
        # Three selected-ancestry matrices, one validation mask, both the
        # buffered and concatenated pair block, and the worst-case file-backed
        # output mappings.  This is an admission bound, not observed RSS.
        memory_bound = (3 * n_people * n_people * 8 + n_people * n_people +
                        resources["block_pairs"] * 64 +
                        (n_role * (n_role - 1) // 2) * 32 + 2**20)
        require(memory_bound <= resources["max_memory_bytes"],
                "Interval target memory admission failed")

        target = request["target"]
        chromosome = _autosome(target["chromosome"])
        predictors = [_autosome(value) for value in request["predictor_chromosomes"]]
        expected_predictors = [str(i) for i in range(1, 23) if str(i) != chromosome]
        require(predictors == expected_predictors and chromosome not in predictors,
                "Predictor chromosome universe must be ordered autosomes excluding target")
        directory = Path(target["directory"]).resolve()
        manifest_path = directory / "manifest.json"
        require(sha256(manifest_path) == target["manifest_sha256"],
                "Target manifest SHA256 differs")
        manifest = json.loads(manifest_path.read_text())
        require(manifest.get("schema") == producer.OUTPUT and
                manifest.get("status") == "COMPLETE_CONDITIONAL_COMMON_TARGET_NOT_MODEL_COMPARISON" and
                _autosome(manifest.get("chromosome")) == chromosome,
                "Unsupported/incomplete target product or chromosome")
        require(manifest.get("catalogue_scope") == "ALL_FIT_SELECTED_WHOLE_CHROMOSOME_COMMON_LOCI" and
                manifest.get("phase_imputed") is False and
                manifest.get("ancestry_truth_calibrated") is False and
                manifest.get("biological_comparison") is False and
                manifest.get("self_pairs") == "EXCLUDED_ZERO_OPPORTUNITIES" and
                manifest.get("zero_denominator") == "NOT_ESTIMABLE" and
                manifest.get("bounds") == "PAIRWISE_OUTER_NOT_JOINTLY_ATTAINABLE_NOT_CONFIDENCE_INTERVAL",
                "Target scientific scope differs")
        require(manifest.get("n_people") == request["expected_n_people"] and
                manifest.get("role_counts") == ROLE_COUNTS,
                "Target person roles differ")
        require(set(manifest.get("outputs_sha256", {})) ==
                {"counts.npz", "persons.tsv", "coordinate-preflight.json"},
                "Target output inventory differs")
        for name, digest in manifest["outputs_sha256"].items():
            require(sha256(directory / name) == digest, "Target output SHA256 differs: " + name)
        persons = _read_persons(directory / "persons.tsv", request["expected_n_people"],
                                ROLE_COUNTS, request["sample_order_sha256"])
        coordinate = json.loads((directory / "coordinate-preflight.json").read_text())
        require(type(coordinate.get("n_loci")) is int and coordinate["n_loci"] == manifest["n_loci"] > 0 and
                coordinate.get("loci_in_msp", 0) > 0 and
                coordinate.get("loci_in_msp") + coordinate.get("loci_outside_msp") == coordinate["n_loci"],
                "Coordinate support accounting differs")
        archive_path = directory / "counts.npz"
        uncompressed = _zip_inventory(archive_path, resources["max_npz_uncompressed_bytes"])
        archive = np.load(archive_path, allow_pickle=False)
        require(set(archive.files) == set(ARRAY_NAMES), "Counts array inventory differs")
        n = len(persons)
        joint = archive["person_joint_loci"]
        observed = archive["person_observed_loci"]
        require(joint.dtype == observed.dtype == np.uint64 and joint.shape == observed.shape == (n,) and
                np.all(joint <= observed) and np.all(observed <= manifest["n_loci"]),
                "Person support arrays are invalid")
        ancestry = request["ancestry"]
        lower = archive[f"lower_count_{ancestry}"]
        upper = archive[f"upper_count_{ancestry}"]
        opportunities = archive[f"opportunities_{ancestry}"]
        for name, value in (("lower", lower), ("upper", upper),
                            ("opportunities", opportunities)):
            _validate_matrix(value, n, ancestry + " " + name)
        require(np.all(lower <= upper) and np.all(upper <= opportunities),
                "Counts must obey 0 <= lower <= upper <= opportunities")
        self.lower, self.upper, self.opportunities = lower, upper, opportunities
        self.archive = archive
        self.request = request
        self.manifest = manifest
        self.persons = persons
        self.chromosome = chromosome
        self.ancestry = request["ancestry"]
        self.role = request["role"]
        self.role_indices = np.asarray([i for i, row in enumerate(persons)
                                        if row["role"] == self.role], dtype=np.int64)
        require(len(self.role_indices) == ROLE_COUNTS[self.role], "Requested role size differs")
        self.uncompressed_bytes = uncompressed
        self.memory_bound_bytes = memory_bound

    def close(self):
        self.archive.close()

    def iter_blocks(self, block_pairs=None):
        """Yield bounded, lexicographically ordered, positive-opportunity pairs."""
        limit = self.request["resources"]["block_pairs"] if block_pairs is None else block_pairs
        require(type(limit) is int and 0 < limit <= self.request["resources"]["block_pairs"],
                "Invalid pair block size")
        buffers = [[] for _ in range(5)]
        buffered = 0
        def emit():
            nonlocal buffers, buffered
            if not buffered:
                return None
            values = tuple(np.concatenate(parts) for parts in buffers)
            buffers = [[] for _ in range(5)]
            buffered = 0
            i, j, lower, upper, opportunities = values
            require(np.all(i < j) and np.all(opportunities > 0) and
                    np.all((0 <= lower) & (lower <= upper) & (upper <= 1)),
                    "Invalid pair block after normalization")
            return dict(person_i=i, person_j=j, lower=lower, upper=upper,
                        opportunities=opportunities)
        ids = self.role_indices
        for offset, person_i in enumerate(ids[:-1]):
            right = ids[offset + 1:]
            den = self.opportunities[person_i, right]
            supported = np.flatnonzero(den > 0)
            cursor = 0
            while cursor < len(supported):
                take_n = min(limit - buffered, len(supported) - cursor)
                take = supported[cursor:cursor + take_n]
                js = right[take]
                d = den[take]
                buffers[0].append(np.full(take_n, person_i, dtype=np.uint32))
                buffers[1].append(js.astype(np.uint32, copy=False))
                buffers[2].append(self.lower[person_i, js].astype(np.float64) / d)
                buffers[3].append(self.upper[person_i, js].astype(np.float64) / d)
                buffers[4].append(d.astype(np.uint64, copy=False))
                buffered += take_n
                cursor += take_n
                if buffered == limit:
                    yield emit()
        block = emit()
        if block is not None:
            yield block

    def count_pairs(self):
        count = 0
        ids = self.role_indices
        for offset, person_i in enumerate(ids[:-1]):
            count += int(np.count_nonzero(self.opportunities[person_i, ids[offset + 1:]] > 0))
        return count


def materialize(request, request_sha256, output):
    request_path = Path(request)
    require(sha256(request_path) == request_sha256, "Request SHA256 differs")
    specification = json.loads(request_path.read_text())
    adapter = CommonIntervalTarget(specification)
    output = Path(output)
    require(not output.exists(), "Output exists; no overwrite allowed")
    n_pairs = adapter.count_pairs()
    estimated = n_pairs * (4 + 4 + 8 + 8 + 8) + 5 * 256 + 16384
    require(estimated <= specification["resources"]["max_output_bytes"],
            "Materialized target exceeds output bound")
    output.mkdir(parents=True, mode=0o700)
    arrays = {
        "person_i": np.lib.format.open_memmap(output / "person_i.npy", mode="w+", dtype="<u4", shape=(n_pairs,)),
        "person_j": np.lib.format.open_memmap(output / "person_j.npy", mode="w+", dtype="<u4", shape=(n_pairs,)),
        "lower": np.lib.format.open_memmap(output / "lower.npy", mode="w+", dtype="<f8", shape=(n_pairs,)),
        "upper": np.lib.format.open_memmap(output / "upper.npy", mode="w+", dtype="<f8", shape=(n_pairs,)),
        "opportunities": np.lib.format.open_memmap(output / "opportunities.npy", mode="w+", dtype="<u8", shape=(n_pairs,)),
    }
    cursor = 0
    for block in adapter.iter_blocks():
        stop = cursor + len(block["lower"])
        for name in arrays:
            arrays[name][cursor:stop] = block[name]
        cursor = stop
    require(cursor == n_pairs, "Pair count changed during materialization")
    for value in arrays.values():
        value.flush()
    del arrays
    outputs = {name: sha256(output / name) for name in OUTPUT_FILES}
    receipt = dict(schema=OUTPUT_SCHEMA,
        status=("COMPLETE_SOLVER_TARGET_ADAPTER_NOT_MODEL_FIT" if n_pairs else
                "NO_POSITIVE_OPPORTUNITY_SUPPORT_NOT_MODEL_FAILURE"),
        request_sha256=request_sha256, implementation_sha256=sha256(__file__),
        source=dict(directory=str(Path(specification["target"]["directory"]).resolve()),
                    manifest_sha256=specification["target"]["manifest_sha256"],
                    counts_sha256=adapter.manifest["outputs_sha256"]["counts.npz"],
                    persons_sha256=adapter.manifest["outputs_sha256"]["persons.tsv"]),
        chromosome=adapter.chromosome, ancestry=adapter.ancestry, role=adapter.role,
        n_people=len(adapter.persons), n_role_people=len(adapter.role_indices), n_pairs=n_pairs,
        sample_order_sha256=specification["sample_order_sha256"],
        predictor_chromosomes=specification["predictor_chromosomes"],
        target_chromosome_excluded_from_declared_predictors=True,
        predictor_bytes_authenticated=False,
        pair_indexing="GLOBAL_PERSON_INDEX_I_LT_J_LEXICOGRAPHIC_WITHIN_ONE_ROLE",
        zero_opportunity="EXCLUDED_NOT_ZERO_RESPONSE", response="COUNT_ENDPOINT_DIVIDED_BY_POSITIVE_OPPORTUNITIES",
        bounds="PAIRWISE_OUTER_NOT_JOINTLY_ATTAINABLE_NOT_CONFIDENCE_INTERVAL",
        ancestry_scope="GNOMIX_HARDCALL_CONDITIONAL_NOT_CALIBRATED_TRUTH",
        dense_all_person_count_matrices_validated=True,
        evaluate_pairs_materialized=False, evaluate_predictions_scored=False,
        cross_role_pairs_used=False,
        estimable_for_interval_fit=bool(n_pairs),
        npz_uncompressed_bytes=adapter.uncompressed_bytes,
        peak_target_memory_estimate_bytes=adapter.memory_bound_bytes,
        outputs_sha256=outputs,
        next_dependency="Authenticate actual C/Q/R predictor columns and prove each uses only declared non-target chromosomes before FIT/SELECT fitting")
    with (output / "manifest.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    adapter.close()
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    materialize(args.request, args.request_sha256, args.output)


if __name__ == "__main__":
    main()
