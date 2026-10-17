#!/usr/bin/env python3
"""Build K03 rare affinities from raw genotypes using one INNER_FIT split.

The module deliberately does not accept an existing affinity matrix.  For every
site it estimates ALT frequency with the people named by ``inner_fit_indices``,
chooses the unique minor allele from that frequency, applies the resulting
orientation to INNER_FIT and INNER_PROJECT genotypes, and then calculates
pairwise carrier Jaccard affinities.  Genotypes of SELECT and EVALUATE are not
valid inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

import r02_k03_inner_fit_contract as contract
import r02_projection_calibration_split as split
import r02_structure_rare_inputs as rare_inputs


SCHEMA = "r02_k03_inner_fit_raw_genotypes_v1"
SHARED_SCHEMA = contract.SHARED_SCHEMA
GT_CODE_ENCODING = contract.GT_CODE_ENCODING
STRICT_INNER_FIT_ASCERTAINMENT = contract.STRICT_INNER_FIT_ASCERTAINMENT
FIXED_PANEL_PENDING = {
    "scope": "FIXED_SOURCE_COHORT_2723_MAC_GE_2_MAF_LE_0_01",
    "inner_fit_reselection": False,
    "select_evaluate_in_source_selection": True,
    "protocol_admission": "PENDING_EXPLICIT_FIXED_PANEL_DECISION",
}
FIXED_PANEL_ADMITTED = {
    **FIXED_PANEL_PENDING,
    "protocol_admission": "EXPLICITLY_ADMITTED_CONDITIONAL_ON_FIXED_SOURCE_PANEL",
}
OUTPUT_SCHEMA = "r02_k03_inner_fit_rare_affinities_v1"
TRANSFORM_DTYPE = np.dtype([("alt_allele_count", "<i8"),
                            ("called_alleles", "<i8"), ("status", "u1")])
TRANSFORM_STATUS = {0: "NO_INNER_FIT_CALLS", 1: "INNER_FIT_MONOMORPHIC",
                    2: "INNER_FIT_EQUAL_FREQUENCY_NO_UNIQUE_MINOR",
                    3: "INCLUDED_ALT", 4: "INCLUDED_REF",
                    5: "INNER_FIT_MAC_BELOW_2", 6: "INNER_FIT_MAF_ABOVE_0_01"}
EDGE_FIELDS = ["left_index", "right_index", "observed", "affinity",
               "joint_callable_sites", "value_denominator_sites", "candidate_sites"]
def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return split.sha256(Path(path))


def ordered_ids_sha256(values: list[str]) -> str:
    return contract.ordered_ids_sha256(values)


def implementation_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    hashes = dict(rare_inputs.implementation())
    for name in ("r02_k03_inner_fit_contract.py", "r02_k03_inner_fit_affinities.py",
                 "r02_projection_calibration_split.py"):
        hashes[name] = sha256(root / name)
    return dict(sorted(hashes.items()))


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def _descriptor(value: Any, label: str, *, base: Path | None = None) -> tuple[Path, str]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}
            and isinstance(value["path"], str)
            and isinstance(value["sha256"], str) and len(value["sha256"]) == 64,
            f"Exact pinned {label} descriptor required")
    path = Path(value["path"])
    if not path.is_absolute():
        require(base is not None and ".." not in path.parts and path.parts,
                f"Safe manifest-relative {label} path required")
        path = Path(base) / path
    require(path.is_file() and sha256(path) == value["sha256"],
            f"Pinned {label} is absent or changed")
    return path, value["sha256"]


def _validate_manifest(path: Path, roster_path: Path, plan_path: Path) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    manifest = load_json(path)
    legacy_required = {"schema", "roster", "fold_plan", "sample_scope", "sample_ids_sha256",
                       "dosage_allele", "variant_universe", "learning_scope",
                       "outer_fit_learned_inputs", "chunks", "resources",
                       "implementation_sha256"}
    shared_required = {"schema", "roster", "sample_scope", "sample_ids_sha256",
                       "dosage_allele", "genotype_encoding", "variant_universe", "territory", "learning_scope",
                       "outer_fit_learned_inputs", "candidate_ascertainment", "chunks",
                       "resources", "producer",
                       "consumer_implementation_sha256"}
    legacy = manifest.get("schema") == SCHEMA
    require((legacy and set(manifest) == legacy_required) or
            (manifest.get("schema") == SHARED_SCHEMA and set(manifest) == shared_required),
            "Exact raw-genotype INNER_FIT or shared outer-FIT ALT manifest required")
    source_dir = Path(path).resolve().parent
    roster_pin, _ = _descriptor(manifest["roster"], "roster", base=source_dir)
    require(roster_pin.resolve() == roster_path.resolve(),
            "CLI roster differs from pinned genotype input")
    if legacy:
        plan_pin, _ = _descriptor(manifest["fold_plan"], "fold plan", base=source_dir)
        require(plan_pin.resolve() == plan_path.resolve(),
                "CLI plan differs from pinned genotype input")
    people, inventory = split.roster_inventory(roster_path)
    plan = load_json(plan_path)
    require(plan.get("schema") == split.PAIR_PLAN_SCHEMA
            and plan.get("roster_sha256") == inventory["roster"]["sha256"],
            "Only the frozen pair-complete INNER_FIT plan is admitted")
    outer_fit = [i for i, row in enumerate(people) if row["role"] == "FIT"]
    train = sorted(plan["inner_fit_indices"]); project = sorted(plan["inner_project_indices"])
    require(len(train) > 1 and project and set(train).isdisjoint(project)
            and sorted(train + project) == outer_fit,
            "Plan must partition outer FIT into nonempty component-disjoint internal roles")
    fit_ids = [people[i]["sample_id"] for i in outer_fit]
    require(manifest["sample_scope"] == "OUTER_FIT_GENOTYPES_ONLY"
            and manifest["sample_ids_sha256"] == ordered_ids_sha256(fit_ids),
            "Genotype source must contain exactly frozen outer FIT in roster order")
    allowed_universe = ("PINNED_ORIGINALLY_BIALLELIC_PASS_NO_INNER_ROLE_SELECTION" if legacy else
                        "PINNED_ORIGINALLY_BIALLELIC_PASS_TARGET_EXCLUDED_NO_ROLE_SELECTION")
    require(manifest["dosage_allele"] == "ALT"
            and manifest["variant_universe"] == allowed_universe
            and manifest["learning_scope"] == "RAW_GENOTYPES_NO_LEARNED_TRANSFORMATIONS"
            and manifest["outer_fit_learned_inputs"] is False,
            "Outer-FIT frequencies, orientations, transformations or affinities are forbidden")
    if not legacy:
        require(manifest["candidate_ascertainment"] == STRICT_INNER_FIT_ASCERTAINMENT
                and manifest["genotype_encoding"] == GT_CODE_ENCODING,
                "Candidate eligibility and partial-call denominator must be learned only in INNER_FIT")
    implementation = (manifest["implementation_sha256"] if legacy else
                      manifest["consumer_implementation_sha256"])
    require(implementation == implementation_sha256(),
            "INNER_FIT producer implementation changed")
    if not legacy:
        territory = manifest["territory"]
        require(territory == {
            "view_id": "BLOCK_AND_LD_MARGIN:chr01_part01",
            "excluded_intervals": [{"chromosome": 1, "start_bp": 1,
                                    "end_bp": 115064116}],
            "target_locus_policy": "EXCLUDED_BEFORE_OUTPUT",
            "source_candidate_rule": "ORIGINALLY_BIALLELIC_PASS_BEFORE_MAC_MAF",
        }, "Shared ALT chunks have a different target-exclusion territory")
        producer = manifest["producer"]
        require(isinstance(producer, dict) and set(producer) == {
                    "implementation_sha256", "source_campaign_sha256", "source_frequency_use",
                    "original_allele_filter", "site_filter", "all_autosomes_complete"}
                and producer["source_frequency_use"] == "NONE_BEFORE_INNER_FIT"
                and producer["original_allele_filter"] == "DNABR_V1_ORIG_NALLELES_EQ_2"
                and producer["site_filter"] == "PASS_BIALLELIC_ACGT_SNV"
                and producer["all_autosomes_complete"] is True,
                "Shared genotype producer provenance is incomplete or admits pre-INNER_FIT frequency learning")
    resources = manifest["resources"]
    require(set(resources) == {"max_input_bytes", "max_output_bytes", "min_free_bytes",
                               "max_memory_bytes", "max_sparse_cells", "site_batch_size"}
            and all(type(resources[k]) is int and resources[k] > 0 for k in resources),
            "Explicit positive producer resource bounds required")
    memory_bound = (136 * len(outer_fit) * len(outer_fit)
                    + 64 * resources["max_sparse_cells"] + 16 * 2**20)
    require(memory_bound <= resources["max_memory_bytes"],
            "Sparse I/U/Q accumulator exceeds declared working-memory bound")
    require(isinstance(manifest["chunks"], list) and manifest["chunks"],
            "At least one raw genotype chunk required")
    total = 0
    for number, descriptor in enumerate(manifest["chunks"]):
        chunk, _ = _descriptor(descriptor, f"genotype chunk {number}", base=source_dir)
        total += chunk.stat().st_size
        require(total <= resources["max_input_bytes"], "Raw genotype inputs exceed byte bound")
    return manifest, people, plan


def _pair_keys(people: list[dict[str, str]], train: list[int], project: list[int]) -> tuple[np.ndarray, np.ndarray]:
    components = np.asarray([row["component"] for row in people])
    fit_left, fit_right = np.triu_indices(len(train), 1)
    fit_left = np.asarray(train, dtype=np.int64)[fit_left]
    fit_right = np.asarray(train, dtype=np.int64)[fit_right]
    keep = components[fit_left] != components[fit_right]
    pairs = [(int(i), int(j)) for i, j in zip(fit_left[keep], fit_right[keep])]
    pairs.extend((min(q, f), max(q, f)) for q in project for f in train)
    pairs.sort()
    array = np.asarray(pairs, dtype=np.int64)
    require(array.ndim == 2 and array.shape[1] == 2
            and len({tuple(row) for row in array.tolist()}) == len(array),
            "INNER_FIT pair universe is not unique")
    return array[:, 0], array[:, 1]


def _load_chunk(path: Path, fit_ids: list[str]) -> dict[str, np.ndarray]:
    required = {"sample_ids", "variant_id", "chromosome", "position_bp", "ref", "alt",
                "genotype_code"}
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == required, "Exact raw genotype chunk arrays required")
        value = {name: data[name].copy() for name in data.files}
    n_sites = len(value["variant_id"])
    require(value["sample_ids"].tolist() == fit_ids,
            "Chunk samples/order differ from frozen outer FIT")
    code = value["genotype_code"]
    require(code.shape == (len(fit_ids), n_sites) and code.dtype == np.uint8
            and np.all(np.isin(code, np.asarray([0, 1, 2, 3, 4, 255], np.uint8))),
            "Chunk genotype code must preserve complete, partial and fully missing diploid calls")
    require(all(value[k].shape == (n_sites,) for k in
                ("chromosome", "position_bp", "ref", "alt", "variant_id"))
            and n_sites > 0, "Variant metadata/chunk dimensions differ")
    chromosome = value["chromosome"]
    position = value["position_bp"]
    require(chromosome.dtype.kind in "iu" and position.dtype.kind in "iu"
            and np.all(chromosome > 0) and np.all(position > 0),
            "Chromosome and one-based position must be positive integers")
    after_previous = ((chromosome[1:] > chromosome[:-1])
                      | ((chromosome[1:] == chromosome[:-1])
                         & (position[1:] > position[:-1])))
    require(np.all(after_previous),
            "Raw variants within a chunk must be sorted and coordinate-unique")
    require(np.all(np.char.str_len(value["ref"].astype(str)) == 1)
            and np.all(np.char.str_len(value["alt"].astype(str)) == 1)
            and np.all(value["ref"] != value["alt"]),
            "Only single-base biallelic records are admitted")
    return value


def materialize(manifest_path: Path, roster_path: Path, plan_path: Path,
                output: Path) -> dict[str, Any]:
    manifest_path, roster_path, plan_path, output = map(Path, (manifest_path, roster_path, plan_path, output))
    manifest, people, plan = _validate_manifest(manifest_path, roster_path, plan_path)
    resources = manifest["resources"]
    require(not output.exists() and output.parent.is_dir()
            and shutil.disk_usage(output.parent).free >= resources["min_free_bytes"] + resources["max_output_bytes"],
            "A new output and declared disk reserve are required")
    output.mkdir(mode=0o700)
    outer_fit = [i for i, row in enumerate(people) if row["role"] == "FIT"]
    local = {global_index: rank for rank, global_index in enumerate(outer_fit)}
    train = sorted(plan["inner_fit_indices"]); project = sorted(plan["inner_project_indices"])
    train_local = np.asarray([local[i] for i in train], dtype=np.int64)
    left, right = _pair_keys(people, train, project)
    left_local = np.asarray([local[int(i)] for i in left], dtype=np.int64)
    right_local = np.asarray([local[int(i)] for i in right], dtype=np.int64)
    counter = rare_inputs.SparseBatchCounter(
        len(outer_fit), batch_sites=resources["site_batch_size"],
        max_sparse_cells=resources["max_sparse_cells"])
    input_sites = 0
    status_counts: Counter[int] = Counter()
    candidate_sites = 0
    previous_key: tuple[Any, ...] | None = None
    fit_ids = [people[i]["sample_id"] for i in outer_fit]
    site_batch = resources["site_batch_size"]
    transform_path = output / "inner_fit_transformations.private.bin"
    with transform_path.open("xb") as transform_stream:
        for descriptor in manifest["chunks"]:
            path, digest = _descriptor(descriptor, "genotype chunk",
                                       base=manifest_path.resolve().parent)
            chunk = _load_chunk(path, fit_ids)
            first_key = (int(chunk["chromosome"][0]), int(chunk["position_bp"][0]))
            last_key = (int(chunk["chromosome"][-1]), int(chunk["position_bp"][-1]))
            require(previous_key is None or previous_key < first_key,
                    "Raw genotype chunks must be globally sorted and coordinate-unique")
            previous_key = last_key
            input_sites += len(chunk["variant_id"])
            for start in range(0, len(chunk["variant_id"]), site_batch):
                stop = min(start + site_batch, len(chunk["variant_id"]))
                code = chunk["genotype_code"][:, start:stop]
                fit_code = code[train_local]
                called = np.where(fit_code <= 2, 2, np.where(fit_code <= 4, 1, 0)).sum(axis=0, dtype=np.int64)
                alt_count = np.where(fit_code <= 2, fit_code,
                                     np.where(fit_code == 4, 1, 0)).sum(axis=0, dtype=np.int64)
                polymorphic = (alt_count > 0) & (alt_count < called)
                unique_minor = 2 * alt_count != called
                mac = np.minimum(alt_count, called - alt_count)
                enough_mac = mac >= 2
                rare_enough = 100 * mac <= called
                eligible = (called > 0) & polymorphic & unique_minor & enough_mac & rare_enough
                minor_alt = 2 * alt_count < called
                status = np.where(called == 0, 0,
                         np.where(~polymorphic, 1,
                         np.where(~unique_minor, 2,
                         np.where(~enough_mac, 5,
                         np.where(~rare_enough, 6, np.where(minor_alt, 3, 4)))))).astype(np.uint8)
                records = np.empty(stop - start, dtype=TRANSFORM_DTYPE)
                records["alt_allele_count"] = alt_count
                records["called_alleles"] = called
                records["status"] = status
                transform_stream.write(records.tobytes())
                status_counts.update(map(int, status))
                require(transform_stream.tell() <= resources["max_output_bytes"]
                        and shutil.disk_usage(output).free >= resources["min_free_bytes"],
                        "Transformation output exceeds byte/disk bound")
                if not eligible.any():
                    continue
                candidate_sites += int(eligible.sum())
                selected_code = code[:, eligible]
                o = selected_code <= 2
                d = np.where(o, selected_code, 255).astype(np.uint8)
                minor_dosage = np.where(
                    o, np.where(minor_alt[eligible], d, 2 - d), 255).astype(np.uint8)
                counter.add(minor_dosage, o)
            require(sha256(path) == digest, "Genotype chunk changed while reading")
    require(candidate_sites > 0, "No site has a unique polymorphic INNER_FIT minor allele")
    require(sum(status_counts.values()) == input_sites
            and status_counts[3] + status_counts[4] == candidate_sites,
            "Transformation status accounting differs from input/candidate sites")
    aggregate = counter.finish()
    require(int(aggregate["n_sites"]) == candidate_sites,
            "Sparse accumulator candidate-site denominator differs")
    counts = np.vstack([aggregate[name][left_local, right_local]
                        for name in ("I", "U", "Q")])
    train_set = set(train)
    fit_pair = np.asarray([(int(i) in train_set and int(j) in train_set) for i, j in zip(left, right)])
    require(np.all(counts[1, fit_pair] > 0),
            "At least one INNER_FIT graph pair has no observed carrier union; complete graph unavailable")
    affinity_path = output / "inner_affinities.private.tsv"
    with affinity_path.open("x", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n"); writer.writerow(EDGE_FIELDS)
        for row, (i, j) in enumerate(zip(left, right)):
            intersection, union, joint = map(int, counts[:, row])
            require(0 <= intersection <= union <= joint <= candidate_sites,
                    "Pair sufficient-statistic invariant failed")
            writer.writerow([int(i), int(j), int(union > 0),
                             format(intersection / union, ".17g") if union else "NA",
                             joint, union, candidate_sites])
            if row % 4096 == 0:
                require(stream.tell() <= resources["max_output_bytes"]
                        and shutil.disk_usage(output).free >= resources["min_free_bytes"],
                        "Affinity output exceeds byte/disk bound")
    require(transform_path.stat().st_size == input_sites * TRANSFORM_DTYPE.itemsize,
            "Packed transformation length differs from input sites")
    raw_manifest_copy = output / "raw_genotype_manifest.json"
    shutil.copyfile(manifest_path, raw_manifest_copy)
    require(sha256(raw_manifest_copy) == sha256(manifest_path),
            "Raw genotype manifest provenance copy differs")
    require(sum(p.stat().st_size for p in output.iterdir()) <= resources["max_output_bytes"],
            "INNER_FIT outputs exceed byte bound")
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "INNER_FIT_FREQUENCIES_ORIENTATION_TRANSFORMATION_AND_AFFINITIES_COMPLETE",
        "fold_id": plan["fold_id"],
        "roster_sha256": sha256(roster_path), "fold_plan_sha256": sha256(plan_path),
        "raw_genotype_manifest": {"path": raw_manifest_copy.name,
                                  "sha256": sha256(raw_manifest_copy)},
        "learning_indices": train, "projection_indices": project,
        "learning_people": len(train), "projection_people": len(project),
        "frequency_scope": "INNER_FIT_ONLY", "minor_allele_scope": "INNER_FIT_ONLY",
        "transformation_scope": "INNER_FIT_ONLY", "affinity_scope": "INNER_FIT_DERIVED",
        "outer_fit_learned_inputs_used": False,
        "select_evaluate_genotypes_read": False,
        "candidate_ascertainment": STRICT_INNER_FIT_ASCERTAINMENT,
        "genotype_encoding": GT_CODE_ENCODING,
        "affinity": {"source_measure": "INNER_FIT_MINOR_CARRIER_JACCARD",
                     "transformation": "INNER_FIT_ALT_FREQUENCY_TO_UNIQUE_MINOR_CARRIER_THEN_INTERSECTION_OVER_UNION",
                     "equal_frequency_policy": "EXCLUDE_NO_UNIQUE_MINOR_ALLELE",
                     "missing_edge_policy": "UNKNOWN_NOT_ZERO", "maximum": 1.0},
        "candidate_sites": candidate_sites, "input_sites": input_sites,
        "transformation_encoding": {
            "file": transform_path.name, "bytes_per_site": TRANSFORM_DTYPE.itemsize,
            "order": "PINNED_INPUT_CHUNK_AND_VARIANT_ORDER",
            "fields": ["alt_allele_count:int64_le", "called_alleles:int64_le", "status:uint8"],
            "status_codes": {str(k): v for k, v in TRANSFORM_STATUS.items()},
            "status_counts": {str(k): status_counts[k] for k in sorted(TRANSFORM_STATUS)}
        },
        "fit_graph_edges": int(fit_pair.sum()), "query_fit_edges": int((~fit_pair).sum()),
        "unknown_query_fit_edges": int(np.sum((~fit_pair) & (counts[1] == 0))),
        "sparse_flushes": counter.flushes,
        "files_sha256": {affinity_path.name: sha256(affinity_path),
                         transform_path.name: sha256(transform_path),
                         raw_manifest_copy.name: sha256(raw_manifest_copy)},
        "implementation_sha256": implementation_sha256(),
        "communities_learned": False, "genetic_error_computed": False,
        "select_opened": False, "evaluate_opened": False, "biological_validation": False,
    }
    write_json(output / "affinity_inputs.json", result)
    require(sha256(manifest_path) == result["raw_genotype_manifest"]["sha256"]
            and implementation_sha256() == result["implementation_sha256"],
            "Inputs or implementation changed during materialization")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--fold-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); os.umask(0o077)
    result = materialize(args.manifest, args.roster, args.fold_plan, args.output)
    print(json.dumps({k: result[k] for k in
        ("status", "fold_id", "candidate_sites", "fit_graph_edges", "query_fit_edges")},
        sort_keys=True))


if __name__ == "__main__":
    main()
