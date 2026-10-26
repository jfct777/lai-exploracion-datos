"""Authenticate and index existing LAI predictors; never read simulated truth.

The index preserves every catalogue locus and references existing genotype files.
It is independent of neural radius, architecture, training and model selection.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r02_lai_event_annotations as event_annotations

SCHEMA = "r02_lai_factorized_predictors_v1"
REFERENCE_SCHEMA = "r02_lai_reference_linkage_receipt_v1"
REFERENCE_HASHES = ("event_catalogue_sha256", "ref_train_receipt_sha256",
                    "ref_support_by_provenance_sha256", "reference_vcf_sha256",
                    "reference_roles_sha256")
MAX_LOCUS_BLOCK = 8192
MAX_PROBABILITY_QUERY_BLOCK = 1024


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def authenticated(path, expected, pins):
    path = str(Path(path).resolve())
    require(sha(path) == expected, "Source hash mismatch: " + path)
    pins[path] = expected
    return Path(path)


def table(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        require(fields is not None, "Missing TSV header")
        return fields, list(reader)


def key(row):
    return (str(row["chromosome"]), int(row["position_bp"]),
            row.get("ref", row.get("REF")), row.get("alt", row.get("ALT")))


def catalogue_index(rows, loci):
    keys = [key(row) for row in loci]
    require(len(set(keys)) == len(keys), "Duplicate source locus")
    source = {value: i for i, value in enumerate(keys)}
    wanted = [key(row) for row in rows]
    require(len(set(wanted)) == len(wanted), "Duplicate catalogue locus")
    require(all(value in source for value in wanted),
            "Catalogue locus absent from observed genotype source; cannot impute REF")
    for row in rows:
        require(row["ref"] in "ACGT" and row["alt"] in "ACGT"
                and len(row["ref"]) == len(row["alt"]) == 1
                and row["ref"] != row["alt"], "Expected biallelic SNV")
        event_annotations.counted_is_alt(row["ref"], row["alt"], row["counted_allele"])
        require(np.isfinite(float(row["position_cm"])), "Non-finite genetic position")
    positions = np.asarray([value[1] for value in wanted], dtype=np.int64)
    require(not len(positions) or np.all(np.diff(positions) > 0), "Unsorted catalogue")
    return np.asarray([source[value] for value in wanted], dtype=np.int64)


def counted_dosage(alt, observed, is_alt):
    require(alt.shape == observed.shape and observed.dtype == np.bool_, "Invalid genotype shape/mask")
    require(np.isin(alt[observed], [0, 1, 2]).all(), "Non-diploid observed dosage")
    # Missing remains missing; the zero here is only a masked placeholder.
    return np.where(observed, np.where(is_alt[None, :], alt, 2 - alt), 0).astype(np.int8)


def reference_linkage(path, expected_sha, task, event_sha, pins):
    """Authenticate DM reference counts and unit separation for one EVENT axis."""
    path = authenticated(path, expected_sha, pins)
    receipt = json.loads(path.read_text())
    require(receipt.get("schema") == REFERENCE_SCHEMA
            and receipt.get("status") == "COUNTS_AND_ORIGINAL_UNITS_MATCH"
            and receipt.get("training_authorized") is False
            and receipt.get("controls_materialized") is False
            and receipt.get("truth_loaded") is False
            and receipt.get("evaluate_opened") is False,
            "DM reference receipt is not PREPARE-only")
    rows = [row for row in receipt.get("cases", []) if row.get("task_id") == task["task_id"]]
    require(len(rows) == 1, "DM reference case absent or duplicated")
    row = rows[0]
    require(row.get("role") == task["role"]
            and row.get("event_catalogue_sha256") == event_sha
            and row.get("event_loci") == task["inputs"]["common_event_core_catalogues"]["event"]["rows"]
            and row.get("target_people") == task["people"]
            and row.get("reference_join_key") == ["chromosome", "position_bp", "REF", "ALT"]
            and row.get("reference_duplicate_keys") == 0
            and row.get("reference_target_person_overlap") == 0
            and row.get("reference_target_original_unit_overlap") == 0
            and row.get("problems") == [], "DM reference case differs")
    identity = {name: row.get(name) for name in REFERENCE_HASHES}
    require(all(isinstance(value, str) and len(value) == 64 for value in identity.values()),
            "DM reference identity is incomplete")
    return identity


def prepare(plan_path, task_id, output):
    start = time.monotonic()
    plan_path, output = Path(plan_path), Path(output)
    plan_hash = sha(plan_path)
    plan = json.loads(plan_path.read_text())
    require(plan["schema"] == "r02_lai_training_inputs_plan_v1", "Unknown input plan")
    require(not plan["training_authorized"] and not plan["evaluate_opened"], "Preparation-only plan required")
    tasks = [task for task in plan["tasks"] if task["task_id"] == task_id]
    require(len(tasks) == 1, "Task missing or duplicated")
    task = tasks[0]
    require(task["role"] in {"FIT", "SELECT"}, "EVALUATE is closed")
    require(task["direction"] == "FOLD_A_TO_FOLD_B", "Direction differs from admitted preparation")
    pins = {str(plan_path.resolve()): plan_hash}
    if (output / "receipt.json").exists():
        record = json.loads((output / "receipt.json").read_text())
        require(record["plan_sha256"] == plan_hash and record["task_id"] == task_id,
                "Existing index belongs to a different task/plan")
        for path, expected in record["inputs_sha256"].items():
            authenticated(path, expected, pins)
        for name, expected in record["outputs_sha256"].items():
            authenticated(output / name, expected, {})
        return record
    require(not output.exists() or not any(output.iterdir()), "Partial output retained; use a new empty destination")
    output.mkdir(parents=True, exist_ok=True)
    inp = task["inputs"]
    revised = "full_material_identity" in inp
    if revised:
        require(plan.get("status") == "REVISED_FACTORIZED_INPUTS_READY"
                and plan.get("selection_authorized") is False
                and plan.get("controls_materialized") is False
                and task.get("status") == "REVISED_FACTORIZED_INDEX_INPUTS_AUTHENTICATED"
                and task.get("training_authorized") is False
                and task.get("selection_authorized") is False
                and task.get("controls_materialized") is False
                and task.get("truth_loaded") is False
                and task.get("evaluate_opened") is False,
                "revised plan/task is not closed PREPARE input")
        require(task["flare"]["primary"].get("initial_gen") == "10"
                and task["flare"]["primary"].get("marker_condition") ==
                    "BASE_EXCLUDES_REVIEWED_EVENT_ALT_REF",
                "revised task FLARE policy differs")
        for name, desc in inp["full_material_identity"].get("files", {}).items():
            authenticated(desc["path"], desc["sha256"], pins)
        if "da_revised_receipt" in inp:
            desc = inp["da_revised_receipt"]
            authenticated(desc["path"], desc["sha256"], pins)
    for name in ("query_overlay", "query_overlay_receipt", "flare_projection_receipt"):
        desc = inp[name]
        authenticated(desc["path"], desc["sha256"], pins)
    overlay_receipt = json.loads(Path(inp["query_overlay_receipt"]["path"]).read_text())
    require(not overlay_receipt["truth_read_for_query_geometry"], "Truth-dependent query axis")
    require(overlay_receipt["output_sha256"] == inp["query_overlay"]["sha256"], "Overlay output binding")
    request_path = authenticated(inp["truthblind_request"], overlay_receipt["request_sha256"], pins)
    request = json.loads(request_path.read_text())
    require(request["case_id"] == task["case_id"] and request["direction"] == task["direction"], "Request identity")
    manifest_path = authenticated(inp["material_manifest"], request["material_manifest_sha256"], pins)
    manifest = json.loads(manifest_path.read_text())
    require(manifest["case_id"] == task["case_id"] and manifest["genome_build"] == "GRCh38", "Material identity")
    require(str(manifest["chromosome"]) == task["chromosome"], "Material chromosome")
    authenticated(inp["genotype_request"], manifest["request_sha256"], pins)
    source = manifest_path.parent
    for name in ("alt_dosage.npy", "genotype_observed.npy", "loci.tsv", "persons.tsv"):
        authenticated(source / name, manifest["outputs_sha256"][name], pins)
    require(request["persons_path_sha256"] == manifest["outputs_sha256"]["persons.tsv"], "Persons binding")
    _, persons = table(source / "persons.tsv")
    person_ids = np.asarray([row["sample_key_sha256"] for row in persons])
    require(len(set(person_ids)) == len(person_ids) == task["people"], "Person count/duplicates")
    if revised:
        expected_fold = {"FIT": "FOLD_A", "SELECT": "FOLD_B"}[task["role"]]
        require(manifest.get("donor_split") == expected_fold
                and inp["full_material_identity"].get("donor_split") == expected_fold
                and all(row.get("donor_fold") == expected_fold for row in persons),
                "donor_split/person donor_fold differs from role")
    _, loci = table(source / "loci.tsv")
    dosage = np.load(source / "alt_dosage.npy", mmap_mode="r", allow_pickle=False)
    observed = np.load(source / "genotype_observed.npy", mmap_mode="r", allow_pickle=False)
    require(dosage.shape == observed.shape == (len(persons), len(loci)), "Genotype dimensions")
    require(observed.dtype == np.bool_, "Expected explicit bool observed mask")
    for offset in range(0, len(loci), MAX_LOCUS_BLOCK):
        obs = observed[:, offset:offset + MAX_LOCUS_BLOCK]
        require(np.isin(dosage[:, offset:offset + MAX_LOCUS_BLOCK][obs], [0, 1, 2]).all(), "Invalid observed genotype")
    arrays = {"person_id": person_ids}
    counts = {}
    event_annotation_schema = event_annotations.LEGACY_SCHEMA
    event_catalogue_sha = inp["common_event_core_catalogues"]["event"]["sha256"]
    reference_identity = None
    if revised:
        require("reference_linkage_receipt" in inp,
                "revised EVENT requires the DM reference linkage receipt")
    if "reference_linkage_receipt" in inp:
        reference_identity = reference_linkage(
            inp["reference_linkage_receipt"]["path"],
            inp["reference_linkage_receipt"]["sha256"], task, event_catalogue_sha, pins)
    for channel in ("common", "event"):
        desc = inp["common_event_core_catalogues"][channel]
        path = authenticated(desc["path"], desc["sha256"], pins)
        fields, rows = table(path)
        require(len(rows) == desc["rows"], "Catalogue row count")
        require(all(str(row["chromosome"]) == task["chromosome"] for row in rows), "Catalogue chromosome")
        arrays[channel + "_source_index"] = catalogue_index(rows, loci)
        if channel == "common" and revised:
            require(all(row["counted_allele"] == "ALT" for row in rows),
                    "revised COMMON must always count ALT")
        arrays[channel + "_is_alt"] = np.asarray([
            event_annotations.counted_is_alt(row["ref"], row["alt"], row["counted_allele"])
            for row in rows], dtype=bool)
        for field in fields:
            arrays[channel + "_" + field] = np.asarray([row[field] for row in rows], dtype=str)
        if channel == "event":
            event_annotation_schema = event_annotations.catalogue_annotation_schema(fields)
            if revised:
                require(event_annotation_schema == event_annotations.SCHEMA,
                        "revised EVENT requires complete typed annotations")
            if event_annotation_schema == event_annotations.SCHEMA:
                arrays.update(event_annotations.build_index_arrays(fields, rows, np))
        counts[channel] = len(rows)
    if revised:
        common_indices = arrays["common_source_index"]
        event_indices = arrays["event_source_index"]
        require(len(np.intersect1d(common_indices, event_indices, assume_unique=True)) == 0
                and len(common_indices) + len(event_indices) == len(loci)
                and np.array_equal(np.sort(np.concatenate((common_indices, event_indices))),
                                   np.arange(len(loci), dtype=np.int64)),
                "revised COMMON/EVENT are not the exact disjoint FULL axis")
    with np.load(inp["query_overlay"]["path"], allow_pickle=False) as query:
        require(np.array_equal(query["person_id"], person_ids), "Query/person order mismatch")
        require(set(np.asarray(query["role"]).reshape(-1)) == {task["role"]}, "Query roles mismatch")
        # Explicit allowlist: truth_state and truth_observed are never loaded.
        for name in ("role", "component_id", "query_pos_bp", "query_cm", "query_chrom", "query_valid", "query_features"):
            value = query[name]
            require(value.shape[0] == len(persons), "Query person dimension: " + name)
        nquery = query["query_valid"].shape[1]
        require(nquery == inp["query_core"]["queries_per_person"], "Query count")
    flare = task["flare"]["primary"]
    authenticated(flare["local_path"], flare["sha256"], pins)
    projection = json.loads(Path(inp["flare_projection_receipt"]["path"]).read_text())
    require(projection["probabilities_sha256"] == flare["sha256"]
            and projection.get("truth_used_for_projection") is False, "FLARE binding")
    if revised:
        require(projection.get("schema") == "r02_lai_flare_query_projection_receipt_v1"
                and projection.get("status") == "EXISTING_FLARE_POSTERIOR_PROJECTED_NO_TRAINING_NO_SELECTION"
                and projection.get("initialization_label") == "gen10"
                and projection.get("marker_condition") == "BASE_EXCLUDES_REVIEWED_EVENT_ALT_REF"
                and projection.get("truth_arrays_loaded") is False
                and projection.get("selection_performed") is False,
                "revised FLARE receipt differs")
    require(projection["overlay_sha256"] == inp["query_overlay"]["sha256"], "FLARE query axis differs")
    require(projection["case_id"] == task["case_id"] and str(projection["chromosome"]) == task["chromosome"], "FLARE case/chromosome")
    require(projection["state_order"] == ["AA", "AE", "AN", "EE", "EN", "NN"], "FLARE state order")
    require(projection["probability_semantics"] == "UNORDERED_DIPLOID_PRODUCT_OF_FLARE_HAPLOTYPE_MARGINALS", "FLARE probability semantics")
    require(projection["people"] == len(persons) and projection["queries"] == nquery, "FLARE axis sizes")
    probabilities = np.load(flare["local_path"], mmap_mode="r", allow_pickle=False)
    require(probabilities.shape == (len(persons), nquery, 6), "FLARE dimensions")
    require(np.issubdtype(probabilities.dtype, np.floating), "FLARE probabilities must be floating point")
    claimed_residual = projection.get("maximum_projected_mass_residual")
    if revised:
        require(isinstance(claimed_residual, (int, float)) and np.isfinite(claimed_residual)
                and claimed_residual >= 0, "FLARE mass-residual bound absent")
    observed_residual = 0.0
    for begin in range(0, nquery, MAX_PROBABILITY_QUERY_BLOCK):
        block = probabilities[:, begin:begin + MAX_PROBABILITY_QUERY_BLOCK, :]
        require(np.isfinite(block).all() and ((block >= 0) & (block <= 1)).all(),
                "FLARE probability outside finite [0,1]")
        observed_residual = max(observed_residual,
                                float(np.max(np.abs(block.sum(axis=2, dtype=np.float64) - 1.0))))
    rounding = 8 * np.finfo(probabilities.dtype).eps
    permitted_residual = float(claimed_residual) if claimed_residual is not None else 0.0
    require(observed_residual <= permitted_residual + rounding,
            "FLARE probability mass differs from authenticated residual")
    np.savez(output / "indices.npz", **arrays)
    record = dict(schema=SCHEMA, status="PREDICTOR_INDEX_COMPLETE_NOT_TRAINING", task_id=task_id,
        case_id=task["case_id"], direction=task["direction"], role=task["role"], chromosome=task["chromosome"],
        people=len(persons), loci=counts, queries_per_person=nquery, plan_sha256=plan_hash,
        inputs_sha256=pins, genotype_directory=str(source), query_overlay=inp["query_overlay"],
        flare=flare, catalogue_scope=("EXPANDED_ALT_REF_WITH_VERSIONED_REFERENCE_AND_CALL_ANNOTATIONS"
            if event_annotation_schema == event_annotations.SCHEMA
            else "EXISTING_MATERIAL_ONLY_REF_MINOR_UNRESOLVED_LOCI_NOT_ADDED"),
        event_annotation_schema=event_annotation_schema,
        reference_linkage_receipt=inp.get("reference_linkage_receipt"),
        reference_identity=reference_identity,
        event_annotation_policy="LOCUS_LEVEL_NOT_CONCATENATED_WITH_TWELVE_EVENT_METADATA_FEATURES",
        truth_arrays_loaded=False, radius_selected=False, training_authorized=False,
        controls_materialized=False, evaluate_opened=False, elapsed_seconds=time.monotonic()-start,
        maximum_loci_per_read=MAX_LOCUS_BLOCK,
        maximum_probability_queries_validated_per_block=MAX_PROBABILITY_QUERY_BLOCK,
        maximum_observed_probability_mass_residual=observed_residual,
        outputs_sha256={"indices.npz": sha(output / "indices.npz")})
    temporary = output / "receipt.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, output / "receipt.json")
    return record


class PredictorIndex:
    """Authenticate once, then read selected people/loci without dense metadata copies."""
    def __init__(self, receipt_path):
        path = Path(receipt_path)
        self.receipt = json.loads(path.read_text())
        require(self.receipt["schema"] == SCHEMA, "Unknown predictor index")
        for filename, expected in self.receipt["inputs_sha256"].items():
            authenticated(filename, expected, {})
        authenticated(path.parent / "indices.npz", self.receipt["outputs_sha256"]["indices.npz"], {})
        with np.load(path.parent / "indices.npz", allow_pickle=False) as data:
            self.index = {name: data[name] for name in data.files}
        source = Path(self.receipt["genotype_directory"])
        self.alt = np.load(source / "alt_dosage.npy", mmap_mode="r", allow_pickle=False)
        self.observed = np.load(source / "genotype_observed.npy", mmap_mode="r", allow_pickle=False)

    def read_loci(self, channel, people, begin=0, end=None):
        require(channel in {"common", "event"}, "Unknown channel")
        indices = self.index[channel + "_source_index"]
        end = len(indices) if end is None else end
        require(0 <= begin <= end <= len(indices), "Invalid locus interval")
        maximum = self.receipt.get("maximum_loci_per_read")
        if maximum is not None:
            require(end - begin <= maximum, "Locus interval exceeds authenticated maximum block")
        people = np.asarray(people)
        require(people.ndim == 1 and np.issubdtype(people.dtype, np.integer), "Person indices must be integers")
        require(len(np.unique(people)) == len(people), "Duplicate person indices require an explicit sampling policy")
        require(np.all((people >= 0) & (people < self.alt.shape[0])), "Invalid person index")
        people = people.astype(np.int64, copy=False)
        selected = np.ix_(people, indices[begin:end])
        obs = self.observed[selected]
        dose = counted_dosage(self.alt[selected], obs, self.index[channel + "_is_alt"][begin:end])
        excluded = (set(event_annotations.NON_LOCUS_ARRAY_FIELDS)
                    if channel == "event" else set())
        metadata = {name[len(channel)+1:]: value[begin:end] for name, value in self.index.items()
                    if name.startswith(channel + "_") and name not in excluded}
        if channel == "event" and "event_annotation_schema" in self.index:
            metadata["annotation_schema"] = str(self.index["event_annotation_schema"].item())
            metadata["reference_ancestry_order"] = self.index["event_reference_ancestry_order"].astype(str)
            metadata["call_provenance_role_order"] = self.index["event_call_provenance_role_order"].astype(str)
            metadata["call_provenance_count_order"] = self.index["event_call_provenance_count_order"].astype(str)
        return dose, obs, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare(args.plan, args.task_id, args.output)
    print(json.dumps({name: result[name] for name in ("task_id", "status", "loci", "elapsed_seconds")}))
