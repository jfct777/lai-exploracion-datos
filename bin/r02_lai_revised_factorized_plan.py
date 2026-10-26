#!/usr/bin/env python3
"""Bind reviewed DA/DM/FULL/projection products into a factorized PREPARE plan.

The program does not read ancestry truth, create genotype matrices, materialize
controls, or authorize training.  Missing projection receipts remain explicit
dependencies; a runnable plan is written only after all twelve are authenticated.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


PLAN_SCHEMA = "r02_lai_training_inputs_plan_v1"
DEPENDENCY_SCHEMA = "r02_lai_revised_factorized_dependencies_v1"
DA_SCHEMA = "r02_df_revised_flare_base_plan_v1"
PROJECTION_SCHEMA = "r02_lai_flare_query_projection_receipt_v1"
DA_RECEIPT_SCHEMA = "r02_lai_expanded_alt_ref_case_v1"
ROLE_DONOR = {"FIT": "FOLD_A", "SELECT": "FOLD_B"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def descriptor(path, expected=None):
    path = Path(path).resolve()
    require(path.is_file(), "file absent: " + str(path))
    observed = sha(path)
    require(expected is None or observed == expected, "SHA-256 differs: " + str(path))
    return {"path": str(path), "sha256": observed, "bytes": path.stat().st_size}


def tsv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames is not None, "TSV header absent: " + str(path))
        return reader.fieldnames, list(reader)


def locus_key(row):
    return (str(row["chromosome"]), int(row["position_bp"]),
            row.get("ref", row.get("REF")), row.get("alt", row.get("ALT")))


def validate_partition(loci_path, common_path, event_path, expected):
    """Prove that revised COMMON and EVENT are an exact disjoint FULL partition."""
    _, loci = tsv(loci_path); common_fields, common = tsv(common_path)
    event_fields, event = tsv(event_path)
    full_keys = [locus_key(row) for row in loci]
    common_keys = [locus_key(row) for row in common]
    event_keys = [locus_key(row) for row in event]
    require(len(full_keys) == expected["FULL_SOURCE"]
            and len(common_keys) == expected["BASE"]
            and len(event_keys) == expected["EVENT_ALT_REF"], "catalogue/FULL row count differs")
    require(len(set(full_keys)) == len(full_keys)
            and len(set(common_keys)) == len(common_keys)
            and len(set(event_keys)) == len(event_keys), "duplicate locus key")
    require(set(common_keys).isdisjoint(event_keys), "COMMON and EVENT overlap")
    require(set(common_keys) | set(event_keys) == set(full_keys),
            "COMMON and EVENT are not the exact FULL locus union")
    source_order = {key: index for index, key in enumerate(full_keys)}
    require(all(source_order[a] < source_order[b] for a, b in zip(common_keys, common_keys[1:]))
            and all(source_order[a] < source_order[b] for a, b in zip(event_keys, event_keys[1:])),
            "catalogue order differs from FULL")
    require("counted_allele" in common_fields
            and all(row["counted_allele"] == "ALT" for row in common),
            "COMMON must always count ALT")
    required_event = {"counted_allele", "copyable_in_case_fold", "source_catalogue_sha256",
        "reference_African_ac", "reference_African_an", "reference_European_ac",
        "reference_European_an", "reference_Native_American_ac", "reference_Native_American_an",
        "ref_train_called_copies", "ref_train_original_observed_copies", "ref_train_completed_copies",
        "source_valid_called_copies", "source_valid_original_observed_copies",
        "source_valid_completed_copies", "event_origin", "phase_status", "quality_status"}
    require(required_event <= set(event_fields), "revised typed EVENT columns incomplete")
    return {"full": len(full_keys), "common": len(common_keys), "event": len(event_keys),
            "join_key": ["chromosome", "position_bp", "REF", "ALT"],
            "common_counted_allele": "ALT", "disjoint_exact_union": True,
            "order_preserved_within_channels": True}


def validate_dm(receipt, task, event_sha, event_rows):
    require(receipt.get("schema") == "r02_lai_reference_linkage_receipt_v1"
            and receipt.get("status") == "COUNTS_AND_ORIGINAL_UNITS_MATCH"
            and receipt.get("training_authorized") is False
            and receipt.get("controls_materialized") is False
            and receipt.get("truth_loaded") is False
            and receipt.get("evaluate_opened") is False, "DM receipt differs")
    rows = [row for row in receipt.get("cases", []) if row.get("task_id") == task["task_id"]]
    require(len(rows) == 1, "DM case absent or duplicated")
    row = rows[0]
    require(row.get("role") == task["role"] and row.get("event_catalogue_sha256") == event_sha
            and row.get("event_loci") == event_rows and row.get("target_people") == task["people"]
            and row.get("reference_join_key") == ["chromosome", "position_bp", "REF", "ALT"]
            and row.get("reference_duplicate_keys") == 0
            and row.get("reference_target_person_overlap") == 0
            and row.get("reference_target_original_unit_overlap") == 0
            and row.get("problems") == [], "DM case identity/checks differ")


def projection_dependency(root, task, expected_source_markers=None):
    directory = Path(root).resolve() / task["task_id"] / "projection"
    receipt_path, probabilities = directory / "receipt.json", directory / "baseline-probabilities.npy"
    result = {"receipt_path": str(receipt_path), "probabilities_path": str(probabilities),
              "status": "PENDING_RECEIPT_AND_HASH"}
    if not receipt_path.is_file() or not probabilities.is_file():
        return result, None
    receipt = load(receipt_path)
    require(receipt.get("schema") == PROJECTION_SCHEMA
            and receipt.get("status") == "EXISTING_FLARE_POSTERIOR_PROJECTED_NO_TRAINING_NO_SELECTION"
            and receipt.get("case_id") == task["case_id"]
            and str(receipt.get("chromosome")) == task["chromosome"]
            and receipt.get("people") == task["people"]
            and receipt.get("queries") == task["inputs"]["query_core"]["queries_per_person"]
            and (expected_source_markers is None
                 or receipt.get("source_markers") == expected_source_markers)
            and receipt.get("initialization_label") == "gen10"
            and receipt.get("marker_condition") == "BASE_EXCLUDES_REVIEWED_EVENT_ALT_REF"
            and receipt.get("state_order") == ["AA", "AE", "AN", "EE", "EN", "NN"]
            and receipt.get("truth_arrays_loaded") is False
            and receipt.get("truth_used_for_projection") is False
            and receipt.get("selection_performed") is False
            and receipt.get("overlay_sha256") == task["inputs"]["query_overlay"]["sha256"]
            and receipt.get("probability_semantics") ==
                "UNORDERED_DIPLOID_PRODUCT_OF_FLARE_HAPLOTYPE_MARGINALS",
            "revised FLARE projection receipt differs")
    probability = descriptor(probabilities, receipt.get("probabilities_sha256"))
    result.update(status="VERIFIED", receipt=descriptor(receipt_path), probabilities=probability)
    return result, receipt


def build(base_plan_path, da_plan_path, dm_paths, projection_root):
    base, da = load(base_plan_path), load(da_plan_path)
    require(base.get("schema") == PLAN_SCHEMA and base.get("direction") == "FOLD_A_TO_FOLD_B"
            and len(base.get("tasks", [])) == 12 and base.get("training_authorized") is False
            and base.get("evaluate_opened") is False, "base factorized plan differs")
    require(da.get("schema") == DA_SCHEMA and len(da.get("cases", [])) == 12
            and da.get("status") == "PREPARED_FOR_ROOT_REVIEW_NO_LAUNCH"
            and da.get("source_minor_definition") == "DNABR2723_FIXED"
            and da.get("truth_available_to_workflow") is False
            and da.get("evaluate_opened") is False and da.get("training_started") is False
            and da.get("inference_started") is False,
            "DA revised plan differs")
    da_cases = {row["task_id"]: row for row in da["cases"]}
    dm_by_chromosome = {str(int(load(path)["cases"][0]["task_id"].split(".chr", 1)[1].split(".", 1)[0])):
                        (Path(path).resolve(), load(path)) for path in dm_paths}
    require(set(dm_by_chromosome) == {"1", "22"}, "exact chr1/chr22 DM receipts required")
    tasks, dependencies = [], []
    for old in sorted(base["tasks"], key=lambda row: row["task_id"]):
        require(old["task_id"] in da_cases, "DA task absent")
        revised = da_cases[old["task_id"]]
        require(revised["case_id"] == old["case_id"] and revised["role"] == old["role"]
                and revised["chromosome"] == old["chromosome"], "DA/base task identity differs")
        require(old["role"] in ROLE_DONOR
                and revised["full_reuse"]["10"]["status"] == "EXACT_FULL_REUSE_AUTHENTICATED"
                and revised["full_reuse"]["10"]["record"]["role"] == old["role"]
                and revised["full_reuse"]["10"]["record"]["donor_split"] == ROLE_DONOR[old["role"]]
                and revised["full_reuse"]["10"]["record"]["initial_generation"] == 10,
                "DA gen10 donor split/role differs")
        counts = revised["marker_counts"]
        require(counts["BASE"] + counts["EVENT_ALT_REF"] == counts["FULL_SOURCE"],
                "revised COMMON/EVENT do not partition FULL")
        material = load(old["inputs"]["material_manifest"])
        require(material.get("case_id") == old["case_id"] and material.get("n_people") == old["people"]
                and material.get("n_loci") == counts["FULL_SOURCE"]
                and material.get("donor_split") == ROLE_DONOR[old["role"]]
                and material.get("outputs_sha256", {}).get("alt_dosage.npy")
                and material.get("outputs_sha256", {}).get("genotype_observed.npy"),
                "FULL material identity/count differs")
        material_path = Path(old["inputs"]["material_manifest"]).resolve()
        full_files = {}
        for name in ("alt_dosage.npy", "genotype_observed.npy", "loci.tsv", "persons.tsv"):
            full_files[name] = descriptor(material_path.parent / name,
                                          material["outputs_sha256"][name])
        common, event = revised["outputs"]["common.revised.tsv.gz"], revised["outputs"]["event.alt_ref.tsv.gz"]
        common = {**descriptor(common["path"], common["sha256"]), "rows": counts["BASE"]}
        event = {**descriptor(event["path"], event["sha256"]), "rows": counts["EVENT_ALT_REF"]}
        partition = validate_partition(full_files["loci.tsv"]["path"], common["path"],
                                       event["path"], counts)
        _, people_rows = tsv(full_files["persons.tsv"]["path"])
        require(len(people_rows) == old["people"]
                and all(row.get("donor_fold") == ROLE_DONOR[old["role"]] for row in people_rows),
                "persons donor_fold differs from task role")
        da_receipt_desc = descriptor(revised["receipt"]["path"], revised["receipt"]["sha256"])
        da_receipt = load(da_receipt_desc["path"])
        require(da_receipt.get("schema") == DA_RECEIPT_SCHEMA
                and da_receipt.get("status") == "PREPARED_NO_INFERENCE"
                and da_receipt.get("task_id") == old["task_id"]
                and da_receipt.get("role") == old["role"]
                and da_receipt.get("truth_read") is False
                and da_receipt.get("training_executed") is False
                and da_receipt.get("inference_executed") is False
                and da_receipt.get("evaluate_opened") is False,
                "DA case receipt differs")
        descriptor(old["inputs"]["query_overlay"]["path"], old["inputs"]["query_overlay"]["sha256"])
        descriptor(old["inputs"]["query_overlay_receipt"]["path"],
                   old["inputs"]["query_overlay_receipt"]["sha256"])
        dm_path, dm = dm_by_chromosome[old["chromosome"]]
        validate_dm(dm, old, event["sha256"], counts["EVENT_ALT_REF"])
        dependency, projection = projection_dependency(projection_root, old, counts["BASE"])
        dependencies.append({"task_id": old["task_id"], **dependency})
        task = json.loads(json.dumps(old))
        task["inputs"]["common_event_core_catalogues"] = {
            "common": common,
            "event": event,
        }
        task["inputs"]["full_material_identity"] = {
            "manifest": descriptor(material_path), "files": full_files,
            "loci": counts["FULL_SOURCE"], "people": old["people"], "partition": partition,
            "donor_split": ROLE_DONOR[old["role"]],
        }
        task["inputs"]["da_revised_receipt"] = da_receipt_desc
        task["inputs"]["reference_linkage_receipt"] = descriptor(dm_path)
        task["product_requests"] = ["REVISED_FLARE_PROJECTION_RECEIPT_AND_HASH_PENDING"]
        task["status"] = "WAITING_REVISED_FLARE_PROJECTION"
        task["training_authorized"] = False
        task["selection_authorized"] = False
        task["controls_materialized"] = False
        task["truth_loaded"] = False
        task["evaluate_opened"] = False
        if projection is not None:
            task["inputs"]["flare_projection_receipt"] = dependency["receipt"]
            task["flare"]["primary"] = {
                "initial_gen": "10", "use": "COMMON_INPUT_FOR_FIT_AND_SELECT",
                "marker_condition": "BASE_EXCLUDES_REVIEWED_EVENT_ALT_REF",
                "local_path": dependency["probabilities"]["path"],
                "sha256": dependency["probabilities"]["sha256"],
                "bytes": dependency["probabilities"]["bytes"],
                "projection_receipt_sha256": dependency["receipt"]["sha256"],
                "source_prediction_sha256": projection["source_prediction_sha256"],
                "projection_policy": projection["projection_policy"],
            }
            task["product_requests"] = ["REGIONAL_LAYOUT_NOT_MATERIALIZED", "CONTROLS_NOT_MATERIALIZED"]
            task["status"] = "REVISED_FACTORIZED_INDEX_INPUTS_AUTHENTICATED"
        tasks.append(task)
    ready = sum(row["status"] == "VERIFIED" for row in dependencies)
    inventory = {"schema": DEPENDENCY_SCHEMA,
                 "status": "COMPLETE" if ready == 12 else "WAITING_FOR_PROJECTIONS",
                 "verified": ready, "required": 12, "tasks": dependencies,
                 "training_authorized": False, "controls_materialized": False,
                 "truth_loaded": False, "evaluate_opened": False}
    plan = json.loads(json.dumps(base))
    plan.update(status="REVISED_FACTORIZED_INPUTS_READY" if ready == 12 else "WAITING_FOR_REVISED_PROJECTIONS",
                tasks=tasks, training_authorized=False, selection_authorized=False,
                evaluate_opened=False, controls_materialized=False)
    plan["scientific_rules"]["common"].update(
        catalogue="EXACT_BASELINE_OTHER_ROWS_MINUS_EXACT_REVIEWED_EVENT_REF_EXTENSION",
        extra_frequency_filter=None)
    plan["scientific_rules"]["event"].update(
        catalogue="DNABR2723_FIXED_REVIEWED_ALT_PLUS_REF_EVENT_CATALOGUE",
        target_frequency_filter=None, truth_filter=None)
    plan["sources"].update(base_plan=descriptor(base_plan_path), da_revised_plan=descriptor(da_plan_path),
                           dm_reference_receipts=[descriptor(path) for path in dm_paths])
    return inventory, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True); parser.add_argument("--expanded-plan", required=True)
    parser.add_argument("--reference-chr1", required=True); parser.add_argument("--reference-chr22", required=True)
    parser.add_argument("--projection-root", required=True)
    parser.add_argument("--dependency-output", required=True); parser.add_argument("--output")
    args = parser.parse_args()
    inventory, plan = build(args.source_plan, args.expanded_plan,
                            [args.reference_chr1, args.reference_chr22], args.projection_root)
    dep = Path(args.dependency_output); require(not dep.exists() and dep.parent.is_dir(), "fresh dependency output required")
    dep.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    if args.output:
        require(inventory["status"] == "COMPLETE", "twelve verified projections required before plan")
        out = Path(args.output); require(not out.exists() and out.parent.is_dir(), "fresh plan output required")
        out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": inventory["status"], "verified": inventory["verified"]}))


if __name__ == "__main__":
    main()
