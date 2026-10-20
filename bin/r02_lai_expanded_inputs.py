#!/usr/bin/env python3
"""Prepare the reviewed ALT+REF event catalogue and changed FLARE BASE input.

This program never reads truth and never runs FLARE.  It reuses the frozen
R02 LAI material producer's VCF and role helpers, while giving the ALT+REF
extension its own schema instead of claiming to be the original L1 catalogue.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import resource
import sys
import time
from collections import Counter
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path

import pysam

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r02_lai_flare_material as frozen_material
import r02_lai_simulate as simulation

SCHEMA = "r02_lai_expanded_alt_ref_input_v1"
BASE = "BASELINE_EXCLUDING_EXPANDED_DNABR_RARE_ALT_REF"
FULL = frozen_material.FULL
EXPECTED_PRE = "d40b64f46a090e757ae37b327de21e5055d75c476d13707d11ae89d1933dddcd"
EXPECTED_DECISION = "d9b1ef63cd45f88733d677b5596280b03775c6bd09e010f23dfd0625bd4dfe50"
EXPECTED_CHANGES = "0d922debe9b89f82b176ca3b01fd04f60db0f913f0734f208eb3ceeb3f9ae023"


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path | str):
    return json.loads(Path(path).read_text())


def dump(path: Path | str, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def require(value, message: str) -> None:
    if not value:
        raise ValueError(message)


def key(row):
    return (str(row["chromosome"]), int(row["position_bp"]),
            row.get("REF", row.get("ref")), row.get("ALT", row.get("alt")))


def table(path: Path | str):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def local_input(package: Path, descriptor: dict) -> Path:
    marker = "/opt/r02-lai-comparison-20261006i/"
    require(descriptor["path"].startswith(marker), "Unexpected frozen package path")
    path = package / descriptor["path"][len(marker):]
    require(path.is_file() and sha(path) == descriptor["sha256"], f"Frozen input changed: {path}")
    return path


def descriptor(path: Path | str) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def pinned(value: dict, label: str) -> Path:
    path = Path(value["path"])
    require(path.is_file() and sha(path) == value["sha256"], f"Pinned {label} changed: {path}")
    return path


def authenticate_full(registry_path: Path, cases: list[dict]) -> dict:
    registry = load(registry_path)
    records = [r for r in registry.get("records", [])
               if r.get("policy") == "ALL" and r.get("marker_condition") == FULL]
    expected = {(c["case_id"], g) for c in cases for g in (5, 10, 16)}
    observed = {(r.get("case_id"), r.get("initial_gen")) for r in records}
    defects = []
    if registry.get("status") != "COMPLETE_108_FITTED_MODELS_RANGE_VERIFIED":
        defects.append("registry_status")
    if len(records) != 36 or observed != expected:
        defects.append("record_axis")
    for r in records:
        if r.get("truth_accessed") is not False or r.get("generation_policy") != "ESTIMATE_PER_CHROMOSOME":
            defects.append(f"scope:{r.get('case_id')}:{r.get('initial_gen')}")
        for item in (r.get("archive"), r.get("archive_inventory"),
                     r.get("members", {}).get("manifest"), r.get("members", {}).get("inference_model")):
            if not item or not item.get("sha256") or not item.get("bytes"):
                defects.append(f"member:{r.get('case_id')}:{r.get('initial_gen')}")
                break
    # The compact registry authenticates all fitted records and archive members,
    # but does not retain FULL material marker identities/order.  Reuse remains
    # closed until those archived material manifests are supplied and checked.
    complete = not defects
    return {
        "schema": "r02_lai_full_reuse_authentication_v1",
        "status": ("RECORDS_36_AUTHENTICATED_FULL_INPUT_EQUIVALENCE_PENDING"
                   if complete else "BLOCKED_REGISTRY_DISCREPANCY"),
        "registry": {"path": str(registry_path.resolve()), "sha256": sha(registry_path)},
        "expected_records": 36,
        "authenticated_records": len(records) if complete else 0,
        "record_axis_complete": observed == expected,
        "archive_members_complete": complete,
        "exact_full_marker_identity_and_order_authenticated": False,
        "reuse_allowed": False,
        "missing_evidence": "Archived FULL material manifest binding reference/target marker identity and order",
        "defects": sorted(set(defects)),
        "truth_read": False,
    }


def locus_label(locus) -> str:
    return f"{locus[0]}:{locus[1]}:{locus[2]}:{locus[3]}"


def compute_copyability(cases: list[dict]) -> dict:
    """Read the pinned SOURCE_VALID phased VCF once per chromosome.

    A REF extension is copyable for a donor fold iff at least one selected
    SOURCE_VALID person carries allele index 0.  Missingness is not converted
    to REF, and the original family-component split is read unchanged.
    """
    result = {}
    for chrom in ("1", "22"):
        current = [c for c in cases if c["chromosome"] == chrom]
        role_hashes = {c["inputs"]["fold_roles"]["sha256"] for c in current}
        require(len(role_hashes) == 1, "Fold-role file differs within chromosome")
        fold_roles = pinned(current[0]["inputs"]["fold_roles"], "fold roles")
        with fold_roles.open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        chosen = {fold: {r["sample_id"] for r in rows if r["role"] == "SOURCE_VALID" and r["split"] == fold}
                  for fold in ("FOLD_A", "FOLD_B")}
        require(all(chosen.values()) and chosen["FOLD_A"].isdisjoint(chosen["FOLD_B"]), "Donor folds invalid")
        added_path = pinned(current[0]["changes"]["event_add_ref.tsv"], "added REF")
        added = {key(r) for r in table(added_path)}
        vcf = pinned(current[0]["metadata_sources"]["source_valid_phased_vcf"], "SOURCE_VALID phased VCF")
        iterator = simulation.vcf_rows(vcf); samples, sentinel = next(iterator)
        require(sentinel is None and set(samples) == chosen["FOLD_A"] | chosen["FOLD_B"], "SOURCE_VALID/fold sample axis differs")
        indices = {fold: [i for i, sample in enumerate(samples) if sample in ids] for fold, ids in chosen.items()}
        values = {}
        for row_samples, row in iterator:
            locus = (row[0].removeprefix("chr"), int(row[1]), row[3], row[4])
            if locus not in added: continue
            calls = simulation.genotypes(row)
            values[locus_label(locus)] = {fold: any(0 in calls[i] for i in idx)
                                          for fold, idx in indices.items()}
        require(set(values) == {locus_label(x) for x in added}, "Added REF copyability axis incomplete")
        result[chrom] = values
    return result


def make_plan(args) -> dict:
    pre, decision, changes = map(Path, (args.pre_resolution, args.decision, args.changes_receipt))
    require(sha(pre) == EXPECTED_PRE and sha(decision) == EXPECTED_DECISION and
            sha(changes) == EXPECTED_CHANGES, "PRE/decision/change receipt hash differs")
    d, receipt = load(decision), load(changes)
    require(d["training_allowed"] is False and d["truth_read"] is False and
            d["evaluate_opened"] is False and len(d["cases"]) == 12, "Decision scope changed")
    require(receipt["individual_genotypes_read"] is False and len(receipt["tasks"]) == 12,
            "Change inventory scope changed")
    by_task = {x["task_id"]: x for x in receipt["tasks"]}
    package = Path(args.package).resolve()
    change_root = changes.parent
    metadata_by_chrom = {}
    for chrom in ("1", "22"):
        product_root = Path(f"/mnt/disks/r02-neural-cx/lai-ref-integration-cy/products")
        metadata = {}
        for role in ("REF_TRAIN", "SOURCE_VALID"):
            product = product_root / f"chr{chrom}_{role}"
            product_receipt = product / "receipt.json"
            support = product / "ref_support_by_provenance.tsv.gz"
            pr = load(product_receipt)
            require(pr["outputs_sha256"][support.name] == sha(support), "Support receipt/output mismatch")
            metadata[role.lower() + "_receipt"] = descriptor(product_receipt)
            metadata[role.lower() + "_support"] = descriptor(support)
            if role == "SOURCE_VALID":
                source_vcf = next(Path(p) for p in pr["input_sha256"] if p.endswith(f"source_valid.chr{chrom}.vcf.gz"))
                require(pr["input_sha256"][str(source_vcf)] == sha(source_vcf), "SOURCE_VALID phased VCF changed")
                metadata["source_valid_phased_vcf"] = descriptor(source_vcf)
        source = Path(f".claude/runs/r02-lai-material-decision-20261004o/results/chr{chrom}/decision/candidate_alleles.tsv.gz").resolve()
        metadata["source_minor"] = descriptor(source)
        metadata_by_chrom[chrom] = metadata
    cases = []
    for chosen in d["cases"]:
        task = by_task[chosen["task_id"]]
        require(task == chosen, f"Decision/change task mismatch: {chosen['task_id']}")
        case_id = chosen["task_id"].removesuffix(".FOLD_A_TO_FOLD_B")
        contract_path = package / "contracts" / f"{case_id}.ALL.json"
        contract = load(contract_path)
        require(contract["schema"] == frozen_material.SCHEMA and
                contract["chromosome"] == chosen["chromosome"] and
                contract["donor_split"] in case_id, "Frozen contract task mismatch")
        inputs = {name: descriptor(local_input(package, value)) for name, value in contract["inputs"].items()}
        cdir = change_root / chosen["task_id"]
        change_files = {}
        for name, expected in chosen["changes"].items():
            p = cdir / name
            require(p.is_file() and sha(p) == expected, f"Change TSV differs: {p}")
            change_files[name] = descriptor(p)
        common = next(x for x in receipt["sources"] if x.get("path", "").endswith(f"/{chosen['task_id']}/common.tsv.gz"))
        event = next(x for x in receipt["sources"] if x.get("path", "").endswith(f"/{chosen['task_id']}/event.tsv.gz"))
        for item in (common, event):
            require(sha(item["path"]) == item["sha256"], "Old catalogue changed")
        cases.append({
            "task_id": chosen["task_id"], "case_id": case_id, "chromosome": chosen["chromosome"],
            "role": chosen["role"], "donor_split": contract["donor_split"],
            "expected": {k: chosen[k] for k in ("old_base_loci", "old_event_loci", "added_ref_loci",
                                                  "expanded_event_loci", "required_base_loci",
                                                  "expanded_event_identity_sha256")},
            "contract": {"path": str(contract_path.resolve()), "sha256": sha(contract_path)},
            "inputs": inputs, "changes": change_files,
            "old_common": common, "old_event": event,
            "metadata_sources": metadata_by_chrom[chosen["chromosome"]],
            "parameters": contract["parameters"], "initial_generations": contract["initial_generations"],
        })
    require(Counter(c["role"] for c in cases) == {"FIT": 6, "SELECT": 6}, "Role axis changed")
    auth = authenticate_full(Path(args.full_registry), cases)
    copyability = compute_copyability(cases)
    plan = {
        "schema": SCHEMA, "status": "PREPARATION_ADMITTED_NO_INFERENCE",
        "pre_resolution": {"path": str(pre.resolve()), "sha256": sha(pre)},
        "decision": {"path": str(decision.resolve()), "sha256": sha(decision)},
        "changes_receipt": {"path": str(changes.resolve()), "sha256": sha(changes)},
        "cases": cases, "full_authentication": auth, "ref_copyability_by_original_donor_fold": copyability,
        "resources": {"cpus_per_case": 1, "memory_mib_per_case": 1536,
                      "nextflow_jvm_memory_mib": 512, "max_parallel_cases": 1,
                      "wall_seconds_per_case": 1200, "campaign_backup_wall_seconds": 7200},
        "truth_read": False, "evaluate_opened": False, "inference_allowed": False,
        "training_allowed": False,
    }
    out = Path(args.output).resolve()
    require(not out.exists() and out.parent.is_dir(), "Fresh plan path required")
    dump(out, plan)
    dump(out.with_name("FULL_AUTHENTICATION.json"), auth)
    return plan


def load_support(case: dict):
    values = {}
    for role in ("REF_TRAIN", "SOURCE_VALID"):
        path = pinned(case["metadata_sources"][role.lower() + "_support"], role + " support")
        for row in table(path):
            values[(key(row), role, row["panel_ancestry"])] = row
    source = pinned(case["metadata_sources"]["source_minor"], "source_minor")
    source_rows = {key(r): r for r in table(source) if r["panel_ancestry"] == "ALL"}
    return source, source_rows, values


def validate_case_sources(case: dict) -> dict[str, Path]:
    pinned(case["contract"], "contract")
    paths = {name: pinned(value, "input " + name) for name, value in case["inputs"].items()}
    for name, value in case["changes"].items(): pinned(value, "change " + name)
    pinned(case["old_common"], "old common")
    pinned(case["old_event"], "old event")
    for name, value in case["metadata_sources"].items(): pinned(value, "metadata " + name)
    return paths


def prepare_case(plan_path: Path, plan_sha: str, task_id: str, output: Path) -> dict:
    started = time.monotonic()
    require(sha(plan_path) == plan_sha, "Plan hash differs")
    plan = load(plan_path)
    require(plan["schema"] == SCHEMA and plan["inference_allowed"] is False and
            plan["truth_read"] is False, "Preparation-only plan required")
    case = next((c for c in plan["cases"] if c["task_id"] == task_id), None)
    require(case is not None and not output.exists() and output.parent.is_dir(), "Unknown task or non-fresh output")
    inputs = validate_case_sources(case)
    exp = case["expected"]
    expanded_rows = list(table(pinned(case["changes"]["event_expanded.tsv"], "expanded event")))
    expanded = {key(r) for r in expanded_rows}
    added = {key(r) for r in table(pinned(case["changes"]["event_add_ref.tsv"], "added REF"))}
    old_event_path = pinned(case["old_event"], "old event")
    old_common_path = pinned(case["old_common"], "old common")
    old_event_rows = list(table(old_event_path))
    old_event = {key(r) for r in old_event_rows}
    require(len(expanded) == exp["expanded_event_loci"] and len(added) == exp["added_ref_loci"] and
            len(old_event) == exp["old_event_loci"] and expanded == old_event | added and
            not old_event & added, "Expanded EVENT exact union differs")
    source_path, source_rows, support = load_support(case)
    old_common_rows = list(table(old_common_path))
    common_by_key = {key(r): r for r in old_common_rows}
    require(added <= set(common_by_key), "Added REF absent from old BASE")
    output.mkdir(mode=0o700)
    # Revised COMMON preserves every original field and removes exactly the added REF keys.
    common_fields = list(old_common_rows[0])
    with gzip.open(output / "common.revised.tsv.gz", "wt") as stream:
        writer = csv.DictWriter(stream, common_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(r for r in old_common_rows if key(r) not in added)
    # Revised EVENT preserves the original reader columns and appends explicit provenance.
    original_fields = list(old_event_rows[0])
    extension_fields = ["event_origin", "ref_train_called_copies", "ref_train_original_observed_copies", "ref_train_completed_copies",
        "source_valid_called_copies", "source_valid_original_observed_copies", "source_valid_completed_copies",
        "phase_status", "quality_status"]
    catalogue_fields = original_fields + [x for x in extension_fields if x not in original_fields]
    with gzip.open(output / "event.alt_ref.tsv.gz", "wt") as stream:
        writer = csv.DictWriter(stream, catalogue_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for locus in sorted(expanded):
            if locus in old_event:
                old = next(r for r in old_event_rows if key(r) == locus)
                row = dict(old)
                row.update({"event_origin": "ORIGINAL_ALT_EVENT",
                    "phase_status": "INHERITED_ORIGINAL_ALT_METADATA", "quality_status": "INHERITED_ORIGINAL_ALT_METADATA"}
                )
            else:
                s = source_rows[locus]; common = common_by_key[locus]
                require(s["counted_allele"] == locus[2], "Added event is not fixed source_minor REF")
                row = {"chromosome": locus[0], "position_bp": locus[1], "ref": locus[2], "alt": locus[3],
                    "position_cm": common["position_cm"], "core": common["core"], "counted_allele": s["counted_allele"],
                    "source_minor_ac": s["source_rare_ac"], "source_an": s["source_rare_an"],
                    "copyable_in_case_fold": str(plan["ref_copyability_by_original_donor_fold"]
                        [case["chromosome"]][locus_label(locus)][case["donor_split"]]),
                    "event_origin": "REVIEWED_REF_EXTENSION", "source_catalogue_sha256": s["source_catalogue_sha256"],
                    "phase_status": "OBSERVED_PHASE_NOT_VALIDATED", "quality_status": "CALLED_GT_ONLY_NOT_QUALITY_VALIDATED"}
                for role, prefix in (("REF_TRAIN", "ref_train"), ("SOURCE_VALID", "source_valid")):
                    q = support[(locus, role, "ALL")]
                    row[prefix + "_called_copies"] = q["ac_counted_called"]
                    row[prefix + "_original_observed_copies"] = q["ac_counted_original_observed"]
                    row[prefix + "_completed_copies"] = q["ac_counted_missing_before_called_after"]
                for ancestry in ("African", "European", "Native_American"):
                    q = support[(locus, "REF_TRAIN", ancestry)]
                    row["reference_" + ancestry + "_ac"] = q["ac_counted_called"]
                    row["reference_" + ancestry + "_an"] = q["an_called"]
            writer.writerow(row)
    sim_manifest = load(inputs["simulation_manifest"])
    ref_it, target_it = simulation.vcf_rows(inputs["reference_vcf"]), simulation.vcf_rows(inputs["target_vcf"])
    reference_samples, marker = next(ref_it); target_samples, marker2 = next(target_it)
    require(marker is marker2 is None, "VCF header mismatch")
    refs, donors, units, order = frozen_material.role_axis(inputs, reference_samples, target_samples,
                                                           case["donor_split"], sim_manifest)
    pair = output / "BASE"; pair.mkdir()
    kept = removed = 0
    genotype_separators = {"reference_phased": 0, "reference_unphased": 0, "reference_missing_tokens": 0,
                           "target_phased": 0, "target_unphased": 0, "target_missing_tokens": 0}
    with ExitStack() as stack:
        rw = stack.enter_context(pysam.BGZFile(str(pair / "reference.vcf.gz"), "wb"))
        tw = stack.enter_context(pysam.BGZFile(str(pair / "target.vcf.gz"), "wb"))
        header = (f"##fileformat=VCFv4.2\n##contig=<ID={case['chromosome']}>\n"
                  "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Original phased genotype\">\n")
        rw.write((header + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" +
                  "\t".join(reference_samples[i] for i in order) + "\n").encode())
        tw.write((header + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" +
                  "\t".join(target_samples) + "\n").encode())
        inv = csv.writer(stack.enter_context((pair / "marker_inventory.tsv").open("x")),
                         delimiter="\t", lineterminator="\n")
        inv.writerow(("chromosome", "position_bp", "ref", "alt", "marker_channel"))
        last = 0
        for a, b in zip_longest(ref_it, target_it):
            require(a is not None and b is not None, "Reference/target record count differs")
            rs, rr = a; ts, tr = b
            locus = (rr[0], int(rr[1]), rr[3], rr[4])
            require(rs == reference_samples and ts == target_samples and
                    locus == (tr[0], int(tr[1]), tr[3], tr[4]) and locus[1] > last,
                    "Reference/target identity/order differs")
            require(rr[8] == tr[8] == "GT" and tr[7] == ".", "GT-only predictor required")
            last = locus[1]
            if locus in expanded:
                removed += 1
                continue
            ref_gt = "\t".join(rr[9:]); target_gt = "\t".join(tr[9:])
            genotype_separators["reference_phased"] += ref_gt.count("|")
            genotype_separators["reference_unphased"] += ref_gt.count("/")
            genotype_separators["reference_missing_tokens"] += ref_gt.count(".")
            genotype_separators["target_phased"] += target_gt.count("|")
            genotype_separators["target_unphased"] += target_gt.count("/")
            genotype_separators["target_missing_tokens"] += target_gt.count(".")
            refrow = [*rr[:7], ".", "GT", *(rr[9:][i] for i in order)]
            rw.write(("\t".join(refrow) + "\n").encode())
            tw.write(("\t".join(tr) + "\n").encode())
            inv.writerow((*locus, "BASELINE_OTHER")); kept += 1
    common_revised_rows = sum(1 for _ in table(output / "common.revised.tsv.gz"))
    event_revised_rows = sum(1 for _ in table(output / "event.alt_ref.tsv.gz"))
    require(removed == exp["expanded_event_loci"] and kept == exp["required_base_loci"] and
            common_revised_rows == kept and event_revised_rows == len(expanded),
            "BASE denominator differs")
    for name in ("reference.vcf.gz", "target.vcf.gz"):
        pysam.tabix_index(str(pair / name), preset="vcf", csi=True)
    with (pair / "sample_map.tsv").open("x") as stream:
        for i in order:
            stream.write(reference_samples[i] + "\t" + frozen_material.ANCESTRY[refs[i]["ancestry"]] + "\n")
    with (pair / "roles.tsv").open("x") as stream:
        w = csv.writer(stream, delimiter="\t", lineterminator="\n")
        w.writerow(("sample_id", "role", "atomic_unit_id", "ancestry"))
        for i in order:
            w.writerow((reference_samples[i], "REF_TRAIN", refs[i]["atomic_unit_id"],
                        frozen_material.ANCESTRY[refs[i]["ancestry"]]))
        for sample in target_samples:
            for unit, ancestry in units:
                w.writerow((sample, "SIMULATED_TARGET", unit, ancestry))
    preparation = {
        "schema": "r02_lai_expanded_alt_ref_case_v1", "status": "PREPARED_NO_INFERENCE",
        "plan_sha256": plan_sha, "producer_code_sha256": sha(__file__),
        "task_id": task_id, "role": case["role"], "chromosome": case["chromosome"],
        "marker_counts": {"EVENT_ALT_REF": len(expanded), "BASE": kept, "FULL_SOURCE": kept + len(expanded)},
        "source_minor_definition": "DNABR2723_FIXED", "source_minor_recalculated": False,
        "reference_people": len(reference_samples), "simulated_people": len(target_samples),
        "original_donor_people": len(donors), "reference_columns_reordered_as_frozen_producer": True,
        "genotypes_modified": False, "phase_modified": False, "missingness_modified": False,
        "genotype_copy_rule": "Retained VCF sample fields copied byte-for-byte; reference columns only use frozen producer order",
        "retained_genotype_separator_counts": genotype_separators,
        "truth_read": False, "evaluate_opened": False, "inference_executed": False, "training_executed": False,
        "parameters_preserved_no_run": case["parameters"], "initial_generations_preserved_no_run": case["initial_generations"],
        "full_reuse_allowed": False,
        "input_descriptors": case["inputs"],
        "catalogue_sources": {"old_event": case["old_event"], "old_common": case["old_common"],
                              "metadata": case["metadata_sources"]},
        "source_context": {"simulation": {k: sim_manifest[k] for k in ("schema", "status", "scope", "donor_split", "input_contract_sha256")},
                           "parental": {k: load(inputs["parental_manifest"])[k] for k in ("schema", "status", "scope", "chromosome")},
                           "original_roles": case["inputs"]["original_roles"],
                           "fold_roles": case["inputs"]["fold_roles"]},
        "outputs_sha256": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    validate_case_sources(case)
    dump(output / "receipt.json", preparation)
    return preparation


def validate_case(plan_path: Path, plan_sha: str, case: dict, case_dir: Path,
                  legacy_code_sha: str | None = None) -> dict:
    require(sha(plan_path) == plan_sha, "Plan hash differs during case validation")
    validate_case_sources(case)
    receipt_path = case_dir / "receipt.json"
    require(receipt_path.is_file(), f"Incomplete case has no receipt: {case['task_id']}")
    receipt = load(receipt_path)
    require(receipt.get("schema") == "r02_lai_expanded_alt_ref_case_v1" and
            receipt.get("status") == "PREPARED_NO_INFERENCE" and
            receipt.get("task_id") == case["task_id"] and receipt.get("role") == case["role"] and
            receipt.get("chromosome") == case["chromosome"], "Case receipt identity/status differs")
    require(receipt.get("source_minor_definition") == "DNABR2723_FIXED" and
            receipt.get("source_minor_recalculated") is False and receipt.get("truth_read") is False and
            receipt.get("evaluate_opened") is False and receipt.get("inference_executed") is False and
            receipt.get("training_executed") is False and receipt.get("genotypes_modified") is False and
            receipt.get("phase_modified") is False and receipt.get("missingness_modified") is False,
            "Case biological/scope invariants differ")
    require(receipt.get("marker_counts") == {"EVENT_ALT_REF": case["expected"]["expanded_event_loci"],
            "BASE": case["expected"]["required_base_loci"],
            "FULL_SOURCE": case["expected"]["required_base_loci"] + case["expected"]["expanded_event_loci"]},
            "Case marker denominators differ")
    require(receipt.get("input_descriptors") == case["inputs"] and
            receipt.get("catalogue_sources", {}).get("old_event") == case["old_event"] and
            receipt.get("catalogue_sources", {}).get("old_common") == case["old_common"] and
            receipt.get("catalogue_sources", {}).get("metadata") == case["metadata_sources"],
            "Case receipt source descriptors differ")
    producer = receipt.get("producer_code_sha256") or legacy_code_sha
    allowed_producers = {sha(__file__)}
    if legacy_code_sha is not None and case["case_id"].endswith("chr1.FOLD_A.r1"):
        allowed_producers.add(legacy_code_sha)
    require(producer in allowed_producers, "Producer code hash is not frozen/authorized")
    if receipt.get("plan_sha256") is not None:
        require(receipt["plan_sha256"] == plan_sha, "Receipt plan hash differs")
    required = {"common.revised.tsv.gz", "event.alt_ref.tsv.gz", "BASE/reference.vcf.gz",
        "BASE/reference.vcf.gz.csi", "BASE/target.vcf.gz", "BASE/target.vcf.gz.csi",
        "BASE/marker_inventory.tsv", "BASE/sample_map.tsv", "BASE/roles.tsv"}
    outputs = receipt.get("outputs_sha256", {})
    require(required <= set(outputs), "Required output missing from receipt")
    for relative, expected in outputs.items():
        path = (case_dir / relative).resolve()
        require(path.is_relative_to(case_dir.resolve()), "Output path escapes case directory")
        require(path.is_file() and sha(path) == expected, f"Output hash differs: {path}")
    validate_case_sources(case)
    verified = {"schema": "r02_lai_expanded_case_verified_v1", "status": "CASE_OUTPUTS_AND_SOURCES_VERIFIED",
        "task_id": case["task_id"], "role": case["role"], "plan_sha256": plan_sha,
        "receipt": {"path": str(receipt_path.resolve()), "sha256": sha(receipt_path)},
        "producer_code_sha256": producer, "verified_output_files": len(outputs),
        "truth_read": False, "inference_executed": False, "training_executed": False}
    marker = case_dir / "CASE_VERIFIED.json"
    if marker.exists():
        require(load(marker) == verified, "Existing case verification differs")
    else:
        dump(marker, verified)
    return verified


def campaign_status(args) -> dict:
    plan_path = Path(args.plan); plan_sha = args.plan_sha256
    require(sha(plan_path) == plan_sha, "Campaign plan hash differs")
    plan = load(plan_path); results = Path(args.results).resolve()
    pending, verified = [], []
    for case in plan["cases"]:
        case_dir = results / case["task_id"]
        if not case_dir.exists():
            pending.append(case["task_id"]); continue
        verified.append(validate_case(plan_path, plan_sha, case, case_dir,
                                      args.legacy_first_code_sha if case["case_id"].endswith("chr1.FOLD_A.r1") else None))
    require(not args.require_complete or not pending, f"Campaign incomplete: {len(pending)} cases")
    result = {"schema": "r02_lai_expanded_campaign_status_v1",
        "status": "TWELVE_CASES_VERIFIED" if not pending else "VERIFIED_EXISTING_WITH_PENDING_CASES",
        "plan_sha256": plan_sha, "verified_cases": len(verified), "pending_cases": len(pending),
        "pending_task_ids": pending, "case_verifications": verified,
        "full_reuse_allowed": False, "truth_read": False, "inference_executed": False, "training_executed": False}
    output = Path(args.output)
    require(not output.exists(), "Fresh campaign status path required")
    dump(output, result); return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    for name in ("pre-resolution", "decision", "changes-receipt", "package", "full-registry", "output"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("prepare-case")
    p.add_argument("--plan", required=True); p.add_argument("--plan-sha256", required=True)
    p.add_argument("--task-id", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("campaign-status")
    p.add_argument("--plan", required=True); p.add_argument("--plan-sha256", required=True)
    p.add_argument("--results", required=True); p.add_argument("--output", required=True)
    p.add_argument("--legacy-first-code-sha")
    p.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(); os.umask(0o077)
    if args.command == "plan":
        result = make_plan(args)
    elif args.command == "prepare-case":
        result = prepare_case(Path(args.plan), args.plan_sha256, args.task_id, Path(args.output))
    else:
        result = campaign_status(args)
    print(json.dumps({k: result[k] for k in result if k in {"status", "schema", "task_id", "marker_counts"}}, sort_keys=True))


if __name__ == "__main__":
    main()
