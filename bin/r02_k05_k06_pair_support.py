#!/usr/bin/env python3
"""Integrate K05 partition labels with authentic K06 reported IBD dyads.

This program computes denominators only.  It does not refit the 1,368 K05
descriptions, rank partitions, calibrate the IBD detector, or treat a partition
label as a population label.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path


ANCESTRY_FIELDS = (
    "Autosomes_African_anc",
    "Autosomes_European_anc",
    "Autosomes_Indigenous_anc",
    "Autosomes_EastAsian_anc",
)
THRESHOLDS = ("0.0221", "0.0442")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_file(), "Missing input: " + str(path))
    return {"path": str(path), "sha256": sha(path)}


def authenticated(item, name):
    require(set(item) == {"path", "sha256"}, "Exact descriptor required: " + name)
    path = Path(item["path"])
    require(path.is_absolute() and path.is_file(), "Missing authenticated input: " + name)
    require(sha(path) == item["sha256"], "Authenticated input differs: " + name)
    return path


def build_plan(individuals, community_summary, ibd_root, output):
    summary_path = Path(community_summary).resolve()
    summary = json.loads(summary_path.read_text())
    require(summary["n_partitions"] == 57, "Expected the 57 recorded K05 partitions")
    chromosomes = {}
    sample_identity = None
    for chrom in range(1, 23):
        base = Path(ibd_root).resolve() / f"chr{chrom}"
        contract_path = base / "ibd_contract.json"
        contract = json.loads(contract_path.read_text())
        require(
            contract["schema"] == "r02_segment_ibd_v2"
            and contract["genome_build"] == "GRCh38"
            and contract["coordinate_system"] == "1-based-inclusive"
            and contract["semantics"] == "ANY_COPY_REPORTED"
            and contract["callset_scope"]
            == "complete_reported_calls_for_declared_chromosome_and_samples"
            and contract["chromosome"] == str(chrom),
            f"K06 contract differs for chromosome {chrom}",
        )
        current_identity = contract["analytical_samples_sha256"]
        sample_identity = current_identity if sample_identity is None else sample_identity
        require(current_identity == sample_identity, "K06 analytical sample identity differs")
        intervals = base / contract["intervals"]["path"]
        require(sha(intervals) == contract["intervals"]["sha256"], "K06 intervals differ")
        chromosomes[str(chrom)] = {
            "contract": descriptor(contract_path),
            "intervals": descriptor(intervals),
        }
    plan = {
        "schema": "r02_k05_k06_pair_support_plan_v1",
        "scope": "AUTHENTIC_REPORTED_DYAD_DENOMINATORS_NO_REFIT_RANKING_OR_CALIBRATION",
        "question": (
            "For each recorded K05 partition, which people and K06 reported IBD dyads "
            "are jointly observable on the authenticated 22-autosome callset?"
        ),
        "people": descriptor(individuals),
        "community_summary": descriptor(summary_path),
        "chromosomes": chromosomes,
        "analytical_samples_sha256": sample_identity,
        "ancestry_fields": list(ANCESTRY_FIELDS),
        "pc_relate_thresholds": list(THRESHOLDS),
        "code_sha256": sha(Path(__file__)),
    }
    output = Path(output)
    require(not output.exists(), "Plan output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def read_people(path, partition_ids, ancestry_fields, thresholds):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"sample_id", *partition_ids, *ancestry_fields}
        required.update("component_phi_" + threshold for threshold in thresholds)
        require(required <= set(reader.fieldnames or []), "K05 people columns differ")
        rows = list(reader)
    by_id = {row["sample_id"]: row for row in rows}
    require(rows and len(by_id) == len(rows), "Empty or duplicate K05 person roster")
    return rows, by_id


def read_reported_dyads(plan, people):
    pairs = defaultdict(lambda: {"rows": 0, "length_cm": 0.0, "chromosomes": set()})
    total_rows = 0
    for chrom in map(str, range(1, 23)):
        item = plan["chromosomes"][chrom]
        contract_path = authenticated(item["contract"], "K06 contract chr" + chrom)
        intervals_path = authenticated(item["intervals"], "K06 intervals chr" + chrom)
        contract = json.loads(contract_path.read_text())
        require(
            contract["schema"] == "r02_segment_ibd_v2"
            and contract["chromosome"] == chrom
            and contract["analytical_samples_sha256"] == plan["analytical_samples_sha256"]
            and contract["intervals"]["sha256"] == sha(intervals_path),
            "K06 identity differs for chromosome " + chrom,
        )
        with gzip.open(intervals_path, "rt", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            require(
                list(reader.fieldnames or [])
                == ["chrom", "sample_a", "sample_b", "start_pos", "end_pos", "ibd_state", "hap_a", "hap_b", "lod", "length_cm"],
                "K06 interval columns differ",
            )
            for row in reader:
                total_rows += 1
                require(row["chrom"] == chrom and row["ibd_state"] == "ANY_COPY_REPORTED", "K06 interval semantics differ")
                a, b = sorted((row["sample_a"], row["sample_b"]))
                require(a != b and a in people and b in people, "K06 dyad outside K05 person roster")
                start, end = int(row["start_pos"]), int(row["end_pos"])
                length = float(row["length_cm"])
                require(0 < start <= end and length >= 0, "Malformed K06 interval")
                item_pair = pairs[(a, b)]
                item_pair["rows"] += 1
                item_pair["length_cm"] += length
                item_pair["chromosomes"].add(chrom)
    require(total_rows > 0 and pairs, "No K06 reported dyads")
    return pairs, total_rows


def pair_fingerprint(keys):
    digest = hashlib.sha256()
    for a, b in sorted(keys):
        digest.update(a.encode() + b"\t" + b.encode() + b"\n")
    return digest.hexdigest()


def run(plan_path, plan_sha256, output):
    require(sha(plan_path) == plan_sha256, "Plan changed")
    plan = json.loads(Path(plan_path).read_text())
    require(
        plan.get("schema") == "r02_k05_k06_pair_support_plan_v1"
        and plan.get("scope")
        == "AUTHENTIC_REPORTED_DYAD_DENOMINATORS_NO_REFIT_RANKING_OR_CALIBRATION",
        "Unsupported plan",
    )
    require(sha(Path(__file__)) == plan["code_sha256"], "Implementation changed")
    people_path = authenticated(plan["people"], "K05 people")
    summary_path = authenticated(plan["community_summary"], "K05 summary")
    summary = json.loads(summary_path.read_text())
    partitions = summary["partition_ledger"]
    partition_ids = [item["partition_id"] for item in partitions]
    require(summary["n_partitions"] == 57 == len(partition_ids), "K05 partition inventory differs")
    require(summary["kinship_thresholds"] == [float(x) for x in plan["pc_relate_thresholds"]], "PC-Relate thresholds differ")
    rows, by_id = read_people(
        people_path, partition_ids, plan["ancestry_fields"], plan["pc_relate_thresholds"]
    )
    require(len(rows) == summary["n_cohort"] == 2619, "K05 cohort size differs")
    reported_pairs, total_rows = read_reported_dyads(plan, by_id)

    output = Path(output)
    require(not output.exists(), "Output already exists")
    output.mkdir(parents=True)
    partition_rows = []
    support_groups = defaultdict(list)
    for item in partitions:
        partition_id = item["partition_id"]
        eligible = {
            row["sample_id"]
            for row in rows
            if int(row[partition_id]) >= 0
            and all(row[field] != "NA" for field in plan["ancestry_fields"])
        }
        keys = [key for key in reported_pairs if set(key) <= eligible]
        same_label = sum(by_id[a][partition_id] == by_id[b][partition_id] for a, b in keys)
        segment_rows = sum(reported_pairs[key]["rows"] for key in keys)
        length_cm = sum(reported_pairs[key]["length_cm"] for key in keys)
        row = {
            "partition_id": partition_id,
            "family": item["family"],
            "graph_id": item["graph_id"],
            "resolution": item["resolution"],
            "eligible_people": len(eligible),
            "people_in_reported_dyads": len({person for key in keys for person in key}),
            "reported_unique_dyads": len(keys),
            "reported_segment_rows": segment_rows,
            "reported_length_cm_sum_not_independent": repr(length_cm),
            "same_partition_label_dyads": same_label,
            "different_partition_label_dyads": len(keys) - same_label,
            "pair_support_sha256": pair_fingerprint(keys),
        }
        for threshold in plan["pc_relate_thresholds"]:
            field = "component_phi_" + threshold
            same_component = sum(by_id[a][field] == by_id[b][field] for a, b in keys)
            row["same_pc_relate_component_phi_" + threshold] = same_component
            row["different_pc_relate_component_phi_" + threshold] = len(keys) - same_component
        partition_rows.append(row)
        support_groups[row["pair_support_sha256"]].append(partition_id)

    table_path = output / "partition_reported_pair_support.tsv"
    with table_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(partition_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(partition_rows)

    pair_group_rows = [
        {
            "pair_support_sha256": fingerprint,
            "partition_count": len(ids),
            "partition_ids": ids,
            "reported_unique_dyads": next(
                row["reported_unique_dyads"]
                for row in partition_rows
                if row["partition_id"] == ids[0]
            ),
        }
        for fingerprint, ids in sorted(support_groups.items(), key=lambda item: (-len(item[1]), item[1]))
    ]
    group_path = output / "exact_pair_support_groups.json"
    group_path.write_text(json.dumps(pair_group_rows, indent=2) + "\n")
    result = {
        "schema": "r02_k05_k06_pair_support_result_v1",
        "status": "COMPLETE_AUTHENTIC_K05_K06_REPORTED_PAIR_DENOMINATORS",
        "people": len(rows),
        "partitions": len(partition_rows),
        "chromosomes": 22,
        "reported_segment_rows": total_rows,
        "reported_unique_dyads": len(reported_pairs),
        "people_in_any_reported_dyad": len({person for key in reported_pairs for person in key}),
        "eligible_people_range": [min(row["eligible_people"] for row in partition_rows), max(row["eligible_people"] for row in partition_rows)],
        "eligible_reported_unique_dyads_range": [min(row["reported_unique_dyads"] for row in partition_rows), max(row["reported_unique_dyads"] for row in partition_rows)],
        "unique_reported_pair_supports": len(support_groups),
        "exact_pair_support_partition_pairs": sum(len(ids) * (len(ids) - 1) // 2 for ids in support_groups.values()),
        "parameters_introduced": [],
        "historical_fits_rerun": 0,
        "detector_called": False,
        "detector_calibration_completed": False,
        "winner_selected": False,
        "evaluate_opened": False,
        "allowed_conclusion": (
            "Exact K05 person and K06 reported-dyad denominators are now authenticated per partition."
        ),
        "limits": [
            "A reported Refined IBD dyad is an observed call, not an independent replicate or documented pedigree relationship.",
            "No reported call means unknown here; it is not a negative IBD label or callable negative territory.",
            "Equal partition labels are descriptive assignments and are not population identities.",
            "These denominators do not compare FIT-learned partitions on SELECT or EVALUATE and do not rank configurations.",
            "Summed segment rows and centimorgans contain dependence across people, chromosomes and overlapping calls.",
        ],
        "inputs_sha256": {
            "plan": plan_sha256,
            "people": sha(people_path),
            "community_summary": sha(summary_path),
        },
        "outputs_sha256": {
            table_path.name: sha(table_path),
            group_path.name: sha(group_path),
        },
        "code_sha256": sha(Path(__file__)),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    require(sha(plan_path) == plan_sha256, "Plan changed during calculation")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-plan")
    build.add_argument("--individuals", required=True)
    build.add_argument("--community-summary", required=True)
    build.add_argument("--ibd-root", required=True)
    build.add_argument("--output", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--plan", required=True)
    execute.add_argument("--plan-sha256", required=True)
    execute.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "build-plan":
        result = build_plan(args.individuals, args.community_summary, args.ibd_root, args.output)
        print(json.dumps({"status": "PLAN_PINNED", "chromosomes": len(result["chromosomes"])}))
    else:
        result = run(args.plan, args.plan_sha256, args.output_dir)
        print(json.dumps({key: result[key] for key in ("status", "partitions", "reported_unique_dyads")}))


if __name__ == "__main__":
    main()
