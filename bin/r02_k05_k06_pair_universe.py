#!/usr/bin/env python3
"""Add the full combinatorial dyad universe to a K05/K06 reported-pair table."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256(Path(path).read_bytes())
    return digest.hexdigest()


def run(summary_path, summary_sha, table_path, table_sha, output):
    if sha(summary_path) != summary_sha or sha(table_path) != table_sha:
        raise ValueError("Authenticated K05/K06 result differs")
    summary = json.loads(Path(summary_path).read_text())
    if summary.get("status") != "COMPLETE_AUTHENTIC_K05_K06_REPORTED_PAIR_DENOMINATORS":
        raise ValueError("Unsupported source result")
    with Path(table_path).open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not (len(rows) == summary["partitions"] == 57):
        raise ValueError("Partition count differs")
    expanded = []
    for row in rows:
        people = int(row["eligible_people"])
        reported = int(row["reported_unique_dyads"])
        possible = people * (people - 1) // 2
        if not 0 <= reported <= possible:
            raise ValueError("Reported dyads exceed the full unordered-pair universe")
        expanded.append({
            **row,
            "all_possible_unordered_dyads": possible,
            "dyads_without_reported_call_unknown": possible - reported,
            "reported_call_dyad_fraction_descriptive": repr(reported / possible),
        })
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(expanded[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(expanded)
    receipt = {
        "schema": "r02_k05_k06_pair_universe_v1",
        "status": "COMPLETE_REPORTED_VERSUS_ALL_POSSIBLE_DYAD_DENOMINATORS",
        "partitions": len(expanded),
        "reported_unique_dyads_range": [
            min(int(row["reported_unique_dyads"]) for row in expanded),
            max(int(row["reported_unique_dyads"]) for row in expanded),
        ],
        "all_possible_unordered_dyads_range": [
            min(int(row["all_possible_unordered_dyads"]) for row in expanded),
            max(int(row["all_possible_unordered_dyads"]) for row in expanded),
        ],
        "reported_call_dyad_fraction_range": [
            min(float(row["reported_call_dyad_fraction_descriptive"]) for row in expanded),
            max(float(row["reported_call_dyad_fraction_descriptive"]) for row in expanded),
        ],
        "absence_semantics": "DYAD_WITHOUT_REPORTED_CALL_IS_UNKNOWN_NOT_A_NEGATIVE_IBD_LABEL",
        "independence_claimed": False,
        "calibration_completed": False,
        "inputs_sha256": {"summary": summary_sha, "table": table_sha},
        "output_sha256": sha(output),
        "code_sha256": sha(Path(__file__)),
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--table-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.summary, args.summary_sha256, args.table, args.table_sha256, args.output)))
