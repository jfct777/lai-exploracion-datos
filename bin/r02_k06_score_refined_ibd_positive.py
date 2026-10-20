#!/usr/bin/env python3
"""Score Refined IBD geometry against eligible K06 positive-copy truth.

This scorer has no negative class.  It reports continuous covered positive
length, unmatched positive intervals, fragmentation, and endpoint errors.
Maximum-overlap matching is an assignment rule, not a binary detection rule.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def descriptor(value: dict, label: str, staged: Path | None = None) -> Path:
    require(set(value) == {"path", "sha256"}, f"Exact descriptor required: {label}")
    original = Path(value["path"])
    require(original.is_absolute(), f"Non-absolute descriptor: {label}")
    path = original
    if staged is not None:
        matches = [p for p in staged.iterdir() if p.is_file() and sha256(p) == value["sha256"]]
        require(len(matches) == 1, f"Staged identity differs: {label}")
        path = matches[0]
    require(path.is_file() and sha256(path) == value["sha256"], f"Input changed: {label}")
    return path


class GeneticMap:
    def __init__(self, path: Path, chromosome: str):
        self.bp, self.cm = [], []
        with open_text(path) as handle:
            for line in handle:
                fields = line.split()
                if not fields or fields[0].lower() in {"chrom", "chromosome"}:
                    continue
                if fields[0].removeprefix("chr") != chromosome:
                    continue
                bp, cm = int(fields[1]), float(fields[2])
                require(bp > 0 and math.isfinite(cm) and cm >= 0, "Invalid map row")
                require(not self.bp or (bp > self.bp[-1] and cm >= self.cm[-1]), "Map not monotone")
                self.bp.append(bp); self.cm.append(cm)
        require(len(self.bp) >= 2, "Insufficient map support")

    def at(self, bp: int) -> float | None:
        if bp < self.bp[0] or bp > self.bp[-1]:
            return None
        index = bisect_left(self.bp, bp)
        if self.bp[index] == bp:
            return self.cm[index]
        lo, hi = self.bp[index - 1], self.bp[index]
        return self.cm[index - 1] + (bp - lo) / (hi - lo) * (self.cm[index] - self.cm[index - 1])


def pair(row: dict) -> tuple:
    left = (row["sample_a"], int(row["haplotype_a"]))
    right = (row["sample_b"], int(row["haplotype_b"]))
    require(left < right and left[1] in (1, 2) and right[1] in (1, 2), "Noncanonical copy pair")
    return row["chromosome"], left, right


def overlap_bp(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)


def union_bp(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return sum(hi - lo + 1 for lo, hi in merged)


def maximum_overlap_assignment(truth: list[dict], calls: list[dict]) -> dict[int, int]:
    """Exact rectangular Hungarian assignment; retain only positive overlaps.

    Rows and columns are sorted before this function. Stable loop order and a
    bounded secondary rank make ties deterministic without changing any total
    overlap that differs by at least one base pair.
    """
    if not truth or not calls:
        return {}
    transposed = len(truth) > len(calls)
    rows, cols = (calls, truth) if transposed else (truth, calls)
    n, m = len(rows), len(cols)
    tie_bound = n * (n * m + 1)
    multiplier = tie_bound + 1
    scores = [[0] * (m + 1)]
    for i, row in enumerate(rows):
        line = [0]
        for j, col in enumerate(cols):
            a, b = (col, row) if transposed else (row, col)
            overlap = overlap_bp((a["start"], a["end"]), (b["start"], b["end"]))
            preference = n * m - (i * m + j)
            line.append(overlap * multiplier + preference)
        scores.append(line)
    u, v, p, way = [0] * (n + 1), [0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0; minimum = [10**100] * (m + 1); used = [False] * (m + 1)
        while True:
            used[j0] = True; i0 = p[j0]; delta = 10**100; j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    current = -scores[i0][j] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j] = current; way[j] = j0
                    if minimum[j] < delta:
                        delta = minimum[j]; j1 = j
            for j in range(m + 1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minimum[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    result = {}
    for column in range(1, m + 1):
        if p[column]:
            row = p[column] - 1; col = column - 1
            ti, ci = (col, row) if transposed else (row, col)
            if overlap_bp((truth[ti]["start"], truth[ti]["end"]),
                          (calls[ci]["start"], calls[ci]["end"])) > 0:
                result[ti] = ci
    return result


def run(request_path: Path, request_sha: str, calls_dir: Path, output: Path,
        staged_input_dir: Path | None = None) -> dict:
    require(sha256(request_path) == request_sha, "Detector request changed")
    request = json.loads(request_path.read_text())
    require(request["schema"] == "r02_k06_refined_ibd_request_v1", "Request schema differs")
    staged = staged_input_dir
    truth_path = descriptor(request["positive_territory"], "positive_territory", staged)
    map_path = descriptor(request["genetic_map"], "genetic_map", staged)
    spec_path = descriptor(request["parameter_spec"], "parameter_spec", staged)
    spec = json.loads(spec_path.read_text())
    require(spec.get("status") == "ADMITTED" and
            spec.get("scientific_scope") == "CONDITIONAL_POSITIVE_COPY_GEOMETRY",
            "Admitted conditional positive geometry profile required")
    minimum_cm = float(spec["parameters"]["length_cm_min"])
    require(minimum_cm == 2.0, "This frozen historical comparison requires length=2 cM")
    receipt_path = calls_dir / "receipt.json"
    calls_path = calls_dir / "refined-ibd.detected.private.tsv.gz"
    require(receipt_path.is_file() and calls_path.is_file(), "Complete normalized calls missing")
    receipt = json.loads(receipt_path.read_text())
    require(receipt["status"] == "COMPLETE" and receipt["case_id"] == request["case_id"] and
            receipt["calls_sha256"] == sha256(calls_path) and
            receipt["parameter_spec_sha256"] == request["parameter_spec"]["sha256"] and
            receipt["positive_territory_sha256"] == request["positive_territory"]["sha256"],
            "Calls receipt identity differs")
    destination = output
    require(not destination.exists(), "Output exists; preserve prior score")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="k06-positive-score-", dir=destination.parent))
    output = temporary / "result"
    output.mkdir()
    genetic = GeneticMap(map_path, request["chromosome"])
    truths, eligible_by_pair, excluded = [], defaultdict(list), Counter()
    with gzip.open(truth_path, "rt", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t"), 1):
            key = pair(row); require(key[0] == request["chromosome"], "Truth chromosome differs")
            relation = row["relationship"]
            require(relation in {"BETWEEN_PERSON_COPY_MATCH", "WITHIN_PERSON_COPY_MATCH"}, "Truth relation differs")
            record = {"index": index, "row": row, "key": key, "relation": relation,
                      "start": int(row["mapped_start_bp"]) if row["mapped_start_bp"] != "NA" else None,
                      "end": int(row["mapped_end_bp"]) if row["mapped_end_bp"] != "NA" else None}
            if row["map_support"] != "FULL_MAP":
                record["eligibility"] = "EXCLUDED_" + row["map_support"]
            elif float(row["mapped_length_cm"]) < minimum_cm:
                record["eligibility"] = "EXCLUDED_LENGTH_LT_2_CM"
            else:
                record["eligibility"] = "ELIGIBLE_FULL_MAP_LENGTH_GE_2_CM"
                eligible_by_pair[key].append(record)
            if record["eligibility"].startswith("EXCLUDED"):
                excluded[(record["eligibility"], relation)] += 1
            truths.append(record)
    for values in eligible_by_pair.values():
        values.sort(key=lambda x: (x["start"], x["end"], x["index"]))
        require(all(values[i]["start"] > values[i-1]["end"] for i in range(1, len(values))),
                "Eligible truth intervals overlap within a copy pair")
    call_rows, calls_by_pair = [], defaultdict(list)
    with gzip.open(calls_path, "rt", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t"), 1):
            key = pair(row); require(key[0] == request["chromosome"], "Call chromosome differs")
            record = {"index": index, "row": row, "key": key,
                      "start": int(row["start_bp_inclusive"]), "end": int(row["end_bp_inclusive"])}
            require(record["start"] <= record["end"], "Invalid call interval")
            call_rows.append(record); calls_by_pair[key].append(record)
    for values in calls_by_pair.values(): values.sort(key=lambda x: (x["start"], x["end"], x["index"]))
    assigned_truth, assigned_call = {}, {}
    for key in sorted(set(eligible_by_pair) | set(calls_by_pair)):
        truth, called = eligible_by_pair.get(key, []), calls_by_pair.get(key, [])
        for ti, ci in maximum_overlap_assignment(truth, called).items():
            assigned_truth[truth[ti]["index"]] = called[ci]
            assigned_call[called[ci]["index"]] = truth[ti]
    truth_fields = ["truth_index", "chromosome", "sample_a", "haplotype_a", "sample_b", "haplotype_b",
                    "start_bp_inclusive", "end_bp_inclusive", "relationship", "map_support", "mapped_length_cm",
                    "eligibility", "positive_bp", "covered_positive_bp", "covered_positive_fraction",
                    "overlapping_call_count", "fragmented_by_calls", "assigned_call_index", "assigned_overlap_bp",
                    "left_endpoint_error_bp", "right_endpoint_error_bp", "left_endpoint_error_cm", "right_endpoint_error_cm"]
    eligible_counts, positive_bp, covered_bp = Counter(), 0, 0
    with gzip.open(output / "truth-intervals.scored.private.tsv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=truth_fields, delimiter="\t", lineterminator="\n"); writer.writeheader()
        for truth in truths:
            row = truth["row"]; result = {name: row.get(name, "NA") for name in truth_fields}
            result.update(truth_index=truth["index"], eligibility=truth["eligibility"])
            if truth["eligibility"].startswith("EXCLUDED"):
                result.update({name: "NA" for name in truth_fields[12:]})
            else:
                calls_here = calls_by_pair.get(truth["key"], [])
                overlaps = [(max(truth["start"], c["start"]), min(truth["end"], c["end"]))
                            for c in calls_here if overlap_bp((truth["start"], truth["end"]), (c["start"], c["end"]))]
                length = truth["end"] - truth["start"] + 1; covered = union_bp(overlaps)
                positive_bp += length; covered_bp += covered
                relation = "within_person" if truth["relation"].startswith("WITHIN") else "between_person"
                eligible_counts[(relation, "intervals")] += 1
                eligible_counts[(relation, "positive_bp")] += length
                eligible_counts[(relation, "covered_bp")] += covered
                assigned = assigned_truth.get(truth["index"])
                result.update(positive_bp=length, covered_positive_bp=covered,
                              covered_positive_fraction=covered / length,
                              overlapping_call_count=len(overlaps), fragmented_by_calls=len(overlaps) > 1,
                              assigned_call_index=assigned["index"] if assigned else "NA",
                              assigned_overlap_bp=(overlap_bp((truth["start"], truth["end"]),
                                                             (assigned["start"], assigned["end"])) if assigned else "NA"))
                if assigned:
                    left_bp, right_bp = assigned["start"] - truth["start"], assigned["end"] - truth["end"]
                    ts, te, cs, ce = (genetic.at(x) for x in (truth["start"], truth["end"], assigned["start"], assigned["end"]))
                    result.update(left_endpoint_error_bp=left_bp, right_endpoint_error_bp=right_bp,
                                  left_endpoint_error_cm=cs-ts if cs is not None and ts is not None else "NA",
                                  right_endpoint_error_cm=ce-te if ce is not None and te is not None else "NA")
                else:
                    result.update(left_endpoint_error_bp="NA", right_endpoint_error_bp="NA",
                                  left_endpoint_error_cm="NA", right_endpoint_error_cm="NA")
            writer.writerow(result)
    call_fields = ["call_index", "chromosome", "sample_a", "haplotype_a", "sample_b", "haplotype_b",
                   "start_bp_inclusive", "end_bp_inclusive", "lod_score", "reported_length_cm",
                   "positive_geometry_class", "eligible_truth_overlap_count", "assigned_truth_index", "assigned_overlap_bp"]
    call_class = Counter()
    with gzip.open(output / "calls.scored.private.tsv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=call_fields, delimiter="\t", lineterminator="\n"); writer.writeheader()
        for call in call_rows:
            overlaps = [t for t in eligible_by_pair.get(call["key"], [])
                        if overlap_bp((t["start"], t["end"]), (call["start"], call["end"]))]
            assigned = assigned_call.get(call["index"])
            category = "ASIGNADA_SOLAPAMIENTO_POSITIVO" if assigned else "SIN_CLASIFICAR"
            call_class[category] += 1
            result = {name: call["row"].get(name, "NA") for name in call_fields}
            result.update(call_index=call["index"], positive_geometry_class=category,
                          eligible_truth_overlap_count=len(overlaps),
                          assigned_truth_index=assigned["index"] if assigned else "NA",
                          assigned_overlap_bp=(overlap_bp((assigned["start"], assigned["end"]),
                                                         (call["start"], call["end"])) if assigned else "NA"))
            writer.writerow(result)
    summary = {
        "schema": "r02_k06_conditional_positive_geometry_v1", "status": "COMPLETE",
        "case_id": request["case_id"], "chromosome": request["chromosome"],
        "policy": {"eligible_truth": "FULL_MAP and mapped_length_cm >= 2.0",
                   "assignment": "maximum total inclusive-bp overlap one-to-one within exact chromosome-copy pair",
                   "tie_break": "sorted coordinates/indexes, bounded secondary rank, stable Hungarian loop order",
                   "assignment_is_detection_claim": False},
        "positive_bp": positive_bp, "covered_positive_bp": covered_bp,
        "covered_positive_fraction": covered_bp / positive_bp if positive_bp else None,
        "eligible_by_relation": {f"{a}:{b}": value for (a, b), value in sorted(eligible_counts.items())},
        "excluded_truth_by_reason_and_relation": {f"{a}:{b}": value for (a, b), value in sorted(excluded.items())},
        "calls_by_positive_geometry_class": dict(call_class),
        "unmatched_positive_intervals": sum(1 for t in truths if t["eligibility"].startswith("ELIGIBLE") and t["index"] not in assigned_truth),
        "fragmented_positive_intervals": sum(1 for t in truths if t["eligibility"].startswith("ELIGIBLE") and
            sum(overlap_bp((t["start"], t["end"]), (c["start"], c["end"])) > 0 for c in calls_by_pair.get(t["key"], [])) > 1),
        "precision": None, "specificity": None, "binary_sensitivity": None,
        "negative_truth_available": False, "demographic_inference": False,
        "limits": ["Unassigned calls are SIN_CLASIFICAR, never false positives.",
                   "Endpoint errors describe only positively overlapping assigned pairs.",
                   "Known copied sequence is positive geometry, not a demographic history.",
                   "Phase accuracy of simulated VCF inputs is not validated."],
        "outputs_sha256": {}
    }
    for name in ("truth-intervals.scored.private.tsv.gz", "calls.scored.private.tsv.gz"):
        summary["outputs_sha256"][name] = sha256(output / name)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    require(sha256(request_path) == request_sha and sha256(calls_path) == receipt["calls_sha256"], "Input changed during scoring")
    shutil.move(output, destination)
    temporary.rmdir()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--calls-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--staged-input-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.request, args.request_sha256, args.calls_dir, args.output,
                         args.staged_input_dir), sort_keys=True))
