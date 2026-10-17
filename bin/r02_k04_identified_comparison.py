#!/usr/bin/env python3
"""K04 deterministic support reduction and partial-identification arithmetic.

These operations do not fit models, impute phase, calibrate Gnomix, or establish
biological validation. The scalar interval functions apply to fixed predictions
and coefficients only. Their aggregate bounds need not be jointly attainable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "r02_k04_support_reduction_v1"
SOURCE_SCHEMA = "r02_rare_ancestry_pair_bounds_v1"
PRODUCT = "PAIR_SHARING_PARTIAL_IDENTIFICATION_V1"
SOURCE_STATUS = "COMPLETE_CONDITIONAL_PAIR_SHARING_BOUNDS_NOT_PHASE_OR_FINE_STRUCTURE_VALIDATION"
ANCESTRIES = ("AFR", "EUR", "NAM")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def counted_allele_bounds(dosage, ancestry_copies):
    require(type(dosage) is int and dosage in (0, 1, 2), "Invalid diploid dosage")
    require(type(ancestry_copies) is int and ancestry_copies in (0, 1, 2),
            "Invalid ancestry copy count")
    return max(0, dosage + ancestry_copies - 2), min(dosage, ancestry_copies)


def mismatch_count_bounds(d_i, a_i, d_j, a_j):
    """Return min/max mismatching ancestral copy pairs and copy opportunities.

    At fixed ancestry copy counts the mismatch count is bilinear in the two
    unknown counted-allele doses, so its extrema are at rectangle vertices.
    Zero opportunities are retained as (0, 0, 0), never as a zero mismatch rate.
    """
    li, ui = counted_allele_bounds(d_i, a_i)
    lj, uj = counted_allele_bounds(d_j, a_j)
    values = [a_j * xi + a_i * xj - 2 * xi * xj
              for xi in (li, ui) for xj in (lj, uj)]
    return min(values), max(values), a_i * a_j


def fixed_linear_bounds(lower, upper, coefficients):
    require(len(lower) == len(upper) == len(coefficients) and len(lower) > 0,
            "Nonempty equal-length vectors required")
    lo, hi = [], []
    for l, u, c in zip(lower, upper, coefficients):
        require(all(math.isfinite(x) for x in (l, u, c)) and l <= u,
                "Invalid interval or coefficient")
        lo.append(c * (l if c >= 0 else u))
        hi.append(c * (u if c >= 0 else l))
    return math.fsum(lo), math.fsum(hi)


def fixed_squared_error_improvement(lower, upper, baseline, candidate, weights):
    """Outer bounds of mean[(y-baseline)^2-(y-candidate)^2].

    Prediction vectors MUST have been fixed without these outcome values.
    Evaluating separately refitted lower/upper models does not satisfy that rule.
    This function is arithmetic, not an independence or provenance validator.
    """
    n = len(lower)
    require(n > 0 and all(len(v) == n for v in (upper, baseline, candidate, weights)),
            "Equal nonempty prediction/target/weight vectors required")
    require(all(math.isfinite(w) and w >= 0 for w in weights) and sum(weights) > 0,
            "Nonnegative finite weights with positive total required")
    require(all(math.isfinite(x) and 0 <= x <= 1
                for values in (lower, upper, baseline, candidate) for x in values),
            "Mismatch rates and predictions must lie in [0,1]")
    scale = math.fsum(weights)
    coefficients = [2 * w * (q - p) / scale
                    for p, q, w in zip(baseline, candidate, weights)]
    constant = math.fsum(w * (p * p - q * q) / scale
                         for p, q, w in zip(baseline, candidate, weights))
    lo, hi = fixed_linear_bounds(lower, upper, coefficients)
    return constant + lo, constant + hi


def reduce_support(specification):
    """Authenticate and reduce only small summary/person products.

    The expected manifest hashes are supplied by a separately authenticated
    GCS read/publication receipt. Matrix bytes are deliberately not opened.
    """
    require(specification["schema"] == "r02_k04_support_inputs_v1", "Input schema differs")
    expected = specification["expected_chromosomes"]
    require(len(expected) == len(set(expected)) and expected, "Duplicate/empty universe")
    require({str(x) for x in expected} <= {str(x) for x in range(1, 23)}, "Autosomes required")
    rows = specification["chromosomes"]
    require(Counter(str(x["chromosome"]) for x in rows) == Counter(map(str, expected)),
            "Chromosome set incomplete or duplicated")
    person_axis = None
    totals = defaultdict(int)
    by_role = defaultdict(lambda: defaultdict(int))
    per_chromosome = []
    authenticated = []
    for entry in sorted(rows, key=lambda x: int(x["chromosome"])):
        directory = Path(entry["directory"])
        manifest_path = directory / "manifest.json"
        require(sha256(manifest_path) == entry["manifest_sha256"], "Manifest SHA256 differs")
        manifest = json.loads(manifest_path.read_text())
        require(manifest["schema"] == SOURCE_SCHEMA and manifest["product"] == PRODUCT,
                "Unexpected source product")
        require(manifest["status"] == SOURCE_STATUS, "Incomplete source product")
        for name in ("summary.json", "persons.tsv", "person_metrics.tsv"):
            require(sha256(directory / name) == manifest["outputs_sha256"][name],
                    f"Source SHA256 differs: {name}")
        summary = json.loads((directory / "summary.json").read_text())
        require(summary["schema"] == SOURCE_SCHEMA and summary["product"] == PRODUCT,
                "Unexpected summary product")
        require(summary["status"] == SOURCE_STATUS, "Incomplete source summary")
        require(str(summary["chromosome"]) == str(entry["chromosome"]), "Chromosome differs")
        require(summary["ancestry_is"] == "GNOMIX_ESTIMATED_AFR_EUR_NAM_HARDCALL_CONDITIONAL_NOT_TRUTH",
                "Ancestry interpretation differs")
        require(summary["coefficient_weight"] == 1 and summary["no_imputation"] is True,
                "Source weighting or imputation differs")
        with (directory / "persons.tsv").open(newline="") as stream:
            persons = list(csv.DictReader(stream, delimiter="\t"))
        axis = [(p["person_index"], p["sample_id"], p["role"]) for p in persons]
        require(len(axis) == summary["n_people"] == specification["expected_n_people"],
                "Person count differs")
        require([x[0] for x in axis] == list(map(str, range(len(axis)))) and
                len({x[1] for x in axis}) == len(axis), "Invalid person order or duplicates")
        require(dict(Counter(x[2] for x in axis)) == specification["expected_role_counts"],
                "Person roles differ")
        if person_axis is None:
            person_axis = axis
        require(axis == person_axis, "Chromosomal person axis differs")
        n_loci = summary["n_loci"]
        require(type(n_loci) is int and n_loci > 0 and
                summary["loci_in_msp"] + summary["loci_outside_msp"] == n_loci,
                "Locus accounting differs")
        with (directory / "person_metrics.tsv").open(newline="") as stream:
            metrics = list(csv.DictReader(stream, delimiter="\t"))
        require(len(metrics) == len(axis), "Metric person count differs")
        chromosome_counts = defaultdict(int)
        for identity, row in zip(axis, metrics):
            require(tuple(row[k] for k in ("person_index", "sample_id", "role")) == identity,
                    "Metric person axis differs")
            values = {k: int(v) for k, v in row.items()
                      if k not in ("person_index", "sample_id", "role")}
            require(all(v >= 0 for v in values.values()), "Negative metric count")
            require(values["observed_gt"] + values["missing_gt"] == n_loci,
                    "Genotype support accounting differs")
            require(values["joint_valid_loci"] + values["outside_msp_observed"] +
                    values["inside_msp_missing_ancestry_observed"] == values["observed_gt"],
                    "Ancestry support accounting differs")
            require(sum(values[f"ancestry_homolog_copies_{a}"] for a in ANCESTRIES) ==
                    2 * values["joint_valid_loci"], "Ancestry copy accounting differs")
            for a in ANCESTRIES:
                for kind in ("presence", "copy"):
                    l = values[f"lower_{kind}_burden_{a}"]
                    u = values[f"upper_{kind}_burden_{a}"]
                    limit = values[f"ancestry_present_loci_{a}"] if kind == "presence" else values[f"ancestry_homolog_copies_{a}"]
                    require(l <= u <= limit, "Burden bounds inconsistent")
                values[f"ambiguous_presence_person_loci_{a}"] = (
                    values[f"upper_presence_burden_{a}"] - values[f"lower_presence_burden_{a}"])
            for key, value in values.items():
                totals[key] += value
                by_role[identity[2]][key] += value
                chromosome_counts[key] += value
        for source_key, metric_key in (
                ("missing_gt_person_locus_rows", "missing_gt"),
                ("observed_gt_outside_msp_person_locus_rows", "outside_msp_observed"),
                ("observed_gt_inside_msp_missing_ancestry_person_locus_rows", "inside_msp_missing_ancestry_observed")):
            require(summary["global_missing_counts"][source_key] == chromosome_counts[metric_key],
                    "Summary/person missing counts differ")
        per_chromosome.append(dict(chromosome=str(entry["chromosome"]), n_loci=n_loci,
                                   loci_in_msp=summary["loci_in_msp"], **chromosome_counts))
        authenticated.append(dict(chromosome=str(entry["chromosome"]),
                                  manifest_sha256=entry["manifest_sha256"],
                                  outputs_sha256=manifest["outputs_sha256"]))
    total_loci = sum(c["n_loci"] for c in per_chromosome)
    require(total_loci == specification["expected_n_loci"], "Catalogue locus count differs")
    return dict(schema=SCHEMA, status="COMPLETE_DESCRIPTIVE_SUPPORT_REDUCTION_NOT_BIOLOGICAL_COMPARISON",
                n_people=len(person_axis), n_loci=total_loci, n_chromosomes=len(rows),
                role_counts=specification["expected_role_counts"], person_loci=len(person_axis)*total_loci,
                totals=dict(totals), by_role={k: dict(v) for k,v in by_role.items()},
                by_chromosome=per_chromosome, authenticated_sources=authenticated,
                matrix_bytes_opened=False, matrix_content_revalidated=False,
                calibrated_ancestry_error=False, phase_imputed=False,
                scientific_comparison_executed=False,
                limitations=["Conditional on Gnomix hardcalls, not true ancestry",
                             "Person-locus ambiguity counts are not independent replicates",
                             "No inference about incremental rare information or IBD"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(sha256(args.inputs) == args.inputs_sha256, "Input specification SHA256 differs")
    result = reduce_support(json.loads(Path(args.inputs).read_text()))
    result["inputs_sha256"] = args.inputs_sha256
    result["implementation_sha256"] = sha256(__file__)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
