#!/usr/bin/env python3
"""Authenticate cohort metadata and dependence units; never create biological labels.

Reuses the PC-Relate reader used for R02 graph diagnostics. Components are an
operational dependence grouping, not independent populations or new kinship.
No fitting, role assignment, sample exclusion or significance test is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import r02_genomic_pair_evidence as evidence
from r02_weighted_kinship import ABSENT, FINITE, NONFINITE, stream_kinship_arrays


MISSING = {"", ".", "na", "nan", "none", "null", "unknown"}
ROLES = {"FIT", "SELECT", "EVALUATE", "EXCLUDE", "UNASSIGNED"}


def read_metadata(path, samples, id_column, columns):
    """Collapse exact duplicate rows only; reject conflicting records for an ID."""
    selected = set(samples)
    found = {}
    duplicates = outside = 0
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = reader.fieldnames
        evidence.require(header and len(header) == len(set(header)) and id_column in header,
                         "Metadata requires unique columns and the declared ID column")
        for row in reader:
            evidence.require(None not in row and None not in row.values(), "Malformed metadata row")
            sid = row[id_column]
            evidence.require(sid and sid == sid.strip(), "Invalid metadata sample ID")
            if sid not in selected:
                outside += 1
                continue
            if sid in found:
                evidence.require(found[sid] == row, "Conflicting duplicate metadata for analytical sample")
                duplicates += 1
            else:
                found[sid] = row
    coverage = []
    for column in columns:
        count = sum(column in row and row[column].strip().lower() not in MISSING for row in found.values())
        coverage.append(dict(field=column, n_measured=count, n_people=len(samples),
                             status="AVAILABLE" if column in header else "COLUMN_ABSENT"))
    return found, coverage, dict(exact_duplicate_rows=duplicates, rows_outside_cohort=outside,
                                samples_without_metadata=len(selected - found.keys()))


def read_roles(path, samples):
    if path is None:
        return {sid: "UNASSIGNED" for sid in samples}, False
    values = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        evidence.require(reader.fieldnames == ["sample_id", "role"], "Roles must have sample_id and role columns")
        for row in reader:
            evidence.require(None not in row and None not in row.values(), "Malformed role row")
            sid, role = row["sample_id"], row["role"]
            evidence.require(sid in samples and sid not in values and role in ROLES, "Invalid or duplicate role")
            values[sid] = role
    evidence.require(set(values) == set(samples), "Roles must cover exactly the analytical cohort")
    return values, True


def components(samples, kinship, states, thresholds, roles):
    """Connected components at each declared phi; unresolved pairs stay unknown."""
    assignments, summary, checks = {}, [], []
    for phi in thresholds:
        adjacency = (states == FINITE) & (kinship >= phi)
        np.fill_diagonal(adjacency, False)
        _, labels = connected_components(csr_matrix(adjacency), directed=False)
        groups = defaultdict(list)
        for sid, label in zip(samples, labels):
            groups[int(label)].append(sid)
        mapping = {}
        collisions = 0
        for members in groups.values():
            key = "dep_" + evidence.sample_hash(sorted(members))[:20]
            for sid in members:
                mapping[sid] = key
            active_roles = {roles[sid] for sid in members} & {"FIT", "SELECT", "EVALUATE"}
            collisions += len(active_roles) > 1
        counts = Counter(len(members) for members in groups.values())
        assignments[phi] = mapping
        summary.append(dict(phi=phi, n_people=len(samples), n_components=len(groups),
                            n_singletons=counts[1], max_component_size=max(counts, default=0),
                            n_components_crossing_roles=collisions,
                            interpretation="DEPENDENCE_GROUPS_NOT_INDEPENDENCE_CERTIFICATES"))
        checks.append(dict(check=f"no_role_crossing_phi_{phi:g}",
                           status="FAIL" if collisions else "PASS" if any(r != "UNASSIGNED" for r in roles.values()) else "NOT_DEFINED",
                           detail=f"{collisions} components cross FIT/SELECT/EVALUATE"))
    return assignments, summary, checks


def write_tsv(path, columns, rows):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    sample_file_sha = evidence.sha256(args.sample_ids)
    samples = evidence.read_samples(args.sample_ids, args.expected_samples)
    evidence.require(len(samples) >= 2, "At least two samples required")
    evidence.require(np.isfinite(args.max_matrix_mb) and args.max_matrix_mb > 0,
                     "max-matrix-mb must be positive and finite")
    # Conservative admission estimate for reader arrays, comparisons and sparse conversion.
    evidence.require(len(samples) ** 2 * 48 / 2**20 <= args.max_matrix_mb,
                     "Pair matrix estimate exceeds declared memory allowance")
    thresholds = args.phi
    evidence.require(thresholds and len(thresholds) == len(set(thresholds))
                     and all(np.isfinite(p) and 0 < p < .5 for p in thresholds), "Invalid phi thresholds")
    output = Path(args.output_dir)
    evidence.require(not output.exists(), "Output directory already exists; will not overwrite")
    metadata_hash = evidence.sha256(args.metadata)
    evidence.require(metadata_hash == args.expected_metadata_sha256, "Metadata SHA256 mismatch")
    columns = list(dict.fromkeys([args.cohort_column, args.region_column, *args.metadata_column]))
    metadata, coverage, metadata_audit = read_metadata(args.metadata, samples, args.id_column, columns)
    roles_file_sha = evidence.sha256(args.roles) if args.roles else None
    roles, roles_supplied = read_roles(args.roles, samples)
    target = np.ones((len(samples), len(samples)), dtype=bool)
    np.fill_diagonal(target, False)
    kinship, states, kin_audit = stream_kinship_arrays(
        args.pcrelate_file, {sid: i for i, sid in enumerate(samples)}, target, args.expected_pcrelate_sha256)
    del target
    assignments, summaries, checks = components(samples, kinship, states, thresholds, roles)
    missing_pairs = kin_audit["n_retained_missing_absent"] + kin_audit["n_retained_missing_nonfinite"]
    checks.extend([
        dict(check="pairwise_kinship_coverage", status="INCOMPLETE" if missing_pairs else "COMPLETE",
             detail=f"{missing_pairs} pairs lack finite kinship; not evidence of being unrelated"),
        dict(check="metadata_semantics", status="NOT_CERTIFIED",
             detail="Field names do not establish birthplace, ancestral origin, population or sequencing batch"),
        dict(check="training_authorization", status="NOT_AUTHORIZED",
             detail="Descriptive audit only; no target, detectable effect, role assignment or training approval created"),
    ])
    def value(sid, column):
        raw = metadata.get(sid, {}).get(column, "")
        return "UNKNOWN" if raw.strip().lower() in MISSING else raw
    cross = Counter((value(sid, args.cohort_column), value(sid, args.region_column)) for sid in samples)
    deterministic = []
    for cohort in sorted({key[0] for key in cross} - {"UNKNOWN"}):
        regions = {region for (c, region), n in cross.items() if c == cohort and region != "UNKNOWN" and n}
        if len(regions) == 1:
            deterministic.append(cohort)
    checks.append(dict(check="cohort_region_confounding", status="WARNING" if deterministic else "NOT_EXCLUDED",
                       detail=f"{len(deterministic)} collections have only one observed region; region is not a primary structure endpoint"))
    evidence.require(evidence.sha256(args.metadata) == metadata_hash, "Metadata changed during audit")
    evidence.require(evidence.sha256(args.sample_ids) == sample_file_sha, "Sample IDs changed during audit")
    if args.roles:
        evidence.require(evidence.sha256(args.roles) == roles_file_sha, "Roles changed during audit")
    output.mkdir(parents=True)
    component_cols = [f"component_phi_{p:g}" for p in thresholds]
    people = []
    for sid in samples:
        row = dict(sample_id=sid, role=roles[sid])
        row.update({c: value(sid, c) for c in columns})
        row.update({c: assignments[p][sid] for c, p in zip(component_cols, thresholds)})
        people.append(row)
    evidence.require(not ({"sample_id", "role", *component_cols} & set(columns)), "Metadata output column collision")
    write_tsv(output / "persons.private.tsv", ["sample_id", "role", *columns, *component_cols], people)
    write_tsv(output / "metadata_coverage.tsv", ["field", "n_measured", "n_people", "status"], coverage)
    write_tsv(output / "components_summary.tsv", list(summaries[0]), summaries)
    write_tsv(output / "cohort_region.tsv", ["cohort", "region", "n_people"],
              [dict(cohort=c, region=r, n_people=n) for (c, r), n in sorted(cross.items())])
    write_tsv(output / "checks.tsv", ["check", "status", "detail"], checks)
    input_hashes = {str(args.sample_ids): sample_file_sha, str(args.metadata): metadata_hash,
                    str(args.pcrelate_file): kin_audit["sha256"]}
    if args.roles:
        input_hashes[str(args.roles)] = roles_file_sha
    manifest = dict(schema="r02_study_design_v1", status="COMPLETE_DESCRIPTIVE_AUDIT",
                    n_samples=len(samples), sample_ids_sha256=evidence.sample_hash(samples),
                    thresholds=thresholds, roles_supplied=roles_supplied,
                    nominal_columns=list(dict.fromkeys([args.cohort_column, args.region_column])),
                    inputs_sha256=input_hashes, metadata_audit=metadata_audit, kinship_audit=kin_audit,
                    checks=checks, training_status="NOT_AUTHORIZED",
                    outputs_sha256={p.name: evidence.sha256(p) for p in sorted(output.glob("*.tsv"))},
                    code_sha256={Path(__file__).name: evidence.sha256(__file__),
                                 "r02_weighted_kinship.py": evidence.sha256(Path(__file__).with_name("r02_weighted_kinship.py")),
                                 "r02_genomic_pair_evidence.py": evidence.sha256(Path(__file__).with_name("r02_genomic_pair_evidence.py"))},
                    limitations=["Components encode a declared cutoff, not independent biological replicates",
                                 "No new PC-Relate estimation or PC interpretation was performed",
                                 "Region labels and discovered communities are not verified populations"])
    (output / "study_contract.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--expected-metadata-sha256", required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--cohort-column", default="Cohort")
    parser.add_argument("--region-column", default="Region")
    parser.add_argument("--metadata-column", action="append", default=[])
    parser.add_argument("--pcrelate-file", required=True, type=Path)
    parser.add_argument("--expected-pcrelate-sha256", required=True)
    parser.add_argument("--phi", nargs="+", type=float, default=[.0221, .0442])
    parser.add_argument("--roles", type=Path)
    parser.add_argument("--max-matrix-mb", type=float, default=1024)
    parser.add_argument("--output-dir", required=True, type=Path)
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "n_samples": result["n_samples"],
                      "training_status": result["training_status"]}))


if __name__ == "__main__":
    main()
