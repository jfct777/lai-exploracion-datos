#!/usr/bin/env python3
"""Prepare authenticated cohort/support inventories and target exclusion plans.

No VCF extraction, fitting, biological threshold choice, or training admission.
Reuses the component-role proposal used by the structure comparator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re

from r02_structure_increment import assign_component_roles, load_json, require, sha, table, validate_roles

ROLES = ("FIT", "SELECT", "EVALUATE")
MISSING = {"", ".", "na", "nan", "none", "null", "unknown"}
BASE_FILES = {"study_contract.json", "persons.private.tsv", "rare_inventory.json", "protocol.md"}
OPTIONAL_FILES = {"frozen_roles.tsv", "blocks.tsv", "genome.fai"}
BLOCK_COLUMNS = ["block_id", "chromosome", "start_bp", "end_bp", "excluded_start_bp", "excluded_end_bp"]


def rows(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "Invalid table columns")
        result = list(reader)
    require(result and all(None not in row and None not in row.values() for row in result), "Malformed or empty table")
    return result


def ordered_ids_hash(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def authenticate(bundle, contract):
    require(contract.get("schema") == "r02_structure_input_plan_v1", "Invalid input planning schema")
    chromosomes = contract["chromosomes"]
    require(isinstance(chromosomes, list) and chromosomes and
            all(type(c) is int and 1 <= c <= 22 for c in chromosomes) and
            chromosomes == sorted(set(chromosomes)), "Unique ordered autosomes required")
    expected = BASE_FILES | {f"common/chr{c:02d}.json" for c in chromosomes}
    names = set(contract["sha256"])
    require(expected <= names and names - expected <= OPTIONAL_FILES, "Unexpected or missing authenticated files")
    require(("blocks.tsv" in names) == ("genome.fai" in names), "Blocks require the authenticated reference FAI")
    budget = contract["max_input_mb"]
    require(type(budget) in (int, float) and 0 < budget < float("inf"), "Positive input size limit required")
    paths = [bundle / name for name in sorted(names)]
    require(all(p.is_file() for p in paths), "Missing authenticated source")
    require(sum(p.stat().st_size for p in paths) <= budget * 2**20, "Metadata size limit exceeded")
    for name in sorted(names):
        expected_hash = contract["sha256"][name]
        require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash), "Invalid source SHA256")
        require(sha(bundle / name) == expected_hash, f"SHA256 mismatch: {name}")
    return chromosomes


def prepare_roles(study, people, contract, frozen=None):
    ids = [r["sample_id"] for r in people]
    require(all(s and s == s.strip() for s in ids) and len(ids) == len(set(ids)), "Invalid cohort IDs")
    require(len(ids) == study["n_samples"] and ordered_ids_hash(ids) == study["sample_ids_sha256"], "Study cohort/order mismatch")
    component = contract["component_column"]
    columns = [f"component_phi_{float(t):g}" for t in study["thresholds"]]
    require(component in columns and all(all(r.get(c) for c in columns) for r in people), "Unknown/missing study components")
    persons = [{"sample_id": r["sample_id"], "component": r[component]} for r in people]
    if frozen is None:
        require(all(r.get("role") == "UNASSIGNED" for r in people), "Existing roles require explicit frozen reuse, not reassignment")
        proposal = contract["proposal"]
        persons = assign_component_roles(persons, proposal["fractions"], proposal["seed"])
        status = "PROPOSED_ROLES_NOT_FROZEN"
    else:
        mapping = {r["sample_id"]: r["role"] for r in frozen}
        require(len(mapping) == len(frozen) and set(mapping) == set(ids), "Frozen roles must cover each cohort member once")
        require(contract.get("roles_status") == "EXPLICITLY_FROZEN" and
                isinstance(contract.get("role_decision_reference"), str) and
                len(contract["role_decision_reference"].strip()) >= 10, "Explicit freezing decision required")
        persons = [{**p, "role": mapping[p["sample_id"]]} for p in persons]
        status = "ROLES_FROZEN_INPUTS_NOT_MATERIALIZED"
    validate_roles(persons)
    crossings = []
    for column in columns:
        groups = defaultdict(set)
        for old, new in zip(people, persons):
            groups[old[column]].add(new["role"])
        crossings.append({"component_column": column, "n_components": len(groups),
                          "n_crossing_roles": sum(len(v) > 1 for v in groups.values())})
    summary = []
    for role in ROLES:
        counts = Counter(p["component"] for p in persons if p["role"] == role)
        n = sum(counts.values())
        summary.append({"role": role, "n_people": n, "n_components": len(counts),
                        "largest_component_people": max(counts.values()),
                        "n_between_component_pairs": (n*n-sum(v*v for v in counts.values()))//2,
                        "n_people_total": len(people)})
    return persons, summary, crossings, status


def metadata_support(people, persons, fields):
    """Descriptions only: collection/region names do not certify ancestry or batch."""
    require(isinstance(fields, list) and fields and len(fields) == len(set(fields)), "Explicit metadata fields required")
    support = []
    for field in fields:
        require(isinstance(field, str) and field not in {"sample_id", "role"} and
                not field.startswith("component_"), "Invalid descriptive metadata field")
        categories = {"__MISSING__" if r.get(field, "").strip().lower() in MISSING else r[field] for r in people}
        for role in ROLES:
            selected = [(r, p) for r, p in zip(people, persons) if p["role"] == role]
            cells = defaultdict(list)
            for row, person in selected:
                raw = row.get(field, "")
                value = "__MISSING__" if raw.strip().lower() in MISSING else raw
                cells[value].append(person)
            for value in sorted(categories):
                members = cells[value]
                n_components = len({p["component"] for p in members})
                support.append({"field": field, "value": value, "role": role,
                    "n_people": len(members), "n_components": n_components,
                    "n_people_role": len(selected),
                    "status": "COLUMN_ABSENT" if field not in people[0] else
                              "CATEGORY_ABSENT_IN_ROLE" if not members else
                              "UNKNOWN" if value == "__MISSING__" else
                              "SINGLE_COMPONENT_NO_REPLICATION" if n_components == 1 else
                              "DESCRIPTIVE_SUPPORT_NOT_POWER_CERTIFIED"})
    return support


def source_support(bundle, chromosomes, study):
    inventory = load_json(bundle / "rare_inventory.json")
    records = inventory["chromosomes"]
    require(len({r["chromosome"] for r in records}) == len(records), "Duplicate rare chromosome")
    by_chr = {r["chromosome"]: r for r in records}
    require(set(chromosomes) <= set(by_chr), "Missing rare chromosome")
    summary = []
    for chrom in chromosomes:
        rare = by_chr[chrom]
        common = load_json(bundle / f"common/chr{chrom:02d}.json")
        require(common["kind"] == "common" and common["chromosomes"] == [str(chrom)] and
                common["n_samples"] == study["n_samples"] and
                common["samples_sha256"] == study["sample_ids_sha256"], "Common evidence cohort/chromosome mismatch")
        n_rare, n_ref, n_alt = (rare[k] for k in ("rare_sites", "rare_ref", "rare_alt"))
        require(all(type(n) is int and n >= 0 for n in (n_rare, n_ref, n_alt)) and
                n_rare == n_ref+n_alt, "Rare allele inventory does not balance")
        summary.append({"chromosome": chrom, "n_common_markers_existing": common["producer"]["marker_selection"]["retained_snps"],
            "common_frequency_scope": common["producer"]["frequency_scope"],
            "n_rare_sites_source": n_rare, "n_rare_ref_minor": n_ref, "n_rare_alt_minor": n_alt,
            "n_rare_source_people": rare["expected_source_samples"], "n_analytical_people": study["n_samples"],
            "rare_vcf_bytes_recorded": rare["vcf"]["bytes"],
            "source_status": "MANIFESTS_AUTHENTICATED_NO_FRESH_VCF_OR_GCS_CHECK",
            "predictor_reuse": "NO_DIRECT_REUSE_FULL_COHORT_SELECTION_AND_TARGET_LOCI"})
    return summary


def plan_regions(blocks, lengths, scope):
    """Closed 1-based intervals; remove target+margin BEFORE any derived feature."""
    require(scope in {"BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME"}, "Unknown exclusion scope")
    require(blocks and len({b["block_id"] for b in blocks}) == len(blocks), "Unique nonempty blocks required")
    previous = defaultdict(list)
    output = []
    for block in blocks:
        name, chrom = block["block_id"], block["chromosome"]
        require(re.fullmatch(r"[A-Za-z0-9_]+", name) and chrom in lengths, "Invalid block ID/chromosome")
        start, end, lo, hi = (int(block[k]) for k in BLOCK_COLUMNS[2:])
        require(1 <= lo <= start <= end <= hi <= lengths[chrom], "Target/margin outside reference chromosome")
        require(not any(start <= b and a <= end for a, b in previous[chrom]), "Overlapping target blocks")
        previous[chrom].append((start, end))
        if scope == "WHOLE_CHROMOSOME":
            lo, hi = 1, lengths[chrom]
        allowed = []
        for c, length in sorted(lengths.items(), key=lambda x: int(x[0])):
            if c != chrom:
                allowed.append([c, 1, length])
            else:
                if lo > 1:
                    allowed.append([c, 1, lo-1])
                if hi < length:
                    allowed.append([c, hi+1, length])
        require(sum(v[2]-v[1]+1 for v in allowed)+(hi-lo+1) == sum(lengths.values()), "Interval partition did not conserve bases")
        output.append({"block_id": name, "target": [chrom, start, end], "excluded": [chrom, lo, hi],
                       "allowed_input_intervals": allowed,
                       "allowed_bp": sum(v[2]-v[1]+1 for v in allowed), "excluded_bp": hi-lo+1})
    return output


def reference_lengths(path, contigs, chromosomes):
    require(set(contigs) == {str(c) for c in chromosomes} and
            len(set(contigs.values())) == len(contigs), "Explicit one-to-one contig map required")
    found = {}
    with Path(path).open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            require(len(parts) >= 2 and parts[0] not in found, "Invalid or duplicate FAI contig")
            found[parts[0]] = int(parts[1])
    require(all(v in found and found[v] > 0 for v in contigs.values()), "FAI lacks declared contigs")
    return {c: found[v] for c, v in contigs.items()}


def write_table(path, data):
    require(data, "Refusing empty output table")
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=list(data[0]))
        w.writeheader(); w.writerows(data)


def run(bundle, contract_path, output_dir):
    contract = load_json(contract_path)
    chromosomes = authenticate(bundle, contract)
    study = load_json(bundle/"study_contract.json")
    require(study["status"] == "COMPLETE_DESCRIPTIVE_AUDIT", "Completed cohort audit required")
    require(sha(bundle/"persons.private.tsv") == study["outputs_sha256"]["persons.private.tsv"], "Study persons hash mismatch")
    people = rows(bundle/"persons.private.tsv")
    frozen = table(bundle/"frozen_roles.tsv", ["sample_id", "component", "role"]) if "frozen_roles.tsv" in contract["sha256"] else None
    if frozen is not None:
        original = {p["sample_id"]: p[contract["component_column"]] for p in people}
        require(all(original.get(p["sample_id"]) == p["component"] for p in frozen), "Frozen roles change study components")
    persons, summary, crossings, status = prepare_roles(study, people, contract, frozen)
    support = source_support(bundle, chromosomes, study)
    metadata = metadata_support(people, persons, contract["metadata_fields"])
    regions = []
    if "blocks.tsv" in contract["sha256"]:
        require(isinstance(contract.get("ld_margin_rationale"), str) and len(contract["ld_margin_rationale"].strip()) >= 10,
                "Development-only margin rationale required; no default window")
        lengths = reference_lengths(bundle/"genome.fai", contract["contig_map"], chromosomes)
        regions = plan_regions(table(bundle/"blocks.tsv", BLOCK_COLUMNS), lengths, contract["exclusion_scope"])
    missing = ["GENOMIC_MATERIALIZER_WITH_FIT_ONLY_STATISTICS", "MATCHED_B0_B1_B2_CHANNELS_AND_MASKS",
               "VERIFIED_FEATURE_LINEAGE_AND_UPSTREAM_RECEIPT", "EVALUATOR_SUPPORT_AND_PRECISION_CONTRACT"]
    if frozen is None:
        missing.append("APPROVED_FROZEN_PERSON_ROLES")
    if not regions:
        missing.append("TARGET_BLOCKS_AND_DEVELOPMENT_LD_MARGINS")
    if any(r["n_crossing_roles"] for r in crossings):
        missing.append("REVIEW_COMPONENT_THRESHOLD_SENSITIVITY_CROSS_ROLE_LINKS")
    require(not output_dir.exists(), "Output directory exists; preserve previous preparation")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    write_table(output_dir/"persons.private.tsv", persons)
    write_table(output_dir/"role_support.tsv", summary)
    write_table(output_dir/"component_sensitivity.tsv", crossings)
    write_table(output_dir/"metadata_support.private.tsv", metadata)
    write_table(output_dir/"chromosome_support.tsv", support)
    if regions:
        (output_dir/"regions.json").write_text(json.dumps(regions, indent=2)+"\n")
        region_dir = output_dir/"regions"; region_dir.mkdir()
        for block in regions:
            for kind, intervals in (("input", block["allowed_input_intervals"]), ("target", [block["target"]]), ("excluded", [block["excluded"]])):
                # BED is zero-based, half-open. Genomic summaries above are closed 1-based.
                with (region_dir/f"{block['block_id']}.{kind}.bed").open("x") as f:
                    for chrom, start, end in intervals:
                        f.write(f"{contract['contig_map'][chrom]}\t{start-1}\t{end}\n")
    result = {"schema": "r02_structure_input_preparation_v1", "status": "NO_ENTRENAMIENTO",
              "role_status": status, "n_people": len(people), "n_chromosomes": len(chromosomes),
              "n_blocks_planned": len(regions), "training_allowed": False, "genotypes_read": False,
              "features_materialized": False, "source_manifest_evidence_only": True,
              "missing_before_training": missing, "contract_sha256": sha(contract_path),
              "inputs_sha256": contract["sha256"], "code_sha256": sha(Path(__file__)),
              "role_algorithm_sha256": sha(Path(__file__).with_name("r02_structure_increment.py")),
              "contains_private_metadata": True, "public_distribution_allowed": False,
              "limitations": ["No independent biological populations or precision certified.",
                 "Metadata names do not verify population of origin or sequencing batch.",
                 "Rare identity remains source-minor M02.1; common selection/statistics must be FIT-only.",
                 "Exclude loci before PCA/GRM/Jaccard/M14 detection; clipping detected chains is insufficient.",
                 "BED plans are geometric complements, not callable territory or verified materialized features."]}
    result["outputs_sha256"] = {str(p.relative_to(output_dir)): sha(p) for p in sorted(output_dir.rglob("*")) if p.is_file()}
    (output_dir/"readiness.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    for path in output_dir.rglob("*"):
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    result = run(args.bundle, args.contract, args.output_dir)
    print(json.dumps({k: result[k] for k in ("status", "role_status", "n_people", "n_chromosomes", "n_blocks_planned")}))


if __name__ == "__main__":
    main()
