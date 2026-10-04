#!/usr/bin/env python3
"""Streaming L1 genotype support for one authenticated panel role.

No imputation, reference-block inference, phasing, role assignment or training.
Raw complete-GT counts and quality-evaluable counts have different denominators.
Only the latter exclude DP/GQ failures; GT_ONLY never claims genotype quality.

Entrypoint: r02_lai_allele_support.py --mode genotype_support, normally through
nextflow run workflows/r02_lai_allele_support.nf -params-file <params.json>.
The parameter file needs the existing catalogue inputs/limits plus mode,
genotype_contract, source_build_verification and panel_build_verification.
input_hashes must authenticate all nine paths named by INPUTS below. A panel
must contain exactly one selected role; every row is emitted for its ALL total
and each ancestry stratum (overlapping denominators, not additional people).

genotype_contract: r02_lai_genotype_contract_v1, panel_role, ordered/membership
sample SHA256, site_filter_policy PASS_ONLY/PASS_OR_UNFILTERED/IGNORE, quality
{contract_id, mode:GT_ONLY} OR {contract_id, mode:DP_GQ, min_dp, min_gq,
missing_policy:exclude|error}; null disables that single threshold, not both.
allele_representation is {status:ORIGINAL_UNSPLIT|DECOMPOSED_OTHER_ALT_MISSING|
UNKNOWN, panel_vcf_sha256, evidence}. Cite a verified processing manifest or
other provenance; never label UNKNOWN as unsplit just from the observed rows.
Reference receipts are separately produced/cached by
workflows/r02_lai_build_verification.nf and bound into build_evidence.

Outputs: soporte_alelos_dnabr.tsv.gz plus manifest.json, written last on
success. No individual IDs/GT are exported; aggregates remain restricted.
Runtime is one reader/CPU, two ordered VCF passes plus authentication hashes,
one-position buffering and per-sample counters. The limits are engineering
bounds, not biological filters. Tests use 1 CPU/2 GB and the pinned pysam
0.23.3 container; production RAM/time/I/O still require a measured pilot.
"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import r02_lai_allele_support as base
from r02_lai_build_verification import METHOD as BUILD_METHOD, reference_contract

INPUTS = (*base.INPUTS, "genotype_contract", "source_build_verification", "panel_build_verification")
FIELDS = (*base.FIELDS, "panel_ancestry", "panel_allele_index", "n_gt_missing", "n_gt_non_diploid",
          "n_site_filter_failed", "n_quality_missing", "n_quality_failed", "called_call_rate",
          "quality_call_rate", "af_counted_called", "af_counted_evaluable", "n_phased_gt_complete",
          "n_phased_carriers_called", "n_phased_carriers_evaluable", "n_homozygous_carriers_evaluable",
          "quality_status", "phase_status")
COUNTERS = ("n_gt_complete", "ac_counted_called", "an_called", "n_carriers_called", "n_gt_missing",
            "n_gt_non_diploid", "n_site_filter_failed", "n_quality_missing", "n_quality_failed",
            "n_quality_evaluable", "ac_counted_evaluable", "an_evaluable", "n_carriers_evaluable",
            "n_phased_gt_complete", "n_phased_carriers_called", "n_phased_carriers_evaluable",
            "n_homozygous_carriers_evaluable")
QC_COUNTERS = ("n_quality_missing", "n_quality_failed", "n_quality_evaluable", "ac_counted_evaluable",
               "an_evaluable", "n_carriers_evaluable", "n_phased_carriers_evaluable",
               "n_homozygous_carriers_evaluable")


def genotype_contract(args, samples):
    data = base.read_json(args.genotype_contract)
    base.require(set(data) == {"schema", "panel_role", "panel_sample_order_sha256",
                              "panel_sample_members_sha256", "site_filter_policy", "quality", "allele_representation"},
                 "Unexpected or missing genotype contract fields")
    base.require(data["schema"] == "r02_lai_genotype_contract_v1" and data["panel_role"] == args.panel_role,
                 "Genotype contract schema or role mismatch")
    base.require(data["panel_sample_order_sha256"] == base.sample_hash(samples), "Panel sample order hash mismatch")
    base.require(data["panel_sample_members_sha256"] == base.sample_hash(sorted(samples)), "Panel membership hash mismatch")
    base.require(data["site_filter_policy"] in ("PASS_ONLY", "PASS_OR_UNFILTERED", "IGNORE"),
                 "Explicit site-filter policy required")
    quality = data["quality"]
    base.require(isinstance(quality, dict) and quality.get("contract_id") and quality.get("mode") in ("GT_ONLY", "DP_GQ"),
                 "Explicit quality contract ID and mode required")
    keys = {"contract_id", "mode"}
    if quality["mode"] == "DP_GQ":
        keys |= {"min_dp", "min_gq", "missing_policy"}
        base.require(quality.get("missing_policy") in ("exclude", "error"), "Explicit missing-quality policy required")
        for name in ("min_dp", "min_gq"):
            value = quality.get(name)
            base.require(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                         and math.isfinite(value) and value >= 0), "DP/GQ thresholds must be nonnegative or null")
        base.require(quality.get("min_dp") is not None or quality.get("min_gq") is not None,
                     "DP_GQ mode requires at least one explicit threshold")
    base.require(set(quality) == keys, "Unexpected, missing or inert quality parameter")
    representation = data["allele_representation"]
    base.require(isinstance(representation, dict) and set(representation) == {"status", "panel_vcf_sha256", "evidence"},
                 "Explicit allele-representation provenance required")
    base.require(representation["status"] in ("ORIGINAL_UNSPLIT", "DECOMPOSED_OTHER_ALT_MISSING", "UNKNOWN"),
                 "Unknown allele-representation status")
    base.require(representation["panel_vcf_sha256"] == base.sha256(args.panel_vcf)
                 and isinstance(representation["evidence"], str) and representation["evidence"].strip(),
                 "Allele-representation provenance must bind the actual panel and cite evidence or uncertainty")
    return data


def verified_build(args, receipts):
    proofs = {"source": args.source_build_verification, "panel": args.panel_build_verification}
    builds, status, proof_receipts = base.build_contract(args.build_evidence, receipts, proofs)
    base.require(status == "BUILD_VERIFIED_BY_SUPPLIED_EVIDENCE", "Genotype support requires verified source and panel builds")
    digests = []
    for name in ("source", "panel"):
        digest = builds[name].get("fasta_sha256")
        base.require(isinstance(digest, str) and base.re.fullmatch(r"[0-9a-f]{64}", digest),
                     "Genotype support requires a source and panel FASTA SHA256")
        proof = base.read_json(proofs[name])
        base.require(proof.get("fasta_sha256") == digest, "Build proof does not bind the FASTA digest")
        audit = proof.get("reference_audit", {})
        base.require(proof.get("method") == BUILD_METHOD
                     and audit.get("chromosome") == args.chromosome
                     and isinstance(audit.get("records_checked"), int) and audit["records_checked"] > 0
                     and audit.get("mismatches") == 0 and audit.get("bcftools_exit_code") == 0
                     and audit.get("scope") == "ALL_VCF_RECORDS"
                     and audit.get("reference_contig") and audit.get("vcf_contigs")
                     and isinstance(proof.get("reference_provenance"), dict),
                     "Genotype support requires an all-record bcftools REF/FASTA audit receipt; hashes alone are insufficient")
        identity = reference_contract(proof["reference_provenance"])
        base.require(identity["fasta_sha256"] == digest and identity["assembly"] == builds[name]["build"]
                     and audit["reference_contig"] in identity["contigs"], "Build audit identity provenance mismatch")
        digests.append(digest)
    base.require(digests[0] == digests[1], "Source and panel FASTA digests differ")
    return builds, status, proof_receipts


def panel_preflight(args, mapping):
    """Bound lines and per-position metadata before htslib parses any calls.

    Unlike catalogue_only, this does not retain a chromosome-wide key index.
    The per-position bound is conservative for encoded records, not a promise
    that total Python/native RSS equals that number; the container is the cap.
    """
    previous, seen, position_bytes, count, peak = 0, set(), 0, 0, 0
    with base.MetadataVCF(args.panel_vcf, args.max_line_bytes) as reader:
        samples = list(reader.header.samples)
        summary = base.check_roles(args.roles, samples, args.panel_role)
        contract = genotype_contract(args, samples)
        for fields in reader:
            key = base.record_key(fields, mapping)
            base.require(key[1] >= previous, "Decreasing panel position")
            if key[1] != previous:
                seen, position_bytes = set(), 0
            base.require(key not in seen, "Duplicate panel allele key")
            seen.add(key)
            # Estimate four times the encoded row plus metadata overhead per
            # record. Native/Python RSS remains separately capped by the job.
            position_bytes += 4 * reader.last_line_bytes + 1024
            base.require(position_bytes <= args.max_panel_index_bytes, "Panel position buffer exceeds max-panel-index-bytes")
            peak = max(peak, position_bytes)
            previous, count = key[1], count + 1
            base.require(count <= args.max_panel_records, "Panel exceeds max-panel-records")
        base.require(count, "Empty panel catalogue")
    return samples, summary, contract, {"records": count, "conservative_position_buffer_bytes": peak}


def ancestry_groups(path, samples, role):
    selected = set(samples)
    with Path(path).open(newline="") as handle:
        ancestry = {row["sample_id"]: row["ancestry"] for row in csv.DictReader(handle, delimiter="\t")
                    if row["role"] == role and row["sample_id"] in selected}
    base.require("ALL" not in ancestry.values(), "Ancestry label ALL is reserved for the role total")
    groups = {"ALL": list(range(len(samples)))}
    for index, sample in enumerate(samples):
        groups.setdefault(ancestry[sample], []).append(index)
    return groups


def choose_record(records, source_key):
    compatible, unsupported = [], False
    for record in records:
        alleles = record.alleles or ()
        if record.ref != source_key[2]:
            continue
        if not all(len(a) == 1 and a in "ACGT" for a in alleles):
            unsupported = True
            continue
        if source_key[3] in alleles[1:]:
            base.require(len(set(alleles)) == len(alleles), "Duplicate allele in panel record")
            compatible.append(record)
    base.require(len(compatible) <= 1, "Multiple compatible panel records; genotype support is ambiguous")
    if compatible:
        return compatible[0], "EXACT_KEY" if len(compatible[0].alleles) == 2 else "PANEL_MULTIALLELIC_ALLELE_MAPPED"
    return None, "UNSUPPORTED_PANEL_ALLELES" if unsupported else ("POSITION_PRESENT_ALLELE_MISMATCH" if records else "NO_RECORD")


def site_allowed(record, policy):
    filters = set(record.filter.keys())
    return policy == "IGNORE" or filters == {"PASS"} or (policy == "PASS_OR_UNFILTERED" and not filters)


def quality_state(call, quality):
    missing, failed = False, False
    for field, parameter in (("DP", "min_dp"), ("GQ", "min_gq")):
        threshold = quality[parameter]
        if threshold is None:
            continue
        value = call.get(field)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            missing = True
        elif value < threshold:
            failed = True
    if missing:
        base.require(quality["missing_policy"] != "error", "Required genotype quality field is missing or invalid")
        return "missing"
    return "failed" if failed else "pass"


def count_record(record, allele_index, samples, contract):
    """One call is inspected once, then summed into role and ancestry strata."""
    counts = []
    allowed = site_allowed(record, contract["site_filter_policy"])
    quality = contract["quality"]
    for sample in samples:
        c = Counter()
        call = record.samples[sample]
        gt = call.get("GT")
        if gt is None or not gt or any(a is None for a in gt):
            c["n_gt_missing"] = 1
        elif len(gt) != 2:
            c["n_gt_non_diploid"] = 1
        else:
            base.require(all(isinstance(a, int) and 0 <= a < len(record.alleles) for a in gt), "Invalid genotype allele index")
            dose = sum(a == allele_index for a in gt)
            c.update(n_gt_complete=1, ac_counted_called=dose, an_called=2, n_carriers_called=int(dose > 0),
                     n_phased_gt_complete=int(call.phased), n_phased_carriers_called=int(call.phased and dose > 0))
            if not allowed:
                c["n_site_filter_failed"] = 1
            elif quality["mode"] == "DP_GQ":
                state = quality_state(call, quality)
                if state != "pass":
                    c["n_quality_" + state] = 1
                else:
                    c.update(n_quality_evaluable=1, ac_counted_evaluable=dose, an_evaluable=2,
                             n_carriers_evaluable=int(dose > 0), n_phased_carriers_evaluable=int(call.phased and dose > 0),
                             n_homozygous_carriers_evaluable=int(dose == 2))
        counts.append(c)
    return counts


def group_support(individual_counts, indices, contract):
    n = len(indices)
    c = Counter()
    for index in indices:
        c.update(individual_counts[index])
    result = {key: c[key] for key in COUNTERS}
    if not c["an_called"]:
        result["ac_counted_called"] = "NA"
    result.update(n_role_members=n, called_call_rate=c["n_gt_complete"] / n,
                  af_counted_called=c["ac_counted_called"] / c["an_called"] if c["an_called"] else "NA",
                  n_phase_usable_carriers="NA", phase_contract_id="NA", phase_status="OBSERVED_PHASE_NOT_VALIDATED",
                  quality_contract_id=contract["quality"]["contract_id"])
    if contract["quality"]["mode"] == "GT_ONLY":
        result.update(dict.fromkeys(QC_COUNTERS, "NA"))
        result.update(quality_call_rate="NA", af_counted_evaluable="NA", n_unresolved=n,
                      quality_status="NOT_EVALUATED_GT_ONLY", support_status="CALLED_GT_ONLY_NOT_QUALITY_VALIDATED")
    else:
        if not c["an_evaluable"]:
            result["ac_counted_evaluable"] = "NA"
        result.update(quality_call_rate=c["n_quality_evaluable"] / n, n_unresolved=n - c["n_quality_evaluable"],
                      af_counted_evaluable=c["ac_counted_evaluable"] / c["an_evaluable"] if c["an_evaluable"] else "NA",
                      quality_status="EVALUATED_UNDER_SUPPLIED_FILTERS")
        result["support_status"] = ("NO_QUALITY_EVALUABLE_GENOTYPES" if not c["an_evaluable"] else
                                    "COUNTED_ALLELE_OBSERVED_EVALUABLE" if c["ac_counted_evaluable"] else
                                    "ZERO_OBSERVED_ALL_ROLE_MEMBERS_EVALUABLE" if c["n_quality_evaluable"] == n else
                                    "ZERO_OBSERVED_PARTIAL_EVALUABILITY")
    return result


def run(args):
    import pysam
    started = time.monotonic()
    base.require(all(getattr(args, key, None) for key in ("input_hashes", "genotype_contract",
                 "source_build_verification", "panel_build_verification")), "Genotype mode requires hashes, contract and explicit build proofs")
    for name in ("expected_source_samples", "expected_rare_sites", "max_panel_records", "max_panel_bytes",
                 "max_panel_index_bytes", "max_line_bytes", "resource_check_rows"):
        base.require(getattr(args, name) > 0, "Counts and resource limits must be positive")
    base.require(args.min_free_gib >= 0 and args.chromosome in {str(i) for i in range(1, 23)}, "Invalid reserve or autosome")
    output = Path(args.outdir)
    base.require(not output.exists() and output.parent.is_dir(), "Output exists or parent directory is absent")
    base.require(Path(args.panel_vcf).stat().st_size <= args.max_panel_bytes, "Panel exceeds max-panel-bytes")
    receipts = base.input_receipts(args, INPUTS)
    builds, build_status, build_receipts = verified_build(args, receipts)
    source_contract = base.read_json(args.rare_contract)
    mapping = base.contig_mapping(args.contig_map, args.chromosome)
    base.free_disk(output.parent, args.min_free_gib)
    samples, role_summary, contract, panel_counts = panel_preflight(args, mapping)
    for proof_path, expected in ((args.source_build_verification, args.expected_rare_sites),
                                 (args.panel_build_verification, panel_counts["records"])):
        base.require(base.read_json(proof_path)["reference_audit"]["records_checked"] == expected,
                     "Build audit record count does not cover this entire input")
    groups = ancestry_groups(args.roles, samples, args.panel_role)
    counts, decoded = Counter(), 0
    with base.MetadataVCF(args.rare_vcf, args.max_line_bytes) as rare, pysam.VariantFile(str(args.panel_vcf), threads=1) as panel:
        base.validate_source_contract(source_contract, rare.header, args)
        base.require(list(panel.header.samples) == samples, "Panel order changed after preflight")
        # Grouping holds at most one position; unmatched records are discarded.
        grouped = itertools.groupby(panel, key=lambda record: record.pos)
        current = next(grouped, None)
        output.mkdir(mode=0o700)
        part = output / "soporte_alelos_dnabr.tsv.gz.partial"
        with gzip.open(part, "xt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            previous = 0
            for fields in rare:
                key = base.record_key(fields, mapping)
                base.require(key[1] > previous, "Duplicate or decreasing source position")
                previous = key[1]
                info = base.rare_info(fields)
                selected = base.validate_rare_site(SimpleNamespace(info=info, alleles=(key[2], key[3])))
                base.require(info["RARE_AN"] <= 2 * args.expected_source_samples, "Source AN exceeds diploid cohort")
                while current is not None and current[0] < key[1]:
                    current = next(grouped, None)
                records = list(current[1]) if current is not None and current[0] == key[1] else []
                record, status = choose_record(records, key)
                if record is not None and selected == 0 and contract["allele_representation"]["status"] == "UNKNOWN":
                    record, status = None, "UNKNOWN_REF_DOSAGE_ALLELE_REPRESENTATION"
                elif record is not None and selected == 0 and len(records) > 1 and contract["allele_representation"]["status"] == "ORIGINAL_UNSPLIT":
                    record, status = None, "UNKNOWN_REF_DOSAGE_MULTIPLE_RECORDS"
                allele = (key[2], key[3])[selected]
                panel_index = record.alleles.index(allele) if record is not None else "NA"
                individual_counts = count_record(record, panel_index, samples, contract) if record is not None else None
                if record is not None:
                    decoded += len(samples)
                row = dict.fromkeys(FIELDS, "NA")
                row.update(dict(zip(("chromosome", "position_bp", "REF", "ALT"), key)))
                row.update(source_build=builds["source"]["build"], counted_allele=allele, source_rare_allele_index=selected,
                           source_rare_ac=info["RARE_AC"], source_rare_an=info["RARE_AN"], source_cohort_n=args.expected_source_samples,
                           source_cohort_sha256=source_contract["cohort_sha256"], source_catalogue_sha256=receipts["rare_vcf"]["sha256"],
                           original_allele_count=2, source_site_id=":".join(map(str, key)), source_record_id=fields[2],
                           panel_role=args.panel_role, panel_source_sha256=receipts["panel_vcf"]["sha256"],
                           panel_records_at_position=len(records), panel_allele_index=panel_index, correspondence_status=status,
                           build_status=build_status, quality_contract_id=contract["quality"]["contract_id"],
                           phase_status="NOT_EVALUATED", quality_status="NOT_EVALUABLE_NO_COMPATIBLE_RECORD",
                           genotype_evidence_status="GT_READ" if record is not None else "NO_GENOTYPE_EVIDENCE_IN_THIS_SOURCE",
                           support_status="UNKNOWN_NO_COMPATIBLE_RECORD")
                for ancestry, indices in groups.items():
                    result = dict(row, panel_ancestry=ancestry, n_role_members=len(indices), n_unresolved=len(indices))
                    if individual_counts is not None:
                        result.update(group_support(individual_counts, indices, contract))
                    writer.writerow(result)
                    counts["output_rows"] += 1
                counts["source_rows"] += 1
                counts[status] += 1
                counts["source_minor_ref" if selected == 0 else "source_minor_alt"] += 1
                if counts["source_rows"] % args.resource_check_rows == 0:
                    base.free_disk(output, args.min_free_gib)
                    print(f"L1 genotype chr{args.chromosome}: source_rows={counts['source_rows']} elapsed_s={time.monotonic()-started:.1f}",
                          file=sys.stderr, flush=True)
            # Force full htslib parsing after the last rare site; preflight has
            # already authenticated the complete ordered catalogue and bounds.
            for _ in panel:
                pass
        base.require(counts["source_rows"] == args.expected_rare_sites, "Observed catalogue count differs from contract")
        for source_name, counter in (("rare_ref", "source_minor_ref"), ("rare_alt", "source_minor_alt")):
            base.require(source_contract["counts"].get(source_name, 0) == counts[counter], "Source REF/ALT counts differ")
    base.verify_inputs_unchanged(receipts)
    base.verify_inputs_unchanged(build_receipts)
    base.free_disk(output, args.min_free_gib)
    table = output / "soporte_alelos_dnabr.tsv.gz"
    part.rename(table)
    manifest = dict(schema="r02_lai_genotype_support_v1", status="COMPLETE_GENOTYPE_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION",
                    mode="genotype_support", evidence_level="FEASIBILITY", public_distribution_allowed=False,
                    contains_sample_identifiers=False, contains_individual_genotypes=False, genotype_fields_decoded=decoded,
                    input_files=receipts, build_evidence=builds, build_status=build_status, build_verification_files=build_receipts,
                    genotype_contract=contract, role_validation=role_summary, ancestry_strata={k: len(v) for k, v in groups.items()},
                    allele_representation_scope="REF dosage requires supplied representation provenance; UNKNOWN is never interpreted as REF support, including single surviving split records. Provenance is authenticated, not independently reconstructed from GT.",
                    strata_are_overlapping="ALL is the role total; ancestry strata partition that same role, not additional people",
                    counts=dict(counts), panel_counts=panel_counts, chromosome=args.chromosome, contig_mapping=mapping,
                    source_cohort_n=args.expected_source_samples, source_frequency_unchanged=True, source_catalogue_all_records_preserved=True,
                    genotype_support="CALLED_GT_COUNTS", quality_support="NOT_EVALUATED_GT_ONLY" if contract["quality"]["mode"] == "GT_ONLY" else "EVALUATED_UNDER_SUPPLIED_FILTERS",
                    donor_phase_support="OBSERVED_PHASE_ONLY_NOT_CERTIFIED", map_coverage="NOT_EVALUATED",
                    denominator_scope="called: complete diploid GT before site/DP/GQ filters; evaluable: complete diploid GT passing all supplied filters; AN=0 yields AC=AF=NA",
                    unresolved_scope="Quality-unresolved role members: n_role_members minus n_quality_evaluable; all are unresolved in GT_ONLY or without a compatible record",
                    missing_record_definition="Unknown, not zero copies; no gVCF/reference-block or absence inference",
                    multiallelic_scope="Map fixed source REF or ALT by base in one compatible SNV record; other ALT genotypes remain observed zero-dose calls; symbolic alleles unsupported",
                    phase_scope="Observed phased flags and homozygosity do not certify regional phase, donor copiability or independence",
                    limits={name: getattr(args, name) for name in ("max_panel_records", "max_panel_bytes", "max_panel_index_bytes", "max_line_bytes", "min_free_gib", "resource_check_rows")},
                    elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    package_versions={"pysam": pysam.__version__}, output_rows=counts["output_rows"],
                    code_sha256={name: base.sha256(Path(__file__).with_name(name)) for name in ("r02_lai_genotype_support.py", "r02_lai_allele_support.py", "r02_lai_build_verification.py", "r02_genomic_pair_evidence.py")},
                    outputs_sha256={table.name: base.sha256(table)})
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest
