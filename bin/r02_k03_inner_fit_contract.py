"""Shared immutable data definitions for the two distinct K03 catalogue questions."""
from __future__ import annotations
import hashlib
import re

SHARED_SCHEMA = "r02_k03_outer_fit_alt_chunks_v1"
GT_CODE_ENCODING = "DIPLOID_GT_CODE_V1_PRESERVES_PARTIAL_CALLED_ALLELES"
FIXED_SOURCE_MINOR_ENCODING = "DIPLOID_SOURCE_MINOR_DOSAGE_UINT8_0_1_2_MISSING_255"
STRICT_INNER_FIT_ASCERTAINMENT = {
    "scope": "INNER_FIT_ONLY_FROM_PRE_FREQUENCY_ORIGINALLY_BIALLELIC_PASS",
    "min_mac_inclusive": 2, "max_maf_inclusive": "0.01",
    "frequency_denominator": "CALLED_ALLELES_INCLUDING_KNOWN_ALLELE_OF_PARTIAL_DIPLOID_GT",
    "inner_fit_reselection": True, "inner_project_used_for_eligibility": False,
    "select_evaluate_used_for_eligibility": False,
    "protocol_admission": "TECHNICAL_NEW_CATALOGUE_MODE_SCIENTIFIC_ADMISSION_REQUIRED",
}
FIXED_SOURCE_MINOR_CATALOG = {
    "scope": "FIXED_SOURCE_COHORT_2723_MAC_GE_2_MAF_LE_0_01",
    "source_cohort_people": 2723,
    "source_minor_redefinition": False,
    "inner_fit_reselection": False,
    "inner_project_used_for_catalog": False,
    "protocol_admission": "TECHNICAL_FIXED_CATALOGUE_MODE_SCIENTIFIC_ADMISSION_REQUIRED",
}

TERRITORY_FIELDS = {"view_id", "excluded_intervals", "target_locus_policy",
                    "source_candidate_rule"}
TERRITORY_VIEW = re.compile(
    r"(BLOCK_AND_LD_MARGIN|WHOLE_CHROMOSOME):"
    r"(chr(0[1-9]|1[0-9]|2[0-2])_part(?:01|02))")


def validate_territory_descriptor(value, source_candidate_rule):
    """Validate shared territory syntax; the producer separately derives coordinates."""
    if not isinstance(value, dict) or set(value) != TERRITORY_FIELDS:
        raise ValueError("Exact K03 territory descriptor required")
    match = TERRITORY_VIEW.fullmatch(
        value["view_id"] if isinstance(value["view_id"], str) else "")
    if match is None:
        raise ValueError("Territory view must name one predeclared block and scope")
    scope, block_id, chromosome_text = match.groups()
    intervals = value["excluded_intervals"]
    if not (isinstance(intervals, list) and len(intervals) == 1
            and isinstance(intervals[0], dict)
            and set(intervals[0]) == {"chromosome", "start_bp", "end_bp"}):
        raise ValueError("Exactly one closed K03 exclusion interval required")
    interval = intervals[0]
    chromosome = int(chromosome_text)
    if not (type(interval["chromosome"]) is int
            and type(interval["start_bp"]) is int
            and type(interval["end_bp"]) is int
            and interval["chromosome"] == chromosome
            and 1 <= interval["start_bp"] <= interval["end_bp"]
            and (scope != "WHOLE_CHROMOSOME" or interval["start_bp"] == 1)):
        raise ValueError("Exclusion interval disagrees with its K03 view")
    if (value["target_locus_policy"] != "EXCLUDED_BEFORE_OUTPUT"
            or value["source_candidate_rule"] != source_candidate_rule):
        raise ValueError("Territory policy disagrees with catalogue mode")
    return scope, block_id, interval

def ordered_ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()
