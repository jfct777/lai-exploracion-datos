"""Shared immutable data definitions for strict K03 INNER_FIT ascertainment."""
from __future__ import annotations
import hashlib

SHARED_SCHEMA = "r02_k03_outer_fit_alt_chunks_v1"
GT_CODE_ENCODING = "DIPLOID_GT_CODE_V1_PRESERVES_PARTIAL_CALLED_ALLELES"
STRICT_INNER_FIT_ASCERTAINMENT = {
    "scope": "INNER_FIT_ONLY_FROM_PRE_FREQUENCY_ORIGINALLY_BIALLELIC_PASS",
    "min_mac_inclusive": 2, "max_maf_inclusive": "0.01",
    "frequency_denominator": "CALLED_ALLELES_INCLUDING_KNOWN_ALLELE_OF_PARTIAL_DIPLOID_GT",
    "inner_fit_reselection": True, "inner_project_used_for_eligibility": False,
    "select_evaluate_used_for_eligibility": False,
    "protocol_admission": "STRICT_INNER_FIT_ELIGIBILITY_REQUIRED",
}

def ordered_ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()
