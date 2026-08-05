#!/usr/bin/env python3
"""Pure helpers for choosing the carrier allele at a biallelic site.

The upstream rare-variant VCF is filtered with bcftools ``:minor`` semantics,
whereas historical downstream modules counted ALT alleles.  These helpers keep
the three audit policies explicit and independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


MODES = ("historical_alt", "minor_allele", "exclude_alt_major")


@dataclass(frozen=True)
class SiteOrientation:
    """Allele-count summary for one diploid, biallelic autosomal site."""

    alt_count: int
    allele_number: int

    @property
    def alt_is_major(self) -> bool:
        return self.allele_number > 0 and 2 * self.alt_count > self.allele_number

    @property
    def is_tie(self) -> bool:
        return self.allele_number > 0 and 2 * self.alt_count == self.allele_number

    @property
    def alt_frequency(self) -> float | None:
        if self.allele_number == 0:
            return None
        return self.alt_count / self.allele_number

    @property
    def minor_allele(self) -> str:
        if self.allele_number == 0:
            return "unknown"
        if self.is_tie:
            return "tie"
        return "REF" if self.alt_is_major else "ALT"


def called_alleles(genotype: Sequence[int]) -> tuple[int, ...]:
    """Return called allele indices, ignoring cyvcf2's trailing phase flag."""

    alleles = genotype[:2]
    return tuple(int(allele) for allele in alleles if allele is not None and int(allele) >= 0)


def summarize_orientation(genotypes: Iterable[Sequence[int]]) -> SiteOrientation:
    """Count ALT copies and called chromosomes from diploid genotypes."""

    alt_count = 0
    allele_number = 0
    for genotype in genotypes:
        alleles = called_alleles(genotype)
        allele_number += len(alleles)
        alt_count += sum(allele == 1 for allele in alleles)
    return SiteOrientation(alt_count=alt_count, allele_number=allele_number)


def carrier_indices(
    genotypes: Sequence[Sequence[int]],
    orientation: SiteOrientation,
    mode: str,
) -> frozenset[int] | None:
    """Return carrier sample indices under one explicit orientation policy.

    ``None`` means the site is excluded by policy.  A returned empty set is a
    retained site with no carriers among the selected samples.
    """

    if mode not in MODES:
        raise ValueError(f"Unknown carrier-allele mode: {mode!r}; expected one of {MODES}")
    if orientation.allele_number == 0:
        return None
    if mode == "minor_allele" and orientation.is_tie:
        return None
    if mode == "exclude_alt_major" and orientation.alt_is_major:
        return None

    carrier_allele = 0 if mode == "minor_allele" and orientation.alt_is_major else 1
    return frozenset(
        sample_idx
        for sample_idx, genotype in enumerate(genotypes)
        if carrier_allele in called_alleles(genotype)
    )


def dosage_for_mode(
    genotype: Sequence[int],
    orientation: SiteOrientation,
    mode: str,
) -> int | None:
    """Return 0/1/2 dosage for a mode, or ``None`` for missing/excluded."""

    if mode not in MODES:
        raise ValueError(f"Unknown carrier-allele mode: {mode!r}; expected one of {MODES}")
    if orientation.allele_number == 0:
        return None
    if mode == "minor_allele" and orientation.is_tie:
        return None
    if mode == "exclude_alt_major" and orientation.alt_is_major:
        return None

    alleles = called_alleles(genotype)
    if len(alleles) != 2:
        return None
    counted_allele = 0 if mode == "minor_allele" and orientation.alt_is_major else 1
    return sum(allele == counted_allele for allele in alleles)
