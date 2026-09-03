#!/usr/bin/env python3
"""Freeze the M38B leave-one-NAM-unit-out locus subset from REF_TRAIN only.

The stage reopens the authenticated phased reference panel, keeps only the
exact S660 locus axis, and freezes the minor-allele orientation from the full
REF_TRAIN set.  It then omits each of the four NAM population/IBD atomic units
in turn and recomputes ancestry-specific allele counts and carrier-unit
support.  A locus enters the primary subset only when NAM has posterior
probability at least 0.8 of carrying the largest allele frequency under both
frozen Beta priors, and at least two remaining NAM units carry the allele, in
all four omissions.

There is deliberately no interface for TARGET genotypes, local-ancestry truth,
predictions or scores.  An empty primary subset is a valid, fail-closed result;
the thresholds are never relaxed after looking at downstream outcomes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO

import numpy as np

from m33_safe_bridge_core import reopen_npz, write_deterministic_npz, write_exclusive_json


ANCESTRIES = ("AFR", "EUR", "NAM")
ANCESTRY_ALIASES = {
    "AFR": "AFR",
    "African": "AFR",
    "EUR": "EUR",
    "European": "EUR",
    "NAM": "NAM",
    "Native_American": "NAM",
}
SELECTED_SCHEMA = {"chrom", "pos", "ref", "alt", "cM", "locus_id"}
REQUIRED_SPLIT_COLUMNS = {
    "sample_id", "ancestry", "atomic_unit_id", "role",
}
HEX_DIGITS = frozenset("0123456789abcdef")


class LooSubsetContractError(ValueError):
    """Raised when an authenticated input or a frozen M38B invariant differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LooSubsetContractError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"input is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate(path: Path, expected_sha256: str, label: str) -> str:
    require(
        isinstance(expected_sha256, str)
        and len(expected_sha256) == 64
        and set(expected_sha256).issubset(HEX_DIGITS),
        f"{label} expected SHA-256 is malformed",
    )
    observed = sha256_file(path)
    require(observed == expected_sha256, f"{label} SHA-256 mismatch")
    return observed


def decode_text(value: object, label: str) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        try:
            result = bytes(value).decode("ascii")
        except UnicodeDecodeError:
            raise LooSubsetContractError(f"{label} is not ASCII") from None
    else:
        result = str(value)
    require(result != "", f"{label} is empty")
    return result


def canonical_chromosome(value: object) -> str:
    return decode_text(value, "chromosome").removeprefix("chr")


VariantKey = tuple[str, int, str, str]


def variant_key(
    chrom: object,
    position: object,
    ref: object,
    alt: object,
    *,
    expected_chromosome: str,
    label: str,
) -> VariantKey:
    canonical_chrom = canonical_chromosome(chrom)
    require(canonical_chrom == expected_chromosome,
            f"{label} chromosome differs")
    if isinstance(position, (int, np.integer)) and not isinstance(position, bool):
        numeric_position = int(position)
    else:
        text = decode_text(position, f"{label} position")
        require(text.isascii() and text.isdecimal(), f"{label} position is invalid")
        numeric_position = int(text)
    require(numeric_position > 0, f"{label} position is invalid")
    ref_text = decode_text(ref, f"{label} REF")
    alt_text = decode_text(alt, f"{label} ALT")
    require(ref_text in {"A", "C", "G", "T"}
            and alt_text in {"A", "C", "G", "T"}
            and ref_text != alt_text,
            f"{label} is not a biallelic A/C/G/T SNV")
    return canonical_chrom, numeric_position, ref_text, alt_text


def axis_sha256(keys: Sequence[VariantKey], *, domain: str) -> str:
    digest = hashlib.sha256(domain.encode("ascii") + b"\0")
    for chrom, position, ref, alt in keys:
        digest.update(f"{chrom}\t{position}\t{ref}\t{alt}\n".encode("ascii"))
    return digest.hexdigest()


def unit_seed_word(unit: str) -> int:
    return int.from_bytes(
        hashlib.sha256(b"M38B_NAM_UNIT_V1\0" + unit.encode("utf-8")).digest()[:4],
        "little",
    )


@contextmanager
def open_text(path: Path) -> Iterator[TextIO]:
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    if compressed:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield handle


@dataclass(frozen=True)
class ReferenceSample:
    sample_id: str
    ancestry: str
    atomic_unit: str


def load_ref_train_samples(
    path: Path, *, expected_nam_units: int,
) -> tuple[dict[str, ReferenceSample], tuple[str, ...]]:
    samples: dict[str, ReferenceSample] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None
                and REQUIRED_SPLIT_COLUMNS.issubset(reader.fieldnames),
                "split TSV lacks required columns")
        for line_number, row in enumerate(reader, 2):
            if row["role"] != "REF_TRAIN":
                continue
            sample_id = row["sample_id"]
            require(sample_id and sample_id not in samples,
                    f"duplicate or empty REF_TRAIN sample at row {line_number}")
            try:
                ancestry = ANCESTRY_ALIASES[row["ancestry"]]
            except KeyError:
                raise LooSubsetContractError(
                    f"unsupported REF_TRAIN ancestry at row {line_number}"
                ) from None
            atomic_unit = row["atomic_unit_id"]
            require(atomic_unit, f"missing REF_TRAIN atomic unit at row {line_number}")
            samples[sample_id] = ReferenceSample(sample_id, ancestry, atomic_unit)
    require(samples, "split TSV contains no REF_TRAIN samples")
    require({sample.ancestry for sample in samples.values()} == set(ANCESTRIES),
            "REF_TRAIN must contain AFR, EUR and NAM")
    nam_units = tuple(sorted({sample.atomic_unit for sample in samples.values()
                              if sample.ancestry == "NAM"}))
    require(len(nam_units) == expected_nam_units,
            f"NAM REF_TRAIN atomic-unit count differs: {len(nam_units)}")
    return samples, nam_units


def load_selected_loci(
    path: Path, *, expected_loci: int, expected_chromosome: str,
) -> tuple[dict[str, np.ndarray], tuple[VariantKey, ...]]:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == SELECTED_SCHEMA,
                "selected-locus NPZ members differ")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    require(all(array.ndim == 1 and len(array) == expected_loci
                for array in arrays.values()),
            "selected-locus arrays have inconsistent dimensions")
    require(expected_loci > 0, "expected selected-locus count must be positive")
    require(np.unique(arrays["locus_id"]).size == expected_loci,
            "selected locus_id values are duplicated")
    require(np.all(np.isfinite(arrays["cM"])) and np.all(np.diff(arrays["cM"]) >= 0),
            "selected genetic positions must be finite and ordered")
    keys = tuple(
        variant_key(
            arrays["chrom"][index], arrays["pos"][index], arrays["ref"][index],
            arrays["alt"][index], expected_chromosome=expected_chromosome,
            label=f"selected locus {index}",
        )
        for index in range(expected_loci)
    )
    require(len(set(keys)) == expected_loci, "selected variants are duplicated")
    return arrays, keys


def parse_phased_gt(
    format_value: str, sample_value: str, *, line_number: int,
) -> tuple[int | None, int | None]:
    fields = format_value.split(":")
    require(len(fields) == len(set(fields)) and "GT" in fields,
            f"VCF row {line_number} lacks one unambiguous GT field")
    values = sample_value.split(":")
    gt_index = fields.index("GT")
    require(gt_index < len(values), f"VCF row {line_number} lacks a sample GT")
    gt = values[gt_index]
    if "/" in gt:
        require(gt == "./.", f"VCF row {line_number} has an unphased called genotype")
        return None, None
    parts = gt.split("|")
    require(len(parts) == 2, f"VCF row {line_number} has a non-diploid genotype")
    alleles: list[int | None] = []
    for value in parts:
        if value == ".":
            alleles.append(None)
        else:
            require(value in {"0", "1"},
                    f"VCF row {line_number} has a non-biallelic genotype")
            alleles.append(int(value))
    return alleles[0], alleles[1]


def collect_reference_counts(
    panel_vcf: Path,
    references: Mapping[str, ReferenceSample],
    selected_keys: Sequence[VariantKey],
    *,
    expected_chromosome: str,
    nam_units: Sequence[str],
) -> dict[str, np.ndarray]:
    locus_count = len(selected_keys)
    selected_index = {key: index for index, key in enumerate(selected_keys)}
    ancestry_index = {ancestry: index for index, ancestry in enumerate(ANCESTRIES)}
    unit_index = {unit: index for index, unit in enumerate(nam_units)}
    full_ac = np.zeros((len(ANCESTRIES), locus_count), dtype=np.int64)
    full_an = np.zeros_like(full_ac)
    nam_unit_ac = np.zeros((len(nam_units), locus_count), dtype=np.int64)
    nam_unit_an = np.zeros_like(nam_unit_ac)
    nam_unit_carrier = np.zeros((len(nam_units), locus_count), dtype=np.uint8)
    minor_code = np.full(locus_count, -1, dtype=np.int8)
    pooled_alt_ac = np.zeros(locus_count, dtype=np.int64)
    pooled_callable_an = np.zeros(locus_count, dtype=np.int64)
    found = np.zeros(locus_count, dtype=np.uint8)

    with open_text(panel_vcf) as handle:
        sample_axis: tuple[str, ...] | None = None
        selected_columns: list[int] = []
        selected_samples: list[ReferenceSample] = []
        previous_position = -1
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##") or not line.strip():
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\r\n").split("\t")
                require(len(fields) >= 10, "panel VCF has no sample columns")
                sample_axis = tuple(fields[9:])
                require(len(sample_axis) == len(set(sample_axis)),
                        "panel VCF sample axis is duplicated")
                by_name = {sample: index for index, sample in enumerate(sample_axis)}
                missing = sorted(set(references) - set(by_name))
                require(not missing,
                        f"REF_TRAIN sample absent from panel VCF: {missing[0] if missing else ''}")
                for sample_id, sample in references.items():
                    selected_columns.append(by_name[sample_id])
                    selected_samples.append(sample)
                continue
            require(sample_axis is not None, "VCF data appeared before #CHROM")
            fields = line.rstrip("\r\n").split("\t")
            require(len(fields) == 9 + len(sample_axis),
                    f"malformed panel VCF row {line_number}")
            chrom = canonical_chromosome(fields[0])
            if chrom != expected_chromosome:
                continue
            require(fields[1].isascii() and fields[1].isdecimal(),
                    f"VCF row {line_number} position is invalid")
            position = int(fields[1])
            require(position >= previous_position, "panel VCF positions are not ordered")
            previous_position = position
            if fields[3] not in {"A", "C", "G", "T"} \
                    or fields[4] not in {"A", "C", "G", "T"}:
                continue
            key = (chrom, position, fields[3], fields[4])
            locus = selected_index.get(key)
            if locus is None:
                continue
            require(found[locus] == 0, f"panel VCF duplicates selected variant {key}")
            genotypes = [
                parse_phased_gt(fields[8], fields[9 + column], line_number=line_number)
                for column in selected_columns
            ]
            alt_ac = sum(allele == 1 for gt in genotypes for allele in gt
                         if allele is not None)
            callable_an = sum(allele is not None for gt in genotypes for allele in gt)
            require(callable_an > 0, f"selected locus has no callable REF_TRAIN alleles: {key}")
            code = 1 if alt_ac <= callable_an - alt_ac else 0
            minor_code[locus] = code
            pooled_alt_ac[locus] = alt_ac
            pooled_callable_an[locus] = callable_an
            for sample, genotype in zip(selected_samples, genotypes, strict=True):
                observed = [allele for allele in genotype if allele is not None]
                dosage = sum(allele == code for allele in observed)
                ancestry = ancestry_index[sample.ancestry]
                full_ac[ancestry, locus] += dosage
                full_an[ancestry, locus] += len(observed)
                if sample.ancestry == "NAM":
                    unit = unit_index[sample.atomic_unit]
                    nam_unit_ac[unit, locus] += dosage
                    nam_unit_an[unit, locus] += len(observed)
                    nam_unit_carrier[unit, locus] |= np.uint8(dosage > 0)
            found[locus] = 1

    require(np.all(found == 1),
            f"selected variants absent from panel VCF: {int(np.sum(found == 0))}")
    require(np.all(np.isin(minor_code, (0, 1))), "minor-allele orientation is incomplete")
    require(np.array_equal(full_ac[2], nam_unit_ac.sum(axis=0))
            and np.array_equal(full_an[2], nam_unit_an.sum(axis=0)),
            "NAM atomic-unit counts do not reconcile with NAM ancestry counts")
    require(np.all(full_an > 0), "an ancestry has no callable allele at a selected locus")
    return {
        "full_minor_ac": full_ac,
        "full_callable_an": full_an,
        "minor_code": minor_code,
        "pooled_alt_ac": pooled_alt_ac,
        "pooled_callable_an": pooled_callable_an,
        "nam_unit_minor_ac": nam_unit_ac,
        "nam_unit_callable_an": nam_unit_an,
        "nam_unit_carrier": nam_unit_carrier,
    }


def leave_one_unit_out_counts(
    counts: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_ac = np.asarray(counts["full_minor_ac"], dtype=np.int64)
    full_an = np.asarray(counts["full_callable_an"], dtype=np.int64)
    unit_ac = np.asarray(counts["nam_unit_minor_ac"], dtype=np.int64)
    unit_an = np.asarray(counts["nam_unit_callable_an"], dtype=np.int64)
    unit_carrier = np.asarray(counts["nam_unit_carrier"], dtype=np.uint8)
    omissions, loci = unit_ac.shape
    loo_ac = np.broadcast_to(full_ac, (omissions, *full_ac.shape)).copy()
    loo_an = np.broadcast_to(full_an, (omissions, *full_an.shape)).copy()
    loo_ac[:, 2, :] -= unit_ac
    loo_an[:, 2, :] -= unit_an
    support = np.empty((omissions, loci), dtype=np.uint8)
    for omission in range(omissions):
        support[omission] = np.delete(unit_carrier, omission, axis=0).sum(axis=0)
    require(np.all((0 <= loo_ac) & (loo_ac <= loo_an)),
            "leave-one-unit-out AC/AN values are invalid")
    require(np.all(loo_an > 0),
            "leave-one-unit-out omission leaves an ancestry without callable alleles")
    return loo_ac, loo_an, support


def posterior_q_top_mc(
    ac: np.ndarray,
    an: np.ndarray,
    locus_id: np.ndarray,
    omitted_units: Sequence[str],
    *,
    prior: float,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    require(prior in (0.5, 1.0), "Beta prior must be 0.5 or 1.0")
    require(draws >= 4096 and draws % 2 == 0,
            "posterior draws must be even and at least 4096")
    require(ac.shape == an.shape
            and ac.shape == (len(omitted_units), len(ANCESTRIES), len(locus_id)),
            "posterior AC/AN dimensions differ")
    q_top = np.empty(ac.shape, dtype=np.float64)
    mc_se = np.empty(ac.shape, dtype=np.float64)
    prior_key = int(prior * 1_000_000)
    for omission, unit in enumerate(omitted_units):
        unit_word = unit_seed_word(unit)
        for locus in range(len(locus_id)):
            identity = int(locus_id[locus])
            sequence = np.random.SeedSequence([
                int(seed), prior_key, unit_word,
                identity & 0xFFFFFFFF, identity >> 32,
            ])
            rng = np.random.default_rng(sequence)
            frequencies = rng.beta(
                ac[omission, :, locus] + prior,
                an[omission, :, locus] - ac[omission, :, locus] + prior,
                size=(draws, len(ANCESTRIES)),
            )
            winners = np.argmax(frequencies, axis=1)
            probabilities = np.bincount(winners, minlength=len(ANCESTRIES)) / draws
            q_top[omission, :, locus] = probabilities
            mc_se[omission, :, locus] = np.sqrt(
                probabilities * (1.0 - probabilities) / draws
            )
    require(np.allclose(q_top.sum(axis=1), 1.0, rtol=0, atol=1e-15),
            "posterior q_top probabilities do not sum to one")
    return q_top, mc_se


def build_primary_mask(
    q_top_by_prior: Mapping[float, np.ndarray],
    remaining_nam_carrier_units: np.ndarray,
    *,
    q_threshold: float,
    min_remaining_units: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(tuple(sorted(q_top_by_prior)) == (0.5, 1.0),
            "both frozen Beta priors are required")
    require(0.5 < q_threshold < 1.0, "q_top threshold must lie in (0.5,1)")
    require(min_remaining_units >= 1, "minimum remaining-unit support must be positive")
    nam_q = np.stack([q_top_by_prior[prior][:, 2, :]
                      for prior in sorted(q_top_by_prior)], axis=0)
    q_min = nam_q.min(axis=(0, 1))
    support_min = remaining_nam_carrier_units.min(axis=0)
    primary = (q_min >= q_threshold) & (support_min >= min_remaining_units)
    return primary.astype(np.uint8), q_min, support_min.astype(np.uint8)


def write_exclusive_text(path: Path, text: str) -> None:
    require(not path.exists(), "refusing to overwrite output")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        os.chmod(path, 0o400)
    finally:
        if temporary.exists():
            temporary.unlink()


def _format_float(value: float) -> str:
    return format(float(value), ".12g")


def build_loo_subset(
    *,
    panel_vcf: Path,
    split_tsv: Path,
    selected_loci: Path,
    expected_panel_sha256: str,
    expected_split_sha256: str,
    expected_selected_sha256: str,
    expected_chromosome: str,
    expected_loci: int,
    expected_nam_units: int,
    beta_priors: Sequence[float],
    q_top_threshold: float,
    min_remaining_nam_units: int,
    posterior_draws: int,
    seed: int,
    output_tsv: Path,
    output_npz: Path,
    output_receipt: Path,
) -> dict[str, Any]:
    outputs = (output_tsv, output_npz, output_receipt)
    require(len(set(outputs)) == len(outputs) and not any(path.exists() for path in outputs),
            "refusing to overwrite or alias M38B outputs")
    expected_chromosome = expected_chromosome.removeprefix("chr")
    require(expected_chromosome == "22", "M38B LOO subset is frozen to chromosome 22")
    require(tuple(beta_priors) == (0.5, 1.0),
            "M38B Beta priors must be exactly 0.5 and 1.0")
    require(expected_nam_units == 4,
            "M38B requires exactly four NAM REF_TRAIN atomic units")
    require(q_top_threshold == 0.8,
            "M38B primary q_top threshold must remain 0.8")
    require(min_remaining_nam_units == 2,
            "M38B primary support threshold must remain two remaining NAM units")

    input_hashes = {
        "panel_vcf": authenticate(panel_vcf, expected_panel_sha256, "panel VCF"),
        "split_tsv": authenticate(split_tsv, expected_split_sha256, "split TSV"),
        "selected_loci": authenticate(
            selected_loci, expected_selected_sha256, "selected-locus NPZ"),
    }
    references, nam_units = load_ref_train_samples(
        split_tsv, expected_nam_units=expected_nam_units)
    selected, selected_keys = load_selected_loci(
        selected_loci, expected_loci=expected_loci,
        expected_chromosome=expected_chromosome,
    )
    counts = collect_reference_counts(
        panel_vcf, references, selected_keys,
        expected_chromosome=expected_chromosome, nam_units=nam_units,
    )
    loo_ac, loo_an, remaining_units = leave_one_unit_out_counts(counts)
    loo_af = loo_ac / loo_an
    q_top_by_prior: dict[float, np.ndarray] = {}
    q_se_by_prior: dict[float, np.ndarray] = {}
    for prior in beta_priors:
        q_top, q_se = posterior_q_top_mc(
            loo_ac, loo_an, selected["locus_id"], nam_units,
            prior=float(prior), draws=posterior_draws, seed=seed,
        )
        q_top_by_prior[float(prior)] = q_top
        q_se_by_prior[float(prior)] = q_se
    primary_mask, q_nam_min, support_min = build_primary_mask(
        q_top_by_prior, remaining_units,
        q_threshold=q_top_threshold,
        min_remaining_units=min_remaining_nam_units,
    )

    primary_indices = np.flatnonzero(primary_mask)
    primary_keys = tuple(selected_keys[index] for index in primary_indices)
    selected_axis_hash = axis_sha256(selected_keys, domain="M38B_S660_AXIS_V1")
    primary_axis_hash = axis_sha256(primary_keys, domain="M38B_PRIMARY_LOO_AXIS_V1")

    npz_payload: dict[str, np.ndarray] = {
        "ancestry": np.asarray(ANCESTRIES),
        "omitted_nam_unit": np.asarray(nam_units),
        "locus_id": selected["locus_id"],
        "chrom": selected["chrom"],
        "pos": selected["pos"],
        "ref": selected["ref"],
        "alt": selected["alt"],
        "cM": selected["cM"],
        "minor_code": counts["minor_code"],
        "pooled_alt_ac": counts["pooled_alt_ac"],
        "pooled_callable_an": counts["pooled_callable_an"],
        "full_minor_ac": counts["full_minor_ac"],
        "full_callable_an": counts["full_callable_an"],
        "loo_minor_ac": loo_ac,
        "loo_callable_an": loo_an,
        "loo_minor_af": loo_af,
        "remaining_nam_carrier_units": remaining_units,
        "q_nam_min_all_priors_omissions": q_nam_min,
        "remaining_nam_carrier_units_min": support_min,
        "primary_mask": primary_mask,
        "primary_locus_id": selected["locus_id"][primary_indices],
        "primary_pos": selected["pos"][primary_indices],
        "primary_ref": selected["ref"][primary_indices],
        "primary_alt": selected["alt"][primary_indices],
        "primary_cM": selected["cM"][primary_indices],
        "primary_minor_code": counts["minor_code"][primary_indices],
    }
    for prior in beta_priors:
        label = str(float(prior)).replace(".", "p")
        npz_payload[f"q_top_prior_{label}"] = q_top_by_prior[float(prior)]
        npz_payload[f"q_top_mc_se_prior_{label}"] = q_se_by_prior[float(prior)]
    write_deterministic_npz(output_npz, npz_payload)
    reopen_npz(output_npz, npz_payload)

    base_fields = [
        "locus_id", "chrom", "position", "ref", "alt", "cM", "minor_code",
        "omitted_nam_unit",
        *(f"{ancestry}_{suffix}" for ancestry in ANCESTRIES
          for suffix in ("minor_ac", "callable_an", "minor_af")),
        "remaining_nam_carrier_units", "NAM_q_top_prior_0P5",
        "NAM_q_top_prior_1P0", "NAM_q_top_min_all_priors_omissions",
        "remaining_nam_carrier_units_min_all_omissions", "primary_mask",
    ]
    lines = ["\t".join(base_fields)]
    for locus in range(expected_loci):
        for omission, unit in enumerate(nam_units):
            row: dict[str, object] = {
                "locus_id": int(selected["locus_id"][locus]),
                "chrom": int(selected["chrom"][locus]),
                "position": int(selected["pos"][locus]),
                "ref": decode_text(selected["ref"][locus], "REF"),
                "alt": decode_text(selected["alt"][locus], "ALT"),
                "cM": _format_float(selected["cM"][locus]),
                "minor_code": int(counts["minor_code"][locus]),
                "omitted_nam_unit": unit,
                "remaining_nam_carrier_units": int(remaining_units[omission, locus]),
                "NAM_q_top_prior_0P5": _format_float(
                    q_top_by_prior[0.5][omission, 2, locus]),
                "NAM_q_top_prior_1P0": _format_float(
                    q_top_by_prior[1.0][omission, 2, locus]),
                "NAM_q_top_min_all_priors_omissions": _format_float(q_nam_min[locus]),
                "remaining_nam_carrier_units_min_all_omissions": int(support_min[locus]),
                "primary_mask": int(primary_mask[locus]),
            }
            for ancestry_index, ancestry in enumerate(ANCESTRIES):
                row[f"{ancestry}_minor_ac"] = int(loo_ac[omission, ancestry_index, locus])
                row[f"{ancestry}_callable_an"] = int(loo_an[omission, ancestry_index, locus])
                row[f"{ancestry}_minor_af"] = _format_float(
                    loo_af[omission, ancestry_index, locus])
            lines.append("\t".join(str(row[field]) for field in base_fields))
    write_exclusive_text(output_tsv, "\n".join(lines) + "\n")

    omission_counts: dict[str, dict[str, object]] = {}
    for omission, unit in enumerate(nam_units):
        q05 = q_top_by_prior[0.5][omission, 2]
        q10 = q_top_by_prior[1.0][omission, 2]
        support = remaining_units[omission]
        omitted_people = sum(
            sample.ancestry == "NAM" and sample.atomic_unit == unit
            for sample in references.values()
        )
        omission_counts[unit] = {
            "omitted_people": omitted_people,
            "loci_q_nam_ge_0_8_prior_0_5": int(np.sum(q05 >= q_top_threshold)),
            "loci_q_nam_ge_0_8_prior_1_0": int(np.sum(q10 >= q_top_threshold)),
            "loci_support_ge_2_remaining_units": int(
                np.sum(support >= min_remaining_nam_units)),
            "loci_passing_prior_and_support_for_this_omission": int(np.sum(
                (q05 >= q_top_threshold)
                & (q10 >= q_top_threshold)
                & (support >= min_remaining_nam_units)
            )),
            "NAM_callable_AN_min": int(loo_an[omission, 2].min()),
            "NAM_callable_AN_max": int(loo_an[omission, 2].max()),
        }

    status = (
        "PASS_PRIMARY_LOO_SUBSET_FROZEN"
        if primary_indices.size
        else "PASS_ZERO_PRIMARY_LOO_SUBSET_NO_RELAXATION"
    )
    receipt: dict[str, Any] = {
        "schema_version": "m38b_loo_subset_receipt_v1",
        "stage": "M38B_REF_TRAIN_LEAVE_ONE_NAM_UNIT_OUT_SUBSET",
        "status": status,
        "scope": {
            "chromosome": expected_chromosome,
            "frequency_role": "REF_TRAIN_only",
            "selected_locus_universe": "authenticated_M34_S660",
            "target_genotypes_read": False,
            "local_ancestry_truth_read": False,
            "predictions_read": False,
            "scores_read": False,
            "king_used": False,
        },
        "orientation": {
            "frozen_before_omissions": True,
            "definition": "ALT_if_REF_TRAIN_ALT_AC_le_REF_AC_else_REF",
            "minor_code_zero_loci": int(np.sum(counts["minor_code"] == 0)),
            "minor_code_one_loci": int(np.sum(counts["minor_code"] == 1)),
        },
        "selection_contract": {
            "beta_priors": list(beta_priors),
            "q_top_definition": "posterior_probability_NAM_AF_is_largest",
            "q_top_threshold": q_top_threshold,
            "minimum_remaining_NAM_carrier_units": min_remaining_nam_units,
            "all_omissions_required": True,
            "all_priors_required": True,
            "post_outcome_relaxation_allowed": False,
            "posterior_method": "deterministic_per_locus_per_omission_Monte_Carlo",
            "posterior_draws": posterior_draws,
            "seed": seed,
        },
        "counts": {
            "REF_TRAIN_people": len(references),
            "REF_TRAIN_people_by_ancestry": {
                ancestry: sum(sample.ancestry == ancestry for sample in references.values())
                for ancestry in ANCESTRIES
            },
            "NAM_atomic_units": len(nam_units),
            "S660_loci": expected_loci,
            "primary_loci": int(primary_indices.size),
            "per_omission": omission_counts,
        },
        "identities": {
            "NAM_atomic_units": list(nam_units),
            "S660_axis_sha256": selected_axis_hash,
            "primary_locus_axis_sha256": primary_axis_hash,
        },
        "inputs": {f"{name}_sha256": digest for name, digest in input_hashes.items()},
        "outputs": {
            "per_locus_per_omission_tsv_sha256": sha256_file(output_tsv),
            "loo_subset_npz_sha256": sha256_file(output_npz),
        },
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_exclusive_json(output_receipt, receipt)
    return receipt


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("Beta priors must be comma-separated numbers") from None
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-vcf", required=True, type=Path)
    parser.add_argument("--split-tsv", required=True, type=Path)
    parser.add_argument("--selected-loci", required=True, type=Path)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-selected-sha256", required=True)
    parser.add_argument("--expected-chromosome", default="22")
    parser.add_argument("--expected-loci", type=int, default=660)
    parser.add_argument("--expected-nam-units", type=int, default=4)
    parser.add_argument("--beta-priors", type=parse_float_list, default=(0.5, 1.0))
    parser.add_argument("--q-top-threshold", type=float, default=0.8)
    parser.add_argument("--min-remaining-nam-units", type=int, default=2)
    parser.add_argument("--posterior-draws", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=3802103)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = build_loo_subset(**vars(args))
    print(json.dumps({
        "status": receipt["status"],
        "primary_loci": receipt["counts"]["primary_loci"],
        "primary_locus_axis_sha256": receipt["identities"]["primary_locus_axis_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
