#!/usr/bin/env python3
"""Prepare exploratory M34 FLARE inputs and rare-variant factors.

The bridge walks the phased panel and mosaic VCFs together exactly once.  It
uses only REF_TRAIN genotypes from the panel, treats SOURCE_VALID solely as the
upstream mosaic donor role, and never reads SOURCE_TEST genotype fields.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import os
import shutil
import struct
import tempfile
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence, TextIO

import numpy as np

from m33_safe_bridge_core import (
    reopen_npz,
    sample_key,
    write_deterministic_npz,
    write_exclusive_json,
)
from m34_generate_mosaics import (
    canonical_contig,
    read_genetic_map,
)


ANCESTRIES = ("AFR", "EUR", "NAM")
ANCESTRY_MAP = {
    "AFR": "AFR",
    "African": "AFR",
    "EUR": "EUR",
    "European": "EUR",
    "NAM": "NAM",
    "Native_American": "NAM",
}
BIOLOGICAL_ROLES = ("REF_TRAIN", "SOURCE_VALID", "SOURCE_TEST", "DISCOVERY")
MOSAIC_DONOR_ROLES = ("SOURCE_VALID", "SOURCE_TEST")
MIN_MAC = 2
MAX_MAF_EXCLUSIVE = 0.01
ANCESTRY_AF_AUDIT_THRESHOLDS = (0.05, 0.10, 0.20)
LOCUS_DOMAIN = b"M34_NAM_LOCUS_V1|"
BGZF_MAX_UNCOMPRESSED = 64 * 1024 - 256
BGZF_EOF = bytes.fromhex(
    "1f8b08040000000000ff0600424302001b0003000000000000000000"
)
OUTPUT_NAMES = {
    "reference_vcf": "m34_ref_train.chr22.vcf.gz",
    "target_vcf": "m34_target.chr22.vcf.gz",
    "sample_map": "m34_ref_train.sample_map.tsv",
    "selected": "m34_selected_loci.npz",
    "target": "m34_target_rare_diploid.npz",
    "reference": "m34_reference_rare_summary.npz",
    "receipt": "m34_panel_factors.receipt.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _DigestRawReader(io.RawIOBase):
    """Read-only raw wrapper that hashes the exact compressed input bytes."""

    def __init__(self, raw: io.BufferedReader) -> None:
        self._raw = raw
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        count = self._raw.readinto(buffer)
        if count:
            block = memoryview(buffer)[:count]
            self.digest.update(block)
            self.byte_count += count
        return count

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


@contextmanager
def open_hashed_text(path: Path) -> Iterator[tuple[TextIO, _DigestRawReader]]:
    """Open gzip/plain text while hashing raw bytes during the same traversal."""
    require(path.is_file() and not path.is_symlink(), f"invalid input file: {path}")
    raw = path.open("rb", buffering=0)
    meter = _DigestRawReader(raw)
    buffered = io.BufferedReader(meter)
    compressed = buffered.peek(2)[:2] == b"\x1f\x8b"
    binary: io.BufferedIOBase
    if compressed:
        binary = gzip.GzipFile(fileobj=buffered, mode="rb")
    else:
        binary = buffered
    text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
    try:
        yield text, meter
    finally:
        text.close()


def _bgzf_block(value: bytes) -> bytes:
    compressor = zlib.compressobj(level=6, method=zlib.DEFLATED, wbits=-15)
    compressed = compressor.compress(value) + compressor.flush()
    block_size = 18 + len(compressed) + 8
    require(block_size <= 65_536, "BGZF block exceeds 64 KiB")
    header = (
        struct.pack("<BBBBIBBH", 31, 139, 8, 4, 0, 0, 255, 6)
        + b"BC"
        + struct.pack("<HH", 2, block_size - 1)
    )
    footer = struct.pack("<II", zlib.crc32(value) & 0xFFFFFFFF, len(value))
    return header + compressed + footer


class _BgzfWriter(io.RawIOBase):
    """Minimal deterministic BGZF writer for sequential FLARE VCF inputs."""

    def __init__(self, raw: io.BufferedWriter) -> None:
        self._raw = raw
        self._buffer = bytearray()
        self._finished = False

    def writable(self) -> bool:
        return True

    def write(self, value: bytes | bytearray) -> int:
        require(not self._finished, "cannot write after BGZF finalization")
        data = bytes(value)
        self._buffer.extend(data)
        while len(self._buffer) >= BGZF_MAX_UNCOMPRESSED:
            chunk = bytes(self._buffer[:BGZF_MAX_UNCOMPRESSED])
            del self._buffer[:BGZF_MAX_UNCOMPRESSED]
            self._raw.write(_bgzf_block(chunk))
        return len(data)

    def finish(self) -> None:
        if self._finished:
            return
        if self._buffer:
            self._raw.write(_bgzf_block(bytes(self._buffer)))
            self._buffer.clear()
        self._raw.write(BGZF_EOF)
        self._finished = True


@contextmanager
def deterministic_bgzf_text(path: Path) -> Iterator[TextIO]:
    """Write a deterministic BGZF stream without bcftools or external codecs."""
    with path.open("xb") as raw:
        writer = _BgzfWriter(raw)
        text = io.TextIOWrapper(writer, encoding="utf-8", newline="")
        try:
            yield text
        finally:
            text.flush()
            text.detach()
            writer.finish()
            raw.flush()
            os.fsync(raw.fileno())


@dataclass(frozen=True)
class SplitContract:
    reference_ancestry: Mapping[str, str]
    role_counts: Mapping[str, int]
    all_biological_samples: frozenset[str]


@dataclass(frozen=True)
class VcfHeader:
    metadata: tuple[str, ...]
    columns: tuple[str, ...]
    samples: tuple[str, ...]


@dataclass(frozen=True)
class VcfRecord:
    line_number: int
    fields: tuple[str, ...]


def _normalize_ancestry(value: str, *, sample: str) -> str:
    try:
        return ANCESTRY_MAP[value]
    except KeyError:
        raise ValueError(f"unsupported ancestry for biological sample {sample}: {value}") from None


def load_split_contract(path: Path) -> SplitContract:
    required = {
        "sample_id", "ancestry", "canonical_population",
        "atomic_unit_id", "role",
    }
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None and required.issubset(reader.fieldnames),
                f"split TSV lacks required fields: {sorted(required)}")
        rows = [dict(row) for row in reader]
    require(rows, "split TSV is empty")

    samples: set[str] = set()
    reference_ancestry: dict[str, str] = {}
    role_counts = {role: 0 for role in BIOLOGICAL_ROLES}
    roles_by_population: dict[str, set[str]] = {}
    roles_by_unit: dict[str, set[str]] = {}
    all_biological_samples: set[str] = set()
    for line_number, row in enumerate(rows, 2):
        sample = row["sample_id"]
        role = row["role"]
        require(sample and sample == sample.strip(),
                f"empty or padded split sample_id at row {line_number}")
        require(sample not in samples, f"duplicate split sample_id: {sample}")
        samples.add(sample)
        if role not in BIOLOGICAL_ROLES:
            continue
        population = row["canonical_population"]
        unit = row["atomic_unit_id"]
        require(population and unit,
                f"empty population or atomic unit for biological sample {sample}")
        ancestry = _normalize_ancestry(row["ancestry"], sample=sample)
        role_counts[role] += 1
        all_biological_samples.add(sample)
        roles_by_population.setdefault(population, set()).add(role)
        roles_by_unit.setdefault(unit, set()).add(role)
        if role == "REF_TRAIN":
            reference_ancestry[sample] = ancestry

    require(reference_ancestry, "split contains no REF_TRAIN samples")
    require(set(reference_ancestry.values()) == set(ANCESTRIES),
            "REF_TRAIN must contain AFR, EUR and NAM")
    for population, roles in roles_by_population.items():
        require(len(roles) == 1,
                f"canonical_population crosses biological roles: {population}")
    for unit, roles in roles_by_unit.items():
        require(len(roles) == 1,
                f"atomic_unit_id crosses biological roles: {unit}")
    return SplitContract(reference_ancestry, role_counts,
                         frozenset(all_biological_samples))


def read_vcf_header(handle: TextIO, label: str) -> VcfHeader:
    metadata: list[str] = []
    for line_number, line in enumerate(handle, 1):
        if line.startswith("##"):
            metadata.append(line.rstrip("\r\n"))
            continue
        if line.startswith("#CHROM"):
            fields = tuple(line.rstrip("\r\n").split("\t"))
            require(len(fields) >= 10 and fields[:9] ==
                    ("#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"),
                    f"{label} VCF header columns differ")
            samples = fields[9:]
            require(samples and len(samples) == len(set(samples)),
                    f"{label} VCF sample axis is empty or duplicated")
            return VcfHeader(tuple(metadata), fields, samples)
        raise ValueError(f"{label} VCF has malformed header at line {line_number}")
    raise ValueError(f"{label} VCF lacks #CHROM header")


def iter_vcf_records(handle: TextIO, header: VcfHeader,
                     label: str) -> Iterator[VcfRecord]:
    line_number = len(header.metadata) + 2
    for line in handle:
        if not line.strip():
            line_number += 1
            continue
        require(not line.startswith("#"),
                f"{label} VCF contains a header line after #CHROM")
        fields = tuple(line.rstrip("\r\n").split("\t"))
        require(len(fields) == 9 + len(header.samples),
                f"malformed {label} VCF row {line_number}")
        yield VcfRecord(line_number, fields)
        line_number += 1


def _gt_from_sample_field(format_value: str, sample_value: str,
                          *, label: str, sample: str, line_number: int) -> tuple[str, tuple[int | None, int | None]]:
    format_fields = format_value.split(":")
    require(len(format_fields) == len(set(format_fields)) and "GT" in format_fields,
            f"{label} VCF row {line_number} lacks one unambiguous GT field")
    values = sample_value.split(":")
    gt_index = format_fields.index("GT")
    require(gt_index < len(values),
            f"{label} VCF row {line_number} lacks GT for sample {sample}")
    gt = values[gt_index]
    parts = gt.split("|")
    require(len(parts) == 2,
            f"{label} VCF row {line_number} has unphased/non-diploid GT for sample {sample}")
    alleles: list[int | None] = []
    for allele in parts:
        if allele == ".":
            alleles.append(None)
        else:
            require(allele in {"0", "1"},
                    f"{label} VCF row {line_number} has non-biallelic GT for sample {sample}")
            alleles.append(int(allele))
    return gt, (alleles[0], alleles[1])


def _canonical_key(fields: Sequence[str], expected_chrom: str,
                   *, label: str, line_number: int) -> tuple[str, int, str, str]:
    chrom = canonical_contig(fields[0])
    require(chrom == expected_chrom,
            f"unexpected chromosome in {label} VCF row {line_number}: {fields[0]}")
    try:
        position = int(fields[1])
    except ValueError:
        raise ValueError(f"invalid position in {label} VCF row {line_number}") from None
    require(position > 0, f"invalid position in {label} VCF row {line_number}")
    ref, alt = fields[3], fields[4]
    require(ref and alt, f"empty REF/ALT in {label} VCF row {line_number}")
    return chrom, position, ref, alt


def _update_axis_hash(digest: "hashlib._Hash", key: tuple[str, int, str, str]) -> None:
    digest.update(("\t".join((key[0], str(key[1]), key[2], key[3])) + "\n").encode("utf-8"))


def _locus_id(chrom: str, position: int, ref: str, alt: str) -> int:
    payload = f"{chrom}|{position}|{ref}|{alt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(LOCUS_DOMAIN + payload).digest()[:8], "little")


def _write_vcf_header(handle: TextIO, source: VcfHeader,
                      samples: Sequence[str], kind: str,
                      mosaic_donor_role: str) -> None:
    for line in source.metadata:
        handle.write(line + "\n")
    if not any(line.startswith("##FORMAT=<ID=GT,") for line in source.metadata):
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype">\n')
    handle.write("##m34_bridge_scope=exploratory_only\n")
    handle.write(f"##m34_bridge_vcf_role={kind}\n")
    handle.write("##m34_reference_and_frequency_role=REF_TRAIN\n")
    handle.write(f"##m34_mosaic_donor_role_upstream={mosaic_donor_role}\n")
    handle.write("##m34_non_reference_panel_genotypes_opened=false\n")
    handle.write(
        "##m34_source_test_mosaic_donors_upstream="
        f"{str(mosaic_donor_role == 'SOURCE_TEST').lower()}\n"
    )
    handle.write(
        "##m34_flare_marker_filter="
        "biallelic_SNV_ACGT_complete_GT_in_REF_TRAIN_and_TARGET_REF_MAC_ge_1\n"
    )
    handle.write("\t".join((*source.columns[:9], *samples)) + "\n")


def _write_readonly_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o400)


def _input_descriptor(path: Path, digest: str, byte_count: int) -> dict[str, object]:
    return {"name": path.name, "sha256": digest, "bytes": byte_count}


def _output_descriptor(path: Path) -> dict[str, object]:
    return {"name": path.name, "sha256": sha256_file(path),
            "bytes": path.stat().st_size}


def _as_bytes(values: Sequence[str], minimum_width: int = 1) -> np.ndarray:
    width = max(minimum_width, *(len(value.encode("ascii")) for value in values))
    return np.asarray([value.encode("ascii") for value in values], dtype=f"|S{width}")


def prepare_panel_factors(
    *,
    panel_vcf: Path,
    mosaic_vcf: Path,
    split_tsv: Path,
    genetic_map_path: Path,
    outdir: Path,
    chromosome: str = "22",
    min_mac: int = MIN_MAC,
    max_maf_exclusive: float = MAX_MAF_EXCLUSIVE,
    mosaic_donor_role: str = "SOURCE_VALID",
) -> dict[str, object]:
    """Create an atomic exploratory bridge directory and return its receipt."""
    paths = (panel_vcf, mosaic_vcf, split_tsv, genetic_map_path)
    require(all(isinstance(path, Path) for path in (*paths, outdir)),
            "all paths must be pathlib.Path values")
    require(type(min_mac) is int and min_mac == MIN_MAC,
            f"this pilot freezes --min-mac at {MIN_MAC}")
    require(type(max_maf_exclusive) is float and
            max_maf_exclusive == MAX_MAF_EXCLUSIVE,
            f"this pilot freezes --max-maf-exclusive at {MAX_MAF_EXCLUSIVE}")
    require(mosaic_donor_role in MOSAIC_DONOR_ROLES,
            "mosaic donor role must be SOURCE_VALID or SOURCE_TEST")
    expected_chrom = canonical_contig(chromosome)
    require(expected_chrom == "22", "this pilot is frozen to chromosome 22")
    require(len({path.resolve() for path in paths}) == len(paths),
            "input paths must be distinct")
    for path in paths:
        require(path.is_file() and not path.is_symlink(), f"invalid input file: {path}")
    require(not outdir.exists(), "output directory already exists")
    outdir.parent.mkdir(parents=True, exist_ok=True)

    split = load_split_contract(split_tsv)
    genetic_map = read_genetic_map(genetic_map_path, expected_chrom)
    split_hash = sha256_file(split_tsv)
    map_hash = sha256_file(genetic_map_path)
    stage = Path(tempfile.mkdtemp(prefix=f".{outdir.name}.stage.", dir=outdir.parent))
    try:
        reference_vcf_path = stage / OUTPUT_NAMES["reference_vcf"]
        target_vcf_path = stage / OUTPUT_NAMES["target_vcf"]
        sample_map_path = stage / OUTPUT_NAMES["sample_map"]
        selected_path = stage / OUTPUT_NAMES["selected"]
        target_path = stage / OUTPUT_NAMES["target"]
        reference_path = stage / OUTPUT_NAMES["reference"]

        total_axis_hash = hashlib.sha256(b"M34_NAM_ALL_AXIS_V1\0")
        flare_axis_hash = hashlib.sha256(b"M34_NAM_FLARE_AXIS_V1\0")
        selected_axis_hash = hashlib.sha256(b"M34_NAM_SELECTED_AXIS_V1\0")
        total_records = 0
        factor_snv_records = 0
        flare_complete_snv_records = 0
        skipped_non_snv = 0
        flare_missing_exclusions = 0
        flare_monomorphic_exclusions = 0
        previous_position = -1
        position_keys: set[tuple[str, int, str, str]] = set()

        selected_rows: list[tuple[int, int, str, str, int, float, bool]] = []
        target_dosage_rows: list[list[int]] = []
        target_mask_rows: list[list[int]] = []
        reference_ac_rows: list[list[int]] = []
        reference_an_rows: list[list[int]] = []
        ref_missing_alleles = 0
        target_missing_genotypes = 0

        with open_hashed_text(panel_vcf) as (panel_handle, panel_meter), \
                open_hashed_text(mosaic_vcf) as (mosaic_handle, mosaic_meter):
            panel_header = read_vcf_header(panel_handle, "panel")
            mosaic_header = read_vcf_header(mosaic_handle, "mosaic")
            collisions = set(mosaic_header.samples) & split.all_biological_samples
            if collisions:
                raise ValueError(
                    "mosaic targets overlap split biological samples: "
                    f"{sorted(collisions)[0]}"
                )
            require(set(panel_header.samples).isdisjoint(mosaic_header.samples),
                    "panel and mosaic VCF sample axes overlap")

            panel_index = {sample: index for index, sample in enumerate(panel_header.samples)}
            missing_reference = set(split.reference_ancestry) - set(panel_index)
            if missing_reference:
                raise ValueError(
                    f"REF_TRAIN sample absent from panel VCF: {sorted(missing_reference)[0]}"
                )
            reference_samples = tuple(
                sample for ancestry in ANCESTRIES for sample in panel_header.samples
                if split.reference_ancestry.get(sample) == ancestry
            )
            require(len(reference_samples) == len(split.reference_ancestry),
                    "REF_TRAIN panel axis is incomplete")
            reference_indices = tuple(panel_index[sample] for sample in reference_samples)
            reference_labels = tuple(split.reference_ancestry[sample]
                                     for sample in reference_samples)
            ancestry_indices = tuple(
                tuple(index for index, label in enumerate(reference_labels)
                      if label == ancestry)
                for ancestry in ANCESTRIES
            )
            require(all(indices for indices in ancestry_indices),
                    "REF_TRAIN ancestry group is empty")
            require(2 * len(reference_samples) <= np.iinfo(np.uint16).max,
                    "REF_TRAIN callable AN exceeds uint16 schema")
            target_samples = mosaic_header.samples

            sample_map_value = "".join(
                f"{sample}\t{split.reference_ancestry[sample]}\n"
                for sample in reference_samples
            )
            _write_readonly_text(sample_map_path, sample_map_value)

            with deterministic_bgzf_text(reference_vcf_path) as reference_out, \
                    deterministic_bgzf_text(target_vcf_path) as target_out:
                _write_vcf_header(reference_out, panel_header, reference_samples,
                                  "REFERENCE_REF_TRAIN", mosaic_donor_role)
                _write_vcf_header(target_out, mosaic_header, target_samples,
                                  f"TARGET_{mosaic_donor_role}_MOSAICS",
                                  mosaic_donor_role)
                panel_records = iter_vcf_records(panel_handle, panel_header, "panel")
                mosaic_records = iter_vcf_records(mosaic_handle, mosaic_header, "mosaic")
                for pair_number, pair in enumerate(
                        itertools.zip_longest(panel_records, mosaic_records), 1):
                    panel_record, mosaic_record = pair
                    require(panel_record is not None and mosaic_record is not None,
                            "panel and mosaic VCF record counts differ")
                    panel_key = _canonical_key(panel_record.fields, expected_chrom,
                                               label="panel",
                                               line_number=panel_record.line_number)
                    mosaic_key = _canonical_key(mosaic_record.fields, expected_chrom,
                                                label="mosaic",
                                                line_number=mosaic_record.line_number)
                    panel_raw_axis = tuple(panel_record.fields[index]
                                           for index in (0, 1, 3, 4))
                    mosaic_raw_axis = tuple(mosaic_record.fields[index]
                                            for index in (0, 1, 3, 4))
                    require(panel_key == mosaic_key and panel_raw_axis == mosaic_raw_axis,
                            f"panel/mosaic VCF axis differs at record {pair_number}")
                    position = panel_key[1]
                    require(position >= previous_position, "VCF positions are not sorted")
                    if position != previous_position:
                        position_keys.clear()
                        previous_position = position
                    require(panel_key not in position_keys,
                            f"duplicate VCF locus axis at {panel_key}")
                    position_keys.add(panel_key)
                    total_records += 1
                    _update_axis_hash(total_axis_hash, panel_key)
                    ref, alt = panel_key[2], panel_key[3]
                    if (ref not in {"A", "C", "G", "T"} or
                            alt not in {"A", "C", "G", "T"} or ref == alt):
                        skipped_non_snv += 1
                        continue
                    factor_snv_records += 1
                    cm = genetic_map.bp_to_cm(position)

                    reference_gts: list[str] = []
                    reference_states: list[tuple[int | None, int | None]] = []
                    for sample, index in zip(reference_samples, reference_indices):
                        gt, states = _gt_from_sample_field(
                            panel_record.fields[8], panel_record.fields[9 + index],
                            label="panel REF_TRAIN", sample=sample,
                            line_number=panel_record.line_number,
                        )
                        reference_gts.append(gt)
                        reference_states.append(states)
                    target_gts: list[str] = []
                    target_states: list[tuple[int | None, int | None]] = []
                    for index, sample in enumerate(target_samples):
                        gt, states = _gt_from_sample_field(
                            mosaic_record.fields[8], mosaic_record.fields[9 + index],
                            label="mosaic TARGET", sample=sample,
                            line_number=mosaic_record.line_number,
                        )
                        target_gts.append(gt)
                        target_states.append(states)

                    flare_complete = all(
                        allele is not None
                        for states in (*reference_states, *target_states)
                        for allele in states
                    )
                    callable_alleles = [allele for pair in reference_states
                                        for allele in pair if allele is not None]
                    ref_ac = callable_alleles.count(0)
                    alt_ac = callable_alleles.count(1)
                    callable_an = ref_ac + alt_ac
                    ref_missing_alleles += 2 * len(reference_states) - callable_an
                    require(callable_an > 0,
                            f"REF_TRAIN has no callable alleles at {panel_key}")
                    ref_mac = min(ref_ac, alt_ac)

                    if flare_complete and ref_mac >= 1:
                        reference_out.write("\t".join(
                            (*panel_record.fields[:8], "GT", *reference_gts)
                        ) + "\n")
                        target_out.write("\t".join(
                            (*mosaic_record.fields[:8], "GT", *target_gts)
                        ) + "\n")
                        flare_complete_snv_records += 1
                        _update_axis_hash(flare_axis_hash, panel_key)
                    elif not flare_complete:
                        flare_missing_exclusions += 1
                    else:
                        flare_monomorphic_exclusions += 1

                    minor_is_alt = alt_ac <= ref_ac
                    mac = ref_mac
                    maf = mac / callable_an
                    if mac < min_mac or not maf < max_maf_exclusive:
                        continue

                    locus_id = _locus_id(*panel_key)
                    selected_rows.append((locus_id, position, ref, alt,
                                          callable_an, cm, minor_is_alt))
                    _update_axis_hash(selected_axis_hash, panel_key)
                    target_dosage: list[int] = []
                    target_mask: list[int] = []
                    minor_code = 1 if minor_is_alt else 0
                    for states in target_states:
                        observed = all(allele is not None for allele in states)
                        target_mask.append(int(observed))
                        target_dosage.append(
                            sum(allele == minor_code for allele in states) if observed else 0
                        )
                        target_missing_genotypes += int(not observed)
                    target_dosage_rows.append(target_dosage)
                    target_mask_rows.append(target_mask)
                    per_ancestry_ac: list[int] = []
                    per_ancestry_an: list[int] = []
                    for indices in ancestry_indices:
                        alleles = [allele for index in indices
                                   for allele in reference_states[index]
                                   if allele is not None]
                        per_ancestry_an.append(len(alleles))
                        per_ancestry_ac.append(sum(allele == minor_code for allele in alleles))
                    reference_ac_rows.append(per_ancestry_ac)
                    reference_an_rows.append(per_ancestry_an)

            panel_hash = panel_meter.digest.hexdigest()
            panel_bytes = panel_meter.byte_count
            mosaic_hash = mosaic_meter.digest.hexdigest()
            mosaic_bytes = mosaic_meter.byte_count

        os.chmod(reference_vcf_path, 0o400)
        os.chmod(target_vcf_path, 0o400)
        require(total_records > 0 and factor_snv_records > 0,
                "VCF axis has no matching biallelic A/C/G/T SNVs")
        require(flare_complete_snv_records > 0,
                "no matching SNV has complete REF_TRAIN and TARGET genotypes for FLARE")
        require(selected_rows, "no REF_TRAIN locus passes frozen MAC/MAF selection")
        locus_ids = [row[0] for row in selected_rows]
        require(len(locus_ids) == len(set(locus_ids)), "stable locus_id collision")

        order = sorted(range(len(selected_rows)),
                       key=lambda index: (selected_rows[index][5],
                                          selected_rows[index][1],
                                          selected_rows[index][0]))
        rows = [selected_rows[index] for index in order]
        locus_axis = np.asarray([row[0] for row in rows], dtype="<u8")
        selected_arrays = {
            "locus_id": locus_axis,
            "chrom": np.full(len(rows), 22, dtype="|u1"),
            "pos": np.asarray([row[1] for row in rows], dtype="<i8"),
            "ref": _as_bytes([row[2] for row in rows]),
            "alt": _as_bytes([row[3] for row in rows]),
            "cM": np.asarray([row[5] for row in rows], dtype="<f8"),
        }
        dosage_locus_sample = np.asarray(
            [target_dosage_rows[index] for index in order], dtype="|i1"
        )
        mask_locus_sample = np.asarray(
            [target_mask_rows[index] for index in order], dtype="|u1"
        )
        target_arrays = {
            "sample_key_sha256": np.asarray(
                [sample_key(sample) for sample in target_samples], dtype="|S64"
            ),
            "locus_id": locus_axis,
            "minor_dosage": np.ascontiguousarray(dosage_locus_sample.T),
            "observed_mask": np.ascontiguousarray(mask_locus_sample.T),
        }
        ac = np.asarray([reference_ac_rows[index] for index in order], dtype="<u2").T
        an = np.asarray([reference_an_rows[index] for index in order], dtype="<u2").T
        af = np.divide(ac, an, out=np.zeros_like(ac, dtype="<f8"), where=an > 0)
        ancestry_af_counts: dict[str, object] = {}
        for threshold in ANCESTRY_AF_AUDIT_THRESHOLDS:
            suffix = f"{threshold:.2f}"
            ancestry_af_counts[
                f"selected_loci_max_ancestry_af_ge_{suffix}"
            ] = int(np.any(af >= threshold, axis=0).sum())
            ancestry_af_counts[
                f"selected_loci_by_ancestry_af_ge_{suffix}"
            ] = {
                ancestry: int((af[index] >= threshold).sum())
                for index, ancestry in enumerate(ANCESTRIES)
            }
        reference_arrays = {
            "ancestry": np.asarray([value.encode("ascii") for value in ANCESTRIES],
                                   dtype="|S4"),
            "locus_id": locus_axis,
            "minor_ac": np.ascontiguousarray(ac),
            "callable_an": np.ascontiguousarray(an),
            "minor_af": np.ascontiguousarray(af),
            "observed_mask": np.ascontiguousarray((an > 0).astype("|u1")),
            "no_support": np.ascontiguousarray(((an > 0) & (ac == 0)).astype("|u1")),
        }
        write_deterministic_npz(selected_path, selected_arrays)
        write_deterministic_npz(target_path, target_arrays)
        write_deterministic_npz(reference_path, reference_arrays)
        reopen_npz(selected_path, selected_arrays)
        reopen_npz(target_path, target_arrays)
        reopen_npz(reference_path, reference_arrays)

        data_outputs = [
            reference_vcf_path, target_vcf_path, sample_map_path,
            selected_path, target_path, reference_path,
        ]
        orientation_alt = sum(row[6] for row in rows)
        receipt: dict[str, object] = {
            "schema_version": "m34_panel_factors_receipt_v1",
            "stage": "M34_EXPLORATORY_VCF_TO_FACTORS_BRIDGE",
            "decision": f"PASS_EXPLORATORY_PANEL_FACTORS_{mosaic_donor_role}_MOSAICS",
            "scope": {
                "exploratory_only": True,
                "confirmatory_validation": False,
                "generalizes_to_dnabr": False,
            },
            "parameters": {
                "chromosome": 22,
                "min_mac": min_mac,
                "max_maf_exclusive": max_maf_exclusive,
                "minor_allele_orientation":
                    "minor_is_alt_when_ALT_count_le_REF_count_tie_ALT",
                "flare_marker_policy":
                    "matching_biallelic_SNV_ACGT_with_complete_REF_TRAIN_and_TARGET_GT_and_REF_MAC_ge_1",
                "rare_factor_policy":
                    "matching_biallelic_SNV_ACGT_selected_by_pooled_REF_TRAIN_MAC_and_MAF",
                "rare_factor_missingness_policy":
                    "preserve_REF_callable_AN_and_TARGET_observed_mask_even_when_excluded_from_FLARE",
                "ancestry_af_audit": {
                    "thresholds": list(ANCESTRY_AF_AUDIT_THRESHOLDS),
                    "source": "REF_TRAIN_only",
                    "used_for_primary_selection": False,
                },
            },
            "roles": {
                "reference_role": "REF_TRAIN",
                "frequency_role": "REF_TRAIN",
                "mosaic_donor_role_upstream": mosaic_donor_role,
                "source_valid_panel_genotypes_opened": False,
                "source_test_panel_genotypes_opened": False,
                "source_test_mosaic_donors_upstream":
                    mosaic_donor_role == "SOURCE_TEST",
                "source_test_open": mosaic_donor_role == "SOURCE_TEST",
            },
            "inputs": {
                "panel_vcf": _input_descriptor(panel_vcf, panel_hash, panel_bytes),
                "mosaic_vcf": _input_descriptor(mosaic_vcf, mosaic_hash, mosaic_bytes),
                "split_tsv": _input_descriptor(split_tsv, split_hash,
                                                split_tsv.stat().st_size),
                "genetic_map": _input_descriptor(genetic_map_path, map_hash,
                                                  genetic_map_path.stat().st_size),
            },
            "outputs": [_output_descriptor(path) for path in data_outputs],
            "counts": {
                "axis_records_total": total_records,
                "biallelic_snv_records_for_factor_evaluation": factor_snv_records,
                "complete_biallelic_snv_records_for_flare": flare_complete_snv_records,
                "snv_records_excluded_from_flare_for_missing_gt": flare_missing_exclusions,
                "complete_snv_records_excluded_from_flare_for_ref_monomorphic":
                    flare_monomorphic_exclusions,
                "non_snv_or_non_biallelic_records_skipped": skipped_non_snv,
                "rare_loci_selected": len(rows),
                "minor_is_alt_loci": orientation_alt,
                "minor_is_ref_loci": len(rows) - orientation_alt,
                "reference_samples": len(reference_samples),
                "target_samples": len(target_samples),
                "reference_samples_by_ancestry": {
                    ancestry: len(indices)
                    for ancestry, indices in zip(ANCESTRIES, ancestry_indices)
                },
                "split_biological_roles": dict(split.role_counts),
                "reference_missing_alleles_on_biallelic_axis": ref_missing_alleles,
                "target_missing_genotypes_on_selected_axis": target_missing_genotypes,
                **ancestry_af_counts,
            },
            "axis_sha256": {
                "input_joint_axis": total_axis_hash.hexdigest(),
                "flare_complete_biallelic_snv_axis": flare_axis_hash.hexdigest(),
                "selected_rare_axis_in_input_order": selected_axis_hash.hexdigest(),
            },
        }
        write_exclusive_json(stage / OUTPUT_NAMES["receipt"], receipt)
        os.replace(stage, outdir)
        return receipt
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-vcf", type=Path, required=True)
    parser.add_argument("--mosaic-vcf", type=Path, required=True)
    parser.add_argument("--split-tsv", type=Path, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--chromosome", default="22")
    parser.add_argument("--min-mac", type=int, default=MIN_MAC)
    parser.add_argument("--max-maf-exclusive", type=float,
                        default=MAX_MAF_EXCLUSIVE)
    parser.add_argument("--mosaic-donor-role", choices=MOSAIC_DONOR_ROLES,
                        default="SOURCE_VALID")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    receipt = prepare_panel_factors(
        panel_vcf=args.panel_vcf,
        mosaic_vcf=args.mosaic_vcf,
        split_tsv=args.split_tsv,
        genetic_map_path=args.genetic_map,
        outdir=args.outdir,
        chromosome=args.chromosome,
        min_mac=args.min_mac,
        max_maf_exclusive=args.max_maf_exclusive,
        mosaic_donor_role=args.mosaic_donor_role,
    )
    print(json.dumps({"decision": receipt["decision"],
                      "outdir": str(args.outdir)}, sort_keys=True))


if __name__ == "__main__":
    main()
