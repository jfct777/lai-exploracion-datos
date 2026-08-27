#!/usr/bin/env python3
"""Audit the ancestry and population-unit distribution of M34 rare loci.

The audit rebuilds the REF_TRAIN-only rare universe directly from the phased
panel VCF.  It never reads mosaic targets, local-ancestry truth or predictions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


ANCESTRIES = ("AFR", "EUR", "NAM")
ANCESTRY_ALIASES = {
    "AFR": "AFR",
    "African": "AFR",
    "EUR": "EUR",
    "European": "EUR",
    "NAM": "NAM",
    "Native_American": "NAM",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    ancestry: str
    population: str
    atomic_unit: str


def load_reference_samples(path: Path) -> dict[str, Sample]:
    required = {
        "sample_id", "ancestry", "canonical_population", "atomic_unit_id", "role"
    }
    result: dict[str, Sample] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None and required.issubset(reader.fieldnames),
                "split TSV lacks required columns")
        for line_number, row in enumerate(reader, 2):
            if row["role"] != "REF_TRAIN":
                continue
            sample_id = row["sample_id"]
            require(sample_id and sample_id not in result,
                    f"duplicate or empty REF_TRAIN sample at row {line_number}")
            try:
                ancestry = ANCESTRY_ALIASES[row["ancestry"]]
            except KeyError:
                raise ValueError(
                    f"unsupported ancestry for REF_TRAIN sample {sample_id}: {row['ancestry']}"
                ) from None
            population = row["canonical_population"]
            atomic_unit = row["atomic_unit_id"]
            require(population and atomic_unit,
                    f"missing population or atomic unit for {sample_id}")
            result[sample_id] = Sample(sample_id, ancestry, population, atomic_unit)
    require(result, "split TSV contains no REF_TRAIN samples")
    require({sample.ancestry for sample in result.values()} == set(ANCESTRIES),
            "REF_TRAIN must contain AFR, EUR and NAM")
    return result


def parse_gt(format_value: str, sample_value: str, *, line_number: int) -> tuple[int | None, int | None]:
    fields = format_value.split(":")
    require(len(fields) == len(set(fields)) and "GT" in fields,
            f"VCF row {line_number} lacks one unambiguous GT field")
    values = sample_value.split(":")
    gt_index = fields.index("GT")
    require(gt_index < len(values), f"VCF row {line_number} lacks a sample GT")
    gt = values[gt_index]
    separator = "|" if "|" in gt else "/"
    parts = gt.split(separator)
    require(len(parts) == 2, f"VCF row {line_number} has a non-diploid GT")
    alleles: list[int | None] = []
    for value in parts:
        if value == ".":
            alleles.append(None)
        else:
            require(value in {"0", "1"},
                    f"VCF row {line_number} has a non-biallelic GT")
            alleles.append(int(value))
    return alleles[0], alleles[1]


def _hhi(counts: Iterable[int]) -> float:
    values = [int(value) for value in counts if value > 0]
    total = sum(values)
    return 0.0 if total == 0 else sum((value / total) ** 2 for value in values)


def _format_float(value: float) -> str:
    return format(value, ".12g")


def audit(
    *, panel_vcf: Path, split_tsv: Path, per_locus_path: Path,
    summary_path: Path, chromosome: str, min_mac: int,
    max_maf_exclusive: float, expected_loci: int | None,
) -> dict[str, object]:
    require(panel_vcf.is_file() and split_tsv.is_file(), "audit inputs must be files")
    require(min_mac >= 1, "minimum MAC must be positive")
    require(0.0 < max_maf_exclusive <= 0.5, "maximum MAF must be in (0, 0.5]")
    references = load_reference_samples(split_tsv)
    ancestry_people = Counter(sample.ancestry for sample in references.values())
    ancestry_units = {
        ancestry: len({sample.atomic_unit for sample in references.values()
                       if sample.ancestry == ancestry})
        for ancestry in ANCESTRIES
    }
    rows: list[dict[str, object]] = []
    vcf_records = 0
    selected_alt_minor = 0
    selected_ref_minor = 0

    with open_text(panel_vcf) as handle:
        sample_axis: tuple[str, ...] | None = None
        selected_indices: list[int] = []
        selected_samples: list[Sample] = []
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##") or not line.strip():
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\r\n").split("\t")
                require(len(fields) >= 10, "VCF has no sample columns")
                sample_axis = tuple(fields[9:])
                require(len(sample_axis) == len(set(sample_axis)),
                        "VCF sample axis is duplicated")
                panel_index = {sample: index for index, sample in enumerate(sample_axis)}
                missing = set(references) - set(panel_index)
                if missing:
                    raise ValueError(
                        f"REF_TRAIN sample absent from panel VCF: {sorted(missing)[0]}"
                    )
                selected_indices = [panel_index[sample] for sample in references]
                selected_samples = [references[sample] for sample in references]
                continue
            require(sample_axis is not None, "VCF data appeared before #CHROM")
            fields = line.rstrip("\r\n").split("\t")
            require(len(fields) == 9 + len(sample_axis),
                    f"malformed VCF row {line_number}")
            chrom = fields[0].removeprefix("chr")
            if chrom != chromosome.removeprefix("chr"):
                continue
            vcf_records += 1
            ref, alt = fields[3], fields[4]
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue

            genotypes = [
                parse_gt(fields[8], fields[9 + index], line_number=line_number)
                for index in selected_indices
            ]
            alt_ac = sum(allele == 1 for gt in genotypes for allele in gt if allele is not None)
            callable_an = sum(allele is not None for gt in genotypes for allele in gt)
            if callable_an == 0:
                continue
            ref_ac = callable_an - alt_ac
            minor_code = 1 if alt_ac <= ref_ac else 0
            minor_ac = min(alt_ac, ref_ac)
            maf = minor_ac / callable_an
            if minor_ac < min_mac or not maf < max_maf_exclusive:
                continue
            if minor_code == 1:
                selected_alt_minor += 1
            else:
                selected_ref_minor += 1

            ancestry_ac = Counter({ancestry: 0 for ancestry in ANCESTRIES})
            ancestry_an = Counter({ancestry: 0 for ancestry in ANCESTRIES})
            ancestry_carriers = Counter({ancestry: 0 for ancestry in ANCESTRIES})
            carrier_units: dict[str, Counter[str]] = {
                ancestry: Counter() for ancestry in ANCESTRIES
            }
            carrier_populations: dict[str, set[str]] = {
                ancestry: set() for ancestry in ANCESTRIES
            }
            for sample, gt in zip(selected_samples, genotypes, strict=True):
                observed = [allele for allele in gt if allele is not None]
                ancestry_an[sample.ancestry] += len(observed)
                dosage = sum(allele == minor_code for allele in observed)
                ancestry_ac[sample.ancestry] += dosage
                if dosage:
                    ancestry_carriers[sample.ancestry] += 1
                    carrier_units[sample.ancestry][sample.atomic_unit] += 1
                    carrier_populations[sample.ancestry].add(sample.population)

            row: dict[str, object] = {
                "chrom": chrom,
                "position": int(fields[1]),
                "ref": ref,
                "alt": alt,
                "minor_allele": alt if minor_code == 1 else ref,
                "minor_code": minor_code,
                "pooled_minor_ac": minor_ac,
                "pooled_callable_an": callable_an,
                "pooled_maf": maf,
                "pooled_callability": callable_an / (2 * len(references)),
            }
            for ancestry in ANCESTRIES:
                an = ancestry_an[ancestry]
                unit_counts = carrier_units[ancestry]
                row.update({
                    f"{ancestry}_minor_ac": ancestry_ac[ancestry],
                    f"{ancestry}_callable_an": an,
                    f"{ancestry}_minor_af": ancestry_ac[ancestry] / an if an else math.nan,
                    f"{ancestry}_callability": an / (2 * ancestry_people[ancestry]),
                    f"{ancestry}_carrier_people": ancestry_carriers[ancestry],
                    f"{ancestry}_carrier_populations": len(carrier_populations[ancestry]),
                    f"{ancestry}_carrier_units": len(unit_counts),
                    f"{ancestry}_max_unit_carrier_share": (
                        max(unit_counts.values()) / sum(unit_counts.values()) if unit_counts else 0.0
                    ),
                    f"{ancestry}_unit_hhi": _hhi(unit_counts.values()),
                })
            rows.append(row)

    rows.sort(key=lambda row: (int(row["position"]), str(row["ref"]), str(row["alt"])))
    require(rows, "rare-locus audit selected no loci")
    if expected_loci is not None:
        require(len(rows) == expected_loci,
                f"selected locus count differs: observed {len(rows)}, expected {expected_loci}")

    fieldnames = list(rows[0])
    per_locus_path.parent.mkdir(parents=True, exist_ok=True)
    with per_locus_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: _format_float(value) if isinstance(value, float) else value
                for key, value in row.items()
            })

    ancestry_summary: dict[str, object] = {}
    for ancestry in ANCESTRIES:
        frequencies = [float(row[f"{ancestry}_minor_af"]) for row in rows]
        callabilities = [float(row[f"{ancestry}_callability"]) for row in rows]
        units = [int(row[f"{ancestry}_carrier_units"]) for row in rows]
        ancestry_summary[ancestry] = {
            "reference_people": ancestry_people[ancestry],
            "reference_atomic_units": ancestry_units[ancestry],
            "loci_af_ge_0_01": sum(value >= 0.01 for value in frequencies),
            "loci_af_ge_0_05": sum(value >= 0.05 for value in frequencies),
            "loci_af_ge_0_10": sum(value >= 0.10 for value in frequencies),
            "loci_af_ge_0_20": sum(value >= 0.20 for value in frequencies),
            "loci_with_carriers_in_ge_2_units": sum(value >= 2 for value in units),
            "loci_with_carriers_in_ge_3_units": sum(value >= 3 for value in units),
            "minimum_callability": min(callabilities),
            "mean_callability": sum(callabilities) / len(callabilities),
        }

    nam_enriched = [
        row for row in rows
        if float(row["NAM_minor_af"]) >= 0.05
        and float(row["AFR_minor_af"]) < 0.01
        and float(row["EUR_minor_af"]) < 0.01
    ]
    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "stage": "M34_RARE_LOCUS_DISTRIBUTION_AUDIT",
        "status": "PASS_DESCRIPTIVE_AUDIT_NO_MODEL_SELECTION",
        "scope": {
            "chromosome": chromosome.removeprefix("chr"),
            "frequency_population": "REF_TRAIN_only",
            "target_mosaics_read": False,
            "local_ancestry_truth_read": False,
            "predictions_read": False,
            "king_used": False,
        },
        "selection": {
            "minimum_mac": min_mac,
            "maximum_maf_exclusive": max_maf_exclusive,
            "selected_loci": len(rows),
            "minor_alt_loci": selected_alt_minor,
            "minor_ref_loci": selected_ref_minor,
            "panel_chr_records": vcf_records,
        },
        "ancestry": ancestry_summary,
        "nam_enrichment": {
            "loci_nam_af_ge_0_05_and_afr_eur_below_0_01": len(nam_enriched),
            "of_these_in_ge_2_nam_units": sum(
                int(row["NAM_carrier_units"]) >= 2 for row in nam_enriched
            ),
            "of_these_in_ge_3_nam_units": sum(
                int(row["NAM_carrier_units"]) >= 3 for row in nam_enriched
            ),
        },
        "inputs": {
            "panel_vcf_sha256": sha256_file(panel_vcf),
            "split_tsv_sha256": sha256_file(split_tsv),
        },
        "outputs": {
            "per_locus_tsv_sha256": sha256_file(per_locus_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-vcf", required=True, type=Path)
    parser.add_argument("--split-tsv", required=True, type=Path)
    parser.add_argument("--per-locus", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--chromosome", default="22")
    parser.add_argument("--min-mac", type=int, default=2)
    parser.add_argument("--max-maf-exclusive", type=float, default=0.01)
    parser.add_argument("--expected-loci", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit(
        panel_vcf=args.panel_vcf,
        split_tsv=args.split_tsv,
        per_locus_path=args.per_locus,
        summary_path=args.summary,
        chromosome=args.chromosome,
        min_mac=args.min_mac,
        max_maf_exclusive=args.max_maf_exclusive,
        expected_loci=args.expected_loci,
    )
    print(json.dumps({
        "status": summary["status"],
        "selected_loci": summary["selection"]["selected_loci"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
