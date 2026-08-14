#!/usr/bin/env python3
"""Create a small structured VCF fixture for the M27D R integration test."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def genotype(rng: random.Random, allele_frequency: float) -> str:
    dosage = int(rng.random() < allele_frequency) + int(rng.random() < allele_frequency)
    return ("0/0", "0/1", "1/1")[dosage]


def build(outdir: Path, base_preregistration: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    samples = [f"S{i:03d}" for i in range(60)]
    alleles = (("A", "G"), ("C", "T"), ("G", "A"), ("T", "C"))
    for chromosome in range(1, 23):
        rng = random.Random(8100 + chromosome)
        path = outdir / f"panel.{chromosome}.vcf"
        with path.open("w", encoding="utf-8") as handle:
            handle.write("##fileformat=VCFv4.2\n")
            handle.write(f"##contig=<ID=chr{chromosome},length=1000000>\n")
            handle.write(
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(samples)
                + "\n"
            )
            for marker in range(1, 101):
                ref, alt = alleles[(marker + chromosome) % len(alleles)]
                calls = []
                for sample_index in range(len(samples)):
                    group_frequency = 0.12 if sample_index < 30 else 0.38
                    frequency = min(0.48, group_frequency + 0.03 * ((marker % 5) - 2))
                    calls.append(genotype(rng, frequency))
                handle.write(
                    f"chr{chromosome}\t{marker * 5000}\tchr{chromosome}:{marker * 5000}\t"
                    f"{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(calls) + "\n"
                )

    metadata = outdir / "metadata.tsv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "IID",
                "Sample_ID(Aliases)",
                "Illumina_ID",
                "original_IID",
                "Source",
                "Ancestry",
                "Population",
                "Country",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for sample_index, sample in enumerate(samples):
            writer.writerow(
                {
                    "IID": sample,
                    "Sample_ID(Aliases)": sample,
                    "Illumina_ID": sample,
                    "original_IID": sample,
                    "Source": "SOURCE_A" if sample_index < 30 else "SOURCE_B",
                    "Ancestry": "AFR" if sample_index < 30 else "NAM",
                    "Population": "POP_A" if sample_index < 30 else "POP_B",
                    "Country": "A" if sample_index < 30 else "B",
                }
            )

    (outdir / "exclude.bed").write_text("chr6\t25000000\t35000000\n", encoding="utf-8")
    prereg = json.loads(base_preregistration.read_text(encoding="utf-8"))
    prereg["scope"]["official_panel_samples_expected"] = len(samples)
    prereg["resource_smoke"]["arm_full_n"] = {
        "max_samples": len(samples),
        "snps": 200,
        "sampling": "synthetic integration fixture",
    }
    prereg["resource_smoke"]["arm_marker_scaling"] = {
        "samples": 30,
        "max_snps": 300,
        "sampling": "synthetic integration fixture",
        "threads": "selected from arm_full_n",
    }
    prereg["resource_smoke"]["thread_screen"] = [1, 2]
    (outdir / "prereg.json").write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(arguments.outdir, arguments.preregistration)
