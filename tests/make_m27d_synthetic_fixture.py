#!/usr/bin/env python3
"""Build a small synthetic cohort that exercises the whole M27D audit.

The fixture is generated, never copied from the cohort: no real genotype belongs in a
repository.  It is deliberately built so that every claim the audit makes has something
to be right or wrong about.

* Two ancestral groups with different allele frequencies, so the PCA has real structure
  to find rather than noise to overfit.
* Explicit parent-offspring trios, so the pairs PC-Relate must recover are known in
  advance instead of being whatever the estimator happens to return.
* Metadata containing the same alias-collision shape observed in the official panel,
  samples with no metadata row at all, and samples flagged ``Exclude``.
* A baseline cohort that shares most of its donors with the panel and keeps one donor
  outside it, which is the situation the real baseline is in.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


N_FOUNDERS = 84
N_TRIOS = 6
N_MARKERS_PER_CHROMOSOME = 200
N_BASELINE_SHARED = 7
ALLELES = (("A", "G"), ("C", "T"), ("G", "A"), ("T", "C"))

# Samples whose metadata row collides with an alias-only decoy, mirroring the 35
# collisions in the official panel.
COLLIDING = ("S004", "S011", "S023", "S037", "S052")
# Samples deliberately left out of the metadata table.
UNMATCHED = ("S060", "S061", "S062")
# Samples the metadata excludes; they must never reach the kinship universe.
EXCLUDED = ("S070", "S071")


def founder_ids() -> list[str]:
    return [f"S{index:03d}" for index in range(N_FOUNDERS)]


def trio_children() -> list[tuple[str, str, str]]:
    """Children and their two parents, all drawn from the second ancestral group."""
    trios = []
    for index in range(N_TRIOS):
        child = f"C{index:03d}"
        father = f"S{42 + 2 * index:03d}"
        mother = f"S{43 + 2 * index:03d}"
        trios.append((child, father, mother))
    return trios


def all_samples() -> list[str]:
    return founder_ids() + [child for child, _, _ in trio_children()]


def build_genotypes(rng: random.Random, n_markers: int) -> tuple[list[list[int]], list[str]]:
    """Return per-marker genotype dosages for every sample plus the sample order.

    Group membership shifts the allele frequency, which is what makes the leading
    principal component an ancestry axis instead of a noise axis.
    """
    founders = founder_ids()
    trios = trio_children()
    samples = founders + [child for child, _, _ in trios]
    index_of = {sample: position for position, sample in enumerate(samples)}
    dosages: list[list[int]] = []

    for _ in range(n_markers):
        ancestral = rng.uniform(0.15, 0.50)
        shift = rng.uniform(0.08, 0.20)
        frequencies = [
            min(0.95, max(0.05, ancestral - shift)),
            min(0.95, max(0.05, ancestral + shift)),
        ]
        haplotypes: dict[str, tuple[int, int]] = {}
        for position, sample in enumerate(founders):
            frequency = frequencies[0 if position < N_FOUNDERS // 2 else 1]
            haplotypes[sample] = (
                int(rng.random() < frequency),
                int(rng.random() < frequency),
            )
        for child, father, mother in trios:
            haplotypes[child] = (
                haplotypes[father][rng.randrange(2)],
                haplotypes[mother][rng.randrange(2)],
            )
        row = [0] * len(samples)
        for sample, (left, right) in haplotypes.items():
            row[index_of[sample]] = left + right
        dosages.append(row)
    return dosages, samples


def write_vcf(path: Path, chromosome: int, samples: list[str], dosages: list[list[int]]) -> None:
    calls = ("0/0", "0/1", "1/1")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(f"##contig=<ID=chr{chromosome},length=100000000>\n")
        handle.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n"
        )
        for marker, row in enumerate(dosages, start=1):
            ref, alt = ALLELES[(marker + chromosome) % len(ALLELES)]
            position = marker * 40000
            handle.write(
                f"chr{chromosome}\t{position}\tchr{chromosome}:{position}\t{ref}\t{alt}\t"
                ".\tPASS\t.\tGT\t" + "\t".join(calls[value] for value in row) + "\n"
            )


def metadata_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    samples = all_samples()
    for position, sample in enumerate(samples):
        if sample in UNMATCHED:
            continue
        group_a = position < N_FOUNDERS // 2
        rows.append(
            {
                "IID": sample,
                "Sample_ID(Aliases)": sample,
                "Illumina_ID": sample,
                "original_IID": sample,
                "Exclude": "TRUE" if sample in EXCLUDED else "FALSE",
                "N_genotypes": "0" if sample in EXCLUDED else "4400",
                "Source": "SOURCE_A" if group_a else "SOURCE_B",
                "Ancestry": "African" if group_a else "Native_American",
                "Population": "POP_A" if group_a else "POP_B",
                "Country": "Kenya" if group_a else "Brazil",
                "Maximum_unrelated_dataset": "TRUE",
            }
        )
    # The decoy rows reproduce the observed collision: excluded, without genotypes and
    # reachable only through an alias column, so the documented rule has to pick the
    # other row on its own rather than on row order.
    for sample in COLLIDING:
        rows.append(
            {
                "IID": f"DECOY_{sample}",
                "Sample_ID(Aliases)": sample,
                "Illumina_ID": "",
                "original_IID": "",
                "Exclude": "TRUE",
                "N_genotypes": "0",
                "Source": "SOURCE_DECOY",
                "Ancestry": "European",
                "Population": "POP_DECOY",
                "Country": "Nowhere",
                "Maximum_unrelated_dataset": "",
            }
        )
    return rows


def build(outdir: Path, base_preregistration: Path) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    panel_dir = outdir / "panel"
    baseline_dir = outdir / "baseline"
    panel_dir.mkdir(exist_ok=True)
    baseline_dir.mkdir(exist_ok=True)

    samples: list[str] = []
    for chromosome in range(1, 23):
        rng = random.Random(9100 + chromosome)
        dosages, samples = build_genotypes(rng, N_MARKERS_PER_CHROMOSOME)
        write_vcf(panel_dir / f"panel.{chromosome}.vcf", chromosome, samples, dosages)

        # The baseline shares donors with the panel by construction, so the identity
        # check has an unambiguous right answer, and keeps one donor outside it.
        shared = samples[:N_BASELINE_SHARED]
        baseline_samples = [f"REF_{name}" for name in shared] + ["REF_ABSENT"]
        extra = random.Random(500 + chromosome)
        baseline_rows = [
            [row[samples.index(name)] for name in shared] + [extra.randrange(3)]
            for row in dosages
        ]
        write_vcf(
            baseline_dir / f"baseline.chr{chromosome}.vcf",
            chromosome,
            baseline_samples,
            baseline_rows,
        )

    metadata = outdir / "metadata.tsv"
    rows = metadata_rows()
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    (outdir / "exclude.bed").write_text("chr6\t25000000\t35000000\n", encoding="utf-8")

    contract = json.loads(base_preregistration.read_text(encoding="utf-8"))
    contract["scope"]["official_panel_samples_expected"] = len(samples)
    contract["scope"]["baseline_samples_expected"] = N_BASELINE_SHARED + 1
    contract["identity_contract"]["expected_shared_baseline_identities"] = N_BASELINE_SHARED
    contract["identity_contract"]["joint_autosomal_genotypes_min_per_shared_identity"] = 100
    contract["resource_smoke"]["arm_full_n"] = {
        "max_samples": len(samples),
        "snps": 200,
        "sampling": "synthetic integration fixture",
    }
    contract["resource_smoke"]["arm_marker_scaling"] = {
        "samples": 30,
        "max_snps": 300,
        "sampling": "synthetic integration fixture",
        "threads": "selected from arm_full_n",
    }
    contract["resource_smoke"]["thread_screen"] = [1, 2]
    (outdir / "prereg.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    expectations = {
        "n_samples": len(samples),
        "n_colliding": len(COLLIDING),
        "n_unmatched": len(UNMATCHED),
        "n_excluded": len(EXCLUDED),
        "n_eligible": len(samples) - len(EXCLUDED),
        "n_expected_pairs": (len(samples) - len(EXCLUDED))
        * (len(samples) - len(EXCLUDED) - 1)
        // 2,
        "n_baseline_shared": N_BASELINE_SHARED,
        "n_baseline_absent": 1,
        "related_pairs": [
            [child, parent] for child, father, mother in trio_children() for parent in (father, mother)
        ],
    }
    (outdir / "expectations.json").write_text(
        json.dumps(expectations, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return expectations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(build(arguments.outdir, arguments.preregistration), indent=2))
