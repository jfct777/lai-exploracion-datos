#!/usr/bin/env python3
"""Select original-biallelic rare records and explicitly encode the counted allele.

REF, ALT and GT retain their VCF meaning. RARE_ALLELE identifies the minor allele
in the *output sample cohort*, not in a future consumer's sample subset. RD is
its integer copy count, not an ALT dosage or a phased haplotype assignment.
Counts use called alleles, including the known allele in a partial genotype;
RD is missing for an incomplete genotype. No imputation or phase inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pysam


CONTRACT = "minor_v1"
# Nextflow passes its configured value explicitly; shared CLI/API fallback.
DEFAULT_MIN_MAC = 2
INFO_FIELDS = {
    "RARE_ALLELE": (1, "Integer", "Selected minor allele index in source cohort: 0=REF; 1=ALT"),
    "RARE_AC": (1, "Integer", "Called copies of selected minor allele in source cohort"),
    "RARE_AN": (1, "Integer", "Called allele denominator in source cohort"),
    "RARE_AF": (1, "Float", "Selected minor allele frequency in source cohort"),
    "AC": ("A", "Integer", "ALT allele count in output samples"),
    "AN": (1, "Integer", "Called allele count in output samples"),
    "AF": ("A", "Float", "ALT allele frequency in output samples"),
}
PROVENANCE_FIELDS = {"ORIG_NALLELES", "ORIG_SITE"}


def count_alleles(genotypes):
    """Return ALT count and called denominator, rejecting non-biallelic GT."""
    ac = an = 0
    for gt in genotypes:
        for allele in gt:
            if allele is None:
                continue
            if allele not in (0, 1):
                raise ValueError("GT contains an allele other than 0/1 in a biallelic record")
            an += 1
            ac += allele
    return ac, an


def minor_dosage(gt, allele):
    return sum(a == allele for a in gt) if gt and all(a is not None for a in gt) else None


def header_value(header, key):
    return [str(row.value) for row in header.records if row.key == key]


def select_rare(input_path, output_path, report_path, counts_path, *, chrom,
                max_maf="0.01", min_mac=DEFAULT_MIN_MAC, samples_path=None,
                remove_info=False, keep_format="GT"):
    max_frequency = Fraction(str(max_maf)) if max_maf is not None else None
    if max_frequency is not None and not 0 <= max_frequency < Fraction(1, 2):
        raise ValueError("max-maf must be >=0 and <0.5; ties have no unique minor allele")
    if min_mac < 1:
        raise ValueError("min-mac must be >=1: monomorphic records are not rare variants")
    output_path, report_path, counts_path = map(Path, (output_path, report_path, counts_path))
    for target in (output_path, report_path, counts_path):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    counts = Counter()
    retained = None if not keep_format else set(keep_format.split(","))
    if retained is not None and "GT" not in retained:
        raise ValueError("GT must be retained; RD supplements GT and never replaces it")
    try:
        with pysam.VariantFile(str(input_path)) as source:
            if header_value(source.header, "dnabr_original_alleles") != ["v1"]:
                raise ValueError("Uncertified original alleles: regenerate M01 -> M02 with provenance; legacy split VCF cannot prove original biallelicity")
            if header_value(source.header, "dnabr_rare_contract") or any(
                    name in source.header.info for name in ("RARE_ALLELE", "RARE_AC", "RARE_AN", "RARE_AF")) or "RD" in source.header.formats:
                raise ValueError("Input already has rare-allele annotations; do not redefine its cohort silently")
            original_n_samples = len(source.header.samples)
            if samples_path:
                requested = Path(samples_path).read_text().split()
                if not requested or len(set(requested)) != len(requested):
                    raise ValueError("Sample list must be nonempty and unique")
                if set(requested) - set(source.header.samples):
                    raise ValueError("Sample list contains IDs absent from input; no silent sample dropping")
                source.subset_samples(requested)
            sample_names = list(source.header.samples)
            if not sample_names:
                raise ValueError("No samples in output cohort")
            cohort_hash = hashlib.sha256(("\n".join(sample_names) + "\n").encode()).hexdigest()
            header = source.header.copy()
            header.add_line(f"##dnabr_rare_contract={CONTRACT}")
            header.add_line(f"##dnabr_rare_cohort_sha256={cohort_hash}")
            header.add_line(f"##dnabr_rare_cohort_n_samples={len(sample_names)}")
            header.add_line("##dnabr_rare_frequency_rule=MAC_ge_min_and_MAF_le_max_called_alleles")
            for name, (number, kind, description) in INFO_FIELDS.items():
                if name not in header.info:
                    header.info.add(name, number, kind, description)
            header.formats.add("RD", 1, "Integer", "Copies of RARE_ALLELE; missing for incomplete GT; GT/REF/ALT unchanged")
            previous = None
            with pysam.VariantFile(str(partial), "wz", header=header) as sink:
                for record in source:
                    counts["input_filtered"] += 1
                    if record.contig.removeprefix("chr") != str(chrom).removeprefix("chr"):
                        raise ValueError("Record chromosome differs from declared chromosome")
                    if "ORIG_NALLELES" not in record.info or "ORIG_SITE" not in record.info:
                        raise ValueError("Record lacks original-allele provenance")
                    original_n = record.info["ORIG_NALLELES"]
                    if not isinstance(original_n, int) or original_n < 2:
                        raise ValueError("Invalid ORIG_NALLELES")
                    if original_n != 2:
                        counts["excluded_original_multiallelic"] += 1
                        continue
                    if len(record.alleles) != 2 or any(len(a) != 1 or a not in "ACGT" for a in record.alleles):
                        counts["excluded_non_biallelic_snv"] += 1
                        continue
                    key = (record.contig, record.pos)
                    if previous is not None and (key[0] != previous[0] or key[1] <= previous[1]):
                        raise ValueError("Non-increasing positions among certified biallelic SNVs; cannot count a site twice")
                    previous = key
                    genotypes = [sample.get("GT", ()) for sample in record.samples.values()]
                    ac, an = count_alleles(genotypes)
                    if an == 0:
                        counts["excluded_no_called_alleles"] += 1
                        continue
                    if 2 * ac == an:
                        counts["excluded_tie"] += 1
                        continue
                    allele = 0 if 2 * ac > an else 1
                    mac = min(ac, an - ac)
                    if mac < min_mac:
                        counts["excluded_mac"] += 1
                        continue
                    if max_frequency is not None and Fraction(mac, an) > max_frequency:
                        counts["excluded_maf"] += 1
                        continue
                    record.translate(header)
                    if remove_info:
                        for name in list(record.info):
                            if name not in PROVENANCE_FIELDS:
                                del record.info[name]
                    if retained is not None:
                        for name in list(record.format):
                            if name not in retained:
                                del record.format[name]
                    record.info["AC"], record.info["AN"], record.info["AF"] = (ac,), an, (ac / an,)
                    record.info["RARE_ALLELE"] = allele
                    record.info["RARE_AC"], record.info["RARE_AN"] = mac, an
                    record.info["RARE_AF"] = mac / an
                    for sample, gt in zip(record.samples.values(), genotypes):
                        sample["RD"] = minor_dosage(gt, allele)
                    counts["rare"] += 1
                    counts["rare_ref" if allele == 0 else "rare_alt"] += 1
                    sink.write(record)
        # Success is recorded only after closing the complete compressed output.
        partial.rename(output_path)
        report = {
            "schema": "m02_1_minor_v1", "input": str(input_path),
            "output": str(output_path), "chromosome": str(chrom),
            "cohort_samples_before": original_n_samples, "cohort_samples_after": len(sample_names),
            "cohort_sha256": cohort_hash, "counts": dict(counts),
            "criteria": {"min_mac_inclusive": min_mac, "max_maf_inclusive": str(max_maf),
                         "original_allele_count": 2, "frequency_denominator": "called_alleles",
                         "partial_genotype_dosage": "missing", "GT_REF_ALT_preserved": True},
            "scope": "Original means input supplied to M01, not unverified upstream calling history",
            "pysam_version": pysam.__version__,
        }
        with report_path.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        with counts_path.open("x") as handle:
            handle.write("chr\tstep\tn_variants\n")
            for name in ("input_filtered", "excluded_original_multiallelic", "excluded_non_biallelic_snv",
                         "excluded_no_called_alleles", "excluded_tie", "excluded_mac", "excluded_maf", "rare", "rare_ref", "rare_alt"):
                handle.write(f"{chrom}\t{name}\t{counts[name]}\n")
        return report
    finally:
        if partial.exists():
            partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "report", "counts", "chrom"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-maf", default="0.01")
    parser.add_argument("--min-mac", type=int, default=DEFAULT_MIN_MAC)
    parser.add_argument("--samples")
    parser.add_argument("--remove-info", action="store_true")
    parser.add_argument("--keep-format", default="GT")
    args = parser.parse_args()
    select_rare(args.input, args.output, args.report, args.counts, chrom=args.chrom,
                max_maf=None if args.max_maf == "none" else args.max_maf, min_mac=args.min_mac,
                samples_path=args.samples, remove_info=args.remove_info, keep_format=args.keep_format)


if __name__ == "__main__":
    main()
