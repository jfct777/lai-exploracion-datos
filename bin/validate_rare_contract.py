#!/usr/bin/env python3
"""Reject new rare-allele inputs in consumers that still count ALT.

Only metadata is inspected here; genotype parsing belongs to the consumer's
VCF library. Renaming a file cannot disable this header-based protection.
"""
import argparse
import gzip


def validate(path, unsupported_consumers, painting_mode=""):
    opener = gzip.open if str(path).endswith(".gz") else open
    contracts = []
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("##dnabr_rare_contract="):
                contracts.append(line.rstrip().split("=", 1)[1])
            if line.startswith("#CHROM\t"):
                break
        else:
            raise ValueError("Missing VCF column header")
    if contracts:
        if contracts != ["minor_v1"]:
            raise ValueError("Unknown or duplicate rare-allele contract")
        if unsupported_consumers:
            raise ValueError("These consumers do not yet support source-cohort minor dosage: " + ",".join(unsupported_consumers))
        if painting_mode and painting_mode != "source_minor":
            raise ValueError("M14 requires source_minor for minor_v1 input")
    elif painting_mode == "source_minor":
        raise ValueError("M14 source_minor requires an M02.1 minor_v1 input, not a historical rare VCF")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--unsupported-consumers", default="")
    parser.add_argument("--painting-mode", default="")
    args = parser.parse_args()
    validate(args.input, [x for x in args.unsupported_consumers.split(",") if x], args.painting_mode)


if __name__ == "__main__":
    main()
