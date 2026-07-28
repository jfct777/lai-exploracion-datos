#!/usr/bin/env python3

import argparse
import re


def parse_args():
    """Define y devuelve los argumentos de línea de comandos."""
    p = argparse.ArgumentParser()
    p.add_argument("--chr", required=True)
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    return p.parse_args()


def main():
    """Convierte la salida de bcftools stats en tablas resumidas."""
    args = parse_args()

    n_sites = None
    n_snps = None
    titv = None

    re_sites = re.compile(r"^SN\s+\d+\s+number of records:\s+(\d+)")
    re_snps = re.compile(r"^SN\s+\d+\s+number of SNPs:\s+(\d+)")
    re_titv = re.compile(r"^TSTV\s+\d+\s+([0-9\.eE+-]+)")

    with open(args.in_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = re_sites.match(line)
            if m:
                n_sites = int(m.group(1))
                continue
            m = re_snps.match(line)
            if m:
                n_snps = int(m.group(1))
                continue
            m = re_titv.match(line)
            if m:
                titv = float(m.group(1))
                continue

    with open(args.out_path, "w", encoding="utf-8") as out:
        out.write("chr\tn_sites\tn_snps\ttitv\n")
        out.write(f"{args.chr}\t{'' if n_sites is None else n_sites}\t{'' if n_snps is None else n_snps}\t{'' if titv is None else titv}\n")


if __name__ == "__main__":
    main()
