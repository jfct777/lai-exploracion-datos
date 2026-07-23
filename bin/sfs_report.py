#!/usr/bin/env python3

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vcf", required=True)
    p.add_argument("--chr", required=True)
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--keep_samples", default=None)

    p.add_argument("--min_an_frac", type=float, default=0.95)
    p.add_argument("--rare_tail_max_ac", type=int, default=20)
    p.add_argument("--sfs_bins_af", default="0,0.001,0.005,0.01,0.05,0.1,0.2,0.5,1.0")
    # Folded minor-allele spectrum. REF/ALT is set by the reference genome, not by
    # biology: an ALT allele at AF 0.95 and one at 0.05 are the same (rare) site
    # folded. The unfolded ALT-AF spectrum puts spurious mass at the high end.
    # Por defecto usa min(AC, AN-AC)/AN en [0, 0.5]. Pass
    # "false" only for an explicitly ALT-based spectrum.
    p.add_argument("--fold", default="true", choices=["true", "false"])

    return p.parse_args()


def _run(cmd):
    r = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.stdout


def _count_samples(vcf: str, keep_samples: str | None) -> int:
    if keep_samples:
        n = 0
        with open(keep_samples, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        return n
    out = _run(["bcftools", "query", "-l", vcf])
    return len([x for x in out.splitlines() if x.strip()])


def main():
    args = parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fold = args.fold == "true"
    bins = [float(x) for x in args.sfs_bins_af.split(",")]
    bins_top = 0.5 if fold else 1.0
    if bins[0] != 0.0 or not math.isclose(bins[-1], bins_top):
        raise SystemExit(
            f"sfs_bins_af must start with 0 and end with {bins_top} "
            f"({'folded MAF' if fold else 'unfolded ALT AF'})"
        )

    n_samples = _count_samples(args.vcf, args.keep_samples)
    max_an = 2 * n_samples
    if max_an <= 0:
        raise SystemExit("Could not determine number of samples for SFS")

    sample_view = []
    if args.keep_samples:
        sample_view = ["bcftools", "view", "-S", args.keep_samples, "-Ou", args.vcf]
        query_in = ["bcftools", "query", "-f", "%CHROM\t%POS\t%INFO/AC\t%INFO/AN\n", "-"]
        p1 = subprocess.Popen(sample_view, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(query_in, stdin=p1.stdout, stdout=subprocess.PIPE, text=True)
        p1.stdout.close()
        out = p2.communicate()[0]
        if p2.returncode != 0:
            raise SystemExit("bcftools query failed")
    else:
        out = _run(["bcftools", "query", "-f", "%CHROM\t%POS\t%INFO/AC\t%INFO/AN\n", args.vcf])

    total_sites = 0
    kept_sites = 0
    rare_tail = Counter()
    hist_bins = [0 for _ in range(len(bins) - 1)]

    for line in out.splitlines():
        if not line:
            continue
        total_sites += 1
        chrom, pos, ac_s, an_s = line.split("\t")
        try:
            ac = int(ac_s.split(",")[0])
            an = int(an_s)
        except Exception:
            continue

        if an <= 0:
            continue

        if an < args.min_an_frac * max_an:
            continue

        # Fold to the minor allele: the spectrum value and the rare-tail count are
        # both keyed on the minor-allele copy count, independent of REF/ALT polarity.
        minor_ac = min(ac, an - ac) if fold else ac
        af = minor_ac / float(an)
        if af < 0 or af > 1:
            continue

        kept_sites += 1

        if 1 <= minor_ac <= args.rare_tail_max_ac:
            rare_tail[minor_ac] += 1

        for i in range(len(bins) - 1):
            lo = bins[i]
            hi = bins[i + 1]
            if (af >= lo and af < hi) or (i == len(bins) - 2 and math.isclose(af, bins[-1])):
                hist_bins[i] += 1
                break

    sfs_rows = []
    for i in range(len(bins) - 1):
        sfs_rows.append(
            {
                "chr": args.chr,
                "af_bin_lo": bins[i],
                "af_bin_hi": bins[i + 1],
                "n_sites": hist_bins[i],
            }
        )

    rare_rows = []
    for ac in range(1, args.rare_tail_max_ac + 1):
        rare_rows.append({"chr": args.chr, "AC": ac, "n_sites": int(rare_tail.get(ac, 0))})

    sfs_path = str(out_prefix) + ".sfs.tsv"
    rare_path = str(out_prefix) + ".rare_tail_ac.tsv"
    summary_path = str(out_prefix) + ".summary.json"

    pd.DataFrame(sfs_rows).to_csv(sfs_path, sep="\t", index=False)
    pd.DataFrame(rare_rows).to_csv(rare_path, sep="\t", index=False)

    summary = {
        "chr": args.chr,
        "n_samples": n_samples,
        "max_an": max_an,
        "total_sites_seen": total_sites,
        "sites_with_ac_an": kept_sites,
        "rare_tail_max_ac": args.rare_tail_max_ac,
        "min_an_frac": args.min_an_frac,
        "folded": fold,
        "keep_samples": bool(args.keep_samples),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
