#!/usr/bin/env python3

import argparse
import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path


def parse_args():
    """Define y devuelve los argumentos de línea de comandos."""
    p = argparse.ArgumentParser()
    p.add_argument("--vcf", required=True)
    p.add_argument("--chr", required=True)
    p.add_argument("--ancestral_fasta", required=True)
    p.add_argument("--out_tsv_gz", required=True)
    p.add_argument("--out_summary_json", required=True)

    p.add_argument("--allow_flip_if_ancestral_is_alt", action="store_true")
    p.add_argument("--ancestral_accept_ambiguous", action="store_true")

    return p.parse_args()


def _run_capture(cmd):
    r = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.stdout


def _load_contigs_from_fai(fai_path: str) -> set[str]:
    contigs: set[str] = set()
    with open(fai_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            contigs.add(line.split("\t", 1)[0])
    return contigs


def _choose_contig(contigs: set[str], requested: str) -> str | None:
    if requested in contigs:
        return requested
    # Ensembl ancestral alleles FASTA uses contig names like:
    #   ANCESTOR_for_chromosome:GRCh38:6:1:170805979:1
    # In that case, match by the chromosome token after 'GRCh38:'.
    req = requested
    if req.startswith("chr"):
        req = req.replace("chr", "", 1)
    for c in contigs:
        if c.startswith("ANCESTOR_for_chromosome:GRCh38:"):
            parts = c.split(":")
            # Expected: ['ANCESTOR_for_chromosome', 'GRCh38', '<CHR>', '<START>', '<END>', '<STRAND>']
            if len(parts) >= 3 and parts[2] == req:
                return c
    if requested.startswith("chr"):
        alt = requested.replace("chr", "", 1)
        if alt in contigs:
            return alt
    else:
        alt = "chr" + requested
        if alt in contigs:
            return alt
    return None


def main():
    """Construye la tabla de alelos ancestrales para las variantes de entrada."""
    args = parse_args()

    fasta_path = args.ancestral_fasta
    fai_path = fasta_path + ".fai"
    if not os.path.isfile(fasta_path) or os.path.getsize(fasta_path) == 0:
        raise SystemExit(f"Missing ancestral FASTA: {fasta_path}")
    if not os.path.isfile(fai_path) or os.path.getsize(fai_path) == 0:
        raise SystemExit(f"Missing FASTA index (.fai): {fai_path}")

    contigs = _load_contigs_from_fai(fai_path)
    contig = _choose_contig(contigs, args.chr)
    if contig is None:
        raise SystemExit(
            f"Contig '{args.chr}' not found in FASTA index. Available examples: {sorted(list(contigs))[:5]}"
        )

    if not shutil.which("samtools"):
        raise SystemExit("samtools is required inside the container to read indexed FASTA (samtools faidx)")

    fa_text = _run_capture(["samtools", "faidx", fasta_path, contig])
    seq = "".join([ln.strip() for ln in fa_text.splitlines() if ln and not ln.startswith(">")]).upper()
    if not seq:
        raise SystemExit(f"Failed to fetch contig '{contig}' from FASTA via samtools faidx")

    out_tsv = Path(args.out_tsv_gz)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "chr": args.chr,
        "fasta": fasta_path,
        "fasta_contig_used": contig,
        "allow_flip_if_ancestral_is_alt": bool(args.allow_flip_if_ancestral_is_alt),
        "ancestral_accept_ambiguous": bool(args.ancestral_accept_ambiguous),
        "n_sites_total": 0,
        "n_sites_snp_biallelic": 0,
        "n_status_ok_ref": 0,
        "n_status_ok_alt": 0,
        "n_status_ambiguous": 0,
        "n_status_mismatch": 0,
    }

    cmd = ["bcftools", "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\n", args.vcf]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    assert p.stdout is not None

    with gzip.open(out_tsv, "wt", encoding="utf-8") as fh:
        fh.write("CHROM\tPOS\tREF\tALT\tANCESTRAL\tSTATUS\n")
        for line in p.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            summary["n_sites_total"] += 1
            chrom, pos_s, ref, alt = line.split("\t")

            # handle multiallelic ALT quickly
            if "," in alt:
                continue
            if len(ref) != 1 or len(alt) != 1:
                continue

            summary["n_sites_snp_biallelic"] += 1

            pos = int(pos_s)
            if pos <= 0 or pos > len(seq):
                summary["n_status_mismatch"] += 1
                fh.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t.\tmismatch\n")
                continue

            anc = seq[pos - 1]
            if anc in ("N", "-") or (not args.ancestral_accept_ambiguous and anc not in ("A", "C", "G", "T")):
                summary["n_status_ambiguous"] += 1
                fh.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t.\tambiguous\n")
                continue

            if anc == ref:
                summary["n_status_ok_ref"] += 1
                fh.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{anc}\tok_ref\n")
            elif anc == alt:
                if args.allow_flip_if_ancestral_is_alt:
                    summary["n_status_ok_alt"] += 1
                    fh.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{anc}\tok_alt\n")
                else:
                    summary["n_status_mismatch"] += 1
                    fh.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t.\tmismatch\n")
            else:
                summary["n_status_mismatch"] += 1
                fh.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t.\tmismatch\n")

    p.wait()
    if p.returncode != 0:
        raise SystemExit("bcftools query failed while building ancestral table")

    out_summary = Path(args.out_summary_json)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
