#!/usr/bin/env python3
"""Build an auditable TRAIN-only rare-minor-allele window feature table.

This descriptive pilot starts from the cohort-ascertained ``lai_rare`` VCF,
subsets genotypes physically to TRAIN with bcftools, re-estimates MAC/MAF and
minor-allele orientation in TRAIN, and aggregates complete diploid genotypes
into fixed, non-overlapping genomic windows.  It does not train or evaluate a
model and must not be reused as a leakage-safe cross-validation matrix.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--vcf-index", required=True)
    parser.add_argument("--reference-fai", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--upstream-qc", required=True)
    parser.add_argument("--upstream-manifest", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--train-label", default="TRAIN")
    parser.add_argument("--test-label", default="TEST")
    parser.add_argument("--expected-train-samples", type=int, required=True)
    parser.add_argument("--expected-input-sites", type=int, required=True)
    parser.add_argument("--min-mac", type=int, default=2)
    parser.add_argument("--max-maf", type=float, default=0.01)
    parser.add_argument("--window-size-bp", type=int, default=250_000)
    parser.add_argument("--outdir", default=".")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


@contextmanager
def deterministic_gzip_writer(path: str | Path) -> Iterator[TextIO]:
    """Write text gzip with a fixed mtime so scientific tables are byte-stable."""
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                yield text


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def normalize_chrom(value: str) -> str:
    return value[3:] if value.lower().startswith("chr") else value


def read_contig_length(fai_path: str | Path, chrom: str) -> tuple[str, int]:
    candidates = {chrom, f"chr{normalize_chrom(chrom)}"}
    with Path(fai_path).open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2 and fields[0] in candidates:
                return fields[0], int(fields[1])
    raise SystemExit(f"Reference FAI lacks chromosome {chrom}")


def load_split_ids(
    path: str | Path,
    sample_id_col: str,
    split_col: str,
    train_label: str,
    test_label: str,
    expected_train_samples: int,
) -> tuple[list[str], set[str], dict[str, int]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {sample_id_col, split_col}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"split_manifest lacks columns {sorted(required)}")
        rows = list(reader)
    sample_ids = [row[sample_id_col] for row in rows]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise SystemExit("split_manifest contains empty or duplicate sample IDs")
    train_ids = [row[sample_id_col] for row in rows if row[split_col] == train_label]
    test_ids = {row[sample_id_col] for row in rows if row[split_col] == test_label}
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row[split_col]] = split_counts.get(row[split_col], 0) + 1
    if len(train_ids) != expected_train_samples:
        raise SystemExit(
            f"TRAIN sample count {len(train_ids)} != expected {expected_train_samples}"
        )
    if not test_ids:
        raise SystemExit(f"No samples found with split={test_label!r}; leakage gate cannot run")
    return train_ids, test_ids, split_counts


def list_emitted_samples(vcf: str | Path, sample_file: str | Path) -> list[str]:
    proc = subprocess.run(
        ["bcftools", "view", "-h", "-S", str(sample_file), str(vcf)],
        capture_output=True,
        text=True,
        check=True,
    )
    header = next((line for line in proc.stdout.splitlines() if line.startswith("#CHROM\t")), None)
    if header is None:
        raise SystemExit("bcftools did not emit a #CHROM header")
    return header.split("\t")[9:]


def site_metrics(allele_bytes: np.ndarray) -> dict:
    """Summarize one biallelic site; AC/AN use called alleles, features full GTs."""
    valid = (allele_bytes == ord("0")) | (allele_bytes == ord("1"))
    unexpected = ~valid & (allele_bytes != ord("."))
    if np.any(unexpected):
        raise ValueError("Unexpected allele index in genotype")
    complete = valid.all(axis=1)
    partial = valid.any(axis=1) & ~complete
    alt_count = int(np.count_nonzero(allele_bytes == ord("1")))
    allele_number = int(valid.sum())
    minor_count = min(alt_count, allele_number - alt_count) if allele_number else 0
    is_tie = allele_number > 0 and alt_count * 2 == allele_number
    maf = minor_count / allele_number if allele_number else math.nan
    counted_allele = None if allele_number == 0 or is_tie else ("REF" if alt_count * 2 > allele_number else "ALT")
    return {
        "valid": valid,
        "complete": complete,
        "partial": partial,
        "alt_count": alt_count,
        "allele_number": allele_number,
        "minor_count": minor_count,
        "maf": maf,
        "counted_allele": counted_allele,
    }


def exclusion_reason(metrics: dict, min_mac: int, max_maf: float) -> str:
    if metrics["allele_number"] == 0:
        return "all_missing"
    if metrics["counted_allele"] is None:
        return "frequency_tie"
    if metrics["minor_count"] < min_mac:
        return "mac_below_min"
    if metrics["maf"] > max_maf:
        return "maf_above_max"
    return "included"


def window_index(pos_1based: int, window_size_bp: int) -> int:
    if pos_1based < 1 or window_size_bp < 1:
        raise ValueError("Coordinates and window size must be positive")
    return (pos_1based - 1) // window_size_bp


def quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {}
    return {
        label: float(value)
        for label, value in zip(
            ("min", "p25", "median", "p75", "max"),
            np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0]),
        )
    }


def validate_upstream_provenance(
    vcf: str | Path,
    vcf_index: str | Path,
    reference_fai: str | Path,
    split_manifest: str | Path,
    upstream_qc_path: str | Path,
    upstream_manifest_path: str | Path,
    chrom: str,
    expected_input_sites: int,
) -> tuple[dict, dict[str, str]]:
    qc = json.loads(Path(upstream_qc_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(upstream_manifest_path).read_text(encoding="utf-8"))
    if qc.get("status") != "PASS" or normalize_chrom(str(qc.get("chrom"))) != normalize_chrom(chrom):
        raise SystemExit("Upstream M24 QC is not PASS for the requested chromosome")
    reference_qc = qc.get("reference_qc", {})
    if reference_qc.get("mismatches") != 0 or reference_qc.get("matches") != expected_input_sites:
        raise SystemExit("Upstream M24 REF/hg38 gate does not cover every expected input site")
    if qc.get("expected_sites_from_m14_summary") != expected_input_sites:
        raise SystemExit("Upstream M24 input-site count differs from this pilot contract")

    inputs = manifest.get("inputs", {})
    expected_qc_hash = manifest.get("sha256", {}).get(Path(upstream_qc_path).name)
    if expected_qc_hash is None or sha256_file(upstream_qc_path) != expected_qc_hash:
        raise SystemExit("Upstream M24 QC hash does not match its manifest")
    paths = [Path(vcf), Path(vcf_index), Path(reference_fai), Path(split_manifest)]
    observed_hashes = {path.name: sha256_file(path) for path in paths}
    for name, observed in observed_hashes.items():
        expected = inputs.get(name)
        if expected is None or observed != expected:
            raise SystemExit(f"Upstream provenance hash mismatch for {name}")
    return qc, observed_hashes


def validate_aggregates(
    panel_sites: np.ndarray,
    callable_sites: np.ndarray,
    carrier_sites: np.ndarray,
    dosage_sum: np.ndarray,
    het_count: np.ndarray,
    hom_count: np.ndarray,
) -> None:
    expected = np.broadcast_to(panel_sites[:, None], callable_sites.shape)
    missing = expected - callable_sites
    if np.any(missing < 0):
        raise SystemExit("Invariant failed: callable exceeds panel sites")
    if np.any(carrier_sites > callable_sites):
        raise SystemExit("Invariant failed: carrier exceeds callable")
    if np.any(dosage_sum < carrier_sites) or np.any(dosage_sum > 2 * carrier_sites):
        raise SystemExit("Invariant failed: dosage is outside [carrier, 2*carrier]")
    if np.any(carrier_sites != het_count + hom_count):
        raise SystemExit("Invariant failed: carrier != het + hom")
    if np.any(dosage_sum != het_count + 2 * hom_count):
        raise SystemExit("Invariant failed: dosage != het + 2*hom")


def main() -> None:
    args = parse_args()
    chrom = normalize_chrom(str(args.chrom))
    if chrom not in {str(value) for value in range(1, 23)}:
        raise SystemExit("This pilot accepts autosomes 1..22 only")
    if args.min_mac < 1 or not (0.0 < args.max_maf <= 0.5) or args.window_size_bp < 1:
        raise SystemExit("Invalid MAC, MAF or window-size parameter")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    contig, contig_length = read_contig_length(args.reference_fai, chrom)
    n_windows = math.ceil(contig_length / args.window_size_bp)

    upstream_qc, input_hashes = validate_upstream_provenance(
        args.vcf,
        args.vcf_index,
        args.reference_fai,
        args.split_manifest,
        args.upstream_qc,
        args.upstream_manifest,
        chrom,
        args.expected_input_sites,
    )
    train_ids, test_ids, split_counts = load_split_ids(
        args.split_manifest,
        args.sample_id_col,
        args.split_col,
        args.train_label,
        args.test_label,
        args.expected_train_samples,
    )
    n_samples = len(train_ids)

    panel_sites = np.zeros(n_windows, dtype=np.int64)
    input_sites = np.zeros(n_windows, dtype=np.int64)
    callable_sites = np.zeros((n_windows, n_samples), dtype=np.int32)
    carrier_sites = np.zeros_like(callable_sites)
    dosage_sum = np.zeros_like(callable_sites)
    het_count = np.zeros_like(callable_sites)
    hom_count = np.zeros_like(callable_sites)
    exclusion_counts: dict[str, int] = {}
    partial_genotypes = 0
    previous_pos = 0
    seen_variants: set[str] = set()

    sites_path = outdir / f"chr{chrom}.train_rare_sites.tsv.gz"
    with tempfile.TemporaryDirectory(prefix="m25_train_") as temp_dir:
        sample_file = Path(temp_dir) / "train_samples.txt"
        sample_file.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
        emitted_samples = list_emitted_samples(args.vcf, sample_file)
        if emitted_samples != train_ids:
            raise SystemExit("bcftools emitted TRAIN samples in a different order")
        leakage = set(emitted_samples) & test_ids
        if leakage:
            raise SystemExit(f"Leakage gate failed: {len(leakage)} TEST genotypes emitted")

        query = subprocess.Popen(
            [
                "bcftools",
                "query",
                "-S",
                str(sample_file),
                "-f",
                r"%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n",
                str(args.vcf),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if query.stdout is None or query.stderr is None:
            raise SystemExit("Could not open bcftools query")

        site_columns = [
            "variant_id", "chrom", "pos_1based", "ref", "alt", "train_ac_alt",
            "train_an", "train_mac", "train_maf", "counted_allele", "included",
            "exclude_reason", "n_complete_diploid", "n_partial_missing",
            "minor_carrier_samples", "minor_dosage_sum", "window_id",
        ]
        with deterministic_gzip_writer(sites_path) as site_handle:
            writer = csv.writer(site_handle, delimiter="\t", lineterminator="\n")
            writer.writerow(site_columns)
            total_input_sites = 0
            total_selected_sites = 0
            for raw_line in query.stdout:
                parts = raw_line.rstrip(b"\n").split(b"\t", 4)
                if len(parts) != 5:
                    raise SystemExit(f"Malformed bcftools row: {raw_line[:160]!r}")
                chrom_b, pos_b, ref_b, alt_b, genotype_bytes = parts
                row_chrom = chrom_b.decode("ascii")
                pos = int(pos_b)
                ref = ref_b.decode("ascii")
                alt = alt_b.decode("ascii")
                if normalize_chrom(row_chrom) != chrom:
                    raise SystemExit(f"Unexpected chromosome {row_chrom}:{pos}")
                if pos < previous_pos:
                    raise SystemExit(f"VCF is unsorted at {row_chrom}:{pos}")
                previous_pos = pos
                if pos > contig_length or len(ref) != 1 or len(alt) != 1 or "," in alt:
                    raise SystemExit(f"Expected in-range biallelic SNV at {row_chrom}:{pos}:{ref}:{alt}")
                win = window_index(pos, args.window_size_bp)
                input_sites[win] += 1
                total_input_sites += 1

                fixed_width = b"\t" + genotype_bytes
                expected_width = 4 * n_samples
                if len(fixed_width) != expected_width:
                    raise SystemExit(
                        f"Non-diploid or malformed GT width at {row_chrom}:{pos}: "
                        f"{len(fixed_width)} != {expected_width}"
                    )
                gt = np.frombuffer(fixed_width, dtype=np.uint8).reshape(n_samples, 4)
                allele_bytes = gt[:, (1, 3)]
                try:
                    metrics = site_metrics(allele_bytes)
                except ValueError as exc:
                    raise SystemExit(f"{exc} at {row_chrom}:{pos}") from exc
                reason = exclusion_reason(metrics, args.min_mac, args.max_maf)
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
                partial_genotypes += int(metrics["partial"].sum())

                carriers = np.zeros(n_samples, dtype=bool)
                minor_dosage = np.zeros(n_samples, dtype=np.int8)
                if reason == "included":
                    complete = metrics["complete"]
                    alt_dosage = np.count_nonzero(allele_bytes == ord("1"), axis=1).astype(np.int8)
                    minor_dosage[complete] = (
                        (2 - alt_dosage[complete])
                        if metrics["counted_allele"] == "REF"
                        else alt_dosage[complete]
                    )
                    carriers = complete & (minor_dosage > 0)
                    panel_sites[win] += 1
                    callable_sites[win] += complete
                    carrier_sites[win] += carriers
                    dosage_sum[win] += minor_dosage
                    het_count[win] += complete & (minor_dosage == 1)
                    hom_count[win] += complete & (minor_dosage == 2)
                    total_selected_sites += 1

                variant_id = f"{row_chrom}:{pos}:{ref}:{alt}"
                if variant_id in seen_variants:
                    raise SystemExit(f"Duplicate variant record: {variant_id}")
                seen_variants.add(variant_id)
                writer.writerow(
                    [
                        variant_id, row_chrom, pos, ref, alt, metrics["alt_count"],
                        metrics["allele_number"], metrics["minor_count"],
                        f"{metrics['maf']:.12g}" if math.isfinite(metrics["maf"]) else "NA",
                        metrics["counted_allele"] or "NA", int(reason == "included"), reason,
                        int(metrics["complete"].sum()), int(metrics["partial"].sum()),
                        int(carriers.sum()), int(minor_dosage.sum()), f"chr{chrom}_w{win:04d}",
                    ]
                )

        stderr = query.stderr.read().decode("utf-8", errors="replace")
        return_code = query.wait()
        if return_code != 0:
            raise SystemExit(f"bcftools query failed ({return_code}): {stderr[-2000:]}")

    if total_input_sites != args.expected_input_sites:
        raise SystemExit(
            f"Input site count {total_input_sites} != expected {args.expected_input_sites}"
        )
    if int(panel_sites.sum()) != total_selected_sites:
        raise SystemExit("Selected-site window totals do not reconcile")
    validate_aggregates(panel_sites, callable_sites, carrier_sites, dosage_sum, het_count, hom_count)

    windows_path = outdir / f"chr{chrom}.windows.tsv"
    window_columns = [
        "chrom", "window_id", "start_0based", "end_0based", "window_bp",
        "n_input_cohort_rare_sites", "n_train_rare_sites", "rare_site_density_per_mb",
        "carrier_incidence", "minor_dosage_total", "n_carrier_samples",
        "mean_minor_dosage", "variance_minor_dosage", "constant_dosage_column",
    ]
    with windows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(window_columns)
        for win in range(n_windows):
            start = win * args.window_size_bp
            end = min((win + 1) * args.window_size_bp, contig_length)
            writer.writerow(
                [
                    contig, f"chr{chrom}_w{win:04d}", start, end, end - start,
                    int(input_sites[win]), int(panel_sites[win]),
                    f"{panel_sites[win] / ((end - start) / 1_000_000):.12g}",
                    int(carrier_sites[win].sum()), int(dosage_sum[win].sum()),
                    int(np.count_nonzero(dosage_sum[win])),
                    f"{dosage_sum[win].mean():.12g}", f"{dosage_sum[win].var():.12g}",
                    int(np.ptp(dosage_sum[win]) == 0),
                ]
            )

    long_path = outdir / f"chr{chrom}.sample_window_features.tsv.gz"
    long_columns = [
        "sample_id", "chrom", "window_id", "start_0based", "end_0based", "window_bp",
        "n_train_rare_sites", "n_callable_rare_sites", "n_missing_rare_sites",
        "minor_carrier_site_count", "minor_dosage_sum", "het_minor_count", "hom_minor_count",
        "carrier_rate", "mean_minor_dosage", "minor_allele_rate", "rare_site_call_rate",
        "zero_panel_sites",
    ]
    with deterministic_gzip_writer(long_path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(long_columns)
        for sample_idx, sample_id in enumerate(train_ids):
            for win in range(n_windows):
                start = win * args.window_size_bp
                end = min((win + 1) * args.window_size_bp, contig_length)
                panel = int(panel_sites[win])
                callable_n = int(callable_sites[win, sample_idx])
                carrier_n = int(carrier_sites[win, sample_idx])
                dosage_n = int(dosage_sum[win, sample_idx])
                if callable_n:
                    rates = (
                        f"{carrier_n / callable_n:.12g}",
                        f"{dosage_n / callable_n:.12g}",
                        f"{dosage_n / (2 * callable_n):.12g}",
                    )
                else:
                    rates = ("NA", "NA", "NA")
                call_rate = f"{callable_n / panel:.12g}" if panel else "NA"
                writer.writerow(
                    [
                        sample_id, contig, f"chr{chrom}_w{win:04d}", start, end, end - start,
                        panel, callable_n, panel - callable_n, carrier_n, dosage_n,
                        int(het_count[win, sample_idx]), int(hom_count[win, sample_idx]),
                        *rates, call_rate, int(panel == 0),
                    ]
                )

    callable_total = callable_sites.sum(axis=0)
    carrier_total = carrier_sites.sum(axis=0)
    dosage_total = dosage_sum.sum(axis=0)
    site_opportunities = int(panel_sites.sum())
    total_cell_count = int(n_samples * n_windows)
    nonzero_windows_per_sample = np.count_nonzero(dosage_sum, axis=0)
    constant_columns = np.ptp(dosage_sum, axis=1) == 0
    hom_dosage = int(2 * hom_count.sum())
    all_dosage = int(dosage_sum.sum())
    callable_denominator = np.broadcast_to(panel_sites[:, None], callable_sites.shape)
    valid_cells = callable_denominator > 0
    cell_call_rates = np.divide(
        callable_sites,
        callable_denominator,
        out=np.full(callable_sites.shape, np.nan, dtype=float),
        where=valid_cells,
    )
    cell_carrier_rates = np.divide(
        carrier_sites,
        callable_sites,
        out=np.full(callable_sites.shape, np.nan, dtype=float),
        where=callable_sites > 0,
    )
    cell_allele_rates = np.divide(
        dosage_sum,
        2 * callable_sites,
        out=np.full(callable_sites.shape, np.nan, dtype=float),
        where=callable_sites > 0,
    )
    finite = np.isfinite(cell_carrier_rates) & np.isfinite(cell_allele_rates)
    rate_correlation_value = (
        float(np.corrcoef(cell_carrier_rates[finite], cell_allele_rates[finite])[0, 1])
        if np.count_nonzero(finite) > 1
        else math.nan
    )
    rate_correlation = rate_correlation_value if math.isfinite(rate_correlation_value) else None

    numeric_matrix = cell_allele_rates.T
    numeric_matrix = np.nan_to_num(numeric_matrix, nan=0.0)
    variable_mask = ~constant_columns
    singular_values: list[float] = []
    effective_rank = 0.0
    numerical_rank = 0
    if np.any(variable_mask):
        centered = numeric_matrix[:, variable_mask] - numeric_matrix[:, variable_mask].mean(axis=0)
        singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
        singular_values = [float(value) for value in singular[:10]]
        numerical_rank = int(np.linalg.matrix_rank(centered))
        weights = singular / singular.sum() if singular.sum() else np.array([])
        effective_rank = float(np.exp(-np.sum(weights * np.log(weights)))) if weights.size else 0.0

    qc_payload = {
        "status": "PASS",
        "scope": "descriptive_train_transductive_chr_window_features_no_training_no_evaluation",
        "ascertainment": {
            "input_universe": "lai_rare VCF selected using the full filter cohort",
            "train_operation": "MAC, MAF and minor-allele orientation recomputed in TRAIN",
            "limitation": "This is a TRAIN subset of a cohort-ascertained rare universe; it cannot recover sites excluded upstream and is not a prospective TRAIN-only rare set.",
        },
        "cross_validation_limit": "For future out-of-fold PCA/NMF/AE, site selection, orientation and scaling must be recomputed inside each outer-training fold.",
        "chrom": chrom,
        "reference_contig": contig,
        "contig_length_bp": contig_length,
        "coordinate_contract": "0-based half-open windows; VCF positions are 1-based; window=floor((POS-1)/window_size_bp)",
        "window_size_bp": args.window_size_bp,
        "n_windows": n_windows,
        "n_input_sites": total_input_sites,
        "n_train_rare_sites": total_selected_sites,
        "n_train_samples": n_samples,
        "n_test_genotypes_emitted": 0,
        "split_counts": split_counts,
        "train_sample_order_sha256": ordered_ids_sha256(train_ids),
        "input_sha256_verified_against_m24": input_hashes,
        "inherited_reference_qc": upstream_qc["reference_qc"],
        "filters": {"min_mac": args.min_mac, "max_maf_inclusive": args.max_maf},
        "site_exclusion_counts": exclusion_counts,
        "partial_genotypes_observed": partial_genotypes,
        "reconciliation": {
            "panel_sites_sum": int(panel_sites.sum()),
            "callable_total": int(callable_sites.sum()),
            "carrier_total": int(carrier_sites.sum()),
            "minor_dosage_total": int(dosage_sum.sum()),
            "het_minor_total": int(het_count.sum()),
            "hom_minor_total": int(hom_count.sum()),
        },
        "diagnostics": {
            "panel_sites_per_window": quantiles(panel_sites),
            "windows_without_panel_sites": int(np.count_nonzero(panel_sites == 0)),
            "constant_dosage_windows": int(constant_columns.sum()),
            "zero_dosage_cell_fraction": float(np.count_nonzero(dosage_sum == 0) / total_cell_count),
            "nonzero_windows_per_sample": quantiles(nonzero_windows_per_sample),
            "sample_callable_sites": quantiles(callable_total),
            "sample_carrier_sites": quantiles(carrier_total),
            "sample_minor_dosage": quantiles(dosage_total),
            "rare_site_call_rate_cells": quantiles(cell_call_rates[np.isfinite(cell_call_rates)]),
            "carrier_rate_vs_minor_allele_rate_correlation": rate_correlation,
            "fraction_minor_dosage_from_homozygotes": hom_dosage / all_dosage if all_dosage else 0.0,
            "numerical_rank_after_centering": numerical_rank,
            "effective_rank_after_centering": effective_rank,
            "first_10_singular_values": singular_values,
            "cohort_association": "not evaluated: cohort/ancestry labels are intentionally absent from M25",
        },
        "field_definitions": {
            "n_train_rare_sites": "Sites in the window passing TRAIN MAC/MAF after cohort-level ascertainment.",
            "n_callable_rare_sites": "Panel sites with a complete diploid GT for the individual; not base-level callability.",
            "minor_carrier_site_count": "Callable panel sites where the individual carries at least one TRAIN-defined minor allele.",
            "minor_dosage_sum": "Sum of 0, 1 or 2 TRAIN-defined minor-allele copies across callable panel sites.",
            "carrier_rate": "minor_carrier_site_count / n_callable_rare_sites.",
            "mean_minor_dosage": "minor_dosage_sum / n_callable_rare_sites, range [0,2].",
            "minor_allele_rate": "minor_dosage_sum / (2*n_callable_rare_sites), range [0,1].",
            "rare_site_call_rate": "n_callable_rare_sites / n_train_rare_sites; callability only within the rare-site panel.",
        },
        "stop_rules": {
            "before_modeling": "Stop if leakage, reconciliation or denominator gates fail; this process fails closed on those conditions.",
            "post_qc": "Do not start PCA/NMF/AE until POST review assesses sparsity, constant columns, callability and carrier/dosage redundancy.",
        },
    }
    qc_path = outdir / f"chr{chrom}.rare_window_qc.json"
    write_json(qc_path, qc_payload)

    print(
        json.dumps(
            {
                "status": "PASS",
                "chrom": chrom,
                "n_train_samples": n_samples,
                "n_windows": n_windows,
                "n_input_sites": total_input_sites,
                "n_train_rare_sites": total_selected_sites,
                "zero_dosage_cell_fraction": qc_payload["diagnostics"]["zero_dosage_cell_fraction"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
