#!/usr/bin/env python3
"""Build fold-specific rare-minor-allele window features for M25B.

The input VCF is physically restricted to the canonical TRAIN partition.  For
each outer validation fold, MAC, MAF and minor-allele orientation are estimated
only from the other TRAIN folds.  The frozen site map is then projected onto
both fit and validation samples.  The upstream ``lai_rare`` universe remains
cohort-ascertained, so this is internal generalisation, not prospective variant
discovery.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from build_rare_window_features import (
    deterministic_gzip_writer,
    exclusion_reason,
    list_emitted_samples,
    normalize_chrom,
    ordered_ids_sha256,
    quantiles,
    read_contig_length,
    site_metrics,
    validate_aggregates,
    validate_upstream_provenance,
    window_index,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--vcf-index", required=True)
    parser.add_argument("--reference-fai", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--upstream-qc", required=True)
    parser.add_argument("--upstream-manifest", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--expected-train-samples", type=int, required=True)
    parser.add_argument("--expected-test-samples", type=int, required=True)
    parser.add_argument("--expected-input-sites", type=int, required=True)
    parser.add_argument("--outer-folds", default="0,1,2,4")
    parser.add_argument("--min-mac", type=int, default=2)
    parser.add_argument("--max-maf", type=float, default=0.01)
    parser.add_argument("--window-size-bp", type=int, default=250_000)
    parser.add_argument("--outdir", default=".")
    return parser.parse_args()


def load_split_manifest(
    path: str | Path,
    outer_folds: tuple[int, ...],
    expected_train: int,
    expected_test: int,
) -> tuple[list[str], np.ndarray, set[str], dict[str, int], dict[int, int]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample_id", "fold", "split", "split_group_key"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"split manifest lacks {sorted(required)}")
        rows = list(reader)
    ids = [row["sample_id"] for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise SystemExit("split manifest contains empty or duplicate sample IDs")
    train_rows = [row for row in rows if row["split"] == "TRAIN"]
    test_ids = {row["sample_id"] for row in rows if row["split"] == "TEST"}
    if len(train_rows) != expected_train or len(test_ids) != expected_test:
        raise SystemExit(
            f"partition cardinality mismatch: TRAIN={len(train_rows)}, TEST={len(test_ids)}"
        )
    train_ids = [row["sample_id"] for row in train_rows]
    try:
        train_folds = np.asarray([int(row["fold"]) for row in train_rows], dtype=np.int16)
    except ValueError as exc:
        raise SystemExit("TRAIN contains a non-integer fold") from exc
    observed = tuple(sorted(int(value) for value in np.unique(train_folds)))
    if observed != tuple(sorted(outer_folds)):
        raise SystemExit(f"TRAIN folds {observed} != expected {outer_folds}")
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    fold_counts = {fold: int(np.count_nonzero(train_folds == fold)) for fold in outer_folds}
    return train_ids, train_folds, test_ids, split_counts, fold_counts


def pairwise_site_stability(selected: dict[int, dict[str, str]]) -> list[dict]:
    rows: list[dict] = []
    folds = sorted(selected)
    for index, left in enumerate(folds):
        for right in folds[index + 1 :]:
            left_set = set(selected[left])
            right_set = set(selected[right])
            intersection = left_set & right_set
            union = left_set | right_set
            flips = sum(selected[left][site] != selected[right][site] for site in intersection)
            rows.append(
                {
                    "fold_a": left,
                    "fold_b": right,
                    "selected_a": len(left_set),
                    "selected_b": len(right_set),
                    "intersection": len(intersection),
                    "union": len(union),
                    "jaccard": len(intersection) / len(union) if union else 1.0,
                    "orientation_flips_in_intersection": flips,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    chrom = normalize_chrom(args.chrom)
    outer_folds = tuple(int(value) for value in args.outer_folds.split(",") if value)
    if chrom not in {str(value) for value in range(1, 23)}:
        raise SystemExit("M25B accepts autosomes 1..22 only")
    if len(outer_folds) < 2 or len(set(outer_folds)) != len(outer_folds):
        raise SystemExit("outer folds must contain at least two unique integers")
    if args.min_mac < 1 or not 0 < args.max_maf <= 0.5 or args.window_size_bp < 1:
        raise SystemExit("invalid MAC, MAF or window size")

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
    train_ids, train_folds, test_ids, split_counts, fold_counts = load_split_manifest(
        args.split_manifest,
        outer_folds,
        args.expected_train_samples,
        args.expected_test_samples,
    )
    n_samples = len(train_ids)
    fit_masks = {fold: train_folds != fold for fold in outer_folds}

    shapes = (n_windows, n_samples)
    panel_sites = {fold: np.zeros(n_windows, dtype=np.int64) for fold in outer_folds}
    input_sites = np.zeros(n_windows, dtype=np.int64)
    callable_sites = {fold: np.zeros(shapes, dtype=np.int32) for fold in outer_folds}
    carrier_sites = {fold: np.zeros(shapes, dtype=np.int32) for fold in outer_folds}
    dosage_sum = {fold: np.zeros(shapes, dtype=np.int32) for fold in outer_folds}
    het_count = {fold: np.zeros(shapes, dtype=np.int32) for fold in outer_folds}
    hom_count = {fold: np.zeros(shapes, dtype=np.int32) for fold in outer_folds}
    exclusion_counts = {fold: {} for fold in outer_folds}
    selected: dict[int, dict[str, str]] = {fold: {} for fold in outer_folds}
    partial_genotypes = {fold: 0 for fold in outer_folds}
    total_input_sites = 0
    previous_pos = 0
    seen_variants: set[str] = set()

    site_columns = [
        "outer_fold", "variant_id", "chrom", "pos_1based", "ref", "alt",
        "fold_train_ac_alt", "fold_train_an", "fold_train_mac", "fold_train_maf",
        "counted_allele", "included", "exclude_reason", "n_fold_train",
        "n_validation", "n_complete_fold_train", "n_partial_fold_train",
        "minor_carrier_fold_train", "minor_dosage_fold_train", "window_id",
    ]
    with tempfile.TemporaryDirectory(prefix="m25b_train_") as temp_dir:
        sample_file = Path(temp_dir) / "train_samples.txt"
        sample_file.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
        emitted_samples = list_emitted_samples(args.vcf, sample_file)
        if emitted_samples != train_ids:
            raise SystemExit("bcftools emitted TRAIN samples in a different order")
        if set(emitted_samples) & test_ids:
            raise SystemExit("leakage gate failed: TEST genotypes were emitted")

        query = subprocess.Popen(
            [
                "bcftools", "query", "-S", str(sample_file), "-f",
                r"%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n", str(args.vcf),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if query.stdout is None or query.stderr is None:
            raise SystemExit("could not open bcftools query")

        with ExitStack() as stack:
            site_writers = {}
            for fold in outer_folds:
                handle = stack.enter_context(
                    deterministic_gzip_writer(outdir / f"chr{chrom}.fold{fold}.rare_sites.tsv.gz")
                )
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(site_columns)
                site_writers[fold] = writer

            for raw_line in query.stdout:
                parts = raw_line.rstrip(b"\n").split(b"\t", 4)
                if len(parts) != 5:
                    raise SystemExit(f"malformed bcftools row: {raw_line[:160]!r}")
                chrom_b, pos_b, ref_b, alt_b, genotype_bytes = parts
                row_chrom = chrom_b.decode("ascii")
                pos = int(pos_b)
                ref = ref_b.decode("ascii")
                alt = alt_b.decode("ascii")
                if normalize_chrom(row_chrom) != chrom or pos < previous_pos:
                    raise SystemExit(f"unexpected chromosome/order at {row_chrom}:{pos}")
                previous_pos = pos
                if pos > contig_length or len(ref) != 1 or len(alt) != 1 or "," in alt:
                    raise SystemExit(f"expected in-range biallelic SNV at {row_chrom}:{pos}")
                win = window_index(pos, args.window_size_bp)
                input_sites[win] += 1
                total_input_sites += 1

                fixed_width = b"\t" + genotype_bytes
                if len(fixed_width) != 4 * n_samples:
                    raise SystemExit(f"non-diploid or malformed GT width at {row_chrom}:{pos}")
                gt = np.frombuffer(fixed_width, dtype=np.uint8).reshape(n_samples, 4)
                allele_bytes = gt[:, (1, 3)]
                full_metrics = site_metrics(allele_bytes)
                complete_all = full_metrics["complete"]
                alt_dosage_all = np.count_nonzero(allele_bytes == ord("1"), axis=1).astype(np.int8)
                variant_id = f"{row_chrom}:{pos}:{ref}:{alt}"
                if variant_id in seen_variants:
                    raise SystemExit(f"duplicate variant record: {variant_id}")
                seen_variants.add(variant_id)

                for fold in outer_folds:
                    fit_mask = fit_masks[fold]
                    metrics = site_metrics(allele_bytes[fit_mask])
                    reason = exclusion_reason(metrics, args.min_mac, args.max_maf)
                    counts = exclusion_counts[fold]
                    counts[reason] = counts.get(reason, 0) + 1
                    partial_genotypes[fold] += int(metrics["partial"].sum())
                    minor_dosage = np.zeros(n_samples, dtype=np.int8)
                    carriers = np.zeros(n_samples, dtype=bool)
                    if reason == "included":
                        minor_dosage[complete_all] = (
                            2 - alt_dosage_all[complete_all]
                            if metrics["counted_allele"] == "REF"
                            else alt_dosage_all[complete_all]
                        )
                        carriers = complete_all & (minor_dosage > 0)
                        panel_sites[fold][win] += 1
                        callable_sites[fold][win] += complete_all
                        carrier_sites[fold][win] += carriers
                        dosage_sum[fold][win] += minor_dosage
                        het_count[fold][win] += complete_all & (minor_dosage == 1)
                        hom_count[fold][win] += complete_all & (minor_dosage == 2)
                        selected[fold][variant_id] = str(metrics["counted_allele"])

                    site_writers[fold].writerow(
                        [
                            fold, variant_id, row_chrom, pos, ref, alt,
                            metrics["alt_count"], metrics["allele_number"],
                            metrics["minor_count"],
                            f"{metrics['maf']:.12g}" if math.isfinite(metrics["maf"]) else "NA",
                            metrics["counted_allele"] or "NA", int(reason == "included"), reason,
                            int(fit_mask.sum()), int((~fit_mask).sum()),
                            int(metrics["complete"].sum()), int(metrics["partial"].sum()),
                            int(carriers[fit_mask].sum()), int(minor_dosage[fit_mask].sum()),
                            f"chr{chrom}_w{win:04d}",
                        ]
                    )

        stderr = query.stderr.read().decode("utf-8", errors="replace")
        return_code = query.wait()
        if return_code != 0:
            raise SystemExit(f"bcftools query failed ({return_code}): {stderr[-2000:]}")

    if total_input_sites != args.expected_input_sites:
        raise SystemExit(f"input sites {total_input_sites} != expected {args.expected_input_sites}")

    fold_qc: dict[str, dict] = {}
    for fold in outer_folds:
        if sum(exclusion_counts[fold].values()) != total_input_sites:
            raise SystemExit(f"fold {fold}: exclusion counts do not reconcile to input sites")
        validate_aggregates(
            panel_sites[fold], callable_sites[fold], carrier_sites[fold], dosage_sum[fold],
            het_count[fold], hom_count[fold],
        )
        if int(panel_sites[fold].sum()) != len(selected[fold]):
            raise SystemExit(f"fold {fold}: selected sites do not reconcile")

        window_path = outdir / f"chr{chrom}.fold{fold}.windows.tsv"
        with window_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    "outer_fold", "chrom", "window_id", "start_0based", "end_0based",
                    "window_bp", "n_input_cohort_rare_sites", "n_fold_train_rare_sites",
                    "informative_in_fold_train",
                ]
            )
            fit_mask = fit_masks[fold]
            for win in range(n_windows):
                start = win * args.window_size_bp
                end = min((win + 1) * args.window_size_bp, contig_length)
                train_rate = np.divide(
                    carrier_sites[fold][win, fit_mask],
                    callable_sites[fold][win, fit_mask],
                    out=np.full(int(fit_mask.sum()), np.nan),
                    where=callable_sites[fold][win, fit_mask] > 0,
                )
                informative = bool(
                    panel_sites[fold][win] > 0
                    and np.all(np.isfinite(train_rate))
                    and np.ptp(train_rate) > 0
                )
                writer.writerow(
                    [
                        fold, contig, f"chr{chrom}_w{win:04d}", start, end, end - start,
                        int(input_sites[win]), int(panel_sites[fold][win]), int(informative),
                    ]
                )

        feature_path = outdir / f"chr{chrom}.fold{fold}.sample_window_features.tsv.gz"
        with deterministic_gzip_writer(feature_path) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    "sample_id", "outer_fold", "fold_role", "chrom", "window_id",
                    "start_0based", "end_0based", "n_fold_train_rare_sites",
                    "n_callable_rare_sites", "n_missing_rare_sites",
                    "minor_carrier_site_count", "minor_dosage_sum", "het_minor_count",
                    "hom_minor_count", "minor_carrier_rate", "minor_allele_rate",
                    "rare_site_call_rate", "zero_panel_sites",
                ]
            )
            for sample_index, sample_id in enumerate(train_ids):
                role = "FIT" if train_folds[sample_index] != fold else "VALIDATION"
                for win in range(n_windows):
                    start = win * args.window_size_bp
                    end = min((win + 1) * args.window_size_bp, contig_length)
                    panel = int(panel_sites[fold][win])
                    callable_n = int(callable_sites[fold][win, sample_index])
                    carrier_n = int(carrier_sites[fold][win, sample_index])
                    dosage_n = int(dosage_sum[fold][win, sample_index])
                    carrier_rate = f"{carrier_n / callable_n:.12g}" if callable_n else "NA"
                    allele_rate = f"{dosage_n / (2 * callable_n):.12g}" if callable_n else "NA"
                    call_rate = f"{callable_n / panel:.12g}" if panel else "NA"
                    writer.writerow(
                        [
                            sample_id, fold, role, contig, f"chr{chrom}_w{win:04d}", start, end,
                            panel, callable_n, panel - callable_n, carrier_n, dosage_n,
                            int(het_count[fold][win, sample_index]),
                            int(hom_count[fold][win, sample_index]), carrier_rate, allele_rate,
                            call_rate, int(panel == 0),
                        ]
                    )

        fit_mask = fit_masks[fold]
        fit_rates = np.divide(
            carrier_sites[fold][:, fit_mask],
            callable_sites[fold][:, fit_mask],
            out=np.full((n_windows, int(fit_mask.sum())), np.nan),
            where=callable_sites[fold][:, fit_mask] > 0,
        )
        informative = (
            (panel_sites[fold] > 0)
            & np.all(np.isfinite(fit_rates), axis=1)
            & (np.nanmax(fit_rates, axis=1) > np.nanmin(fit_rates, axis=1))
        )
        call_rate_cells = np.divide(
            callable_sites[fold], panel_sites[fold][:, None],
            out=np.full(shapes, np.nan), where=panel_sites[fold][:, None] > 0,
        )
        fold_qc[str(fold)] = {
            "n_fit": int(fit_mask.sum()),
            "n_validation": int((~fit_mask).sum()),
            "n_selected_sites": int(panel_sites[fold].sum()),
            "n_windows": n_windows,
            "n_informative_windows": int(informative.sum()),
            "n_zero_panel_windows": int(np.count_nonzero(panel_sites[fold] == 0)),
            "selected_sites_per_window": quantiles(panel_sites[fold]),
            "rare_site_call_rate_cells": quantiles(call_rate_cells[np.isfinite(call_rate_cells)]),
            "exclusion_counts": exclusion_counts[fold],
            "partial_genotypes_in_fold_train": partial_genotypes[fold],
        }

    qc_payload = {
        "status": "PASS",
        "stage": "M25B_FOLD_FEATURES",
        "scope": "internal TRAIN-only outer-fold generalisation; TEST genotypes not emitted",
        "ascertainment": {
            "input_universe": "cohort-ascertained lai_rare VCF",
            "fold_operation": "MAC, MAF and minor orientation fit on outer-fold FIT only",
            "limitation": "not prospective discovery because the upstream candidate universe used the filter cohort",
        },
        "chrom": chrom,
        "reference_contig": contig,
        "contig_length_bp": contig_length,
        "coordinate_contract": "0-based half-open windows; VCF POS is 1-based",
        "window_size_bp": args.window_size_bp,
        "n_windows": n_windows,
        "n_input_sites": total_input_sites,
        "n_train_samples": n_samples,
        "n_test_samples_in_manifest": len(test_ids),
        "n_test_genotypes_emitted": 0,
        "outer_folds": list(outer_folds),
        "fold_counts": fold_counts,
        "split_counts": split_counts,
        "train_sample_order_sha256": ordered_ids_sha256(train_ids),
        "input_sha256_verified_against_m24": input_hashes,
        "inherited_reference_qc": upstream_qc["reference_qc"],
        "filters": {"min_mac": args.min_mac, "max_maf_inclusive": args.max_maf},
        "folds": fold_qc,
        "pairwise_site_stability": pairwise_site_stability(selected),
        "stop_rules": {
            "technical": "fail closed on TEST emission, cardinality, provenance, coordinate or count reconciliation",
            "scientific": "do not infer prospective discovery, biology, LAI improvement or select PCA rank here",
        },
    }
    write_json(outdir / f"chr{chrom}.fold_features_qc.json", qc_payload)
    print(json.dumps({"status": "PASS", "folds": fold_qc}, sort_keys=True))


if __name__ == "__main__":
    main()
