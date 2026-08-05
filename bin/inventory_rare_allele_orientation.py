#!/usr/bin/env python3
"""Inventory ALT-versus-minor orientation without rebuilding M14 topology.

The upstream rare VCF was filtered before the 2,619-sample M14 subset was
selected.  This scanner therefore reports orientation in two explicit
universes: every sample in the VCF header (the filter cohort) and the canonical
M14 subset.  TRAIN is used only for descriptive burden summaries.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import resource
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rare_allele_orientation import SiteOrientation


MODES = (
    "historical_alt",
    "minor_filter_cohort",
    "minor_m14_subset",
    "exclude_alt_major_filter_cohort",
    "exclude_alt_major_m14_subset",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--canonical-summary", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--train-label", default="TRAIN")
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--expected-filter-samples", type=int, default=2723)
    parser.add_argument("--expected-m14-samples", type=int, default=2619)
    parser.add_argument("--outdir", default=".")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ordered_ids_sha256(values: list[str]) -> str:
    payload = ("\n".join(values) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_reference_contig(fasta_path: str | Path, chrom: str) -> tuple[str, str]:
    fai_path = Path(f"{fasta_path}.fai")
    if not fai_path.exists():
        raise SystemExit(f"Missing FASTA index: {fai_path}")
    entries = {}
    with fai_path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                raise SystemExit(f"Malformed FASTA index line: {line[:200]!r}")
            entries[fields[0]] = tuple(map(int, fields[1:5]))
    candidates = (chrom, f"chr{chrom}" if not chrom.startswith("chr") else chrom[3:])
    for candidate in candidates:
        if candidate not in entries:
            continue
        length, offset, line_bases, line_width = entries[candidate]
        n_lines = (length + line_bases - 1) // line_bases
        with open(fasta_path, "rb") as fasta_handle:
            fasta_handle.seek(offset)
            raw = fasta_handle.read(n_lines * line_width)
        sequence = raw.replace(b"\n", b"").replace(b"\r", b"")[:length].decode("ascii").upper()
        if len(sequence) != length:
            raise SystemExit(f"Incomplete FASTA contig {candidate}: {len(sequence)} != {length}")
        return candidate, sequence
    raise SystemExit(f"Reference FASTA lacks chromosome {chrom}; first contigs: {list(entries)[:5]}")


def load_canonical_summary(path: str | Path, chrom: str) -> tuple[list[str], int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    observed = str(payload.get("chrom", "")).removeprefix("chr")
    if observed != chrom:
        raise SystemExit(f"Canonical summary chromosome {observed!r} != {chrom!r}")
    samples = [str(value) for value in payload.get("selected_samples", [])]
    if not samples or len(samples) != len(set(samples)):
        raise SystemExit("Canonical summary lacks unique selected_samples")
    expected_sites = int(payload.get("total_variants_in_input", -1))
    if expected_sites < 1:
        raise SystemExit("Canonical summary lacks total_variants_in_input")
    return samples, expected_sites


def load_split_mask(
    path: str | Path,
    selected_samples: list[str],
    sample_id_col: str,
    split_col: str,
    train_label: str,
) -> np.ndarray:
    split = pd.read_csv(path, sep="\t", dtype={sample_id_col: str, split_col: str})
    missing_columns = {sample_id_col, split_col} - set(split.columns)
    if missing_columns:
        raise SystemExit(f"split_manifest lacks columns: {sorted(missing_columns)}")
    if split[sample_id_col].duplicated().any():
        raise SystemExit("split_manifest contains duplicate sample IDs")
    role = split.set_index(sample_id_col)[split_col]
    missing = [sample for sample in selected_samples if sample not in role.index]
    if missing:
        raise SystemExit(f"split_manifest lacks {len(missing)} M14 samples; first: {missing[:5]}")
    mask = np.array([role.at[sample] == train_label for sample in selected_samples], dtype=bool)
    if not mask.any():
        raise SystemExit(f"No selected sample has split={train_label!r}")
    return mask


def list_vcf_samples(vcf_path: str | Path) -> list[str]:
    proc = subprocess.run(
        ["bcftools", "query", "-l", str(vcf_path)], capture_output=True, text=True, check=True
    )
    samples = proc.stdout.splitlines()
    if not samples or len(samples) != len(set(samples)):
        raise SystemExit("VCF header has no samples or contains duplicate sample IDs")
    return samples


def orientation_state(value: SiteOrientation) -> str:
    if value.allele_number == 0:
        return "all_missing"
    if value.is_tie:
        return "tie"
    return "alt_major" if value.alt_is_major else "alt_minor"


def counted_allele(mode: str, filter_orientation: SiteOrientation, subset_orientation: SiteOrientation) -> int | None:
    if mode == "historical_alt":
        return 1
    orientation = filter_orientation if "filter_cohort" in mode else subset_orientation
    if orientation.allele_number == 0 or orientation.is_tie:
        return None
    if mode.startswith("exclude_alt_major") and orientation.alt_is_major:
        return None
    return 0 if orientation.alt_is_major else 1


def bcftools_version() -> str:
    proc = subprocess.run(["bcftools", "--version"], capture_output=True, text=True, check=True)
    return proc.stdout.splitlines()[0]


def main() -> None:
    started = time.monotonic()
    args = parse_args()
    chrom = str(args.chrom).removeprefix("chr")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected_samples, expected_sites = load_canonical_summary(args.canonical_summary, chrom)
    if len(selected_samples) != args.expected_m14_samples:
        raise SystemExit(
            f"M14 sample count {len(selected_samples)} != expected {args.expected_m14_samples}"
        )
    train_mask = load_split_mask(
        args.split_manifest,
        selected_samples,
        args.sample_id_col,
        args.split_col,
        args.train_label,
    )
    filter_samples = list_vcf_samples(args.vcf)
    if len(filter_samples) != args.expected_filter_samples:
        raise SystemExit(
            f"VCF sample count {len(filter_samples)} != expected {args.expected_filter_samples}"
        )
    filter_index = {sample: idx for idx, sample in enumerate(filter_samples)}
    missing_selected = [sample for sample in selected_samples if sample not in filter_index]
    if missing_selected:
        raise SystemExit(f"VCF lacks {len(missing_selected)} M14 samples; first: {missing_selected[:5]}")
    subset_indices = np.array([filter_index[sample] for sample in selected_samples], dtype=np.int64)

    reference_contig, reference_sequence = load_reference_contig(args.reference_fasta, chrom)
    query = subprocess.Popen(
        ["bcftools", "query", "-f", r"%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n", str(args.vcf)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if query.stdout is None or query.stderr is None:
        raise SystemExit("Could not open bcftools query")

    n_selected = len(selected_samples)
    dosage = {mode: np.zeros(n_selected, dtype=np.int64) for mode in MODES}
    carriers = {mode: np.zeros(n_selected, dtype=np.int64) for mode in MODES}
    callable_sites = {mode: np.zeros(n_selected, dtype=np.int64) for mode in MODES}
    counts = {"filter_cohort": Counter(), "m14_subset": Counter()}
    totals = {mode: defaultdict(int) for mode in MODES}
    orientation_disagreements = Counter()
    previous_variant_id: str | None = None

    exceptions_path = outdir / f"chr{chrom}.orientation_exceptions.tsv.gz"
    exception_columns = [
        "variant_id", "chrom", "pos", "ref", "alt", "filter_state", "filter_alt_count",
        "filter_allele_number", "m14_state", "m14_alt_count", "m14_allele_number",
        "historical_alt_carriers_m14", "minor_filter_carriers_m14",
        "minor_subset_carriers_m14", "historical_alt_dosage_train",
        "minor_filter_dosage_train", "minor_subset_dosage_train",
    ]
    with gzip.open(exceptions_path, "wt", encoding="utf-8", newline="") as exceptions:
        exceptions.write("\t".join(exception_columns) + "\n")
        for raw_line in query.stdout:
            parts = raw_line.rstrip(b"\n").split(b"\t", 4)
            if len(parts) != 5:
                raise SystemExit(f"Malformed bcftools row: {raw_line[:200]!r}")
            chrom_b, pos_b, ref_b, alt_b, genotype_bytes = parts
            row_chrom = chrom_b.decode("ascii")
            pos = int(pos_b)
            ref = ref_b.decode("ascii")
            alt = alt_b.decode("ascii")
            if row_chrom.removeprefix("chr") != chrom:
                raise SystemExit(f"Unexpected chromosome {row_chrom}:{pos}; expected chr{chrom}")
            if "," in alt or len(ref) != 1 or len(alt) != 1:
                raise SystemExit(f"Expected a biallelic SNV, found {ref}>{alt} at {row_chrom}:{pos}")
            variant_id = f"{row_chrom}:{pos}:{ref}:{alt}"
            if variant_id == previous_variant_id:
                raise SystemExit(f"Adjacent duplicate variant record: {variant_id}")
            previous_variant_id = variant_id
            if pos > len(reference_sequence) or reference_sequence[pos - 1] != ref.upper():
                raise SystemExit(f"REF_QC_FAIL at {variant_id}")

            fixed_width = b"\t" + genotype_bytes
            expected_width = 4 * len(filter_samples)
            if len(fixed_width) != expected_width:
                raise SystemExit(
                    f"Non-diploid or malformed GT width at {variant_id}: "
                    f"{len(fixed_width)} != {expected_width}"
                )
            gt = np.frombuffer(fixed_width, dtype=np.uint8).reshape(len(filter_samples), 4)
            allele_bytes = gt[:, (1, 3)]
            valid = (allele_bytes == ord("0")) | (allele_bytes == ord("1"))
            if np.any(~valid & (allele_bytes != ord("."))):
                raise SystemExit(f"Unexpected allele index in GT at {variant_id}")
            filter_orientation = SiteOrientation(
                alt_count=int(np.count_nonzero(allele_bytes == ord("1"))),
                allele_number=int(valid.sum()),
            )
            subset_alleles = allele_bytes[subset_indices]
            subset_valid = valid[subset_indices]
            subset_orientation = SiteOrientation(
                alt_count=int(np.count_nonzero(subset_alleles == ord("1"))),
                allele_number=int(subset_valid.sum()),
            )
            filter_state = orientation_state(filter_orientation)
            subset_state = orientation_state(subset_orientation)
            counts["filter_cohort"][filter_state] += 1
            counts["m14_subset"][subset_state] += 1
            counts["filter_cohort"]["total_sites"] += 1
            counts["m14_subset"]["total_sites"] += 1
            counts["filter_cohort"]["partially_missing_genotypes"] += int(
                np.count_nonzero(valid.sum(axis=1) == 1)
            )
            counts["m14_subset"]["partially_missing_genotypes"] += int(
                np.count_nonzero(subset_valid.sum(axis=1) == 1)
            )
            orientation_disagreements[f"{filter_state}__{subset_state}"] += 1

            complete = subset_valid.all(axis=1)
            per_mode_values: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
            for mode in MODES:
                allele = counted_allele(mode, filter_orientation, subset_orientation)
                if allele is None:
                    per_mode_values[mode] = None
                    continue
                allele_byte = ord(str(allele))
                mode_dosage = np.count_nonzero(subset_alleles == allele_byte, axis=1).astype(np.int8)
                mode_dosage = np.where(complete, mode_dosage, 0)
                mode_carrier = mode_dosage > 0
                mode_carrier_any_called = np.any(subset_alleles == allele_byte, axis=1)
                dosage[mode] += mode_dosage
                carriers[mode] += mode_carrier
                callable_sites[mode] += complete
                totals[mode]["dosage_m14"] += int(mode_dosage.sum())
                totals[mode]["carrier_incidence_m14"] += int(mode_carrier.sum())
                totals[mode]["carrier_incidence_any_called_m14"] += int(
                    mode_carrier_any_called.sum()
                )
                totals[mode]["dosage_train"] += int(mode_dosage[train_mask].sum())
                totals[mode]["carrier_incidence_train"] += int(mode_carrier[train_mask].sum())
                totals[mode]["carrier_incidence_any_called_train"] += int(
                    np.count_nonzero(mode_carrier_any_called & train_mask)
                )
                totals[mode]["nnz_train"] += int(mode_carrier[train_mask].sum())
                totals[mode]["retained_sites"] += 1
                totals[mode]["sites_with_missing_rate_train_gt_0_1"] += int(
                    np.count_nonzero(~complete & train_mask) / int(train_mask.sum()) > 0.1
                )
                totals[mode]["sites_with_mac_train_ge_2"] += int(mode_dosage[train_mask].sum() >= 2)
                totals[mode]["sites_with_two_train_carriers"] += int(
                    np.count_nonzero(mode_carrier & train_mask) >= 2
                )
                per_mode_values[mode] = mode_dosage, mode_carrier

            historical_dosage, historical_carrier = per_mode_values["historical_alt"]
            for universe_name, orientation in (
                ("filter_cohort", filter_orientation),
                ("m14_subset", subset_orientation),
            ):
                state = orientation_state(orientation)
                if state in ("alt_minor", "alt_major"):
                    totals["historical_alt"][f"dosage_m14_on_{universe_name}_orientable"] += int(
                        historical_dosage.sum()
                    )
                    totals["historical_alt"][f"carrier_incidence_m14_on_{universe_name}_orientable"] += int(
                        historical_carrier.sum()
                    )
                if state == "alt_major":
                    totals["historical_alt"][f"dosage_m14_at_{universe_name}_alt_major"] += int(
                        historical_dosage.sum()
                    )
                    totals["historical_alt"][f"carrier_incidence_m14_at_{universe_name}_alt_major"] += int(
                        historical_carrier.sum()
                    )

            if filter_state != "alt_minor" or subset_state != "alt_minor" or filter_state != subset_state:
                hist_dosage, hist_carrier = historical_dosage, historical_carrier
                filter_values = per_mode_values["minor_filter_cohort"]
                subset_values = per_mode_values["minor_m14_subset"]
                filter_dosage = filter_values[0] if filter_values else np.zeros(n_selected, dtype=np.int8)
                filter_carrier = filter_values[1] if filter_values else np.zeros(n_selected, dtype=bool)
                subset_dosage = subset_values[0] if subset_values else np.zeros(n_selected, dtype=np.int8)
                subset_carrier = subset_values[1] if subset_values else np.zeros(n_selected, dtype=bool)
                row = (
                    variant_id, chrom, pos, ref, alt, filter_state, filter_orientation.alt_count,
                    filter_orientation.allele_number, subset_state, subset_orientation.alt_count,
                    subset_orientation.allele_number, int(hist_carrier.sum()), int(filter_carrier.sum()),
                    int(subset_carrier.sum()), int(hist_dosage[train_mask].sum()),
                    int(filter_dosage[train_mask].sum()), int(subset_dosage[train_mask].sum()),
                )
                exceptions.write("\t".join(map(str, row)) + "\n")

    query.stdout.close()
    stderr = query.stderr.read().decode("utf-8", errors="replace")
    query.stderr.close()
    return_code = query.wait()
    if return_code != 0:
        raise SystemExit(f"bcftools query failed ({return_code}): {stderr.strip()}")
    observed_sites = counts["filter_cohort"]["total_sites"]
    if observed_sites != expected_sites:
        raise SystemExit(f"Site count {observed_sites} != canonical M14 input {expected_sites}")

    burden = pd.DataFrame({"sample_id": selected_samples, "is_train": train_mask.astype(int)})
    for mode in MODES:
        burden[f"{mode}_dosage_sum"] = dosage[mode]
        burden[f"{mode}_carrier_site_count"] = carriers[mode]
        burden[f"{mode}_callable_sites"] = callable_sites[mode]
    burden.to_csv(
        outdir / f"chr{chrom}.sample_burden_by_mode.tsv.gz",
        sep="\t", index=False, compression="gzip",
    )

    def universe_summary(name: str) -> dict:
        universe_counts = counts[name]
        orientable = universe_counts["alt_minor"] + universe_counts["alt_major"]
        total = universe_counts["total_sites"]
        return {
            "counts": dict(sorted(universe_counts.items())),
            "alt_major_fraction_of_orientable": universe_counts["alt_major"] / orientable,
            "alt_major_fraction_of_total": universe_counts["alt_major"] / total,
            "orientable_sites": orientable,
        }

    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    summary = {
        "status": "PASS",
        "scope": "orientation_and_marginal_burden_only_no_topology_no_training",
        "chrom": chrom,
        "reference_contig": reference_contig,
        "reference_qc": {"matches": observed_sites, "mismatches": 0},
        "n_filter_cohort_samples": len(filter_samples),
        "n_m14_samples": n_selected,
        "n_train_samples": int(train_mask.sum()),
        "sample_order_sha256": {
            "filter_cohort": ordered_ids_sha256(filter_samples),
            "m14_subset": ordered_ids_sha256(selected_samples),
            "train_subset": ordered_ids_sha256(
                [sample for sample, is_train in zip(selected_samples, train_mask) if is_train]
            ),
        },
        "expected_sites_from_m14_summary": expected_sites,
        "orientation_universes": {
            "filter_cohort": universe_summary("filter_cohort"),
            "m14_subset": universe_summary("m14_subset"),
        },
        "orientation_cross_tab": dict(sorted(orientation_disagreements.items())),
        "burden_totals": {mode: dict(sorted(values.items())) for mode, values in totals.items()},
        "field_definitions": {
            "filter_cohort": "Todas las muestras del VCF raro; corresponde al universo usado para filtrar rareza aguas arriba.",
            "m14_subset": "Las 2619 muestras canónicas usadas por M14.",
            "historical_alt": "Cuenta copias ALT como hicieron M14, M20 y la matriz cruda de M23.",
            "minor_filter_cohort": "Cuenta el alelo menor definido en todas las muestras del VCF; empates y sitios sin alelos llamados se excluyen.",
            "minor_m14_subset": "Cuenta el alelo menor definido dentro de las 2619 muestras de M14; es una sensibilidad interna.",
            "callable_sites": "Sitios retenidos con GT diploide completo en el individuo; no es callability por base.",
            "carrier_incidence_m14": "Eventos individuo×sitio con GT diploide completo y al menos una copia del alelo contado.",
            "carrier_incidence_any_called_m14": "Sensibilidad para M14: eventos portadores aceptando GT parcialmente faltante si la copia llamada contiene el alelo contado.",
            "nnz_train": "Entradas no cero en TRAIN bajo la orientación indicada, antes del prefiltro de missingness de M23.",
        },
        "resource_usage": {
            "analysis_wall_seconds": time.monotonic() - started,
            "self_max_rss_kib": int(self_usage.ru_maxrss),
            "largest_reaped_child_max_rss_kib": int(child_usage.ru_maxrss),
            "self_cpu_seconds": self_usage.ru_utime + self_usage.ru_stime,
            "reaped_children_cpu_seconds": child_usage.ru_utime + child_usage.ru_stime,
            "measurement_limit": "Linux ru_maxrss for children is the largest reaped child, not aggregate process-tree RSS.",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "bcftools": bcftools_version(),
        },
        "interpretation_limits": [
            "This inventory measures orientation and additive burdens, not M14 pairs, segments, graph topology or M16.5 communities.",
            "TRAIN is a descriptive stratum and does not choose orientation, thresholds or models.",
            "M14-derived labels remain an internal construction, not independent biological truth.",
        ],
        "analysis_date": datetime.now(timezone.utc).isoformat(),
    }
    write_json(outdir / f"chr{chrom}.orientation_inventory.json", summary)


if __name__ == "__main__":
    main()
