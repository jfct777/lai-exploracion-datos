#!/usr/bin/env python3
"""Exact masked rare-sharing and import of established PLINK common GRMs.

This is descriptive pair evidence, not an IBD estimator or a population test.
Rare sites remain eligible even with zero or one analytical carrier. Missing
genotypes are excluded jointly, never interpreted as absence of the rare allele.
PLINK computes the common GRM; this module only imports and combines its output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse


SCHEMA = "r02_pair_evidence_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_hash(samples):
    return hashlib.sha256(("\n".join(samples) + "\n").encode()).hexdigest()


def read_samples(path, expected=None):
    values = Path(path).read_text().splitlines()
    require(values and len(set(values)) == len(values), "Sample list empty or duplicated")
    require(all(v and not any(c.isspace() for c in v) for v in values), "Invalid sample ID")
    require(expected is None or len(values) == expected, "Unexpected analytical sample count")
    return values


def chromosome(value):
    value = str(value).removeprefix("chr")
    require(value.isdigit() and 1 <= int(value) <= 22, "Expected an autosome from 1 to 22")
    return str(int(value))


class RareCounts:
    """Sparse products in site chunks; dense output is only persons x persons.

    C indicates observed carriage; D indicates missing GT. With V sites,
    I=CC', Q=V-miss_i-miss_j+DD', a=carrier_i-CD', U=a+a'-I.
    Products use int64: uint8 multiplication would overflow at 256 sites.
    """
    def __init__(self, n_samples):
        require(n_samples > 0, "No analytical samples")
        self.n = n_samples
        self.n_sites = 0
        self.intersection = np.zeros((n_samples, n_samples), dtype=np.int64)
        self.carrier_missing = np.zeros_like(self.intersection)
        self.missing_joint = np.zeros_like(self.intersection)
        self.carriers = np.zeros(n_samples, dtype=np.int64)
        self.missing = np.zeros(n_samples, dtype=np.int64)

    def add_sparse(self, carrier_rows, carrier_cols, missing_rows, missing_cols, n_sites):
        require(n_sites > 0, "Empty site chunk")
        shape = (self.n, n_sites)
        c = sparse.csr_matrix((np.ones(len(carrier_rows), dtype=np.int64),
                               (carrier_rows, carrier_cols)), shape=shape)
        d = sparse.csr_matrix((np.ones(len(missing_rows), dtype=np.int64),
                               (missing_rows, missing_cols)), shape=shape)
        require(not c.nnz or c.data.max() == 1, "Duplicate carrier cell")
        require(not d.nnz or d.data.max() == 1, "Duplicate missing cell")
        require(c.multiply(d).nnz == 0, "A missing GT cannot be a carrier")
        self.intersection += (c @ c.T).toarray()
        if d.nnz:
            self.carrier_missing += (c @ d.T).toarray()
            self.missing_joint += (d @ d.T).toarray()
        self.carriers += np.asarray(c.sum(axis=1)).ravel()
        self.missing += np.asarray(d.sum(axis=1)).ravel()
        self.n_sites += n_sites

    def finish(self):
        a = self.carriers[:, None] - self.carrier_missing
        union = a + a.T - self.intersection
        joint = self.n_sites - self.missing[:, None] - self.missing[None, :] + self.missing_joint
        validate_rare_counts(self.intersection, union, joint)
        return dict(I=self.intersection, U=union, Q=joint,
                    individual_carrier=self.carriers,
                    individual_callable=self.n_sites - self.missing,
                    n_sites=np.asarray(self.n_sites, dtype=np.int64))


def validate_rare_counts(intersection, union, joint):
    require(intersection.ndim == 2 and intersection.shape[0] == intersection.shape[1], "Nonsquare rare matrix")
    require(intersection.shape == union.shape == joint.shape, "Rare matrix shape disagreement")
    for matrix in (intersection, union, joint):
        require(matrix.dtype.kind in "iu" and np.all(matrix >= 0), "Counts must be nonnegative integers")
        require(np.array_equal(matrix, matrix.T), "Asymmetric pair counts")
    require(np.all(intersection <= union) and np.all(union <= joint), "Expected 0 <= I <= U <= Q")


def masked_jaccard(intersection, union):
    result = np.full(union.shape, np.nan, dtype=np.float64)
    np.divide(intersection, union, out=result, where=union > 0)
    return result


def _header_scalar(header, key):
    values = [line.split("=", 1)[1] for line in str(header).splitlines()
              if line.startswith("##" + key + "=")]
    require(len(values) == 1, f"Expected one {key} header")
    return values[0]


def rare_evidence(vcf, samples, chrom, expected_source_samples, chunk_sites=2048):
    import pysam
    require(chunk_sites > 0, "chunk-sites must be positive")
    chrom = chromosome(chrom)
    counter, started = Counter(), time.monotonic()
    accumulator = RareCounts(len(samples))
    with pysam.VariantFile(str(vcf)) as source:
        source_samples = list(source.header.samples)
        require(len(source_samples) == expected_source_samples, "Unexpected source sample count")
        require(_header_scalar(source.header, "dnabr_original_alleles") == "v1", "Uncertified original alleles")
        require(_header_scalar(source.header, "dnabr_rare_contract") == "minor_v1", "Expected M02.1 minor_v1")
        require(int(_header_scalar(source.header, "dnabr_rare_cohort_n_samples")) == len(source_samples), "Source N header mismatch")
        source_hash = sample_hash(source_samples)
        require(_header_scalar(source.header, "dnabr_rare_cohort_sha256") == source_hash, "Source sample identity mismatch")
        require(set(samples) <= set(source_samples), "Analytical sample absent from VCF")
        for key in ("RARE_ALLELE", "ORIG_NALLELES", "RARE_AC", "RARE_AN"):
            require(key in source.header.info, f"Missing INFO/{key}")
        require("RD" in source.header.formats and "GT" in source.header.formats, "Missing GT/RD")
        # pysam retains input order; explicitly map to the requested analysis order.
        source.subset_samples(samples)
        current_order = list(source.header.samples)
        analytical_index = {name: i for i, name in enumerate(samples)}
        sample_indices = [analytical_index[name] for name in current_order]
        cr, cc, mr, mc, block_n, previous = [], [], [], [], 0, 0
        first = last = None
        for record in source:
            require(chromosome(record.contig) == chrom, "VCF contains unexpected chromosome")
            require(record.pos > previous, "Duplicate or decreasing genomic position")
            previous = record.pos
            require(record.info.get("ORIG_NALLELES") == 2, "Originally multiallelic site reached evidence reader")
            require(len(record.alleles) == 2 and all(len(a) == 1 and a in "ACGT" for a in record.alleles), "Expected biallelic SNV")
            allele = record.info["RARE_ALLELE"]
            require(allele in (0, 1), "Invalid selected source allele")
            ac, an = record.info["RARE_AC"], record.info["RARE_AN"]
            require(ac >= 2 and an > 0 and 100 * ac <= an, "Source rare criterion MAC>=2, MAF<=1% violated")
            n_carriers = 0
            for i, call in zip(sample_indices, record.samples.values()):
                gt, rd = call.get("GT", ()), call.get("RD")
                require(len(gt) == 2 and all(a in (None, 0, 1) for a in gt), "Non-diploid/non-biallelic GT")
                if None in gt:
                    require(rd is None, "Incomplete GT must have missing RD")
                    mr.append(i); mc.append(block_n)
                    counter["incomplete_genotypes"] += 1
                    continue
                expected_rd = sum(a == allele for a in gt)
                require(rd == expected_rd, "RD disagrees with GT and fixed source allele")
                if rd > 0:
                    cr.append(i); cc.append(block_n)
                    n_carriers += 1
            counter[f"sites_{'zero' if n_carriers == 0 else 'one' if n_carriers == 1 else 'multiple'}_analytical_carriers"] += 1
            counter["source_minor_ref" if allele == 0 else "source_minor_alt"] += 1
            counter["sites"] += 1
            first = record.pos if first is None else first
            last = record.pos
            block_n += 1
            if block_n == chunk_sites:
                accumulator.add_sparse(cr, cc, mr, mc, block_n)
                cr, cc, mr, mc, block_n = [], [], [], [], 0
                print(f"rare evidence chr{chrom}: sites={counter['sites']} elapsed_s={time.monotonic()-started:.1f}", file=sys.stderr, flush=True)
        if block_n:
            accumulator.add_sparse(cr, cc, mr, mc, block_n)
    require(counter["sites"] > 0, "No rare sites; do not fabricate an empty chromosome result")
    return accumulator.finish(), dict(counts=dict(counter), source_cohort_n=expected_source_samples,
                                     source_cohort_sha256=source_hash, first_position_bp=first,
                                     source_cohort_members_sha256=sample_hash(sorted(source_samples)),
                                     last_position_bp=last, pysam_version=pysam.__version__,
                                     mask="complete diploid GT; upstream site filters; not new DP/GQ filtering",
                                     allele="fixed M02.1 RARE_ALLELE; no reorientation in analytical subset",
                                     method="exact pairwise jointly called Jaccard over ALL source rare catalogue sites",
                                     elapsed_seconds=time.monotonic()-started)


def read_grm_ids(path):
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    require(rows, "Empty GRM IDs")
    header = rows.pop(0) if rows[0][0].startswith("#") else None
    if header is not None:
        require("IID" in header or "#IID" in header, "GRM ID header lacks IID")
        index = header.index("IID") if "IID" in header else header.index("#IID")
    else:
        require(all(len(row) in (1, 2) for row in rows), "GRM ID format needs IID or FID/IID")
        index = len(rows[0]) - 1
    require(rows and all(len(row) > index for row in rows), "Malformed GRM IDs")
    ids = [row[index] for row in rows]
    require(len(set(ids)) == len(ids), "IID is ambiguous across GRM rows")
    return ids


def _triangle(path, n):
    values = np.fromfile(path, dtype="<f4")
    require(values.size == n * (n + 1) // 2 and Path(path).stat().st_size == values.size * 4,
            "GRM binary does not contain the full lower triangle with diagonal")
    matrix = np.zeros((n, n), dtype=np.float64)
    indices = np.tril_indices(n)
    matrix[indices] = values
    matrix[(indices[1], indices[0])] = values
    return matrix


def import_common(prefix, samples):
    """Import PLINK --make-grm-bin (no meanimpute) and retain denominators.

    Official formats: https://www.cog-genomics.org/plink/2.0/formats
    This matrix is not PC-Relate kinship and does not certify unrelatedness.
    """
    prefix = str(prefix)
    ids = read_grm_ids(prefix + ".grm.id")
    require(len(ids) == len(samples) and set(ids) == set(samples), "GRM and rare analytical cohorts differ")
    k, counts = (_triangle(prefix + suffix, len(ids)) for suffix in (".grm.bin", ".grm.N.bin"))
    require(np.all(np.isfinite(counts)) and np.all(counts >= 0), "Invalid GRM observation counts")
    require(np.all(counts == np.rint(counts)), "Noninteger GRM counts; expected complete-call denominator")
    require(np.all(np.isfinite(k[counts > 0])), "Nonfinite GRM where observations exist")
    order = {name: i for i, name in enumerate(ids)}
    take = np.asarray([order[name] for name in samples])
    k, counts = k[np.ix_(take, take)], counts[np.ix_(take, take)].astype(np.int64)
    numerator = np.zeros(k.shape, dtype=np.float64)
    np.multiply(k, counts, out=numerator, where=counts > 0)
    k[counts == 0] = np.nan
    require(np.any(counts > 0), "No common observations")
    return dict(K=k, common_N=counts, common_numerator=numerator), dict(
        original_grm_sample_sha256=sample_hash(ids), reordered_to_analytical_samples=ids != samples,
        method="PLINK variance-standardized GRM; no mean imputation; aggregation weighted by pair-specific N",
        definition="K is similarity of common genotypes, not PC-Relate kinship",
        inputs_sha256={suffix: sha256(prefix + suffix) for suffix in (".grm.bin", ".grm.N.bin", ".grm.id")})


def write_bundle(prefix, arrays, samples, chromosomes, kind, metadata):
    prefix = Path(prefix)
    npz, manifest = Path(str(prefix) + ".npz"), Path(str(prefix) + ".manifest.json")
    require(not npz.exists() and not manifest.exists(), "Output exists; no overwrite allowed")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema=SCHEMA, kind=kind, chromosomes=chromosomes,
                   samples_sha256=sample_hash(samples), n_samples=len(samples),
                   contains_individual_identifiers=True, public_distribution_allowed=False,
                   evidence_level="DESCRIPTIVE", **metadata)
    with npz.open("xb") as handle:
        np.savez_compressed(handle, samples=np.asarray(samples, dtype=str),
                            sample_hash=np.asarray(payload["samples_sha256"]),
                            chromosomes=np.asarray(chromosomes, dtype=str),
                            kind=np.asarray(kind), schema=np.asarray(SCHEMA), **arrays)
    payload["output_npz_sha256"] = sha256(npz)
    with manifest.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return payload


def load_bundle(path, samples, kind):
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    metadata = json.loads(manifest_path.read_text())
    require(metadata["output_npz_sha256"] == sha256(path), "NPZ hash differs from manifest")
    require(metadata["kind"] == kind and metadata["schema"] == SCHEMA, "Wrong evidence type/schema")
    with np.load(path, allow_pickle=False) as archive:
        require(archive["samples"].tolist() == samples and str(archive["sample_hash"]) == sample_hash(samples), "Evidence sample order differs")
        require(str(archive["kind"]) == kind and str(archive["schema"]) == SCHEMA, "NPZ/manifest type disagreement")
        require(archive["chromosomes"].tolist() == metadata["chromosomes"], "Chromosome identity mismatch")
        arrays = {key: archive[key] for key in archive.files
                  if key not in {"samples", "sample_hash", "chromosomes", "kind", "schema"}}
    return arrays, metadata


def aggregate_evidence(rare_paths, common_paths, samples, chromosomes):
    expected = sorted([chromosome(c) for c in chromosomes], key=int)
    require(len(set(expected)) == len(expected), "Duplicate expected chromosome")
    outputs, source_hashes = {}, {}
    for kind, paths in (("rare", rare_paths), ("common", common_paths)):
        require(paths, f"Missing {kind} evidence")
        seen, signatures = [], []
        keys = ("I", "U", "Q", "individual_carrier", "individual_callable", "n_sites") if kind == "rare" else ("common_numerator", "common_N")
        for path in paths:
            arrays, metadata = load_bundle(path, samples, kind)
            require(len(metadata["chromosomes"]) == 1, "Aggregate expects one chromosome per input")
            chrom = chromosome(metadata["chromosomes"][0])
            require(chrom not in seen, "Duplicated chromosome evidence")
            seen.append(chrom)
            if kind == "rare":
                validate_rare_counts(arrays["I"], arrays["U"], arrays["Q"])
                require({"source_cohort_n", "source_cohort_members_sha256", "mask", "allele"} <= set(metadata),
                        "Incomplete rare source-cohort/mask provenance")
                signatures.append(tuple(metadata[key] for key in
                                        ("source_cohort_n", "source_cohort_members_sha256", "mask", "allele")))
            else:
                require(np.all(arrays["common_N"] >= 0) and np.all(np.isfinite(arrays["common_numerator"])), "Invalid common sufficient statistics")
                require("producer" in metadata, "Missing common producer/LD definition")
                producer = metadata["producer"]
                require(chromosome(producer["chromosome"]) == chrom, "Common producer chromosome mismatch")
                marker = producer["marker_selection"]
                signature_keys = ("original_biallelic", "PASS", "min_maf", "max_missing", "window_kb", "step_variants", "r2", "indep_order")
                require(set(signature_keys) <= set(marker), "Incomplete common marker selection")
                signatures.append((producer["tool"], producer["version"], producer["frequency_scope"],
                                   producer["missingness"], *(marker[key] for key in signature_keys)))
            for key in keys:
                if key not in outputs:
                    outputs[key] = arrays[key].copy()
                else:
                    require(outputs[key].shape == arrays[key].shape, "Evidence dimensions differ")
                    outputs[key] += arrays[key]
            source_hashes[str(path)] = metadata["output_npz_sha256"]
        require(sorted(seen, key=int) == expected, f"Incomplete {kind} autosomes: {seen}")
        require(all(signature == signatures[0] for signature in signatures),
                f"Inconsistent {kind} cohort, masks or marker/LD definitions across chromosomes")
    validate_rare_counts(outputs["I"], outputs["U"], outputs["Q"])
    outputs["J"] = masked_jaccard(outputs["I"], outputs["U"])
    outputs["K"] = np.full(outputs["common_N"].shape, np.nan)
    np.divide(outputs["common_numerator"], outputs["common_N"], out=outputs["K"], where=outputs["common_N"] > 0)
    outputs["pair_eligible_R_C"] = (outputs["U"] > 0) & (outputs["Q"] > 0) & (outputs["common_N"] > 0)
    # This is an eligibility indicator, not an imputation of unmeasured pairs.
    return outputs, dict(inputs_npz_sha256=source_hashes,
                         aggregation="J=sum(I)/sum(U); K=sum(N*K)/sum(N), each chromosome once",
                         missing="J is NaN when U=0; K is NaN when common_N=0; never fill undefined similarity with zero",
                         limitations="Global identity similarity and common-GRM comparison; no segment annotation, RC filtering, IBD or confirmed populations")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("rare", "common", "aggregate"):
        sub = modes.add_parser(mode)
        sub.add_argument("--samples", required=True)
        sub.add_argument("--expected-samples", type=int, required=True)
        sub.add_argument("--output-prefix", required=True)
        if mode != "aggregate":
            sub.add_argument("--chrom", required=True)
        if mode == "rare":
            sub.add_argument("--vcf", required=True)
            sub.add_argument("--expected-source-samples", type=int, required=True)
            sub.add_argument("--chunk-sites", type=int, default=2048)
        elif mode == "common":
            sub.add_argument("--grm-prefix", required=True)
            sub.add_argument("--method-manifest", required=True,
                             help="JSON with tool, version, command, marker_selection, frequency_scope and missingness")
        else:
            sub.add_argument("--rare", nargs="+", required=True)
            sub.add_argument("--common", nargs="+", required=True)
            sub.add_argument("--chromosomes", required=True, help="Comma-separated exact chromosome set")
    args = parser.parse_args(argv)
    samples = read_samples(args.samples, args.expected_samples)
    if args.mode == "rare":
        arrays, metadata = rare_evidence(args.vcf, samples, args.chrom, args.expected_source_samples, args.chunk_sites)
        metadata.update(input_vcf=str(args.vcf), input_vcf_sha256=sha256(args.vcf), chunk_sites=args.chunk_sites)
        chroms = [chromosome(args.chrom)]
    elif args.mode == "common":
        method = json.loads(Path(args.method_manifest).read_text())
        require({"tool", "version", "command", "marker_selection", "frequency_scope", "missingness", "chromosome", "analytical_samples_sha256"} <= set(method), "Incomplete common method manifest")
        require(chromosome(method["chromosome"]) == chromosome(args.chrom), "Common method chromosome mismatch")
        require(method["analytical_samples_sha256"] == sample_hash(samples), "Common method sample identity/order mismatch")
        require("meanimpute" not in str(method["command"]), "Mean-imputed GRM not permitted in this complete-call contract")
        arrays, metadata = import_common(args.grm_prefix, samples)
        metadata.update(producer=method, method_manifest_sha256=sha256(args.method_manifest))
        chroms = [chromosome(args.chrom)]
    else:
        chroms = [chromosome(c) for c in args.chromosomes.split(",")]
        arrays, metadata = aggregate_evidence(args.rare, args.common, samples, chroms)
    metadata["code_sha256"] = sha256(__file__)
    result = write_bundle(args.output_prefix, arrays, samples, chroms, args.mode, metadata)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
