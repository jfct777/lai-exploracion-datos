#!/usr/bin/env python3
"""Authenticate and stream a correctly oriented ordered-rare M31 smoke input.

This is a technical preflight, not an analysis.  Site eligibility is recomputed
from FREQ haplotypes only.  TARGET is read only after that universe is fixed.
The output deliberately remains model-agnostic: it freezes neither an
architecture nor an evaluation metric.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


MISSING = {"", ".", "NA", "NaN", "nan"}


@dataclass(frozen=True)
class SelectedSite:
    locus_index: int
    chrom: str
    position: int
    minor_code: int
    mac: int
    an: int
    maf: float
    freq_carrier_individuals: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


@contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                yield text


def parse_observed_state(value: str) -> int | None:
    if value in MISSING:
        return None
    observed = int(value)
    if observed not in (0, 1):
        raise ValueError(f"target haplotype state must be 0/1/missing, observed {value!r}")
    return observed


def minor_presence(observed_state: int | None, minor_code: int) -> int | None:
    """Return minor-allele presence; allele state 0 is not synonymous with absence."""
    if minor_code not in (0, 1):
        raise ValueError("minor_code must be 0 or 1")
    if observed_state is None:
        return None
    if observed_state not in (0, 1):
        raise ValueError("observed_state must be 0, 1 or None")
    return int(observed_state == minor_code)


def known_answers() -> dict[str, int | None]:
    observed = {
        "state0_minor0": minor_presence(0, 0),
        "state1_minor0": minor_presence(1, 0),
        "state0_minor1": minor_presence(0, 1),
        "state1_minor1": minor_presence(1, 1),
        "missing_minor0": minor_presence(None, 0),
        "missing_minor1": minor_presence(None, 1),
    }
    expected = {
        "state0_minor0": 1,
        "state1_minor0": 0,
        "state0_minor1": 0,
        "state1_minor1": 1,
        "missing_minor0": None,
        "missing_minor1": None,
    }
    if observed != expected:
        raise AssertionError(f"minor-presence known answers failed: {observed}")
    return observed


def validate_git_commit(value: str | None) -> str | None:
    if value is not None and not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("git_commit must be an exact lowercase 40-character hexadecimal commit")
    return value


def authenticate_script(path: Path, expected_sha256: str | None) -> str:
    observed = sha256_file(path)
    if expected_sha256 is not None and expected_sha256 != observed:
        raise ValueError(f"preflight script sha256 mismatch: {observed} != {expected_sha256}")
    return observed


def load_contract(path: Path, root_label: str, root_seed: int) -> tuple[dict, dict]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("stage") != "M31_ORDERED_RARE_PREFLIGHT"
        or contract.get("version") != 1
        or contract.get("status") != "PREFLIGHT_ONLY_NOT_SCIENTIFIC_EVIDENCE"
    ):
        raise ValueError("M31 preflight contract is not the supported frozen smoke contract")
    roots = contract.get("roots", {})
    if root_label not in roots or roots[root_label].get("root_seed") != root_seed:
        raise ValueError("root label/seed is not authenticated by the M31 contract")
    rare = contract["rare_universe"]
    if rare.get("selector") != "FREQ_only":
        raise ValueError("site selection must be FREQ-only")
    prohibited = set(rare.get("prohibited_selectors", []))
    required_prohibited = {"TARGET", "truth", "Gnomix_prediction", "FLARE_prediction"}
    if not required_prohibited.issubset(prohibited):
        raise ValueError("contract does not prohibit target/truth/prediction site selection")
    materialization = contract["materialization"]
    if materialization.get("sample_identity_key") != ["root_seed", "sample_id"]:
        raise ValueError("sample identity key must be (root_seed, sample_id)")
    if materialization.get("row_primary_key") != ["root_seed", "sample_id", "locus_index"]:
        raise ValueError("row primary key must include root, sample and locus")
    return contract, roots[root_label]


def authenticate_inputs(paths: dict[str, Path], spec: dict) -> dict[str, str]:
    expected = spec.get("sha256", {})
    if set(expected) != {"tree", "pools", "catalog", "haplotypes"}:
        raise ValueError("root contract must pin exactly tree/pools/catalog/haplotypes")
    observed: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing {label} input: {path}")
        observed[label] = sha256_file(path)
        if observed[label] != expected[label]:
            raise ValueError(f"{label} sha256 mismatch: {observed[label]} != {expected[label]}")
    return observed


def load_freq_people(path: Path) -> list[tuple[str, tuple[int, int]]]:
    grouped: dict[str, list[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"role", "individual_id", "node_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("pool manifest has an unexpected header")
        seen_nodes: set[int] = set()
        for row in reader:
            node = int(row["node_id"])
            if node in seen_nodes:
                raise ValueError("pool manifest duplicates a node")
            seen_nodes.add(node)
            if row["role"] == "FREQ":
                grouped.setdefault(row["individual_id"], []).append(node)
    people: list[tuple[str, tuple[int, int]]] = []
    for individual, nodes in sorted(grouped.items()):
        if len(nodes) != 2:
            raise ValueError(f"FREQ individual {individual} lacks two haplotypes")
        people.append((individual, tuple(sorted(nodes))))
    if not people:
        raise ValueError("pool manifest has no FREQ individuals")
    return people


def load_catalog(path: Path, chrom: str) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    previous = -1
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chrom", "position", "minor_code", "mac", "an", "maf"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("rare catalog has an unexpected header")
        for row in reader:
            position = int(row["position"])
            if row["chrom"].removeprefix("chr") != chrom.removeprefix("chr"):
                raise ValueError("rare catalog chromosome mismatch")
            if position <= previous or position in rows:
                raise ValueError("rare catalog positions are not unique and strictly ordered")
            previous = position
            rows[position] = row
    if not rows:
        raise ValueError("rare catalog is empty")
    return rows


def derive_freq_sites(
    tree_path: Path,
    pools_path: Path,
    catalog_path: Path,
    contract: dict,
) -> tuple[list[SelectedSite], dict]:
    """Recompute eligibility without consulting TARGET, truth or predictions."""
    import tskit

    chrom = contract["chromosome_domain"]["chrom"]
    genomic_offset = int(contract["chromosome_domain"]["start_bp"])
    rare = contract["rare_universe"]
    catalog = load_catalog(catalog_path, chrom)
    people = load_freq_people(pools_path)
    nodes = [node for _, pair in people for node in pair]
    ts = tskit.load(str(tree_path))
    sample_index = {int(node): index for index, node in enumerate(ts.samples())}
    try:
        genotype_indexes = [sample_index[node] for node in nodes]
    except KeyError as exc:
        raise ValueError(f"FREQ node is not a tree-sequence sample: {exc.args[0]}") from exc

    selected: list[SelectedSite] = []
    catalog_seen: set[int] = set()
    catalog_eligible = 0
    excluded_single_carrier = 0
    for variant in ts.variants():
        position = genomic_offset + int(variant.site.position)
        row = catalog.get(position)
        if row is None:
            continue
        catalog_seen.add(position)
        if len(variant.alleles) != 2:
            raise ValueError(f"catalog position {position} is not biallelic")
        minor_code = int(row["minor_code"])
        if minor_code not in (0, 1):
            raise ValueError(f"invalid minor_code at {position}")
        states = [int(variant.genotypes[index]) for index in genotype_indexes]
        if any(state not in (0, 1) for state in states):
            raise ValueError("authenticated simulation has non-binary or missing FREQ states")
        presence = [int(state == minor_code) for state in states]
        mac = sum(presence)
        an = len(presence)
        maf = mac / an
        carrier_count = sum(
            int(presence[2 * index] + presence[2 * index + 1] > 0)
            for index in range(len(people))
        )
        if (
            mac != int(row["mac"])
            or an != int(row["an"])
            or not math.isclose(maf, float(row["maf"]), abs_tol=1e-12)
        ):
            raise ValueError(f"FREQ catalog mismatch at {position}")
        if mac < rare["minimum_mac"] or not maf < rare["maximum_maf_exclusive"]:
            raise ValueError(f"catalog contains a non-rare FREQ site at {position}")
        catalog_eligible += 1
        if carrier_count < rare["minimum_carrier_individuals"]:
            excluded_single_carrier += 1
            continue
        selected.append(
            SelectedSite(
                len(selected), chrom, position, minor_code, mac, an, maf, carrier_count
            )
        )
    if catalog_seen != set(catalog):
        missing = sorted(set(catalog) - catalog_seen)[:5]
        raise ValueError(f"catalog sites absent from tree sequence: {missing}")
    if not selected:
        raise ValueError("FREQ-only universe is empty after the carrier rule")
    if any(left.position >= right.position for left, right in zip(selected, selected[1:])):
        raise AssertionError("selected loci are not strictly ordered")
    return selected, {
        "catalog_sites": len(catalog),
        "freq_rare_sites": catalog_eligible,
        "selected_freq_two_carrier_sites": len(selected),
        "excluded_freq_single_carrier_sites": excluded_single_carrier,
        "selection_inputs": ["tree.FREQ_haplotypes", "FREQ_rare_catalog"],
        "selection_inputs_excluded": contract["rare_universe"]["prohibited_selectors"],
    }


def target_header(fieldnames: list[str] | None) -> tuple[list[str], list[str]]:
    if fieldnames is None:
        raise ValueError("target haplotypes lack a header")
    fixed = {"chrom", "position", "minor_code"}
    haplotypes = [field for field in fieldnames if field not in fixed]
    samples = sorted({field.rsplit("_h", 1)[0] for field in haplotypes})
    expected = [f"{sample}_h{hap}" for sample in samples for hap in (0, 1)]
    if set(haplotypes) != set(expected) or len(haplotypes) != len(expected):
        raise ValueError("target table must contain exactly two haplotypes per sample")
    if len(samples) != len(set(samples)):
        raise ValueError("target sample IDs are not unique")
    return samples, haplotypes


def materialize_target(
    target_path: Path,
    selected: list[SelectedSite],
    root_seed: int,
    outdir: Path,
) -> dict:
    selected_by_position = {site.position: site for site in selected}
    seen: set[int] = set()
    target_rows = 0
    missing_haplotypes = 0
    minor0_sites = 0
    minor0_cells = 0
    mismatched_legacy_cells = 0
    coincident_legacy_cells = 0
    sites_path = outdir / "m31_ordered_rare.sites.tsv.gz"
    target_out = outdir / "m31_ordered_rare.target.tsv.gz"
    samples_path = outdir / "m31_ordered_rare.samples.tsv"

    with open_text(target_path) as source, deterministic_gzip_text(sites_path) as sites_handle, deterministic_gzip_text(target_out) as target_handle:
        reader = csv.DictReader(source, delimiter="\t")
        samples, _ = target_header(reader.fieldnames)
        site_writer = csv.writer(sites_handle, delimiter="\t", lineterminator="\n")
        target_writer = csv.writer(target_handle, delimiter="\t", lineterminator="\n")
        site_writer.writerow(("root_seed", "locus_index", "chrom", "position", "minor_code", "mac", "an", "maf", "freq_carrier_individuals"))
        target_writer.writerow(("root_seed", "sample_id", "locus_index", "chrom", "position", "minor_code", "h0_minor_presence", "h1_minor_presence", "minor_dosage", "missing_haplotypes"))
        previous_position = -1
        for row in reader:
            position = int(row["position"])
            site = selected_by_position.get(position)
            if site is None:
                continue
            if position <= previous_position:
                raise ValueError("selected target loci are not strictly ordered")
            previous_position = position
            if position in seen:
                raise ValueError("target table duplicates a selected position")
            seen.add(position)
            if int(row["minor_code"]) != site.minor_code:
                raise ValueError(f"target/catalog minor_code mismatch at {position}")
            site_writer.writerow((root_seed, site.locus_index, site.chrom, site.position, site.minor_code, site.mac, site.an, format(site.maf, ".17g"), site.freq_carrier_individuals))
            if site.minor_code == 0:
                minor0_sites += 1
            for sample in samples:
                states = [parse_observed_state(row[f"{sample}_h{hap}"]) for hap in (0, 1)]
                presence = [minor_presence(state, site.minor_code) for state in states]
                missing = sum(value is None for value in presence)
                dosage = "" if missing else str(sum(int(value) for value in presence))
                target_writer.writerow((
                    root_seed, sample, site.locus_index, site.chrom, site.position,
                    site.minor_code,
                    "" if presence[0] is None else presence[0],
                    "" if presence[1] is None else presence[1],
                    dosage, missing,
                ))
                target_rows += 1
                missing_haplotypes += missing
                if site.minor_code == 0 and not missing:
                    minor0_cells += 1
                    legacy = int(states[0]) + int(states[1])
                    corrected = int(dosage)
                    if legacy != corrected:
                        mismatched_legacy_cells += 1
                    else:
                        coincident_legacy_cells += 1

    if seen != set(selected_by_position):
        missing_positions = sorted(set(selected_by_position) - seen)[:5]
        raise ValueError(f"target table does not cover selected FREQ loci: {missing_positions}")
    with samples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("root_seed", "sample_id", "sample_index"))
        for index, sample in enumerate(samples):
            writer.writerow((root_seed, sample, index))
    if target_rows != len(selected) * len(samples):
        raise AssertionError("ordered target row count is not sites times samples")
    return {
        "samples": len(samples),
        "target_rows": target_rows,
        "missing_target_haplotypes": missing_haplotypes,
        "minor_code_zero_sites": minor0_sites,
        "minor_code_zero_sample_cells": minor0_cells,
        "m29_legacy_mismatched_sample_cells": mismatched_legacy_cells,
        "m29_legacy_equal_only_by_heterozygote_coincidence_cells": coincident_legacy_cells,
        "m29_semantic_bug_present": minor0_sites > 0,
        "m29_legacy_formula": "int(h0_state) + int(h1_state)",
        "correct_formula": "I(h0_state == minor_code) + I(h1_state == minor_code)",
        "layout_order": "site-major; within each locus sample_id lexical",
        "sample_identity_key": ["root_seed", "sample_id"],
        "row_primary_key": ["root_seed", "sample_id", "locus_index"],
        "sequence_order": ["position", "locus_index"],
    }


def write_manifest(
    outdir: Path,
    preregistration: Path,
    input_hashes: dict[str, str],
    script_sha256: str,
    git_commit: str | None,
) -> Path:
    outputs = {
        path.name: sha256_file(path)
        for path in sorted(outdir.iterdir())
        if path.name != "m31_ordered_rare.manifest.json"
    }
    manifest = {
        "stage": "M31_ORDERED_RARE_PREFLIGHT",
        "preregistration_sha256": sha256_file(preregistration),
        "code_sha256": {"preflight_script": script_sha256},
        "git_commit": git_commit,
        "input_sha256": input_hashes,
        "output_sha256": outputs,
    }
    path = outdir / "m31_ordered_rare.manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> dict:
    contract, root_spec = load_contract(args.preregistration, args.root_label, args.root_seed)
    input_paths = {
        "tree": args.tree,
        "pools": args.pools,
        "catalog": args.catalog,
        "haplotypes": args.haplotypes,
    }
    input_hashes = authenticate_inputs(input_paths, root_spec)
    script_sha256 = authenticate_script(Path(__file__).resolve(), args.preflight_script_sha256)
    git_commit = validate_git_commit(args.git_commit)
    answers = known_answers()
    selected, freq_audit = derive_freq_sites(args.tree, args.pools, args.catalog, contract)
    if args.outdir.exists():
        raise ValueError(f"refusing to overwrite output directory: {args.outdir}")
    args.outdir.mkdir(parents=True)
    target_audit = materialize_target(args.haplotypes, selected, args.root_seed, args.outdir)
    report = {
        "stage": "M31_ORDERED_RARE_PREFLIGHT",
        "status": "PASS_TECHNICAL_SMOKE_NOT_EVIDENCE",
        "scope": contract["scope"],
        "root_label": args.root_label,
        "root_seed": args.root_seed,
        "source_attempt": root_spec["source_attempt"],
        "scientific_evidence": False,
        "architecture_frozen": False,
        "metrics_frozen": False,
        "known_answers": answers,
        "freq_only_audit": freq_audit,
        "target_materialization_audit": target_audit,
        "input_sha256": input_hashes,
        "code_sha256": {"preflight_script": script_sha256},
        "git_commit": git_commit,
        "materialization_keys": {
            "sample_identity_key": ["root_seed", "sample_id"],
            "row_primary_key": ["root_seed", "sample_id", "locus_index"],
        },
        "preregistration_sha256": sha256_file(args.preregistration),
        "memory_policy": "stream TARGET rows; retain only the FREQ site registry, never an N_sample_by_N_site dense matrix",
        "claims_excluded": contract["claims_excluded"],
    }
    report_path = args.outdir / "m31_ordered_rare.preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(args.outdir, args.preregistration, input_hashes, script_sha256, git_commit)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--root-label", required=True, choices=("root17", "root18"))
    parser.add_argument("--root-seed", required=True, type=int)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--pools", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--haplotypes", required=True, type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--preflight-script-sha256")
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({
        "status": report["status"],
        "root_seed": report["root_seed"],
        "m29_semantic_bug_present": report["target_materialization_audit"]["m29_semantic_bug_present"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
