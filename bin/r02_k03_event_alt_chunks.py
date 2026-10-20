#!/usr/bin/env python3
"""Recover exact outer-FIT source-minor dosages for K03 from rare events and tails.

The source archives store dosage of the source cohort's minor allele.  This
producer preserves that dosage and the catalogue identity fixed in 2,723 source
people.  ``counted_allele`` is checked against REF/ALT provenance, but the dosage
is never reoriented from the genotypes of an INNER_FIT subset.  Source frequencies
never become an INNER_FIT estimate.
The requested, predeclared target territory is removed before any reusable chunk
is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable, Iterator

import numpy as np

import r02_genomic_event_inputs as events
import r02_k03_inner_fit_affinities as affinities
import r02_k03_inner_fit_contract as contract
import r02_structure_materialize as material
import r02_structure_input_plan as input_plan
import r02_structure_rare_inputs as rare_inputs


CAMPAIGN_SCHEMA = "r02_k03_event_alt_campaign_v1"
CHROMOSOME_SCHEMA = "r02_k03_outer_fit_alt_chromosome_v1"
SHARED_SCHEMA = affinities.SHARED_SCHEMA
BASES = np.asarray(list(events.BASES))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return material.sha(Path(path))


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def implementation_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ("r02_k03_event_alt_chunks.py", "r02_k03_inner_fit_affinities.py",
             "r02_genomic_event_inputs.py", "r02_structure_rare_inputs.py",
             "r02_structure_materialize.py", "r02_structure_input_plan.py",
             "r02_k03_inner_fit_contract.py")
    return {name: sha256(root / name) for name in sorted(names)}


def _pin(value: Any, label: str, base: Path) -> Path:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}
            and isinstance(value["path"], str)
            and isinstance(value["sha256"], str) and len(value["sha256"]) == 64,
            f"Exact pinned {label} descriptor required")
    path = Path(value["path"])
    if not path.is_absolute():
        require(path.parts and ".." not in path.parts,
                f"Safe campaign-relative {label} path required")
        path = base / path
    require(path.is_file() and sha256(path) == value["sha256"],
            f"Pinned {label} is absent or changed")
    return path


def _ordered_ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def select_source_minor_dosage(dosage: np.ndarray, observed: np.ndarray,
                               counted_allele: np.ndarray,
                               sample_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select people while preserving the source-minor dosage and missingness."""
    require(dosage.ndim == 2 and dosage.shape == observed.shape
            and dosage.dtype == np.uint8 and observed.dtype == np.bool_
            and np.array_equal(dosage == 255, ~observed)
            and np.all(dosage[observed] <= 2),
            "Source-minor dosage, missingness or diploid range differs")
    require(counted_allele.shape == (dosage.shape[1],)
            and counted_allele.dtype.kind in "iu"
            and np.all((counted_allele == 0) | (counted_allele == 1)),
            "One REF/ALT source-minor orientation is required per site")
    require(sample_indices.ndim == 1 and sample_indices.dtype.kind in "iu"
            and len(sample_indices) and len(set(map(int, sample_indices))) == len(sample_indices)
            and int(sample_indices.min()) >= 0 and int(sample_indices.max()) < dosage.shape[0],
            "Unique in-range outer-FIT sample indices required")
    chosen_observed = observed[sample_indices].copy()
    chosen_minor = dosage[sample_indices].copy()
    require(np.array_equal(chosen_minor == 255, ~chosen_observed)
            and np.all(chosen_minor[chosen_observed] <= 2),
            "Selecting people changed source-minor missingness or dosage")
    return chosen_minor, chosen_observed


def _read_roster(path: Path) -> tuple[list[dict[str, str]], list[int], list[str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    require(rows and set(rows[0]) == {"sample_id", "component", "role"}
            and len({row["sample_id"] for row in rows}) == len(rows),
            "Frozen roster schema or sample identity differs")
    fit = [i for i, row in enumerate(rows) if row["role"] == "FIT"]
    require(len(fit) == 1460, "The admitted outer FIT must contain exactly 1,460 people")
    return rows, fit, [row["sample_id"] for row in rows]


def _territory(blocks_path: Path, source: dict[str, Any],
               requested: Any) -> dict[str, Any]:
    scope, block_id, _ = contract.validate_territory_descriptor(
        requested,
        "FIXED_SOURCE_COHORT_RARE_CANDIDATES_PRESERVED_NO_INNER_ROLE_FILTER")
    with blocks_path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == input_plan.BLOCK_COLUMNS,
                "Frozen block table columns differ")
        blocks = list(reader)
    sources = source.get("sources")
    require(isinstance(sources, dict) and set(sources) == {str(c) for c in range(1, 23)},
            "Frozen source contract must define all 22 autosomes")
    lengths = {}
    for chromosome, descriptor in sources.items():
        require(isinstance(descriptor, dict)
                and type(descriptor.get("length_bp")) is int
                and descriptor["length_bp"] > 0,
                "Frozen source chromosome length differs")
        lengths[chromosome] = descriptor["length_bp"]
    planned = {item["block_id"]: item
               for item in input_plan.plan_regions(blocks, lengths, scope)}
    require(block_id in planned, "Territory block is absent from the frozen block table")
    chromosome, start_bp, end_bp = planned[block_id]["excluded"]
    expected = {
        "view_id": f"{scope}:{block_id}",
        "excluded_intervals": [{"chromosome": int(chromosome),
                                "start_bp": start_bp, "end_bp": end_bp}],
        "target_locus_policy": "EXCLUDED_BEFORE_OUTPUT",
        "source_candidate_rule": "FIXED_SOURCE_COHORT_RARE_CANDIDATES_PRESERVED_NO_INNER_ROLE_FILTER",
    }
    require(requested == expected,
            "Requested exclusion differs from the frozen block/source-derived territory")
    return expected


def _campaign(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path, list[int], list[str]]:
    value = load_json(path); base = path.resolve().parent
    required = {"schema", "source_contract", "roster", "blocks", "territory",
                "chromosomes", "resources", "consumer_resources",
                "consumer_implementation_sha256", "converter_implementation_sha256"}
    require(set(value) == required and value["schema"] == CAMPAIGN_SCHEMA,
            "Exact K03 event-to-ALT campaign required")
    source_path = _pin(value["source_contract"], "source contract", base)
    roster_path = _pin(value["roster"], "roster", base)
    blocks_path = _pin(value["blocks"], "blocks", base)
    source = load_json(source_path)
    require(source.get("schema") == "r02_structure_materialize_v1"
            and source.get("genome_build") == "GRCh38"
            and source.get("source_n_samples") == 2723
            and source.get("source_rare_rule") == {
                "min_mac": 2, "max_maf": "0.01",
                "ascertainment": "FIXED_SOURCE_COHORT_NOT_FIT_RESELECTED"},
            "Frozen GRCh38 source rare candidate definition differs")
    _, fit, all_ids = _read_roster(roster_path)
    require(sha256(roster_path) == source.get("metadata_sha256", {}).get("persons.tsv"),
            "Roster authentication differs from the frozen source contract")
    value["territory"] = _territory(blocks_path, source, value["territory"])
    require(value["consumer_implementation_sha256"] == affinities.implementation_sha256(),
            "Pinned INNER_FIT consumer implementation changed")
    require(value["converter_implementation_sha256"] == implementation_sha256(),
            "Pinned event-to-ALT converter implementation changed")
    chroms = value["chromosomes"]
    require(isinstance(chroms, list) and [x.get("chromosome") for x in chroms] == list(range(1, 23))
            and all(set(x) == {"chromosome", "event_archive", "event_manifest",
                               "tail_vcf", "tail_manifest"} for x in chroms),
            "Exactly one ordered input set for each of 22 autosomes is required")
    r = value["resources"]
    require(set(r) == {"max_archive_bytes", "max_chunk_bytes", "max_array_bytes",
                       "max_input_bytes", "max_output_bytes", "min_free_bytes",
                       "output_chunk_sites", "timeout_seconds"}
            and all(type(x) is int and x > 0 for x in r.values())
            and r["output_chunk_sites"] * len(fit) * 4 <= r["max_array_bytes"],
            "Explicit positive converter bounds or memory sizing required")
    cr = value["consumer_resources"]
    require(set(cr) == {"max_input_bytes", "max_output_bytes", "min_free_bytes",
                        "max_memory_bytes", "max_sparse_cells", "site_batch_size"}
            and all(type(x) is int and x > 0 for x in cr.values()),
            "Explicit positive INNER_FIT consumer bounds required")
    return value, source, roster_path, blocks_path, fit, all_ids


def _excluded(chromosome: int, positions: np.ndarray,
              territory: dict[str, Any]) -> np.ndarray:
    interval = territory["excluded_intervals"][0]
    return ((chromosome == interval["chromosome"])
            & (positions >= interval["start_bp"])
            & (positions <= interval["end_bp"]))


def chromosome_identity(campaign: dict[str, Any], chromosome: int) -> dict[str, Any]:
    """Bind a reusable chromosome to its effective exclusion and exact inputs.

    Different target views may reuse a chromosome only when they remove exactly
    the same coordinates from it. File names and chronological proximity do not
    establish that equivalence.
    """
    item = campaign["chromosomes"][chromosome - 1]
    require(item["chromosome"] == chromosome, "Chromosome source order differs")
    return {
        "source_contract_sha256": campaign["source_contract"]["sha256"],
        "roster_sha256": campaign["roster"]["sha256"],
        "sources_sha256": {k: item[k]["sha256"] for k in
                           ("event_archive", "event_manifest", "tail_vcf", "tail_manifest")},
        "excluded_intervals": [dict(i) for i in campaign["territory"]["excluded_intervals"]
                               if i["chromosome"] == chromosome],
        "dosage_allele": "SOURCE_MINOR",
    }


def validate_chromosome_binding(record: dict[str, Any], campaign: dict[str, Any],
                                chromosome: int, fit_ids: list[str]) -> None:
    require(record.get("input_identity") == chromosome_identity(campaign, chromosome),
            "Chromosome product belongs to different sources, people or target exclusion")
    require(record.get("sample_ids_sha256") == _ordered_ids_sha256(fit_ids)
            and record.get("outer_fit_people") == len(fit_ids),
            "Chromosome product FIT people/order differs")
    require(record["source_sites"] == record["included_sites"] + record["excluded_target_sites"]
            and sum(c["n_sites"] for c in record["chunks"]) == record["included_sites"],
            "Chromosome retained/excluded counts differ")


class ChunkWriter:
    def __init__(self, output: Path, chromosome: int, fit_ids: list[str],
                 sites_per_chunk: int, max_output_bytes: int, min_free_bytes: int):
        self.output, self.chromosome, self.fit_ids = output, chromosome, fit_ids
        self.sites_per_chunk = sites_per_chunk
        self.max_output_bytes, self.min_free_bytes = max_output_bytes, min_free_bytes
        self.parts: list[dict[str, np.ndarray]] = []
        self.pending = self.total_sites = self.total_bytes = 0
        self.chunks: list[dict[str, Any]] = []
        (output / "chunks").mkdir()

    def add(self, positions: np.ndarray, ref: np.ndarray, alt: np.ndarray,
            source_minor_dosage: np.ndarray, observed: np.ndarray) -> None:
        n = len(positions)
        require(n and positions.shape == ref.shape == alt.shape == (n,)
                and source_minor_dosage.shape == observed.shape == (len(self.fit_ids), n),
                "Source-minor chunk input shapes differ")
        start = 0
        while start < n:
            take = min(n - start, self.sites_per_chunk - self.pending)
            sl = slice(start, start + take)
            self.parts.append({"position_bp": positions[sl].copy(), "ref": ref[sl].copy(),
                               "alt": alt[sl].copy(),
                               "genotype_code": source_minor_dosage[:, sl].copy(),
                               "observed": observed[:, sl].copy()})
            self.pending += take; start += take
            if self.pending == self.sites_per_chunk:
                self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        positions = np.concatenate([p["position_bp"] for p in self.parts])
        ref = np.concatenate([p["ref"] for p in self.parts]).astype("U1")
        alt = np.concatenate([p["alt"] for p in self.parts]).astype("U1")
        genotype_code = np.concatenate([p["genotype_code"] for p in self.parts], axis=1)
        observed = np.concatenate([p["observed"] for p in self.parts], axis=1)
        require(np.array_equal(genotype_code == 255, ~observed),
                "Source-minor chunk missingness changed before writing")
        require(np.all(positions[1:] > positions[:-1]),
                "Source rare rows are not coordinate-unique and sorted")
        variant_id = np.asarray([f"{self.chromosome}:{p}:{r}:{a}"
                                 for p, r, a in zip(positions, ref, alt)])
        number = len(self.chunks); relative = f"chunks/{number:06d}.npz"
        path = self.output / relative
        require(shutil.disk_usage(self.output).free >= self.min_free_bytes,
                "Converter disk reserve exhausted")
        np.savez_compressed(path, sample_ids=np.asarray(self.fit_ids), variant_id=variant_id,
                            chromosome=np.full(len(positions), self.chromosome, np.int16),
                            position_bp=positions.astype(np.int64), ref=ref, alt=alt,
                            genotype_code=genotype_code)
        size = path.stat().st_size; self.total_bytes += size
        require(self.total_bytes <= self.max_output_bytes,
                "Compressed ALT chunks exceed declared output bound")
        self.chunks.append({"path": relative, "sha256": sha256(path), "bytes": size,
                            "n_sites": len(positions), "start_bp": int(positions[0]),
                            "end_bp": int(positions[-1])})
        self.total_sites += len(positions); self.parts.clear(); self.pending = 0


def _tail_arrays(rows: Iterable[list[str]], source: dict[str, Any], n_people: int,
                 expected_sites: int) -> dict[str, np.ndarray]:
    positions: list[int] = []; refs: list[str] = []; alts: list[str] = []
    orientations: list[int] = []; doses: list[np.ndarray] = []; masks: list[np.ndarray] = []
    previous = 0
    for fields in rows:
        require(len(fields) == 9 + 2 * n_people, "Rare tail GT/RD column count differs")
        position, eligible = material.basic_site(fields, source, previous)
        require(position > previous and eligible, "Rare tail is not a sorted eligible source SNV")
        previous = position
        allele, ac, an = map(int, fields[6:9])
        require(allele in (0, 1) and ac >= 2 and an > 0 and ac <= an // 2
                and 100 * ac <= an, "Rare tail source-candidate identity differs")
        gt = material.decode_gt(fields[9::2]); rd = np.asarray(fields[10::2])
        observed = np.isfinite(gt)
        require(np.array_equal(rd == ".", ~observed), "Rare tail RD/GT missingness differs")
        dose = np.full(n_people, 255, np.uint8)
        dose[observed] = rd[observed].astype(np.uint8)
        require(np.array_equal(dose[observed],
                               (gt if allele else 2 - gt)[observed]),
                "Rare tail source-minor dosage disagrees with GT and REF/ALT orientation")
        positions.append(position); refs.append(fields[2]); alts.append(fields[3])
        orientations.append(allele); doses.append(dose); masks.append(observed)
        require(len(positions) <= expected_sites, "Tail VCF exceeds recorded outside-map rows")
    require(len(positions) == expected_sites, "Tail VCF does not recover every outside-map source row")
    return {"position_bp": np.asarray(positions, np.int64),
            "ref": np.asarray(refs, dtype="U1"), "alt": np.asarray(alts, dtype="U1"),
            "counted_allele": np.asarray(orientations, np.uint8),
            "dosage": np.column_stack(doses) if doses else np.empty((n_people, 0), np.uint8),
            "observed": np.column_stack(masks) if masks else np.empty((n_people, 0), np.bool_)}


def _query_tail(tool: str, path: Path, source: dict[str, Any], all_ids: list[str],
                expected_sites: int, log: Path) -> dict[str, np.ndarray]:
    require(subprocess.check_output([tool, "query", "-l", str(path)], text=True).splitlines() == all_ids,
            "Tail VCF samples/order differ from the frozen analytical roster")
    header = subprocess.check_output([tool, "view", "-h", str(path)], text=True)
    require("##dnabr_original_alleles=v1\n" in header
            and "##dnabr_rare_contract=minor_v1\n" in header,
            "Tail VCF lacks original-allele/source-minor provenance")
    fmt = "%CHROM\t%POS\t%REF\t%ALT\t%FILTER\t%INFO/ORIG_NALLELES\t%INFO/RARE_ALLELE\t%INFO/RARE_AC\t%INFO/RARE_AN[\t%GT\t%RD]\n"
    with log.open("x") as error:
        process = subprocess.Popen([tool, "query", "-f", fmt, str(path)],
                                   stdout=subprocess.PIPE, stderr=error, text=True,
                                   bufsize=2**20)
        try:
            rows = (line.rstrip("\n").split("\t") for line in process.stdout)
            result = _tail_arrays(rows, source, len(all_ids), expected_sites)
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            code = process.wait()
    require(code == 0, "bcftools failed while reading authenticated rare tails")
    return result


def _write_source(writer: ChunkWriter, chromosome: int, positions: np.ndarray,
                  ref: np.ndarray, alt: np.ndarray, source_minor: np.ndarray,
                  observed: np.ndarray, counted: np.ndarray, fit: np.ndarray,
                  territory: dict[str, Any]) -> tuple[int, int]:
    source_minor_dose, fit_observed = select_source_minor_dosage(
        source_minor, observed, counted, fit)
    remove = _excluded(chromosome, positions, territory); keep = ~remove
    if np.any(keep):
        writer.add(positions[keep], ref[keep], alt[keep],
                   source_minor_dose[:, keep], fit_observed[:, keep])
    return int(keep.sum()), int(remove.sum())


def convert_chromosome(campaign_path: Path, chromosome: int, output: Path) -> dict[str, Any]:
    campaign_path, output = Path(campaign_path), Path(output)
    campaign, source, roster_path, _, fit_list, all_ids = _campaign(campaign_path)
    require(chromosome in range(1, 23), "Autosome must be in 1..22")
    require(not output.exists() and output.parent.is_dir(), "A new chromosome output is required")
    r = campaign["resources"]
    require(shutil.disk_usage(output.parent).free >= r["min_free_bytes"] + r["max_output_bytes"],
            "Declared converter output plus disk reserve is unavailable")
    output.mkdir(mode=0o700); started = time.monotonic(); base = campaign_path.resolve().parent
    item = campaign["chromosomes"][chromosome - 1]
    archive = _pin(item["event_archive"], f"chr{chromosome} event archive", base)
    event_path = _pin(item["event_manifest"], f"chr{chromosome} event manifest", base)
    tail_vcf = _pin(item["tail_vcf"], f"chr{chromosome} tail VCF", base)
    tail_manifest_path = _pin(item["tail_manifest"], f"chr{chromosome} tail manifest", base)
    require(archive.stat().st_size <= r["max_archive_bytes"]
            and archive.stat().st_size + tail_vcf.stat().st_size <= r["max_input_bytes"],
            "Chromosome source inputs exceed declared byte bounds")
    event = load_json(event_path); tail_manifest = load_json(tail_manifest_path)
    fit = np.asarray(fit_list, np.int64); fit_ids = [all_ids[i] for i in fit_list]
    full_ids_hash = _ordered_ids_sha256(all_ids)
    require(event.get("schema") == events.CHROM_SCHEMA
            and event.get("stage") == "COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING"
            and event.get("chromosome") == str(chromosome)
            and event.get("source_contract_sha256") == campaign["source_contract"]["sha256"]
            and event.get("n_people") == len(all_ids)
            and event.get("sample_ids_sha256") == full_ids_hash
            and event.get("raw_contains_targets") is True
            and event.get("requires_excluded_window_reader") is True
            and event["audit"]["rare"]["source_rows"] ==
                event["audit"]["rare"]["retained_sites"] + event["audit"]["rare"].get("outside_map", 0),
            "Event source/cohort/completeness provenance differs")
    require(tail_manifest.get("schema") == "r02_structure_indexed_source_slice_v1"
            and tail_manifest.get("stage") == "INDEXED_SOURCE_SLICE_NOT_FIT"
            and tail_manifest.get("output_sha256") == item["tail_vcf"]["sha256"],
            "Tail slice identity or non-FIT provenance differs")
    src = source["sources"][str(chromosome)]
    interval = campaign["territory"]["excluded_intervals"][0]
    if (interval["chromosome"] == chromosome and interval["start_bp"] == 1
            and interval["end_bp"] == src["length_bp"]):
        # All source identities were checked above. There are no eligible sites
        # to read when this entire chromosome is the prediction target.
        writer = ChunkWriter(output, chromosome, fit_ids, r["output_chunk_sites"],
                             r["max_output_bytes"], r["min_free_bytes"])
        return _finish_chromosome(campaign, chromosome, output, item, event, fit_ids,
                                  writer, 0, int(event["audit"]["rare"]["source_rows"]))
    tool = material.checked_tool("bcftools", source["tools"]["bcftools"])
    tails = _query_tail(tool, tail_vcf, src, all_ids,
                        event["audit"]["rare"].get("outside_map", 0),
                        output / "tails.bcftools.log")
    entries = event["channels"]["rare"]
    lower, upper = entries[0]["start_bp"], entries[-1]["end_bp"]
    require(np.all((tails["position_bp"] < lower) | (tails["position_bp"] > upper)),
            "Tail rows overlap mapped event support")
    writer = ChunkWriter(output, chromosome, fit_ids, r["output_chunk_sites"],
                         r["max_output_bytes"], r["min_free_bytes"])
    included = excluded = 0
    for selection in (tails["position_bp"] < lower,):
        if np.any(selection):
            a, b = _write_source(writer, chromosome, tails["position_bp"][selection],
                                 tails["ref"][selection], tails["alt"][selection],
                                 tails["dosage"][:, selection], tails["observed"][:, selection],
                                 tails["counted_allele"][selection], fit,
                                 campaign["territory"])
            included += a; excluded += b
    deadline = started + r["timeout_seconds"]
    for array in rare_inputs.iter_archive(archive, event, n_people=len(all_ids),
            source_n_people=source["source_n_samples"], max_chunk_bytes=r["max_chunk_bytes"],
            max_array_bytes=r["max_array_bytes"], deadline=deadline):
        require(time.monotonic() < deadline, "Chromosome conversion deadline exhausted")
        a, b = _write_source(writer, chromosome, array["position_bp"],
                             BASES[array["ref"]], BASES[array["alt"]], array["dosage"],
                             array["observed"], array["counted_allele"], fit,
                             campaign["territory"])
        included += a; excluded += b
    selection = tails["position_bp"] > upper
    if np.any(selection):
        a, b = _write_source(writer, chromosome, tails["position_bp"][selection],
                             tails["ref"][selection], tails["alt"][selection],
                             tails["dosage"][:, selection], tails["observed"][:, selection],
                             tails["counted_allele"][selection], fit,
                             campaign["territory"])
        included += a; excluded += b
    writer.flush()
    return _finish_chromosome(campaign, chromosome, output, item, event, fit_ids,
                              writer, included, excluded)


def _finish_chromosome(campaign, chromosome, output, item, event, fit_ids,
                       writer, included, excluded):
    source_sites = int(event["audit"]["rare"]["source_rows"])
    require(included + excluded == source_sites and writer.total_sites == included,
            "Source, target-exclusion and emitted-site counts do not reconcile")
    result = {
        "schema": CHROMOSOME_SCHEMA, "chromosome": chromosome,
        "status": "OUTER_FIT_SOURCE_MINOR_DOSAGE_COMPLETE_TARGET_EXCLUDED",
        "source_sites": source_sites, "included_sites": included,
        "excluded_target_sites": excluded,
        "mapped_event_sites": int(event["audit"]["rare"]["retained_sites"]),
        "recovered_outside_map_sites": int(event["audit"]["rare"].get("outside_map", 0)),
        "outer_fit_people": len(fit_ids), "sample_ids_sha256": _ordered_ids_sha256(fit_ids),
        "source_frequency_used_as_inner_estimate": False,
        "conversion": "SOURCE_MINOR_DOSAGE_PRESERVED_WITHOUT_INNER_SUBSET_REORIENTATION",
        "missingness": "OBSERVED_MASK_AND_UINT8_255_SENTINEL_PRESERVED",
        "event_archive_sha256": item["event_archive"]["sha256"],
        "event_manifest_sha256": item["event_manifest"]["sha256"],
        "tail_vcf_sha256": item["tail_vcf"]["sha256"],
        "tail_manifest_sha256": item["tail_manifest"]["sha256"],
        "input_identity": chromosome_identity(campaign, chromosome),
        "chunks": writer.chunks, "compressed_chunk_bytes": writer.total_bytes,
        "implementation_sha256": implementation_sha256(),
        "biological_validation": False,
    }
    write_json(output / "chromosome.json", result)
    return result


def assemble_shared(campaign_path: Path, chromosome_outputs: list[Path], output: Path) -> dict[str, Any]:
    campaign_path, output = Path(campaign_path), Path(output)
    campaign, _, roster_path, _, fit_list, all_ids = _campaign(campaign_path)
    require(len(chromosome_outputs) == 22 and not output.exists() and output.parent.is_dir(),
            "Exactly 22 completed chromosome outputs and a new shared output are required")
    output.mkdir(mode=0o700); (output / "chunks").mkdir()
    shutil.copyfile(roster_path, output / "persons.tsv")
    descriptors: list[dict[str, str]] = []; expected_chromosome = 1
    source_sites = included_sites = excluded_sites = recovered = compressed = 0
    for directory in map(Path, chromosome_outputs):
        record_path = directory / "chromosome.json"; record = load_json(record_path)
        require(record.get("schema") == CHROMOSOME_SCHEMA
                and record.get("chromosome") == expected_chromosome
                and record.get("implementation_sha256") == implementation_sha256()
                and record.get("source_frequency_used_as_inner_estimate") is False,
                "Chromosome source-minor product order, implementation or learning scope differs")
        validate_chromosome_binding(record, campaign, expected_chromosome,
                                    [all_ids[i] for i in fit_list])
        target = output / "chunks" / f"chr{expected_chromosome:02d}"
        target.mkdir()
        for item in record["chunks"]:
            require(set(item) == {"path", "sha256", "bytes", "n_sites", "start_bp", "end_bp"},
                    "Chromosome chunk descriptor differs")
            source_path = directory / item["path"]
            require(source_path.is_file() and source_path.stat().st_size == item["bytes"]
                    and sha256(source_path) == item["sha256"], "Chromosome chunk changed")
            destination = target / Path(item["path"]).name
            try:
                os.link(source_path, destination)
            except OSError:
                shutil.copyfile(source_path, destination)
            descriptors.append({"path": str(destination.relative_to(output)),
                                "sha256": item["sha256"]})
        source_sites += record["source_sites"]; included_sites += record["included_sites"]
        excluded_sites += record["excluded_target_sites"]
        recovered += record["recovered_outside_map_sites"]
        compressed += record["compressed_chunk_bytes"]
        expected_chromosome += 1
    require(source_sites == included_sites + excluded_sites and recovered == 22922,
            "All source rare rows, target exclusions and 22,922 outside-map rows must reconcile")
    fit_ids = [all_ids[i] for i in fit_list]
    manifest = {
        "schema": SHARED_SCHEMA,
        "roster": {"path": "persons.tsv", "sha256": sha256(output / "persons.tsv")},
        "sample_scope": "OUTER_FIT_GENOTYPES_ONLY",
        "sample_ids_sha256": _ordered_ids_sha256(fit_ids),
        "dosage_allele": "SOURCE_MINOR",
        "genotype_encoding": contract.FIXED_SOURCE_MINOR_ENCODING,
        "variant_universe": (
            "PINNED_SOURCE_RARE_BLOCK_AND_LD_MARGIN_TARGET_EXCLUDED_NO_INNER_ROLE_SELECTION"
            if campaign["territory"]["view_id"].startswith("BLOCK_AND_LD_MARGIN:")
            else "PINNED_SOURCE_RARE_WHOLE_CHROMOSOME_TARGET_EXCLUDED_NO_INNER_ROLE_SELECTION"),
        "candidate_ascertainment": contract.FIXED_SOURCE_MINOR_CATALOG,
        "territory": campaign["territory"],
        "learning_scope": "FIXED_SOURCE_MINOR_DOSAGE_NO_INNER_TRANSFORMATIONS",
        "outer_fit_learned_inputs": False, "chunks": descriptors,
        "resources": campaign["consumer_resources"],
        "producer": {"implementation_sha256": implementation_sha256(),
                     "source_contract_sha256": campaign["source_contract"]["sha256"],
                     "source_frequency_use": "ORIENTATION_VALIDATION_ONLY_NOT_AN_INNER_ESTIMATE",
                     "source_minor_dose_use": "PRESERVE_FIXED_CATALOG_ORIENTATION",
                     "tail_recovery": "ALL_OUTSIDE_MAP_SOURCE_RARE_ROWS_RECOVERED",
                     "all_autosomes_complete": True},
        "consumer_implementation_sha256": campaign["consumer_implementation_sha256"],
    }
    write_json(output / "manifest.json", manifest)
    summary = {"schema": "r02_k03_outer_fit_alt_shared_summary_v1",
               "status": "FIXED_SOURCE_MINOR_CATALOG_READY_FOR_INNER_FIT_CALIBRATION",
               "source_sites": source_sites, "included_sites": included_sites,
               "excluded_target_sites": excluded_sites,
               "recovered_outside_map_sites": recovered,
               "outer_fit_people": len(fit_ids), "chunks": len(descriptors),
               "compressed_chunk_bytes": compressed,
               "manifest_sha256": sha256(output / "manifest.json"),
               "ten_context_copies_created": False,
               "human_affinity_consumer_admitted": False,
               "biological_validation": False}
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("chromosome")
    one.add_argument("--campaign", type=Path, required=True)
    one.add_argument("--chromosome", type=int, required=True)
    one.add_argument("--output", type=Path, required=True)
    combine = sub.add_parser("assemble")
    combine.add_argument("--campaign", type=Path, required=True)
    combine.add_argument("--chromosome-output", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); os.umask(0o077)
    if args.command == "chromosome":
        result = convert_chromosome(args.campaign, args.chromosome, args.output)
    else:
        result = assemble_shared(args.campaign, args.chromosome_output, args.output)
    print(json.dumps({k: result[k] for k in result if k in {
        "status", "chromosome", "included_sites", "excluded_target_sites",
        "recovered_outside_map_sites", "compressed_chunk_bytes"}}, sort_keys=True))


if __name__ == "__main__":
    main()
