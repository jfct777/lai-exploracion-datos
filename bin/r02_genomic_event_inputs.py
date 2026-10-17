#!/usr/bin/env python3
"""Bounded individual genotype stores and excluded-window neural input views.

This is a data producer, not a trainer. Common context uses FIT-only eligibility
without LD pruning; the separate K02 target selection is never changed. Rare
rows retain the fixed source-minor allele, including observed zero doses.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import gzip
import math
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np

import r02_structure_materialize as material
import r02_structure_regions as maps
import r02_genomic_event_cache as fit_cache

require, sha, load_json = material.require, material.sha, material.load_json
SCHEMA = "r02_genomic_event_inputs_v1"
CHROM_SCHEMA = "r02_genomic_event_chromosome_v1"
CODE_FILES = (*material.CODE_FILES, "r02_structure_regions.py", "r02_genomic_event_inputs.py", "r02_genomic_event_cache.py")
BASES = "ACGT"
COMMON_FEATURES = ("alt_dose_fraction", "fit_alt_af", "ref_A", "ref_C", "ref_G", "ref_T",
                   "alt_A", "alt_C", "alt_G", "alt_T", "bp_fraction", "map_fraction")
EVENT_FEATURES = ("minor_dose_fraction", "source_minor_af", "counted_is_alt", "ref_A", "ref_C",
                  "ref_G", "ref_T", "alt_A", "alt_C", "alt_G", "alt_T", "bp_fraction", "map_fraction")
ARRAY_FIELDS = {"dosage", "observed", "position_bp", "position_cm", "ref", "alt",
                "counted_allele", "allele_count", "allele_number", "frequency"}


def code_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def load_contract(bundle, expected_sha):
    bundle = Path(bundle)
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha or "") and
            sha(bundle / "event_inputs.json") == expected_sha, "Event contract SHA256 mismatch")
    cfg = load_json(bundle / "event_inputs.json")
    require(set(cfg) - {"fit_caches"} == {"schema", "materialize_contract_sha256", "common_policy", "map_policy",
            "overlap_policy", "maps", "resources"} and cfg["schema"] == SCHEMA, "Event input schema differs")
    require(sha(bundle / "source/materialize.json") == cfg["materialize_contract_sha256"],
            "Inherited materialization contract changed")
    contract, persons, roles, blocks, regions = material.load_context(bundle / "source")
    require(contract["roles_state"] == "EXPLICITLY_FROZEN", "Explicitly frozen family-separated roles required")
    require(cfg["common_policy"] == "ALL_FIT_ELIGIBLE_WITHOUT_LD_PRUNING", "Explicit common-context policy required")
    require(cfg["map_policy"] == "OMIT_OUTSIDE_MAP_NO_EXTRAPOLATION", "Explicit map coverage policy required")
    require(cfg["overlap_policy"] == "REJECT_SELECTED_COMMON_RARE_OVERLAP", "Common/rare overlap policy required")
    require(set(cfg["maps"]) == set(contract["sources"]), "Map/source chromosome inventory differs")
    if "fit_caches" in cfg:
        require(set(cfg["fit_caches"]) == set(contract["sources"]), "FIT eligibility/source chromosome inventory differs")
    for row in cfg["maps"].values():
        require(set(row) == {"path", "sha256", "format"} and Path(row["path"]).is_absolute() and
                re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "Authenticated map descriptor required")
    r = cfg["resources"]
    require(set(r) == {"chunk_sites", "max_sites_per_chromosome", "max_array_mb", "max_output_mb",
                       "min_free_disk_mb", "max_window_sites", "max_batch_people", "max_map_points"}, "Explicit event resource limits required")
    require(all(type(v) is int and v > 0 for v in r.values()), "Positive integer resource limits required")
    require(r["chunk_sites"] * len(persons) * 48 <= r["max_array_mb"] * 2**20,
            "Chunk construction exceeds conservative memory budget")
    require(r["max_batch_people"] <= len(persons), "Batch person limit exceeds cohort")
    require(r["max_map_points"]*256 <= r["max_array_mb"]*2**20, "Map point memory budget exceeded")
    require(r["max_map_points"]*256+r["chunk_sites"]*len(persons)*48 <= r["max_array_mb"]*2**20,
            "Combined map/chunk memory budget exceeded")
    return cfg, contract, persons, roles, blocks, regions


def bounded_map(path, chromosome, length_bp, format, settings):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as handle:
        for index,line in enumerate(handle):
            require(index <= settings['max_map_points'] and len(line) <= 1024, "Map metadata limit exceeded")
    return maps.read_map(path,chromosome,length_bp,format)


def verify_source(bundle, expected_sha, chromosome, output):
    cfg, contract, _, _, _, _ = load_contract(bundle, expected_sha)
    require(chromosome in contract["sources"], "Unknown chromosome")
    out = material.new_directory(output)
    source = material.verify_source(Path(bundle) / "source", chromosome, out / "source")
    desc = cfg["maps"][chromosome]
    before = material.signature(desc["path"])
    require(sha(desc["path"]) == desc["sha256"], "Genetic map SHA256 mismatch")
    points = bounded_map(desc["path"], chromosome, contract["sources"][chromosome]["length_bp"], desc["format"], cfg["resources"])
    if "fit_caches" in cfg:
        _, _, persons, roles, _, _ = load_contract(bundle, expected_sha)
        fit_cache.authenticate(cfg["fit_caches"][chromosome], contract, persons, roles, chromosome, desc["sha256"])
    require(material.signature(desc["path"]) == before, "Map changed during verification")
    result = dict(schema=SCHEMA, stage="SOURCE_VERIFIED", chromosome=chromosome,
                  event_contract_sha256=expected_sha, code_sha256=code_hashes(), source=source,
                  map_descriptor=desc, map_signature=before, map_points=len(points))
    material.write_json(out / "manifest.json", result)
    return result


def decode_rare(row, n, contract):
    """Keep GT and stored minor dosage consistent, including partial missing GT."""
    require(len(row) == 9 + 2*n, "Rare GT/RD column count differs")
    allele, ac, an = map(int, row[6:9]); rule = contract["source_rare_rule"]
    maximum = Fraction(rule["max_maf"])
    require(allele in (0, 1) and rule["min_mac"] <= ac <= an // 2 and
            0 < an <= 2*contract["source_n_samples"] and
            ac * maximum.denominator <= an * maximum.numerator,
            "Rare source minor count/frequency differs from frozen rule")
    gt = material.decode_gt(row[9::2]); rd = np.asarray(row[10::2]); observed = np.isfinite(gt)
    require(np.array_equal(rd == ".", ~observed), "RD/GT missingness disagreement")
    require(all(x in {"0", "1", "2"} for x in rd[observed]), "Invalid observed rare dosage")
    dose = np.full(n, 255, dtype=np.uint8); dose[observed] = rd[observed].astype(np.uint8)
    require(np.array_equal(dose[observed], (gt if allele else 2-gt)[observed]),
            "RD disagrees with fixed source minor allele")
    return dose, observed, allele, ac, an


class ChunkWriter:
    """Append at most chunk_sites rows, then release genotype buffers."""
    def __init__(self, output, kind, n_people, settings, shared_bytes=None):
        self.out, self.kind, self.n, self.r = Path(output), kind, n_people, settings
        self.rows, self.chunks, self.total_bytes, self.nsites = [], [], 0, 0
        self.shared_bytes = shared_bytes if shared_bytes is not None else [0]

    def add(self, row):
        self.nsites += 1
        require(self.nsites <= self.r["max_sites_per_chromosome"], "Site limit exceeded; no silent truncation")
        self.rows.append(row)
        if len(self.rows) == self.r["chunk_sites"]:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        nsites = len(self.rows)
        fields = ("position_bp", "position_cm", "ref", "alt", "counted_allele", "allele_count", "allele_number", "frequency")
        arrays = {key: np.asarray([r[key] for r in self.rows], dtype=(np.float64 if key in {"position_cm", "frequency"}
                  else np.uint8 if key in {"ref", "alt", "counted_allele"} else np.int64)) for key in fields}
        arrays["dosage"] = np.column_stack([r["dosage"] for r in self.rows])
        arrays["observed"] = np.column_stack([r["observed"] for r in self.rows])
        name = f"{self.kind}.{len(self.chunks):06d}.npz"
        path = self.out / name
        require(sum(a.nbytes for a in arrays.values())*3 <= self.r["max_array_mb"]*2**20, "Chunk allocation exceeds limit")
        estimate = sum(a.nbytes for a in arrays.values())+8192
        require(self.shared_bytes[0]+estimate <= self.r["max_output_mb"]*2**20, "Combined chromosome output budget exhausted")
        require(shutil.disk_usage(self.out).free >= self.r["min_free_disk_mb"]*2**20+estimate,
                "Chunk would consume reserved disk space")
        material.check_disk(self.out, self.r)
        np.savez(path, **arrays)
        self.total_bytes += path.stat().st_size
        self.shared_bytes[0] += path.stat().st_size
        require(self.total_bytes <= self.r["max_output_mb"]*2**20, "Chromosome output budget exhausted")
        self.chunks.append(dict(path=name, sha256=sha(path), n_sites=nsites, bytes=path.stat().st_size,
                               start_bp=int(arrays["position_bp"][0]), end_bp=int(arrays["position_bp"][-1])))
        self.rows.clear()


def extract(bundle, expected_sha, chromosome, verified, output):
    cfg, contract, persons, roles, _, _ = load_contract(bundle, expected_sha)
    receipt = load_json(verified)
    require(receipt["schema"] == SCHEMA and receipt["stage"] == "SOURCE_VERIFIED" and
            receipt["chromosome"] == chromosome and receipt["event_contract_sha256"] == expected_sha and
            receipt["code_sha256"] == code_hashes(), "Verification receipt contract/code/chromosome differs")
    require(receipt["source"]["contract_sha256"] == cfg["materialize_contract_sha256"] and
            receipt["source"]["code_sha256"] == material.code_hashes(), "Underlying source verification differs")
    material.assert_source_unchanged(receipt["source"], contract, chromosome)
    desc = cfg["maps"][chromosome]
    require(receipt["map_descriptor"] == desc and material.signature(desc["path"]) == receipt["map_signature"],
            "Verified map changed")
    src, n = contract["sources"][chromosome], len(persons)
    points = bounded_map(desc["path"], chromosome, src["length_bp"], desc["format"], cfg["resources"])
    out = material.new_directory(output); fs = material.local_intermediate_filesystem(out)
    (out / "source").mkdir()
    for name in ("materialize.json", *material.META):
        shutil.copyfile(Path(bundle) / "source" / name, out / "source" / name)
    shutil.copyfile(Path(bundle) / "event_inputs.json", out / "event_inputs.json")
    material.write_json(out / "source_verification.json", receipt)
    sample_file = out / "samples.txt"
    sample_file.write_text("".join(p["sample_id"] + "\n" for p in persons))
    with (out / "map.tsv").open("x") as handle:
        handle.write("chrom\tpos\tcm\n")
        for bp, cm in points:
            handle.write(f"{chromosome}\t{bp}\t{cm:.17g}\n")
    tool = material.checked_tool("bcftools", contract["tools"]["bcftools"])
    channels, audit = {}, {}
    targets = cached_rows = None
    if "fit_caches" in cfg:
        descriptors = cfg["fit_caches"][chromosome]
        upstream, expected_sites = fit_cache.authenticate(descriptors, contract, persons, roles, chromosome, desc["sha256"])
        for label, item in descriptors.items():
            shutil.copyfile(item["path"], out / ("fit_" + label + (".tsv" if label == "eligible" else ".json")))
        targets = out / "common_targets.tsv"
        fit_cache.write_targets(out / "fit_eligible.tsv", targets, src, upstream["blocks"], points, expected_sites)
        cached_rows = fit_cache.rows(out / "fit_eligible.tsv")
    marker_file = out / "common_loci.tsv"
    shared_bytes = [sum(p.stat().st_size for p in out.rglob("*") if p.is_file())]
    for kind in ("common", "rare"):
        writer = ChunkWriter(out, kind, n, cfg["resources"], shared_bytes)
        counts, previous, eligible_previous = Counter(), 0, 0
        comparison = None
        if kind == "common":
            marker_handle = marker_file.open("x")
            marker_handle.write("position_bp\tref\talt\n")
        else:
            marker_handle = marker_file.open()
            comparison = csv.DictReader(marker_handle, delimiter="\t")
            next_common = next(comparison, None)
        try:
            with material.query_rows(tool, src[kind + "_vcf"]["path"], sample_file, src["contig"],
                                     [1, src["length_bp"]], kind == "rare", out / f"{kind}.bcftools.log",
                                     targets=targets if kind == "common" else None) as rows:
                for row in rows:
                    counts["source_rows"] += 1
                    require(len(row) == (6+n if kind == "common" else 9+2*n), "Source column count differs")
                    pos, eligible = material.basic_site(row, src, previous); previous = pos
                    if kind == "rare":
                        require(eligible, "Rare row failed original-biallelic PASS SNV contract")
                    if not eligible:
                        counts["ineligible_rows"] += 1
                        continue
                    require(pos > eligible_previous and row[2] != row[3], "Duplicate or invalid eligible SNV")
                    eligible_previous = pos
                    if kind == "common":
                        gt = material.decode_gt(row[6:])
                        if cached_rows is None:
                            keep, freq = material.common_eligible(gt, roles["FIT"], contract["common"])
                        else:
                            freq = fit_cache.check_genotype(row, next(cached_rows, None), gt, roles["FIT"], contract["common"])
                            keep = True
                        if not keep:
                            counts["failed_fit_selection"] += 1
                            continue
                        observed = np.isfinite(gt); dose = np.where(observed, gt, 255).astype(np.uint8)
                        allele, an = 1, int(2*observed[roles["FIT"]].sum())
                        ac = int(gt[roles["FIT"]][observed[roles["FIT"]]].sum())
                    else:
                        dose, observed, allele, ac, an = decode_rare(row, n, contract); freq = ac/an
                        while next_common is not None and int(next_common["position_bp"]) < pos:
                            next_common = next(comparison, None)
                        require(next_common is None or int(next_common["position_bp"]) != pos,
                                "Selected common locus overlaps fixed rare catalogue; cannot isolate rare channel")
                    if not points[0][0] <= pos <= points[-1][0]:
                        counts["outside_map"] += 1
                        continue
                    if kind == "common":
                        marker_handle.write(f"{pos}\t{row[2]}\t{row[3]}\n")
                    counts["retained_sites"] += 1
                    counts["observed_genotypes"] += int(observed.sum())
                    counts["missing_genotypes"] += int((~observed).sum())
                    writer.add(dict(position_bp=pos, position_cm=maps.interpolate_cm(points, pos),
                        ref=BASES.index(row[2]), alt=BASES.index(row[3]), counted_allele=allele,
                        allele_count=ac, allele_number=an, frequency=freq, dosage=dose, observed=observed))
            if kind == "common" and cached_rows is not None:
                require(next(cached_rows, None) is None, "Source omitted cached FIT loci")
            writer.flush(); channels[kind] = writer.chunks; audit[kind] = dict(counts)
        finally:
            marker_handle.close()
    material.assert_source_unchanged(receipt["source"], contract, chromosome)
    require(material.signature(desc["path"]) == receipt["map_signature"], "Map changed during extraction")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    require(total <= cfg["resources"]["max_output_mb"]*2**20, "Total chromosome output exceeds budget")
    result = dict(schema=CHROM_SCHEMA, stage="COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING", chromosome=chromosome,
        event_contract_sha256=expected_sha, source_contract_sha256=cfg["materialize_contract_sha256"],
        code_sha256=code_hashes(), n_people=n, role_counts={k:len(v) for k,v in roles.items()},
        sample_ids_sha256=material.evidence.sample_hash([p["sample_id"] for p in persons]),
        common_policy=cfg["common_policy"], catalogue_policy="FIXED_LOCUS_GRID_ZERO_DOSES_RETAINED",
        missing_policy="OBSERVED_MASK_AND_UINT8_255_SENTINEL", context_policy="COMPUTED_AT_READ_TIME_WITH_SEPARATE_R_C_R_R",
        raw_contains_targets=True, requires_excluded_window_reader=True, training_allowed=False,
        common_features=list(COMMON_FEATURES), event_features=list(EVENT_FEATURES), rare_dosage_feature_index=0,
        channels=channels, audit=audit, local_intermediate_filesystem=fs,
        outputs_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()})
    material.write_json(out / "manifest.json", result)
    return result


class EventChromosome:
    """Authenticated chromosome cache. Windows are masked before tensor creation.

    The returned dictionaries have the fields of genomic_context_runtime.Sites.
    Callers must pool only core masks, not halos. This API never chooses a model,
    target, genomic tiling scheme, or training/evaluation hyperparameters.
    """
    def __init__(self, directory, manifest_sha256):
        self.root = Path(directory).resolve()
        require(sha(self.root / "manifest.json") == manifest_sha256, "Chromosome manifest SHA256 differs")
        self.manifest = m = load_json(self.root / "manifest.json")
        self.manifest_sha256 = manifest_sha256
        require(m["schema"] == CHROM_SCHEMA and m["stage"] == "COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING"
                and m["training_allowed"] is False and m["code_sha256"] == code_hashes(), "Chromosome schema/code differs")
        self.cfg, self.source, self.persons, self.roles, self.blocks, _ = load_contract(self.root, m["event_contract_sha256"])
        require(m["source_contract_sha256"] == self.cfg["materialize_contract_sha256"] and
                m["n_people"] == len(self.persons) and m["sample_ids_sha256"] ==
                material.evidence.sample_hash([p["sample_id"] for p in self.persons]), "Chromosome cohort differs")
        self.chrom = str(m["chromosome"]); self.length = self.source["sources"][self.chrom]["length_bp"]
        self.people = {p["sample_id"]:i for i,p in enumerate(self.persons)}
        self.r = self.cfg["resources"]
        required = {"event_inputs.json", "source_verification.json", "samples.txt", "map.tsv", "common_loci.tsv",
                    "common.bcftools.log", "rare.bcftools.log", *{"source/"+n for n in ("materialize.json", *material.META)}}
        required |= {c["path"] for channel in m["channels"].values() for c in channel}
        if "fit_caches" in self.cfg:
            required |= {"fit_eligible.tsv", "fit_manifest.json", "fit_contract.json", "common_targets.tsv"}
        require(set(m["outputs_sha256"]) == required, "Chromosome artifact inventory incomplete")
        for name, digest in m["outputs_sha256"].items():
            path = self.safe_path(name)
            require(path.is_file() and (name.endswith(".npz") or sha(path) == digest), "Metadata missing or changed")
        self.points = bounded_map(self.root / "map.tsv", self.chrom, self.length, "tsv_chrom_pos_cm", self.r)
        for kind in ("common", "rare"):
            previous = 0
            for chunk in m["channels"][kind]:
                require(set(chunk) == {"path", "sha256", "n_sites", "bytes", "start_bp", "end_bp"}, "Chunk descriptor fields differ")
                require(chunk["start_bp"] > previous and chunk["end_bp"] >= chunk["start_bp"] and
                        0 < chunk["n_sites"] <= self.r["chunk_sites"] and
                        m["outputs_sha256"].get(chunk["path"]) == chunk["sha256"], "Chunk inventory/order differs")
                previous = chunk["end_bp"]

    def safe_path(self, name):
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe chromosome artifact path")
        path = (self.root / rel).resolve()
        require(path.is_relative_to(self.root), "Chromosome artifact escaped its root")
        return path

    def read(self, kind, ids, lower, upper, excluded):
        columns = {k: [] for k in ARRAY_FIELDS}; n_sites = 0
        for chunk in self.manifest["channels"][kind]:
            if chunk["end_bp"] < lower or chunk["start_bp"] > upper:
                continue
            path = self.safe_path(chunk["path"])
            require(path.stat().st_size == chunk["bytes"] and sha(path) == chunk["sha256"], "Genotype chunk changed")
            with zipfile.ZipFile(path) as archive:
                require(sum(r.file_size for r in archive.infolist())*4 <= self.r["max_array_mb"]*2**20,
                        "Chunk decompression exceeds memory budget")
            with np.load(path, allow_pickle=False) as data:
                require(set(data.files) == ARRAY_FIELDS, "Tensor array inventory differs")
                pos, dose, observed = data["position_bp"], data["dosage"], data["observed"]
                s = chunk["n_sites"]
                require(pos.dtype == np.int64 and pos.shape == (s,) and
                        np.all(np.diff(pos) > 0) and pos[0] == chunk["start_bp"] and pos[-1] == chunk["end_bp"], "Chunk position contract differs")
                require(dose.dtype == np.uint8 and observed.dtype == np.bool_ and
                        dose.shape == observed.shape == (len(self.persons), s) and
                        np.all(dose[observed] <= 2) and np.all(dose[~observed] == 255), "Genotype dosage/mask contract differs")
                allowed = (pos >= lower) & (pos <= upper)
                if excluded is not None:
                    allowed &= ~((pos >= excluded[0]) & (pos <= excluded[1]))
                n_sites += int(allowed.sum())
                require(n_sites <= self.r["max_window_sites"], "Window limit exceeded; no truncation")
                for key in ARRAY_FIELDS:
                    value = data[key]
                    require(value.shape == ((len(self.persons),s) if key in {"dosage","observed"} else (s,)), "Array shape differs")
                    columns[key].append(value[np.ix_(ids, np.flatnonzero(allowed))] if key in {"dosage","observed"} else value[allowed])
                require(np.all(data["ref"] < 4) and np.all(data["alt"] < 4) and np.all(data["ref"] != data["alt"])
                        and np.all(data["counted_allele"] <= 1), "Invalid nucleotide/count identity")
                require(np.all(data["allele_number"] > 0) and np.all(data["allele_count"] >= 0) and
                        np.all(data["allele_count"] <= data["allele_number"]) and
                        np.allclose(data["frequency"],data["allele_count"]/data["allele_number"],rtol=0,atol=1e-12),
                        "Frequency numerator/denominator disagree")
        result = {}
        for key, chunks in columns.items():
            axis = 1 if key in {"dosage", "observed"} else 0
            result[key] = np.concatenate(chunks, axis=axis) if chunks else np.empty((len(ids),0) if axis else (0,))
        return result

    def load_window(self, sample_ids, core_start_bp, core_end_bp, block_id, exclusion_scope,
                    common_radius_cm, rare_radius_cm, attention_depth):
        """Return fixed-locus Sites arrays plus disjoint core masks and B1 counts.

        Radii are in cM. rare_radius_cm=None means no rare-event interaction.
        depth*rR is a conservative input halo for stacked local attention. A
        caller must use at most attention_depth such layers; the common halo
        extends another rC. Excluded sites never enter either halo or counts.
        """
        require(sample_ids and len(sample_ids) == len(set(sample_ids)) and set(sample_ids) <= set(self.people), "Unknown or duplicate batch person")
        require(len(sample_ids) <= self.r["max_batch_people"], "Person batch exceeds budget")
        require(type(core_start_bp) is int and type(core_end_bp) is int and
                self.points[0][0] <= core_start_bp <= core_end_bp <= self.points[-1][0], "Core outside mapped territory")
        require(type(common_radius_cm) in (float,int) and math.isfinite(common_radius_cm) and common_radius_cm > 0,
                "Explicit positive common radius in cM required")
        require(type(attention_depth) is int and attention_depth >= 0, "Invalid attention depth")
        require((rare_radius_cm is None and attention_depth == 0) or
                (type(rare_radius_cm) in (float,int) and math.isfinite(rare_radius_cm) and rare_radius_cm > 0 and attention_depth > 0),
                "Rare interaction radius and depth must agree")
        require(exclusion_scope in {self.source["exclusion_scope"], *self.source.get("additional_exclusion_scopes", [])}, "Unapproved exclusion scope")
        matches = [b for b in self.blocks if b["block_id"] == block_id]
        require(len(matches) == 1, "Unknown target block")
        block = matches[0]; excluded = None
        if block["chromosome"] == self.chrom:
            excluded = [1,self.length] if exclusion_scope == "WHOLE_CHROMOSOME" else [int(block["excluded_start_bp"]), int(block["excluded_end_bp"])]
        # This bounds both channels, padding and feature construction before loading.
        require(len(sample_ids)*self.r["max_window_sites"]*400+self.r["max_map_points"]*256+
                len(self.persons)*self.r["chunk_sites"]*48 <= self.r["max_array_mb"]*2**20, "Window tensor allocation exceeds budget")
        lo, hi = maps.interpolate_cm(self.points, core_start_bp), maps.interpolate_cm(self.points, core_end_bp)
        event_halo = (rare_radius_cm or 0)*attention_depth
        def bounds(halo):
            a,b = max(self.points[0][1],lo-halo),min(self.points[-1][1],hi+halo)
            return math.floor(maps.inverse_cm(self.points,a,"left")), math.ceil(maps.inverse_cm(self.points,b,"right"))
        ids = [self.people[p] for p in sample_ids]
        common = self.read("common", ids, *bounds(event_halo+common_radius_cm), excluded)
        rare = self.read("rare", ids, *bounds(event_halo), excluded)
        result = dict(sample_ids=list(sample_ids), chromosome=self.chrom, block_id=block_id,
            input_lineage=dict(chromosome_manifest_sha256=self.manifest_sha256,
                event_contract_sha256=self.manifest['event_contract_sha256'],
                source_contract_sha256=self.manifest['source_contract_sha256'],code_sha256=code_hashes()),
            person_id=np.asarray(sample_ids), component_id=np.asarray([self.persons[i]["component"] for i in ids]),
            role=np.asarray([self.persons[i]["role"] for i in ids]),
            exclusion_scope=exclusion_scope, core_bp=[core_start_bp,core_end_bp],
            common_radius_cm=common_radius_cm, rare_radius_cm=rare_radius_cm, attention_depth=attention_depth,
            common_feature_names=list(COMMON_FEATURES), event_feature_names=list(EVENT_FEATURES),
            rare_dosage_feature_index=0, excluded_bp=excluded, training_allowed=False)
        for kind,data in (("common",common),("events",rare)):
            s = len(data["position_bp"]); width = len(COMMON_FEATURES if kind == "common" else EVENT_FEATURES)
            if not s:
                features = np.zeros((len(ids),1,width),np.float32); valid = np.zeros((len(ids),1),bool)
                cm = np.full((len(ids),1),lo,np.float64); core = valid.copy()
                result[kind] = dict(features=features,cm=cm,chrom=np.full(cm.shape,int(self.chrom),np.int64),valid=valid)
                result[kind+"_core"] = core
                result[kind+"_pos_bp"] = np.full(cm.shape,core_start_bp,np.int64)
                result[kind+"_padding"] = np.ones(cm.shape,bool)
                continue
            require(np.isfinite(data["position_cm"]).all() and np.isfinite(data["frequency"]).all() and
                    np.all((data["frequency"] >= 0)&(data["frequency"] <= 1)), "Invalid frequency/map coordinates")
            fixed = [data["frequency"]]
            if kind == "events": fixed.append(data["counted_allele"])
            fixed += [np.eye(4,dtype=np.float32)[data["ref"]],np.eye(4,dtype=np.float32)[data["alt"]],
                      data["position_bp"]/self.length,
                      (data["position_cm"]-self.points[0][1])/(self.points[-1][1]-self.points[0][1])]
            fixed = np.column_stack(fixed).astype(np.float32)
            features = np.empty((len(ids),s,width),np.float32)
            features[:,:,0] = np.where(data["observed"],data["dosage"],0)/2
            features[:,:,1:] = fixed[None,:,:]
            valid = data["observed"].astype(bool)
            cm = np.broadcast_to(data["position_cm"],valid.shape).copy()
            core = np.broadcast_to((data["position_bp"] >= core_start_bp) & (data["position_bp"] <= core_end_bp),valid.shape).copy()
            result[kind] = dict(features=features,cm=cm,chrom=np.full(cm.shape,int(self.chrom),np.int64),valid=valid)
            result[kind+"_core"] = core
            result[kind+"_pos_bp"] = np.broadcast_to(data["position_bp"],valid.shape).copy()
            result[kind+"_padding"] = np.zeros(valid.shape,bool)
        mask = result["events_core"] & result["events"]["valid"]; dosage = result["events"]["features"][:,:,0]*2
        result["rare_core_counts"] = np.column_stack(((mask & (dosage>0)).sum(1),mask.sum(1),np.where(mask,dosage,0).sum(1)))
        result["rare_core_count_names"] = ["carrier_sites", "observed_sites", "minor_copies"]
        return result


def export_training_batch(view, pairs, target, path, target_contract_sha256):
    """Export one bounded view to an explicitly supplied supervised task.

    Targets are supplied, never derived here. A receipt is not admission for a
    real fit: an external protocol must justify the task and its targets.
    """
    require(re.fullmatch(r"[0-9a-f]{64}",target_contract_sha256 or ""), "Explicit target-definition contract SHA256 required")
    path = Path(path)
    require(path.suffix == ".npz", "Explicit .npz output required")
    require(not path.exists() and not path.with_suffix(path.suffix+".json").exists(), "Batch output exists")
    require((view["common"]["valid"] & view["common_core"]).any() or
            (view["events"]["valid"] & view["events_core"]).any(), "Entirely unavailable core cannot become training batch")
    require(set(view["role"].tolist()) <= {"FIT", "SELECT"}, "Only development roles may be exported for training")
    pairs, target = np.asarray(pairs), np.asarray(target)
    n = len(view["person_id"])
    require(pairs.dtype.kind in "iu" and pairs.ndim == 2 and pairs.shape[1] == 2 and
            np.all(pairs >= 0) and np.all(pairs < n) and np.all(pairs[:,0] < pairs[:,1]) and
            len(set(map(tuple,pairs.tolist()))) == len(pairs), "Canonical unique in-batch pairs required")
    require(target.shape == (len(pairs),) and np.isfinite(target).all(), "Finite explicit pair targets required")
    require(np.all(view["component_id"][pairs[:,0]] != view["component_id"][pairs[:,1]]) and
            np.all(view["role"][pairs[:,0]] == view["role"][pairs[:,1]]), "Pairs cross roles or repeat family components")
    arrays = {k:np.asarray(view[k]) for k in ("person_id","component_id","role")}
    for source,prefix in (("common","common"),("events","event")):
        for field in ("cm","chrom","valid"):
            arrays[prefix+"_"+field] = view[source][field]
        arrays[prefix+"_pos_bp"] = view[source+"_pos_bp"]
        arrays[prefix+"_core"] = view[source+"_core"]
        arrays[prefix+"_padding"] = view[source+"_padding"]
    arrays["common_features"] = view["common"]["features"]
    arrays["event_metadata"] = view["events"]["features"][:,:,1:]
    arrays["event_dosage"] = view["events"]["features"][:,:,0]*2
    arrays["pairs"], arrays["target"] = pairs.astype(np.int64), target.astype(np.float64)
    np.savez(path, **arrays)
    receipt = dict(schema="r02_genomic_event_training_batch_v1",training_allowed=False,
        target_contract_sha256=target_contract_sha256, arrays_sha256=sha(path), n_people=n,n_pairs=len(pairs),
        block_id=view["block_id"],exclusion_scope=view["exclusion_scope"],core_bp=view["core_bp"],
        common_radius_cm=view["common_radius_cm"],rare_radius_cm=view["rare_radius_cm"],attention_depth=view["attention_depth"],
        input_lineage=view["input_lineage"],excluded_bp=view['excluded_bp'],
        common_features=list(COMMON_FEATURES),event_metadata_features=list(EVENT_FEATURES[1:]),
        missing_policy="VALID_MASK_NOT_CARRIER",target_provenance="EXTERNALLY_SUPPLIED_REQUIRES_TASK_ADMISSION")
    material.write_json(path.with_suffix(path.suffix+".json"),receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--contract-sha256",required=True)
    parser.add_argument("--chromosome",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--verified",type=Path)
    parser.add_argument("--stage",choices=("verify-source","extract"),required=True)
    args = parser.parse_args(argv)
    if args.stage == "verify-source":
        verify_source(args.bundle,args.contract_sha256,args.chromosome,args.output_dir)
    else:
        require(args.verified is not None,"Source verification receipt required")
        extract(args.bundle,args.contract_sha256,args.chromosome,args.verified,args.output_dir)


if __name__ == "__main__":
    main()
