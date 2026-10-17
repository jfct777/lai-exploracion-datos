#!/usr/bin/env python3
"""Measure FIT-only marker support for frozen geometric target intervals.

One authenticated chromosome per Nextflow task. Reuses the materializer's GT
filter and interval-specific PLINK2 pruning; never reads SELECT/EVALUATE GTs
into a table or selects a margin from prediction results. Diagnostic LD is a
bounded, unadjusted sample, not a certificate of independence or rare value.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

import r02_structure_materialize as material

SCHEMA = "r02_structure_region_support_v1"
require, sha = material.require, material.sha


def authenticated(descriptor):
    require(set(descriptor) == {"path", "sha256"}, "Explicit path/SHA256 required")
    path = Path(descriptor["path"])
    require(path.is_absolute() and path.is_file(), "Missing absolute source path")
    require(re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]), "Invalid SHA256")
    before = material.signature(path)
    require(sha(path) == descriptor["sha256"] and material.signature(path) == before,
            "Source digest mismatch or changed while authenticating")
    return path, before


def validate_contract(contract):
    require(contract.get("schema") == SCHEMA, "Wrong support schema")
    require(contract.get("roles_state") == "EXPLICITLY_FROZEN", "Frozen roles required")
    src, resources, common = contract["source"], contract["resources"], contract["common"]
    require(src["chromosome"] in {str(c) for c in range(1, 23)}, "Invalid autosome")
    require(src["genome_build"] == "GRCh38", "Explicit GRCh38 source required")
    require(type(src["length_bp"]) is int and src["length_bp"] > 0, "Invalid contig length")
    require(src["contig"] in {src["chromosome"], "chr" + src["chromosome"]}, "Wrong contig")
    require(src["common_index"]["path"] in {src["common_vcf"]["path"] + s for s in (".tbi", ".csi")},
            "Index must accompany source")
    require(re.fullmatch(r"[0-9a-f]{64}", src["source_samples_sha256"]), "Source sample hash required")
    for key in ("threads", "plink_memory_mb", "max_common_sites", "max_intermediate_mb", "max_array_mb",
                "max_ld_markers", "max_ld_pairs"):
        require(type(resources[key]) is int and resources[key] > 0, "Positive integer resource required: " + key)
    require(resources["plink_memory_mb"] >= 640, "PLINK2 needs >=640 MiB workspace")
    require(resources["max_ld_markers"] <= 256, "Diagnostic LD is capped at 256 markers")
    require(resources["max_ld_markers"] * (resources["max_ld_markers"] - 1) // 2 <= resources["max_ld_pairs"] <= 32640,
            "LD pair budget incompatible with marker cap")
    material.finite_number(resources["min_free_disk_mb"], 0, float("inf"), "disk reserve")
    material.finite_number(common["min_maf"], np.finfo(float).eps, .5, "common MAF")
    material.finite_number(common["max_fit_missing"], 0, 1 - np.finfo(float).eps, "FIT missingness")
    material.finite_number(common["ld_r2"], np.finfo(float).eps, 1 - np.finfo(float).eps, "LD pruning r2")
    require(common["ld_mode"] == "PLINK2_FIT_PER_INTERVAL" and common["ld_step_variants"] == 1,
            "Reuse interval-specific FIT PLINK2 pruning")
    require(type(common["ld_window_kb"]) is int and common["ld_window_kb"] > 0 and common["ld_order"] in (1, 2),
            "Invalid pruning window/order")


def validate_blocks(blocks, src, points, policy):
    require(type(policy["parts"]) is int and policy["parts"] > 0, "Explicit number of target parts required")
    material.finite_number(policy["margin_cm"], 0, float("inf"), "geometric margin")
    require(type(policy["min_target_sites"]) is int and policy["min_target_sites"] > 0, "Positive target support floor required")
    require(len(blocks) == policy["parts"] and len({b["block_id"] for b in blocks}) == len(blocks),
            "All policy target parts required exactly once")
    previous = points[0][0] - 1
    for block in blocks:
        require(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", block["block_id"]), "Unsafe block ID")
        require(str(block["chromosome"]) == src["chromosome"], "Wrong target chromosome")
        a, b, x, y = (block[k] for k in ("start_bp", "end_bp", "excluded_start_bp", "excluded_end_bp"))
        require(all(type(v) is int for v in (a, b, x, y)), "Integer target coordinates required")
        require(points[0][0] <= a <= b <= points[-1][0] and 1 <= x <= a <= b <= y <= src["length_bp"],
                "Target outside map or exclusion outside chromosome")
        require(a == previous + 1, "Target parts must be ordered, disjoint, and gap-free")
        previous = b
    require(previous == points[-1][0], "Targets must cover complete authenticated map domain")
    from r02_structure_regions import build_regions, BLOCK_COLUMNS
    expected = build_regions(points, src["chromosome"], src["length_bp"], policy["parts"], policy["margin_cm"])
    require([{k: b[k] for k in BLOCK_COLUMNS} for b in blocks] ==
            [{k: b[k] for k in BLOCK_COLUMNS} for b in expected],
            "Targets/exclusions do not match authenticated geometric policy")


def cm_at(bp, positions, cms):
    require(positions[0] <= bp <= positions[-1], "No map extrapolation")
    j = bisect_right(positions, bp) - 1
    if j == len(positions) - 1 or positions[j] == bp:
        return cms[j]
    return cms[j] + (cms[j + 1] - cms[j]) * (bp - positions[j]) / (positions[j + 1] - positions[j])


class DiagnosticSample:
    """One nearest-center FIT-eligible site per equal-cM bin; O(cap) GT storage."""
    def __init__(self, minimum, maximum, cap):
        require(maximum > minimum and 1 <= cap <= 256, "Invalid diagnostic genetic grid")
        self.minimum, self.maximum, self.cap = minimum, maximum, cap
        self.rows = {}

    def add(self, cm, position, variant_id, row):
        require(self.minimum <= cm <= self.maximum, "Diagnostic site outside map")
        index = min(self.cap - 1, int((cm - self.minimum) / (self.maximum - self.minimum) * self.cap))
        center = self.minimum + (index + .5) * (self.maximum - self.minimum) / self.cap
        item = (abs(cm - center), position, variant_id, cm, row)
        if index not in self.rows or item[:3] < self.rows[index][:3]:
            self.rows[index] = item

    def selected(self):
        return sorted(self.rows.values(), key=lambda row: (row[1], row[2]))


def table_writer(path, columns):
    handle = Path(path).open("x", newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    return handle, writer


def disk_budget(output, resources):
    require(sum(p.stat().st_size for p in Path(output).rglob("*") if p.is_file()) <= resources["max_intermediate_mb"] * 2**20,
            "Combined chromosome intermediate budget exceeded")
    material.check_disk(output, resources)


def summarize_candidate(candidate, retained, block, people, common, points, sampler, eligible_writer, retained_writer):
    positions, cms = zip(*points)
    n, found, called_min = 0, set(), len(people)
    with Path(candidate).open() as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                require(line.rstrip("\n").split("\t")[9:] == people, "Candidate contains non-FIT or reordered people")
            if line.startswith("#"):
                continue
            row = line.rstrip("\n").split("\t")
            require(len(row) == 9 + len(people), "Candidate genotype dimension mismatch")
            pos = int(row[1]); require(block["start_bp"] <= pos <= block["end_bp"], "Candidate outside target")
            gt = material.decode_gt(row[9:])
            keep, p = material.common_eligible(gt, np.arange(len(people)), common)
            require(keep, "Materializer emitted an ineligible candidate")
            called = int(np.isfinite(gt).sum()); called_min = min(called_min, called)
            cm = cm_at(pos, positions, cms)
            record = dict(block_id=block["block_id"], chromosome=block["chromosome"], position_bp=pos,
                          variant_id=row[2], ref=row[3], alt=row[4], position_cm=format(cm, ".12g"),
                          fit_called=called, fit_an=2*called, fit_alt_af=format(p, ".12g"))
            eligible_writer.writerow(record); n += 1
            if row[2] in retained:
                require(row[2] not in found, "Duplicate retained marker")
                found.add(row[2]); retained_writer.writerow(record)
            sampler.add(cm, pos, row[2], line)
    require(found == retained, "Pruned IDs do not match eligible candidates")
    return dict(candidate_sites=n, retained_sites=len(found), min_fit_calls_per_candidate=called_min if n else 0)


def ld_command(plink, sampled_vcf, prefix, src, resources):
    return [plink, "--vcf", str(sampled_vcf), "--double-id", "--vcf-require-gt", "--vcf-half-call", "missing",
            "--r2-unphased", "--ld-window-r2", "-1", "--ld-window-kb", str(math.ceil(src["length_bp"] / 1000) + 1),
            "--ld-window", str(resources["max_ld_markers"] + 1), "--threads", str(resources["threads"]),
            "--memory", str(resources["plink_memory_mb"]), "--out", str(prefix)]


def annotate_ld(path, output, selected, resources):
    sites = {r[2]: (r[1], r[3]) for r in selected}; seen = set(); finite = 0
    handle, writer = table_writer(output, ["variant_a", "variant_b", "distance_bp", "distance_cm", "unphased_r2", "status"])
    with handle, Path(path).open() as data:
        rows = csv.DictReader(data, delimiter="\t")
        require(rows.fieldnames and {"ID_A", "ID_B", "UNPHASED_R2"} <= set(rows.fieldnames), "Unsupported PLINK2 LD columns")
        for row in rows:
            a, b = row["ID_A"], row["ID_B"]
            require(a in sites and b in sites and a != b, "Wrong/self LD pair")
            key = tuple(sorted((a, b))); require(key not in seen, "Duplicated LD pair")
            seen.add(key); require(len(seen) <= resources["max_ld_pairs"], "Diagnostic LD pair budget exceeded")
            value = float(row["UNPHASED_R2"])
            require(math.isnan(value) or math.isfinite(value) and 0 <= value <= 1 + 1e-7, "Invalid LD r2")
            finite += int(math.isfinite(value))
            writer.writerow(dict(variant_a=a, variant_b=b, distance_bp=abs(sites[a][0]-sites[b][0]),
                                 distance_cm=format(abs(sites[a][1]-sites[b][1]), ".12g"),
                                 unphased_r2="NA" if math.isnan(value) else format(value, ".12g"),
                                 status="UNDEFINED" if math.isnan(value) else "UNADJUSTED_FIT_DESCRIPTIVE"))
    require(len(seen) == len(sites)*(len(sites)-1)//2, "PLINK2 LD report omitted sampled pairs")
    return dict(n_sampled_sites=len(sites), n_pairs=len(seen), n_finite_pairs=finite,
                ancestry_adjusted=False, independence_certified=False, margin_selected=False)


def measure(contract_path, output):
    require(not Path(output).exists(), "Output exists; use completed Nextflow cache, never overwrite")
    contract_path = Path(contract_path)
    contract_hash = sha(contract_path); contract = material.load_json(contract_path)
    validate_contract(contract)
    src, resources, common = contract["source"], contract["resources"], contract["common"]
    descriptors = {**{k: src[k] for k in ("common_vcf", "common_index")},
                   **{k: contract[k] for k in ("persons", "map", "policy")}}
    checked = {k: authenticated(v) for k, v in descriptors.items()}
    persons = material.evaluator.table(checked["persons"][0], ["sample_id", "component", "role"])
    roles = material.evaluator.validate_roles(persons)
    people = [persons[i]["sample_id"] for i in roles["FIT"]]
    require(len(people) >= 50 and all(not any(c.isspace() for c in s) for s in people), "At least 50 valid FIT IDs required for PLINK2 LD")
    require(type(common["min_fit_called"]) is int and 2 <= common["min_fit_called"] <= len(people), "Invalid FIT call minimum")
    require(resources["max_common_sites"]*256 + resources["max_ld_markers"]*len(people)*32 < resources["max_array_mb"]*2**20,
            "Marker IDs/diagnostic sample exceed conservative array budget")
    policy = material.load_json(checked["policy"][0])
    from r02_structure_regions import read_map
    points = read_map(checked["map"][0], src["chromosome"], src["length_bp"], "shapeit3")
    blocks = contract["blocks"]
    validate_blocks(blocks, src, points, policy)
    bcftools, plink = (material.checked_tool(k, contract["tools"][k]) for k in ("bcftools", "plink2"))
    header, source_people = material.header_info(src["common_vcf"]["path"], bcftools)
    require(material.evidence.sample_hash(source_people) == src["source_samples_sha256"], "Source sample/order digest mismatch")
    require({p["sample_id"] for p in persons} <= set(source_people), "Cohort absent from source header")
    require("##dnabr_original_alleles=v1\n" in header and "##INFO=<ID=ORIG_NALLELES," in header,
            "Original-site provenance absent")
    require("##FORMAT=<ID=GT," in header, "GT definition absent")
    contigs = [r for r in header.splitlines() if r.startswith(f"##contig=<ID={src['contig']},")]
    require(len(contigs) == 1 and re.search(r"(?:,|<)length=" + str(src["length_bp"]) + r"(?:,|>)", contigs[0]),
            "Source contig length mismatch")
    out = material.new_directory(output)
    filesystem = material.local_intermediate_filesystem(out); disk_budget(out, resources)
    work = out/"private_work"; work.mkdir(mode=0o700)
    sample_file = work/"fit.samples"; sample_file.write_text("\n".join(people) + "\n")
    sampler = DiagnosticSample(points[0][1], points[-1][1], resources["max_ld_markers"])
    columns = ["block_id", "chromosome", "position_bp", "variant_id", "ref", "alt", "position_cm", "fit_called", "fit_an", "fit_alt_af"]
    eh, ew = table_writer(out/"fit_eligible.tsv", columns); rh, rw = table_writer(out/"retained.tsv", columns)
    supports, pruning = [], []
    with eh, rh:
        for block in blocks:
            directory = work/block["block_id"]; directory.mkdir()
            candidate, retained, detail = material.prepare_common_interval(
                bcftools, plink, src, [block["start_bp"], block["end_bp"]], people, np.arange(len(people)),
                common, resources, directory, sample_file)
            stats = summarize_candidate(candidate, retained, block, people, common, points, sampler, ew, rw)
            require(stats["candidate_sites"] == detail["candidate_sites"], "Candidate count mismatch")
            supports.append(dict(block_id=block["block_id"], chromosome=src["chromosome"], **stats,
                                 support_floor=policy["min_target_sites"], n_fit_people=len(people),
                                 status="FIT_TARGET_SUPPORT_AVAILABLE" if stats["retained_sites"] >= policy["min_target_sites"]
                                 else "INSUFFICIENT_TARGET_SUPPORT"))
            pruning.append(dict(block_id=block["block_id"], **detail)); disk_budget(out, resources)
    selected = sampler.selected(); sampled = work/"diagnostic.vcf"
    with sampled.open("x") as handle:
        handle.write("##fileformat=VCFv4.2\n" + f"##contig=<ID={src['contig']},length={src['length_bp']}>\n" +
                     '##FORMAT=<ID=GT,Number=1,Type=String,Description="Complete FIT diploid hard calls">\n' +
                     "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(people) + "\n")
        for row in selected:
            handle.write(row[4])
    ld_path = out/"ld_diagnostic.tsv"; command = None
    if len(selected) >= 2:
        prefix = work/"diagnostic"; command = ld_command(plink, sampled, prefix, src, resources)
        with (out/"ld_tool.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        require(result.returncode == 0, "PLINK2 diagnostic failed; see ld_tool.log")
        ld = annotate_ld(str(prefix) + ".vcor", ld_path, selected, resources)
    else:
        handle, _ = table_writer(ld_path, ["variant_a", "variant_b", "distance_bp", "distance_cm", "unphased_r2", "status"])
        handle.close(); ld = dict(n_sampled_sites=len(selected), n_pairs=0, n_finite_pairs=0,
                                 status="INSUFFICIENT_DIAGNOSTIC_SITES", ancestry_adjusted=False,
                                 independence_certified=False, margin_selected=False)
    material.input_plan.write_table(out/"target_support.tsv", supports)
    disk_budget(out, resources)
    require(sha(contract_path) == contract_hash, "Contract changed during execution")
    for path, before in checked.values():
        require(material.signature(path) == before, "Authenticated source changed during execution")
    public = ["fit_eligible.tsv", "retained.tsv", "target_support.tsv", "ld_diagnostic.tsv"]
    if command: public.append("ld_tool.log")
    result = dict(schema=SCHEMA, stage="FIT_SUPPORT_MEASURED_NO_BIOLOGICAL_VALIDATION", chromosome=src["chromosome"],
                  contract_sha256=contract_hash, source_sha256={k:v["sha256"] for k,v in descriptors.items()},
                  n_fit_people=len(people), fit_ids_sha256=material.evidence.sample_hash(people),
                  genotype_roles_read=["FIT"], training_allowed=False, excludes_unmapped_territory=True,
                  supports=supports, common=common, policy=policy, pruning=pruning, ld=ld,
                  ld_command=command, resources=resources, filesystem=filesystem,
                  tools=contract["tools"], code_sha256={**material.code_hashes(), Path(__file__).name:sha(__file__),
                      "r02_structure_regions.py":sha(Path(__file__).with_name("r02_structure_regions.py"))},
                  publish_files={name:sha(out/name) for name in public},
                  limitations=["No SELECT/EVALUATE genotypes or rare variants were analyzed.",
                               "FIT support does not certify pairwise evaluation support or statistical power.",
                               "Geometric margins are not selected or validated by this sparse LD diagnostic.",
                               "At most 256 sites cannot detect all local or long-range LD regions.",
                               "PLINK founders are computational pedigree status, not certified unrelated people."])
    material.write_json(out/"manifest.json", result)
    return result


def export_public(output, public_path):
    """Export only aggregate/marker tables; keep FIT VCF/PGEN in task storage."""
    source = Path(output)
    result = material.load_json(source/"manifest.json")
    require(result.get("schema") == SCHEMA and not result.get("training_allowed", True), "Invalid support manifest")
    allowed = {"fit_eligible.tsv", "retained.tsv", "target_support.tsv", "ld_diagnostic.tsv", "ld_tool.log"}
    require(set(result["publish_files"]) <= allowed and
            {"fit_eligible.tsv", "retained.tsv", "target_support.tsv", "ld_diagnostic.tsv"} <= set(result["publish_files"]),
            "Unexpected/missing publication artifact; private genotypes must stay local")
    hashes = {**result["publish_files"], "manifest.json": sha(source/"manifest.json")}
    require(all(sha(source/name) == digest for name, digest in hashes.items()), "Source publication digest mismatch")
    out = material.new_directory(public_path)
    for name, digest in hashes.items():
        shutil.copyfile(source/name, out/name)
        require(sha(out/name) == digest, "Exported publication digest mismatch")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--public-output")
    args = parser.parse_args(); os.umask(0o077)
    measure(args.contract, args.output)
    if args.public_output:
        export_public(args.output, args.public_output)


if __name__ == "__main__":
    main()
