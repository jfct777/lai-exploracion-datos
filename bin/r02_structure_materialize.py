#!/usr/bin/env python3
"""Stream FIT-selected common genotypes and source-minor rare pair evidence.

Produces prepared comparator inputs, never fits/evaluates a prediction model.
Chromosome/interval partials are shared across target blocks when identical.
The block index stores independently authenticated shards instead of requiring
an autosomal genotype matrix in memory. Source staging is chromosome-local.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

import r02_genomic_pair_evidence as evidence
import r02_structure_increment as evaluator
import r02_structure_input_plan as input_plan

require, sha, load_json = evaluator.require, evaluator.sha, evaluator.load_json
SCHEMA = "r02_structure_materialize_v1"
PARTIAL = "r02_structure_materialize_partial_v1"
KINDS = ("features", "target")
META = ("persons.tsv", "blocks.tsv", "protocol.md")
BLOCK_INDEX = "r02_structure_block_index_v1"
PREPARED_BLOCK = "r02_structure_prepared_block_v1"
CODE_FILES = ("r02_structure_materialize.py", "r02_structure_increment.py",
              "r02_structure_input_plan.py", "r02_genomic_pair_evidence.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def code_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")


def new_directory(path):
    path = Path(path)
    require(not path.exists(), "Output exists; use completed Nextflow cache, never overwrite")
    path.mkdir(parents=True, mode=0o700)
    return path


def signature(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def check_disk(path, settings):
    require(shutil.disk_usage(path).free >= settings["min_free_disk_mb"]*2**20, "Local free-space reserve exhausted")


def local_intermediate_filesystem(path):
    fs=subprocess.check_output(["findmnt","-n","-o","FSTYPE","--target",str(path)],text=True).strip()
    require(fs in {"ext2","ext3","ext4","xfs","btrfs","overlay"},
            "Intermediates require verified local/block storage, not FUSE, network or tmpfs")
    return fs


def finite_number(value, lower, upper, name):
    require(type(value) in (int, float) and np.isfinite(value) and lower <= value <= upper, f"Invalid {name}")


def load_context(bundle):
    bundle = Path(bundle)
    contract = load_json(bundle/"materialize.json")
    require(contract.get("schema") == SCHEMA, "Unsupported materialization contract")
    require(set(contract["metadata_sha256"]) == set(META), "Exact metadata hashes required")
    for name in META:
        require(sha(bundle/name) == contract["metadata_sha256"][name], f"Metadata SHA256 mismatch: {name}")
    persons = evaluator.table(bundle/"persons.tsv", ["sample_id", "component", "role"])
    roles = evaluator.validate_roles(persons)
    require(all(not any(c.isspace() for c in p["sample_id"]) for p in persons), "Sample IDs cannot contain whitespace")
    require(contract["roles_state"] in {"PROPOSED_ROLES_NOT_FROZEN", "EXPLICITLY_FROZEN"}, "Explicit role state required")
    require(isinstance(contract.get("role_decision_reference"), str) and len(contract["role_decision_reference"]) >= 10,
            "Role decision/proposal reference required")
    require(contract["genotype_policy"] == "PASS_ORIGINALLY_BIALLELIC_COMPLETE_DIPLOID_GT_NO_NEW_DP_GQ",
            "Genotype policy must be explicit")
    require(contract["rare_allele_convention"] == "SOURCE_MINOR_M02_1", "Source-minor convention required")
    rare_rule=contract["source_rare_rule"]
    require(type(rare_rule["min_mac"]) is int and rare_rule["min_mac"]>=1, "Explicit source MAC rule required")
    require(isinstance(rare_rule["max_maf"],str) and 0<Fraction(rare_rule["max_maf"])<=Fraction(1,2),
            "Explicit source MAF rule required as exact decimal string")
    require(rare_rule["ascertainment"]=="FIXED_SOURCE_COHORT_NOT_FIT_RESELECTED", "Rare ascertainment must remain declared")
    sources = contract["sources"]
    require(sources and all(c in {str(v) for v in range(1,23)} for c in sources), "Explicit autosomal source map required")
    for chrom, src in sources.items():
        require(src["chromosome"] == chrom and src["genome_build"] == contract["genome_build"], "Source build/chromosome mismatch")
        require(type(src["length_bp"]) is int and src["length_bp"] > 0 and isinstance(src["contig"], str), "Invalid contig definition")
        for kind in ("common_vcf", "common_index", "rare_vcf", "rare_index"):
            d = src[kind]
            require(set(d) == {"path", "sha256"} and Path(d["path"]).is_absolute() and
                    re.fullmatch(r"[0-9a-f]{64}", d["sha256"]), "Explicit absolute source and SHA256 required")
        for vcf, index in (("common_vcf", "common_index"), ("rare_vcf", "rare_index")):
            require(src[index]["path"] in {src[vcf]["path"]+".tbi", src[vcf]["path"]+".csi"}, "Index must match its VCF/BCF")
    blocks = evaluator.table(bundle/"blocks.tsv", input_plan.BLOCK_COLUMNS)
    require(all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", b["block_id"]) for b in blocks), "Unsafe block ID")
    regions = input_plan.plan_regions(blocks, {c:s["length_bp"] for c,s in sources.items()}, contract["exclusion_scope"])
    common, resources = contract["common"], contract["resources"]
    finite_number(common["min_maf"], np.finfo(float).eps, .5, "FIT common MAF")
    finite_number(common["max_fit_missing"], 0, 1-np.finfo(float).eps, "FIT missingness")
    require(type(common["min_fit_called"]) is int and 2 <= common["min_fit_called"] <= len(roles["FIT"]), "Invalid FIT call minimum")
    require(common["ld_mode"] == "PLINK2_FIT_PER_INTERVAL", "Only explicit interval-wise FIT PLINK2 pruning supported")
    require(type(common["ld_window_kb"]) is int and common["ld_window_kb"] > 0 and common["ld_step_variants"] == 1,
            "LD windows are kb; PLINK2 requires step=1")
    finite_number(common["ld_r2"], np.finfo(float).eps, 1-np.finfo(float).eps, "LD r2")
    require(common["ld_order"] in (1,2), "Explicit PLINK2 pruning order required")
    for key in ("chunk_sites", "threads", "plink_memory_mb", "max_array_mb", "max_intermediate_mb", "max_target_sites", "max_common_sites", "max_pairs"):
        require(type(resources[key]) is int and resources[key] > 0, f"Positive integer resource required: {key}")
    require(resources["plink_memory_mb"] >= 640, "PLINK2 needs >=640 MiB workspace")
    finite_number(resources["min_free_disk_mb"], 0, float("inf"), "disk reserve")
    require(len(persons)**2*144 + len(persons)*resources["chunk_sites"]*128 <= resources["max_array_mb"]*2**20,
            "Chromosome accumulators/chunks exceed conservative memory budget")
    require(len(persons)**2*32 + len(persons)*resources["max_target_sites"]*24 <= resources["max_array_mb"]*2**20,
            "Target genotype list, stacking and archive exceed memory budget")
    for role, indices in roles.items():
        require(len(indices)*(len(indices)-1)//2 <= resources["max_pairs"], f"{role} pair allocation exceeds budget")
    require(contract["evaluator"]["max_pairs"] == resources["max_pairs"], "Materializer/evaluator pair limits differ")
    require(contract["evaluator"]["exclusion_scope"] == contract["exclusion_scope"], "Exclusion scope differs")
    require(contract["evaluator"]["scientific_protocol_sha256"] == contract["metadata_sha256"]["protocol.md"], "Scientific protocol hash differs")
    require(type(contract["evaluator"]["max_matrix_mb"]) is int and contract["evaluator"]["max_matrix_mb"] > 0,
            "Prepared-block consumer memory budget required")
    require(contract.get("target_selection", "RUN_FIT_SELECTION") in
            {"RUN_FIT_SELECTION", "AUTHENTICATED_FIT_CACHE_REQUIRED"}, "Unknown target selection policy")
    require(contract.get("additional_exclusion_scopes", []) in ([], ["WHOLE_CHROMOSOME"]),
            "Only the declared whole-chromosome sensitivity is supported")
    return contract, persons, roles, blocks, regions


def make_tasks(contract, regions):
    tasks, membership = {}, {}
    def add(kind, chrom, intervals):
        body = {"kind":kind, "chromosome":chrom, "intervals":intervals}
        name = kind+"_chr"+chrom+"_"+digest(body)[:16]
        tasks[name] = {"task_id":name, **body}
        return name
    for block in regions:
        grouped = {}
        for c,a,b in block["allowed_input_intervals"]:
            grouped.setdefault(c, []).append([a,b])
        feature_tasks = [add("features",c,intervals) for c,intervals in sorted(grouped.items(),key=lambda v:int(v[0]))]
        require(feature_tasks, "Block leaves no input territory")
        c,a,b = block["target"]
        membership[block["block_id"]] = {"features":feature_tasks, "target":add("target",c,[[a,b]])}
    return tasks, membership


def plan(bundle, output):
    contract, persons, roles, blocks, regions = load_context(bundle)
    tasks, membership = make_tasks(contract, regions)
    out = new_directory(output); (out/"tasks").mkdir()
    for name, task in tasks.items():
        write_json(out/"tasks"/(name+".json"), task)
    result = dict(schema=SCHEMA, stage="PLAN", contract_sha256=sha(Path(bundle)/"materialize.json"),
        code_sha256=code_hashes(), tasks=tasks, blocks=membership, regions=regions,
        n_tasks=len(tasks), n_people=len(persons), training_allowed=False)
    write_json(out/"manifest.json", result)
    return result


def checked_tool(name, expected):
    path = shutil.which(name)
    require(path is not None and sha(Path(path).resolve()) == expected["sha256"], f"{name} executable digest mismatch")
    version = subprocess.check_output([path,"--version"], text=True).splitlines()[0].strip()
    require(version == expected["version"], f"{name} version mismatch")
    return path


def header_info(path, bcftools):
    text = subprocess.check_output([bcftools,"view","-h",str(path)], text=True)
    lines = text.splitlines()
    names = [line.split("\t")[9:] for line in lines if line.startswith("#CHROM\t")]
    require(len(names) == 1 and names[0] and len(set(names[0])) == len(names[0]), "Invalid source sample header")
    return text, names[0]


def verify_source(bundle, chrom, output):
    contract, persons, _, _, _ = load_context(bundle)
    require(chrom in contract["sources"], "Unknown source chromosome")
    bcftools = checked_tool("bcftools", contract["tools"]["bcftools"])
    src, checked = contract["sources"][chrom], {}
    for name in ("common_vcf","common_index","rare_vcf","rare_index"):
        d=src[name]; before=signature(d["path"])
        local_intermediate_filesystem(d["path"])
        require(sha(d["path"]) == d["sha256"] and signature(d["path"]) == before, "Source changed or digest mismatch")
        checked[name]={**d,"signature":before}
    for kind in ("common_vcf","rare_vcf"):
        header, samples=header_info(src[kind]["path"],bcftools)
        require(len(samples)==contract["source_n_samples"] and evidence.sample_hash(samples)==contract["source_samples_sha256"], "Source cohort mismatch")
        require({p["sample_id"] for p in persons} <= set(samples), "Analytical sample absent from source")
        require("##dnabr_original_alleles=v1\n" in header and "##INFO=<ID=ORIG_NALLELES," in header, "Original-biallelic provenance missing")
        contigs=[line for line in header.splitlines() if line.startswith(f"##contig=<ID={src['contig']},")]
        require(len(contigs)==1 and re.search(r"(?:,|<)length="+str(src["length_bp"])+r"(?:,|>)",contigs[0]),
                "Source contig length mismatch or undocumented")
        if kind=="rare_vcf":
            require("##dnabr_rare_contract=minor_v1\n" in header and
                    f"##dnabr_rare_cohort_sha256={contract['source_samples_sha256']}\n" in header and
                    f"##dnabr_rare_cohort_n_samples={contract['source_n_samples']}\n" in header, "Rare source contract mismatch")
            require(all(f"##INFO=<ID={key}," in header for key in ("RARE_ALLELE","RARE_AC","RARE_AN")) and
                    "##FORMAT=<ID=RD," in header, "Rare counted allele/dosage header missing")
    result=dict(schema=SCHEMA, stage="SOURCE_VERIFIED",chromosome=chrom, checked=checked,
                contract_sha256=sha(Path(bundle)/"materialize.json"), code_sha256=code_hashes())
    out=new_directory(output); write_json(out/"manifest.json",result)
    return result


def assert_source_unchanged(receipt, contract, chrom):
    require(receipt["schema"]==SCHEMA and receipt["stage"]=="SOURCE_VERIFIED" and receipt["chromosome"]==chrom,
            "Wrong source verification receipt")
    require(set(receipt["checked"])=={"common_vcf","common_index","rare_vcf","rare_index"},
            "Source verification receipt is incomplete")
    for name, old in receipt["checked"].items():
        require({k:old[k] for k in ("path","sha256")}==contract["sources"][chrom][name] and
                signature(old["path"])==old["signature"], "Verified local source changed")


def decode_gt(tokens):
    """Vectorized complete diploid hard calls; missing or partial stays NaN."""
    a=np.asarray(tokens,dtype="U8")
    valid={"0/0":0,"0|0":0,"0/1":1,"1/0":1,"0|1":1,"1|0":1,"1/1":2,"1|1":2}
    missing={"./.",".|.","0/.","./0","1/.","./1","0|.",".|0","1|.",".|1"}
    require(np.isin(a,list(valid)+list(missing)).all(), "Non-diploid, nonbiallelic or malformed GT")
    gt=np.full(len(a),np.nan)
    for token,value in valid.items(): gt[a==token]=value
    return gt


@contextmanager
def query_rows(tool, source, sample_file, contig, interval, rare, log, targets=None):
    fixed="%CHROM\t%POS\t%REF\t%ALT\t%FILTER\t%INFO/ORIG_NALLELES"
    fmt=fixed+("\t%INFO/RARE_ALLELE\t%INFO/RARE_AC\t%INFO/RARE_AN[\t%GT\t%RD]\n" if rare else "[\t%GT]\n")
    command=[tool,"query","--samples-file",str(sample_file),"-r",f"{contig}:{interval[0]}-{interval[1]}","-f",fmt,source]
    if targets is not None:
        command[2:2] = ["-T", str(targets)]
    with Path(log).open("a") as error:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=error,text=True,bufsize=1024*1024)
        try:
            yield (line.rstrip("\n").split("\t") for line in process.stdout)
        except BaseException:
            process.kill(); process.wait(); raise
        finally:
            process.stdout.close()
        require(process.wait()==0, "bcftools query failed; see private tool log")


def common_eligible(gt, fit, settings):
    observed=np.isfinite(gt[fit]); n=int(observed.sum())
    p=float(np.nansum(gt[fit])/(2*n)) if n else 0.
    return (n >= settings["min_fit_called"] and n/len(fit) >= 1-settings["max_fit_missing"] and
            min(p,1-p) >= settings["min_maf"]), p


def basic_site(fields, src, previous):
    require(fields[0]==src["contig"], "Wrong source contig")
    # Decomposed originally multiallelic records can legitimately repeat a
    # coordinate; reject repeated *eligible* sites in the caller instead.
    pos=int(fields[1]); require(pos>=previous and 1<=pos<=src["length_bp"], "Unsorted/out-of-bounds position")
    eligible=fields[4]=="PASS" and fields[5]=="2" and len(fields[2])==len(fields[3])==1 and all(a in "ACGT" for a in fields[2:4])
    return pos,eligible


def prepare_common_interval(tool, plink, src, interval, people, fit, settings, resources, directory, sample_file):
    """Filter on FIT only, then let PLINK2 prune each connected allowed interval."""
    candidate=directory/"candidate.vcf"
    nsites=0; nbytes=0; previous=0; previous_eligible=0; last_id=None
    with candidate.open("x") as out:
        out.write("##fileformat=VCFv4.2\n"+f"##contig=<ID={src['contig']},length={src['length_bp']}>\n"+
                  '##FORMAT=<ID=GT,Number=1,Type=String,Description="Complete diploid hard calls">\n'+
                  "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"+"\t".join(people)+"\n")
        with query_rows(tool,src["common_vcf"]["path"],sample_file,src["contig"],interval,False,directory/"bcftools.log") as rows:
            for fields in rows:
                require(len(fields)==6+len(people), "Common GT column count mismatch")
                pos,eligible=basic_site(fields,src,previous); previous=pos
                if not eligible: continue
                require(pos>previous_eligible,"Duplicate originally biallelic eligible position");previous_eligible=pos
                gt=decode_gt(fields[6:]); keep,_=common_eligible(gt,fit,settings)
                if not keep: continue
                nsites+=1; require(nsites<=resources["max_common_sites"], "Common candidate site limit exceeded")
                last_id=f"{src['chromosome']}:{pos}:{fields[2]}:{fields[3]}"
                genotype=np.take(np.array(["0/0","0/1","1/1","./."]),np.where(np.isfinite(gt),gt,3).astype(int))
                row=f"{src['contig']}\t{pos}\t{src['chromosome']}:{pos}:{fields[2]}:{fields[3]}\t{fields[2]}\t{fields[3]}\t.\tPASS\t.\tGT\t"+"\t".join(genotype)+"\n"
                # Reserve space for candidate VCF plus PLINK files/logs before
                # launching PLINK, not just for one file's bytes.
                nbytes+=len(row); require(nbytes*3<=resources["max_intermediate_mb"]*2**20,"Common intermediate size limit exceeded")
                out.write(row)
                if nsites%4096==0:check_disk(directory,resources)
    if not nsites:return candidate,set(),dict(candidate_sites=0,retained_sites=0,commands=[])
    if nsites==1:
        # PLINK emits no prune list when no marker pair exists; the unique FIT
        # eligible site has no LD competitor and is retained explicitly.
        return candidate,{last_id},dict(candidate_sites=1,retained_sites=1,commands=[],
                                      ld_status="ONE_MARKER_NO_PAIRWISE_LD_DEFINED")
    keepfile=directory/"fit.keep"
    keepfile.write_text("".join(f"{people[i]}\t{people[i]}\n" for i in fit))
    # PLINK's <50-founder refusal remains active; never inject --bad-ld silently.
    prefix=directory/"fit"
    commands=[ [plink,"--vcf",str(candidate),"--double-id","--keep",str(keepfile),"--vcf-require-gt",
                "--vcf-half-call","missing","--make-pgen","erase-phase","erase-dosage",
                "--threads",str(resources["threads"]),"--memory",str(resources["plink_memory_mb"]),"--out",str(prefix)],
               [plink,"--pfile",str(prefix),"--indep-pairwise",str(settings["ld_window_kb"])+"kb","1",str(settings["ld_r2"]),
                "--indep-order",str(settings["ld_order"]),"--threads",str(resources["threads"]),
                "--memory",str(resources["plink_memory_mb"]),"--out",str(directory/"ld")] ]
    for idx,command in enumerate(commands):
        with (directory/f"plink{idx}.log").open("w") as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        require(result.returncode==0,"PLINK2 FIT filtering/pruning failed; see private logs")
        require(sum(p.stat().st_size for p in directory.iterdir() if p.is_file())<=resources["max_intermediate_mb"]*2**20,
                "PLINK interval intermediates exceed disk budget")
        check_disk(directory,resources)
    retained=(directory/"ld.prune.in").read_text().splitlines()
    require(retained and len(set(retained))==len(retained),"Empty or duplicated PLINK retained list")
    return candidate,set(retained),dict(candidate_sites=nsites,retained_sites=len(retained),commands=commands,
        retained_ids_sha256=sha(directory/"ld.prune.in"),fit_ids_sha256=evidence.sample_hash(sorted(people[i] for i in fit)))


class CommonCounts:
    def __init__(self,n):
        self.n=n;self.sites=0;self.numerator=np.zeros((n,n));self.joint=np.zeros((n,n),dtype=np.int64)
        self.zsum=np.zeros(n);self.called=np.zeros(n,dtype=np.int64)
    def add(self,gt,p):
        mask=np.isfinite(gt);z=np.where(mask,(gt-2*p)/np.sqrt(2*p*(1-p)),0.)
        self.numerator+=z@z.T
        m=mask.astype(np.float64);self.joint+=(m@m.T).astype(np.int64)
        self.zsum+=z.sum(axis=1);self.called+=mask.sum(axis=1);self.sites+=gt.shape[1]


def selected_common_rows(path):
    """Sorted streaming join; no chromosome-wide Python marker set."""
    with Path(path).open() as f:
        yield from csv.DictReader(f,delimiter="\t")


def authenticated_fit_cache(descriptor_path, contract, persons, roles, request):
    """Authenticate a completed FIT selection without rerunning interval LD.

    The frozen support result can be reused only for exactly the same target,
    source bytes, FIT people/order, filters and tool executables. A pruning list
    for two separate halves is not substituted for a full-chromosome list.
    """
    descriptor = load_json(descriptor_path)
    require(set(descriptor) == {"manifest", "contract", "retained"}, "Exact FIT cache descriptors required")
    for label, item in descriptor.items():
        require(set(item) == {"path", "sha256"} and Path(item["path"]).is_absolute()
                and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "Invalid FIT cache descriptor")
        require(sha(item["path"]) == item["sha256"], f"FIT cache SHA256 mismatch: {label}")
    recorded = load_json(descriptor["manifest"]["path"])
    source_contract = load_json(descriptor["contract"]["path"])
    chrom = request["chromosome"]
    require(recorded.get("schema") == "r02_structure_region_support_v1"
            and recorded.get("stage") == "FIT_SUPPORT_MEASURED_NO_BIOLOGICAL_VALIDATION"
            and recorded["chromosome"] == chrom and recorded["genotype_roles_read"] == ["FIT"]
            and recorded["training_allowed"] is False, "Wrong FIT cache scope")
    require(recorded["contract_sha256"] == descriptor["contract"]["sha256"]
            and recorded["publish_files"]["retained.tsv"] == descriptor["retained"]["sha256"],
            "FIT cache manifest does not authenticate the supplied files")
    fit_people = [persons[i]["sample_id"] for i in roles["FIT"]]
    require(recorded["fit_ids_sha256"] == evidence.sample_hash(fit_people)
            and recorded["n_fit_people"] == len(fit_people)
            and source_contract["persons"]["sha256"] == contract["metadata_sha256"]["persons.tsv"],
            "FIT cache people/order/roles differ")
    require(recorded["common"] == contract["common"] == source_contract["common"]
            and recorded["tools"] == contract["tools"] == source_contract["tools"],
            "FIT cache filters or tool executables differ")
    source = contract["sources"][chrom]
    for key in ("chromosome", "contig", "length_bp", "genome_build"):
        require(source_contract["source"][key] == source[key], "FIT cache source coordinates differ")
    for key in ("common_vcf", "common_index"):
        require(recorded["source_sha256"][key] == source[key]["sha256"]
                == source_contract["source"][key]["sha256"], "FIT cache source content differs")
    require(source_contract["source"]["source_samples_sha256"] == contract["source_samples_sha256"],
            "FIT cache source cohort differs")
    require(request["kind"] == "target" and len(request["intervals"]) == 1, "FIT cache only applies to one target interval")
    interval = request["intervals"][0]
    matching = [b for b in source_contract["blocks"] if b["chromosome"] == chrom
                and [int(b["start_bp"]), int(b["end_bp"])] == interval]
    require(len(matching) == 1, "No exact FIT cache target interval")
    block_id = matching[0]["block_id"]
    supports = [row for row in recorded["supports"] if row["block_id"] == block_id]
    require(len(supports) == 1 and supports[0]["retained_sites"] > 0, "FIT cache target has no retained markers")
    selected = {}
    with Path(descriptor["retained"]["path"]).open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        require(reader.fieldnames == ["block_id", "chromosome", "position_bp", "variant_id", "ref", "alt",
                                     "position_cm", "fit_called", "fit_an", "fit_alt_af"], "Invalid FIT retained columns")
        previous = 0
        for row in reader:
            if row["block_id"] != block_id:
                continue
            pos = int(row["position_bp"])
            require(row["chromosome"] == chrom and interval[0] <= pos <= interval[1] and pos > previous,
                    "Unsorted or out-of-target FIT marker")
            previous = pos
            key = f"{chrom}:{pos}:{row['ref']}:{row['alt']}"
            require(row["variant_id"] == key and key not in selected, "Ambiguous FIT marker alleles/key")
            require(len(selected) < contract["resources"]["max_target_sites"], "Cached target exceeds marker limit")
            selected[key] = row
    require(len(selected) == supports[0]["retained_sites"], "FIT retained count differs from manifest")
    return selected, dict(selection_source="AUTHENTICATED_EXISTING_FIT_TARGET",
                          cache_manifest_sha256=descriptor["manifest"]["sha256"],
                          cache_contract_sha256=descriptor["contract"]["sha256"],
                          retained_sha256=descriptor["retained"]["sha256"],
                          block_id=block_id, retained_sites=len(selected), commands=[], ld_recomputed=False)


def prepare_cached_target(tool, src, interval, people, fit, settings, resources,
                          directory, sample_file, selected, audit):
    """Read only selected GT rows; retain and check missingness and FIT frequency."""
    positions = directory/"selected.positions.tsv"
    with positions.open("x") as f:
        for row in selected.values():
            f.write(f"{src['contig']}\t{row['position_bp']}\n")
    candidate = directory/"candidate.vcf"
    found = set(); previous = 0; nbytes = 0
    with candidate.open("x") as out:
        with query_rows(tool, src["common_vcf"]["path"], sample_file, src["contig"], interval,
                        False, directory/"bcftools.log", targets=positions) as rows:
            for fields in rows:
                require(len(fields) == 6 + len(people), "Cached target GT column mismatch")
                pos, eligible = basic_site(fields, src, previous); previous = pos
                key = f"{src['chromosome']}:{pos}:{fields[2]}:{fields[3]}"
                require(eligible and key in selected and key not in found, "Cached target marker lost identity or eligibility")
                found.add(key)
                gt = decode_gt(fields[6:]); keep, p = common_eligible(gt, fit, settings)
                row = selected[key]; called = int(np.isfinite(gt[fit]).sum())
                require(keep and called == int(row["fit_called"]) and 2*called == int(row["fit_an"])
                        and np.isclose(p, float(row["fit_alt_af"]), rtol=0, atol=1e-12),
                        "Cached target FIT genotype support/frequency changed")
                calls = np.take(np.array(["0/0", "0/1", "1/1", "./."]), np.where(np.isfinite(gt), gt, 3).astype(int))
                text = f"{src['contig']}\t{pos}\t{key}\t{fields[2]}\t{fields[3]}\t.\tPASS\t.\tGT\t" + "\t".join(calls) + "\n"
                nbytes += len(text)
                require(nbytes <= resources["max_intermediate_mb"]*2**20, "Cached target intermediate limit exceeded")
                out.write(text)
    require(found == set(selected), "Cached target marker absent from source")
    return candidate, set(selected), audit


def pair_keys(persons,roles):
    components=np.asarray([p["component"] for p in persons]);chunks=[]
    for role in evaluator.ROLES:
        indices=roles[role];a,b=np.triu_indices(len(indices),1);a,b=indices[a],indices[b]
        keep=components[a]!=components[b];chunks.append(np.column_stack((a[keep],b[keep])))
    pairs=np.concatenate(chunks);order=np.argsort(pairs[:,0]*len(persons)+pairs[:,1]);return pairs[order]


def process_slice(bundle,request_path,verified_path,output,fit_cache=None):
    start=time.monotonic();contract,persons,roles,_,regions=load_context(bundle)
    tasks,_=make_tasks(contract,regions);request=load_json(request_path);name=request["task_id"]
    require(tasks.get(name)==request,"Request not generated by this contract")
    chrom=request["chromosome"];src=contract["sources"][chrom];receipt=load_json(verified_path)
    require(receipt["contract_sha256"]==sha(Path(bundle)/"materialize.json") and receipt["code_sha256"]==code_hashes(),"Source receipt implementation/contract mismatch")
    assert_source_unchanged(receipt,contract,chrom)
    tool=checked_tool("bcftools",contract["tools"]["bcftools"]);plink=checked_tool("plink2",contract["tools"]["plink2"])
    out=new_directory(output);fs=local_intermediate_filesystem(out)
    scratch=out/"scratch";scratch.mkdir();check_disk(out,contract["resources"])
    people=[p["sample_id"] for p in persons];sample_file=scratch/"samples.txt";sample_file.write_text("\n".join(people)+"\n")
    n=len(people);r=contract["resources"];pairs=pair_keys(persons,roles);a,b=pairs.T
    common=CommonCounts(n);target_gt=[];target_markers=[];selection=[]
    common_markers_file=out/"common_markers.tsv"
    with common_markers_file.open("x") as marker_file:
        writer=csv.writer(marker_file,delimiter="\t");writer.writerow(["variant_id","chromosome","position_bp","fit_alt_frequency"])
        for index,interval in enumerate(request["intervals"]):
            directory=scratch/f"common_{index}";directory.mkdir()
            if request["kind"] == "target" and fit_cache is not None:
                selected, audit = authenticated_fit_cache(fit_cache, contract, persons, roles, request)
                candidate, retained, audit = prepare_cached_target(tool, src, interval, people, roles["FIT"],
                    contract["common"], r, directory, sample_file, selected, audit)
            else:
                require(request["kind"] != "target" or contract.get("target_selection", "RUN_FIT_SELECTION") !=
                        "AUTHENTICATED_FIT_CACHE_REQUIRED", "Authenticated FIT target cache required; never silently repeat LD")
                candidate,retained,audit=prepare_common_interval(tool,plink,src,interval,people,roles["FIT"],contract["common"],r,directory,sample_file)
            selection.append(audit);batch=[];freq=[];found=set()
            with candidate.open() as f:
                for line in f:
                    if line.startswith("#"):continue
                    row=line.rstrip("\n").split("\t")
                    if row[2] not in retained:continue
                    require(row[2] not in found,"Duplicate retained marker");found.add(row[2])
                    gt=decode_gt(row[9:]);keep,p=common_eligible(gt,roles["FIT"],contract["common"])
                    require(keep,"FIT-selected marker lost eligibility")
                    writer.writerow([row[2],chrom,row[1],format(p,".17g")])
                    if request["kind"]=="target":
                        target_gt.append(gt);target_markers.append(dict(variant_id=row[2],chromosome=chrom,position_bp=row[1],counted_allele="ALT"))
                        require(len(target_gt)<=r["max_target_sites"],"Target marker memory limit exceeded")
                    else:
                        batch.append(gt);freq.append(p)
                        if len(batch)==r["chunk_sites"]:
                            common.add(np.column_stack(batch),np.asarray(freq));batch=[];freq=[]
            require(found==retained,"Pruned marker is absent from candidate source")
            if batch:common.add(np.column_stack(batch),np.asarray(freq))
            # Release only files generated for this finished interval, before
            # starting another interval. Retain recipes, marker list and logs.
            for generated in directory.iterdir():
                if generated.name.endswith((".vcf",".pgen",".pvar",".psam")):generated.unlink()
    if request["kind"]=="target":
        require(target_gt,"No eligible target markers after FIT-only selection")
        arrays=dict(genotypes=np.column_stack(target_gt),sample_ids=np.asarray(people),variant_ids=np.asarray([m["variant_id"] for m in target_markers]))
        input_plan.write_table(out/"markers.tsv",target_markers)
    else:
        rare=evidence.RareCounts(n);doses=np.zeros(n,dtype=np.int64)
        common_rows=selected_common_rows(common_markers_file);next_common=next(common_rows,None);overlap_sites=0
        cr=[];cc=[];mr=[];mc=[];sites=0
        for interval in request["intervals"]:
            previous=0
            with query_rows(tool,src["rare_vcf"]["path"],sample_file,src["contig"],interval,True,scratch/"rare.bcftools.log") as rows:
                for row in rows:
                    require(len(row)==9+2*n,"Rare GT/RD column count mismatch")
                    pos,eligible=basic_site(row,src,previous);require(pos>previous,"Duplicate rare position");previous=pos
                    require(eligible,"Rare source site failed PASS/original-biallelic SNV contract")
                    while next_common is not None and int(next_common["position_bp"])<pos:
                        next_common=next(common_rows,None)
                    if next_common is not None and int(next_common["position_bp"])==pos:
                        require(next_common["variant_id"]==f"{chrom}:{pos}:{row[2]}:{row[3]}",
                                "Common/rare overlap has inconsistent alleles")
                        overlap_sites+=1
                    allele,ac,an=map(int,row[6:9]);rule=contract["source_rare_rule"];max_maf=Fraction(rule["max_maf"])
                    require(allele in (0,1) and ac>=rule["min_mac"] and an>0 and ac*max_maf.denominator<=an*max_maf.numerator,
                            "Invalid source-minor allele/frequency: source rule is checked, not refit on analytical people")
                    gt=decode_gt(row[9::2]);rd=np.asarray(row[10::2]);missing=~np.isfinite(gt)
                    require(np.array_equal(rd==".",missing),"RD/GT missingness disagreement")
                    observed=np.zeros(n,dtype=np.int64);observed[~missing]=rd[~missing].astype(np.int64)
                    expected=gt if allele==1 else 2-gt
                    require(np.array_equal(observed[~missing],expected[~missing]),"RD disagrees with fixed source allele and GT")
                    carried=np.flatnonzero((observed>0)&~missing);unknown=np.flatnonzero(missing)
                    cr.extend(carried);cc.extend([sites]*len(carried));mr.extend(unknown);mc.extend([sites]*len(unknown));doses+=observed;sites+=1
                    if sites==r["chunk_sites"]:
                        rare.add_sparse(cr,cc,mr,mc,sites);cr=[];cc=[];mr=[];mc=[];sites=0;check_disk(out,r)
        if sites:rare.add_sparse(cr,cc,mr,mc,sites)
        common_rows.close()
        counts=rare.finish()
        arrays=dict(pairs=pairs,common_numerator=common.numerator[a,b],common_joint=common.joint[a,b],
            common_zsum=common.zsum,common_called=common.called,common_sites=np.asarray(common.sites,dtype=np.int64),
            rare_I=counts["I"][a,b],rare_U=counts["U"][a,b],rare_Q=counts["Q"][a,b],
            rare_sites=counts["n_sites"],rare_carriers=counts["individual_carrier"],rare_called=counts["individual_callable"],
            rare_copies=doses,common_rare_overlap_sites=np.asarray(overlap_sites,dtype=np.int64),sample_ids=np.asarray(people))
    assert_source_unchanged(receipt,contract,chrom);check_disk(out,r)
    np.savez(out/"data.npz",**arrays)
    write_json(out/"selection.json",selection)
    # Only generated intermediates under this new private output are removed.
    # Their recipe, selected marker IDs/frequencies, and tool logs are retained.
    for directory in sorted(scratch.iterdir()):
        if directory.is_dir():
            for path in directory.iterdir():
                if path.name.endswith((".vcf",".pgen",".pvar",".psam")):path.unlink()
    result=dict(schema=PARTIAL,stage="COMPLETE_SLICE",request=request,contract_sha256=sha(Path(bundle)/"materialize.json"),
        code_sha256=code_hashes(),source_receipt_sha256=sha(verified_path),source_intervals=[[chrom,*v] for v in request["intervals"]],
        sample_ids_sha256=evidence.sample_hash(people),fit_ids_sha256=evidence.sample_hash(sorted(people[i] for i in roles["FIT"])),
        selection_used="FIT_ONLY",evaluated_outcomes_for_selection=False,roles_state=contract["roles_state"],
        training_allowed=False,local_intermediate_filesystem=fs,elapsed_seconds=time.monotonic()-start,
        outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write_json(out/"manifest.json",result);return result


def ratio(num,den):
    out=np.full(np.shape(num),np.nan,dtype=float);np.divide(num,den,out=out,where=np.asarray(den)>0);return out


def load_partial(path,expected,contract_hash,people):
    path=Path(path);record=load_json(path/"manifest.json")
    require(record["schema"]==PARTIAL and record["stage"]=="COMPLETE_SLICE" and record["request"]==expected and
            record["contract_sha256"]==contract_hash and record["code_sha256"]==code_hashes(),"Partial scope/code mismatch")
    require(record["sample_ids_sha256"]==evidence.sample_hash(people) and record["selection_used"]=="FIT_ONLY", "Partial samples/selection mismatch")
    expected_files={"data.npz","selection.json","common_markers.tsv"} | ({"markers.tsv"} if expected["kind"]=="target" else set())
    require(set(record["outputs_sha256"])==expected_files,"Partial artifact inventory is incomplete")
    for name,h in record["outputs_sha256"].items():
        require(Path(name).name==name and sha(path/name)==h,"Partial artifact hash mismatch")
    with np.load(path/"data.npz",allow_pickle=False) as z:
        require(z["sample_ids"].tolist()==people,"Partial sample order mismatch")
        return {k:z[k] for k in z.files if k!="sample_ids"},record


def assembly_scope(contract, blocks, regions, block_id=None, exclusion_scope=None):
    """Select storage scope, preserving the original scientific data contract."""
    selected_scope = exclusion_scope or contract["exclusion_scope"]
    require(selected_scope in [contract["exclusion_scope"], *contract.get("additional_exclusion_scopes", [])],
            "Unregistered exclusion sensitivity")
    if selected_scope != contract["exclusion_scope"]:
        regions = input_plan.plan_regions(blocks, {c:s["length_bp"] for c,s in contract["sources"].items()}, selected_scope)
    if block_id is not None:
        require(block_id in {b["block_id"] for b in blocks}, "Unknown assembly block")
        blocks = [b for b in blocks if b["block_id"] == block_id]
        regions = [r for r in regions if r["block_id"] == block_id]
    return blocks, regions, selected_scope


def assemble(bundle,plan_path,partials,output,block_id=None,exclusion_scope=None):
    contract,persons,roles,blocks,regions=load_context(bundle);chash=sha(Path(bundle)/"materialize.json")
    planned=load_json(plan_path);tasks,membership=make_tasks(contract,regions)
    require(planned["contract_sha256"]==chash and planned["code_sha256"]==code_hashes() and
            planned["tasks"]==tasks and planned["blocks"]==membership,"Plan is not bound to this contract")
    blocks, regions, selected_scope = assembly_scope(contract, blocks, regions, block_id, exclusion_scope)
    scoped_tasks, membership = make_tasks(contract, regions)
    require(set(scoped_tasks) <= set(tasks), "Sensitivity requests an interval not materialized in the original plan")
    people=[p["sample_id"] for p in persons];expected_pairs=pair_keys(persons,roles)
    fit_pair_mask=np.isin(expected_pairs[:,0],roles["FIT"]) & np.isin(expected_pairs[:,1],roles["FIT"])
    locations={}
    for path in partials:
        r=load_json(Path(path)/"manifest.json");name=r["request"]["task_id"]
        require(name in tasks and name not in locations,"Missing/duplicate/unexpected partial task");locations[name]=Path(path)
    require(set(scoped_tasks) <= set(locations),"Partial task coverage incomplete")
    if block_id is None and exclusion_scope is None:
        require(set(locations)==set(tasks),"Partial task coverage incomplete")
    n=len(people);budget=contract["resources"]["max_array_mb"]*2**20
    require(len(blocks)*len(expected_pairs)*256 + n*contract["resources"]["max_target_sites"]*32 < budget,
            "Assembled archives exceed conservative memory admission")
    out=new_directory(output);features={"sample_ids":np.asarray(people)};paired={"sample_ids":np.asarray(people)}
    lineage={};pair_lineage={};all_gt=[];markers=[];support={};sources={}
    fit_hash=evidence.sample_hash(sorted(people[i] for i in roles["FIT"]))
    def column(name,kind,intervals,derivation,ispair=False):
        row=dict(id=name,kind=kind,source_intervals=intervals,fit_sample_ids_sha256=fit_hash,derivation=derivation)
        if ispair:row["symmetry"]="PAIR_ORDER_INVARIANT"
        return row
    for region in regions:
        name=region["block_id"];totals=None
        for task_id in membership[name]["features"]:
            data,record=load_partial(locations[task_id],tasks[task_id],chash,people)
            require(np.array_equal(data.pop("pairs"),expected_pairs),"Partial pair keys differ")
            require(record["source_intervals"]==[[tasks[task_id]["chromosome"],*v] for v in tasks[task_id]["intervals"]],"Partial interval lineage differs")
            sources[task_id]=sha(locations[task_id]/"manifest.json")
            if totals is None:totals={k:v.copy() for k,v in data.items()}
            else:
                require(set(totals)==set(data),"Partial sufficient-statistic schema differs")
                for key in totals:totals[key]+=data[key]
        require(int(totals["common_sites"])>0 and int(totals["rare_sites"])>0,"No common or rare input support")
        I,U,Q=(totals[k] for k in ("rare_I","rare_U","rare_Q"))
        require(np.all((0<=I)&(I<=U)&(U<=Q)) and np.all(Q<=totals["rare_sites"]),"Invalid aggregate I/U/Q")
        J=evidence.masked_jaccard(I,U);C=ratio(totals["common_numerator"],totals["common_joint"])
        rare_fraction=totals["rare_called"]/totals["rare_sites"];common_fraction=totals["common_called"]/totals["common_sites"]
        mean_common=ratio(totals["common_zsum"],totals["common_called"])
        # Zero fill here is paired with explicit individual availability shared by all arms.
        x0=np.column_stack((np.nan_to_num(mean_common,nan=0.),common_fraction,rare_fraction))
        x1=np.column_stack((x0,totals["rare_carriers"],totals["rare_copies"]))
        w0=np.column_stack((C,totals["common_joint"]/totals["common_sites"],Q/totals["rare_sites"]))
        w2=np.column_stack((w0,J));intervals=region["allowed_input_intervals"]
        require(np.isfinite(w2[fit_pair_mask]).any(axis=0).all(),
                "Paired column has no observed support in FIT; no training-ready contract can be emitted")
        l0=[column("common_mean_standardized_dosage","common",intervals,"Observed common standardized dosage mean, FIT allele frequencies; zero only with explicit zero availability"),
            column("common_callable_fraction","measurement",intervals,"Observed complete common GT count divided by FIT-selected input common sites"),
            column("rare_callable_fraction","measurement",intervals,"Observed complete rare GT count divided by fixed source rare input sites")]
        l1=l0+[column("rare_carrier_sites","rare_count",intervals,"Count of evaluable fixed source-minor sites with one or two copies"),
               column("rare_copy_count","rare_count",intervals,"Sum of observed fixed source-minor diploid copies; unknown excluded, availability shared")]
        p0=[column("common_GRM","common",intervals,"Sum of FIT-standardized genotype products divided by jointly observed FIT-selected common markers",True),
            column("common_joint_fraction","measurement",intervals,"Common pairwise jointly observed marker count divided by input common sites",True),
            column("rare_joint_fraction","measurement",intervals,"Rare pairwise jointly observed catalogue site count divided by fixed source rare input sites",True)]
        p2=p0+[column("rare_Jaccard","rare_pattern",intervals,"Shared source-minor carriage I divided by union U among jointly observed rare sites; U=0 is NaN",True)]
        for arm,x,w,lin,plin in (("B0",x0,w0,l0,p0),("B1",x1,w0,l1,p0),("B2",x1,w2,l1,p2)):
            features[f"{name}_{arm}"]=x;paired[f"{name}_{arm}"]=w
            lineage.setdefault(name,{})[arm]=copy.deepcopy(lin);pair_lineage.setdefault(name,{})[arm]=copy.deepcopy(plin)
        paired[f"{name}_pairs"]=expected_pairs
        support[f"{name}_I"]=I;support[f"{name}_U"]=U;support[f"{name}_Q"]=Q
        support[f"{name}_common_rare_overlap_sites"]=totals["common_rare_overlap_sites"]
        support[f"{name}_J_NA_reason"]=np.where(Q==0,1,np.where(U==0,2,0)).astype(np.int8)
        target_id=membership[name]["target"]
        target,_=load_partial(locations[target_id],tasks[target_id],chash,people)
        target_markers=evaluator.table(locations[target_id]/"markers.tsv",["variant_id","chromosome","position_bp","counted_allele"])
        require(target["variant_ids"].tolist()==[m["variant_id"] for m in target_markers],"Target marker order mismatch")
        require(all(m["chromosome"]==region["target"][0] and region["target"][1]<=int(m["position_bp"])<=region["target"][2] for m in target_markers),"Target marker outside block")
        all_gt.append(target["genotypes"]);markers+=target_markers;sources[target_id]=sha(locations[target_id]/"manifest.json")
        require(len(markers)<=contract["resources"]["max_target_sites"],"Total target marker limit exceeded")
        check_disk(out,contract["resources"])
    require(len({m["variant_id"] for m in markers})==len(markers),"Target marker reused across blocks")
    shutil.copyfile(Path(bundle)/"persons.tsv",out/"persons.tsv")
    input_plan.write_table(out/"blocks.tsv", blocks)
    input_plan.write_table(out/"markers.tsv",markers)
    np.savez(out/"targets.npz",genotypes=np.column_stack(all_gt),sample_ids=np.asarray(people),variant_ids=np.asarray([m["variant_id"] for m in markers]))
    np.savez(out/"features.npz",**features);np.savez(out/"pair_features.npz",**paired)
    np.savez(out/"pair_support.private.npz",pairs=expected_pairs,sample_ids=np.asarray(people),**support)
    write_json(out/"lineage.json",lineage);write_json(out/"pair_lineage.json",pair_lineage)
    indicator_scope={name:{column["id"]:("rare_pattern" if column["kind"]=="rare_pattern" else "measurement")
                          for column in arms["B2"]} for name,arms in pair_lineage.items()}
    final=copy.deepcopy(contract["evaluator"])
    final["exclusion_scope"] = selected_scope
    final["schema"] = PREPARED_BLOCK if block_id is not None else "r02_structure_increment_v3"
    verification_contract = {**final, "schema":"r02_structure_increment_v3"}
    validation=dict(schema=evaluator.verification_schema(verification_contract),status="COMPLETE_FEATURE_EXCLUSION_AND_SPLIT_CHECK",
        materializer_code_sha256=sha(__file__),scientific_protocol_sha256=contract["metadata_sha256"]["protocol.md"],
        input_sha256={name:sha(out/name) for name in evaluator.DATA_FILES+evaluator.PAIR_FILES},
        checks={k:True for k in evaluator.verification_checks(verification_contract)},pair_missing_indicator_scope=indicator_scope,
        partial_manifests_sha256=sources,plan_sha256=sha(plan_path),source_contract_sha256=chash,
        verification_scope="Actual allowed-interval-only extraction, FIT-only selection, additive counts, exact nested arms, masked denominators; no biological validation")
    write_json(out/"upstream_verification.json",validation)
    final.update(feature_materializer_verified=True,materializer_code_sha256=sha(__file__),
        materializer_verification_sha256=sha(out/"upstream_verification.json"),
        rare_allele_convention="SOURCE_MINOR_M02_1",unphased_rare_policy="DIPLOID_NO_HAPLOTYPE_ASSIGNMENT",
        model_family="SYMMETRIC_STANDARDIZED_RIDGE",evaluation_claim="CONDITIONAL_PREDICTIVE_INCREMENT_NOT_POPULATIONS",
        pair_missing_policy="FIT_UNWEIGHTED_MEAN_WITH_PER_COLUMN_INDICATOR",
        pair_missing_indicator_scope=indicator_scope,
        training_allowed=False,roles_state=contract["roles_state"],
        common_representation="EXPLICIT_GRM_RELATION_COMPARATOR_NOT_ALL_COMMON_INFORMATION",
        sha256={name:sha(out/name) for name in (*evaluator.DATA_FILES,*evaluator.PAIR_FILES,"upstream_verification.json")})
    write_json(out/"contract.json",final)
    # Runs the existing consumer's admission/lineage checks only; no model fitting.
    if block_id is None:
        evaluator.validate_bundle(out,final,with_pair_features=True)
    else:
        evaluator.validate_prepared_shard(out,final,with_pair_features=True)
    result=dict(schema=SCHEMA,stage="COMPLETE_PREPARED_INPUTS_NO_TRAINING",training_allowed=False,roles_state=contract["roles_state"],
        source_contract_sha256=chash,code_sha256=code_hashes(),n_people=len(persons),n_blocks=len(blocks),n_target_sites=len(markers),
        block_id=block_id,exclusion_scope=selected_scope,
        n_pairs=len(expected_pairs),partial_manifests_sha256=sources,
        common_rare_overlap_sites={region["block_id"]:int(support[region["block_id"]+"_common_rare_overlap_sites"]) for region in regions},
        limitations=["GRM is an explicit common relationship comparator, not every common-genotype representation.",
          "I/U are biological counts reported only in audit; Q is technical evaluability shared in all arms.",
          "J undefined because U=0 is preserved; its indicator enters B1/B2 only, never B0. Thus B2-vs-B1 isolates value beyond availability.",
          "Rare ascertainment remains the declared fixed source cohort, not a new FIT-only discovery catalogue.",
          "Retained FIT-common versus source-rare marker overlap is reported, not assumed absent or silently removed.",
          "No DP/GQ rethresholding, population validation, ancestry-specific assignment or training was performed.",
          "Roles may remain proposed; a valid file contract is not permission to train."],
        outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write_json(out/"manifest.json",result);return result


def partial_locations(path, expected_sha=None):
    """Resolve only authenticated completed partials, never source VCF paths."""
    if expected_sha is not None:
        require(sha(path) == expected_sha, "Partial locations descriptor changed")
    document = load_json(path)
    require(set(document) == {"schema", "partials"} and document["schema"] == "r02_structure_partial_locations_v1",
            "Invalid partial locations descriptor")
    result = []
    for row in document["partials"]:
        require(set(row) == {"path", "manifest_sha256"} and Path(row["path"]).is_absolute(),
                "Absolute partial path and manifest SHA256 required")
        p = Path(row["path"])
        require(sha(p/"manifest.json") == row["manifest_sha256"], "Partial manifest mismatch")
        result.append(p)
    require(result and len(set(result)) == len(result), "Missing or duplicated partial locations")
    return result


def assemble_index(bundle, shards, output, exclusion_scope=None):
    """Publish a complete ordered block index; never concatenate target matrices.

    Every shard keeps the v3 matrix, missingness and lineage semantics. The v4
    container changes storage and reading order, not statistical selection.
    Hyperparameters are deliberately absent unless an external evaluator
    contract was explicitly supplied by the caller for historical fixtures.
    """
    contract, persons, _, blocks, regions = load_context(bundle)
    _, _, selected_scope = assembly_scope(contract, blocks, regions, exclusion_scope=exclusion_scope)
    locations = {}; scope = {}; selected_final = None
    for directory in map(Path, shards):
        receipt = load_json(directory/"manifest.json")
        name = receipt.get("block_id")
        require(name in {b["block_id"] for b in blocks} and name not in locations,
                "Duplicate or unexpected prepared shard")
        require(receipt["schema"] == SCHEMA and receipt["stage"] == "COMPLETE_PREPARED_INPUTS_NO_TRAINING"
                and receipt["source_contract_sha256"] == sha(Path(bundle)/"materialize.json")
                and receipt["code_sha256"] == code_hashes() and receipt["exclusion_scope"] == selected_scope
                and receipt["n_blocks"] == 1 and receipt["training_allowed"] is False,
                "Prepared shard scope/code/contract mismatch")
        require(set(receipt["outputs_sha256"]) == set(evaluator.DATA_FILES + evaluator.PAIR_FILES)
                | {"pair_support.private.npz", "upstream_verification.json", "contract.json"},
                "Prepared shard inventory incomplete")
        for filename, expected in receipt["outputs_sha256"].items():
            require(Path(filename).name == filename and sha(directory/filename) == expected,
                    "Prepared shard artifact changed")
        shard_contract = load_json(directory/"contract.json")
        require(shard_contract["schema"] == PREPARED_BLOCK and shard_contract["training_allowed"] is False,
                "Expected a data-only prepared shard")
        require(sha(directory/"persons.tsv") == contract["metadata_sha256"]["persons.tsv"], "Shard people/roles changed")
        single_block = evaluator.table(directory/"blocks.tsv", input_plan.BLOCK_COLUMNS)
        require(single_block == [b for b in blocks if b["block_id"] == name], "Shard target coordinates changed")
        expected_evaluator = {**contract["evaluator"], "exclusion_scope": selected_scope}
        require(all(shard_contract.get(k) == v for k, v in expected_evaluator.items()), "Shard preparation parameters differ")
        scope.update(shard_contract["pair_missing_indicator_scope"])
        locations[name] = directory
        selected_final = shard_contract
    require(set(locations) == {b["block_id"] for b in blocks}, "Prepared shard coverage incomplete")
    out = new_directory(output); (out/"shards").mkdir()
    rows = []; nsites = 0
    for block in blocks:
        name = block["block_id"]; source = locations[name]; dest = out/"shards"/name
        # A copied publication is self-contained; no links into disposable task
        # storage. Caller performs bucket checksum verification before cleanup.
        source_manifest_sha = sha(source/"manifest.json")
        shutil.copytree(source, dest)
        require(sha(dest/"manifest.json") == source_manifest_sha == sha(source/"manifest.json"),
                "Prepared shard manifest changed while publishing")
        receipt = load_json(dest/"manifest.json")
        for filename, expected in receipt["outputs_sha256"].items():
            require(sha(dest/filename) == expected, "Copied shard failed integrity check")
        rows.append(dict(block_id=name, path=f"shards/{name}", contract_sha256=sha(dest/"contract.json"),
                         manifest_sha256=sha(dest/"manifest.json")))
        nsites += receipt["n_target_sites"]; check_disk(out, contract["resources"])
    for filename in META:
        shutil.copyfile(Path(bundle)/filename, out/filename)
    write_json(out/"block_index.json", dict(schema=BLOCK_INDEX, blocks=rows))
    final = copy.deepcopy(selected_final)
    for key in ("materializer_verification_sha256", "sha256"):
        final.pop(key, None)
    final.update(schema="r02_structure_increment_v4", block_schema=PREPARED_BLOCK,
        pair_missing_indicator_scope=scope, source_contract_sha256=sha(Path(bundle)/"materialize.json"),
        training_allowed=False, evaluation_contract_state="NOT_ATTACHED",
        sha256={name:sha(out/name) for name in (*META, "block_index.json")})
    write_json(out/"contract.json", final)
    result = dict(schema=SCHEMA, stage="COMPLETE_BLOCK_INDEX_NO_TRAINING", training_allowed=False,
        roles_state=contract["roles_state"], source_contract_sha256=sha(Path(bundle)/"materialize.json"),
        code_sha256=code_hashes(), exclusion_scope=selected_scope, n_people=len(persons), n_blocks=len(blocks),
        n_target_sites=nsites, storage="INDEPENDENT_AUTHENTICATED_BLOCK_SHARDS",
        outputs_sha256={name:sha(out/name) for name in (*META, "block_index.json", "contract.json")})
    write_json(out/"manifest.json",result)
    return result


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest="mode",required=True)
    for name in ("plan","verify-source","slice","assemble","assemble-block","assemble-index"):
        p=subs.add_parser(name);p.add_argument("--bundle",required=True,type=Path);p.add_argument("--out",required=True,type=Path)
        if name=="verify-source":p.add_argument("--chromosome",required=True)
        if name=="slice":
            p.add_argument("--request",required=True,type=Path);p.add_argument("--verified",required=True,type=Path)
            p.add_argument("--fit-cache",type=Path)
        if name in ("assemble", "assemble-block"):
            p.add_argument("--plan",required=True,type=Path);p.add_argument("--partials",required=True,nargs="+",type=Path)
        if name=="assemble-block":
            p.add_argument("--block-id",required=True);p.add_argument("--exclusion-scope")
        if name=="assemble-index":
            p.add_argument("--shards",required=True,nargs="+",type=Path);p.add_argument("--exclusion-scope")
    args=parser.parse_args()
    if args.mode=="plan":r=plan(args.bundle,args.out)
    elif args.mode=="verify-source":r=verify_source(args.bundle,args.chromosome,args.out)
    elif args.mode=="slice":r=process_slice(args.bundle,args.request,args.verified,args.out,args.fit_cache)
    elif args.mode=="assemble-block":r=assemble(args.bundle,args.plan,args.partials,args.out,args.block_id,args.exclusion_scope)
    elif args.mode=="assemble-index":r=assemble_index(args.bundle,args.shards,args.out,args.exclusion_scope)
    else:r=assemble(args.bundle,args.plan,args.partials,args.out)
    print(json.dumps({k:r[k] for k in ("schema","stage") if k in r}))


if __name__=="__main__":main()
