#!/usr/bin/env python3
"""Bounded source-minor dosage/ancestry bands, never haplotype imputation.

One authenticated event chunk per invocation. Reuses the existing MSP/FB
reader, retrieving BOTH homologues of EACH person. The principal output is
aggregated identification counts; optional private detail has an explicit cap.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path
import resource
import tempfile
import time
import zipfile

import numpy as np

import r02_ibd_local_ancestry as ancestry
from r02_ibd_compatibility import input_snapshot, require, sha256, verify_unchanged
from r02_genomic_pair_evidence import sample_hash

SCHEMA = "r02_rare_ancestry_identifiability_v1"
IMPLEMENTATION_VERSION = "1.3.0"
FILES = ("ancestry_contract", "event_manifest", "material_contract", "persons",
         "msp", "fb", "crosswalk", "event_chunk")
CODE_FILES = ("r02_rare_ancestry_identifiability.py", "r02_ibd_local_ancestry.py",
              "r02_ibd_compatibility.py", "r02_ibd_prepare.py",
              "r02_genomic_pair_evidence.py", "r02_segment_evidence.py")
FIELDS = {"dosage", "observed", "position_bp", "position_cm", "ref", "alt",
          "counted_allele", "allele_count", "allele_number", "frequency"}
DETAIL = ("sample_id", "chromosome", "position_bp", "ref", "alt", "counted_base",
          "observed", "minor_dosage", "ancestry", "hard_h0", "hard_h1",
          "ancestry_dosage_conditional", "lower", "upper", "status",
          "posterior_h0", "posterior_h1")


def dose_band(dose, observed, hard0, hard1, population):
    """Set identification conditional on two local ancestry hardcalls."""
    if not observed:
        return None, None, None, "MISSING_GT"
    require(dose in (0, 1, 2), "Invalid observed minor dosage")
    if hard0 < 0 or hard1 < 0:
        return None, None, None, "MISSING_ANCESTRY"
    require(hard0 in (0, 1, 2) and hard1 in (0, 1, 2), "Invalid ancestry label")
    a = int(hard0 == population) + int(hard1 == population)
    lo, hi = max(0, int(dose) + a - 2), min(int(dose), a)
    return a, lo, hi, ("POINT_CONDITIONAL" if lo == hi else "AMBIGUOUS_CONDITIONAL")


def joint_state(dose, observed, hard0, hard1):
    """One state per person/locus, rather than three marginal ancestry states."""
    if not observed:
        return "MISSING_GT"
    require(dose in (0, 1, 2), "Invalid observed minor dosage")
    if hard0 < 0 or hard1 < 0:
        return "MISSING_ANCESTRY"
    require(hard0 in (0, 1, 2) and hard1 in (0, 1, 2), "Invalid ancestry label")
    if dose == 0:
        return "NONCARRIER"
    if dose == 2:
        return "COPY_COUNTS_POINT_CONDITIONAL"
    return ("ORIGIN_POINT_CONDITIONAL" if hard0 == hard1
            else "ORIGIN_AMBIGUOUS_CONDITIONAL")


def locate(position, starts, ends):
    index = int(np.searchsorted(starts, position, side="right")) - 1
    return index if index >= 0 and position <= ends[index] else None


def aggregate_chunk(data, starts, ends, hard):
    """Exact bounded array equivalent of the scalar person/locus loop.

    Dtype promotion precedes subtraction: uint8 dosage must never underflow.
    Missing genotypes take priority over missing ancestry, as in dose_band.
    """
    d=data['dosage']; observed=data['observed']; positions=data['position_bp']
    indexes=np.searchsorted(starts,positions,side='right')-1
    covered=(indexes>=0)&(positions<=ends[np.maximum(indexes,0)])
    # Advanced indexing copies only this event chunk, never the whole genome.
    h0=hard[np.maximum(indexes,0),0::2].T.copy()
    h1=hard[np.maximum(indexes,0),1::2].T.copy()
    h0[:,~covered]=-1; h1[:,~covered]=-1
    valid_hard=(h0>=0)&(h1>=0); joint=observed&valid_hard
    require(np.all(np.isin(h0[valid_hard],[0,1,2])) and
            np.all(np.isin(h1[valid_hard],[0,1,2])), 'Invalid ancestry label')
    counts=Counter(); by_dose=Counter(); joint_counts=Counter()
    ancestry_copies=Counter(); lower=Counter(); upper=Counter()
    def add(counter,key,n):
        if n: counter[key]+=int(n)
    missing_gt=int(np.count_nonzero(~observed))
    missing_ancestry=observed&~valid_hard
    add(joint_counts,('MISSING_GT','MISSING_GT'),missing_gt)
    for dose in range(3):
        mask=observed&(d==dose); usable=mask&valid_hard
        add(joint_counts,(str(dose),'MISSING_ANCESTRY'),np.count_nonzero(mask&~valid_hard))
        if dose==1:
            add(joint_counts,('1','ORIGIN_POINT_CONDITIONAL'),np.count_nonzero(usable&(h0==h1)))
            add(joint_counts,('1','ORIGIN_AMBIGUOUS_CONDITIONAL'),np.count_nonzero(usable&(h0!=h1)))
        else:
            add(joint_counts,(str(dose),'NONCARRIER' if dose==0 else 'COPY_COUNTS_POINT_CONDITIONAL'),np.count_nonzero(usable))
    for population,label in enumerate(ancestry.POPULATIONS):
        ad=(h0==population).astype(np.int16)+(h1==population).astype(np.int16)
        lo=np.maximum(0,d.astype(np.int16)+ad-2); hi=np.minimum(d,ad)
        ambiguous=joint&(lo!=hi); point=joint&~ambiguous
        for state,mask in [('MISSING_GT',~observed),('MISSING_ANCESTRY',missing_ancestry),
                           ('AMBIGUOUS_CONDITIONAL',ambiguous),('POINT_CONDITIONAL',point)]:
            add(counts,(label,state),np.count_nonzero(mask))
            if state=='MISSING_GT':add(by_dose,(label,'MISSING_GT',state),missing_gt)
            else:
                for dose in range(3):add(by_dose,(label,str(dose),state),np.count_nonzero(mask&(d==dose)))
        if np.any(joint):
            ancestry_copies[label]=int(ad[joint].sum())
            lower[label]=int(lo[joint].sum());upper[label]=int(hi[joint].sum())
    return counts,by_dose,joint_counts,ancestry_copies,lower,upper


def coordinate_preflight(chunk, msp, chromosome, entry, max_array_mb, *, require_overlap=True, intervals=None):
    """Exact locus/MSP overlap, reading only coordinates, never dosage/mask arrays.

    Positive overlap is necessary for this question, not a coverage-quality
    threshold or a claim that the selected fragment represents the genome.
    """
    with zipfile.ZipFile(chunk) as archive:
        require(sum(x.file_size for x in archive.infolist()) * 4 <= max_array_mb * 2**20,
                "Chunk exceeds memory admission")
    with np.load(chunk, allow_pickle=False) as archive:
        positions = archive["position_bp"]
    require(positions.shape == (entry["n_sites"],) and
            np.issubdtype(positions.dtype, np.integer) and np.all(positions > 0) and
            np.all(np.diff(positions) > 0), "Invalid preflight locus coordinates")
    require(int(positions[0]) == entry["start_bp"] and int(positions[-1]) == entry["end_bp"],
            "Preflight chunk bounds differ")
    if intervals is None:
        intervals = []
        with Path(msp).open() as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.split("\t", 3)
                require(len(fields) == 4 and fields[0] == str(chromosome), "Preflight MSP chromosome differs")
                start, end = int(fields[1]), int(fields[2])
                require(0 < start <= end and (not intervals or start > intervals[-1][1]),
                        "Preflight MSP overlap/order/coordinates invalid")
                intervals.append((start, end))
    else:
        intervals = [(int(w[0]),int(w[1])) for w in intervals]
        require(all(0 < a <= b and (i==0 or a>intervals[i-1][1])
                    for i,(a,b) in enumerate(intervals)), "Preflight reused intervals overlap/order invalid")
    starts = np.asarray([x[0] for x in intervals]); ends = np.asarray([x[1] for x in intervals])
    covered = sum(locate(int(p), starts, ends) is not None for p in positions)
    if require_overlap:
        require(covered > 0, "NO_COORDINATE_OVERLAP: no locus intersects MSP; no genotype arrays read")
    return dict(n_loci=int(len(positions)), loci_in_msp=int(covered),
                loci_outside_msp=int(len(positions)-covered), genotype_arrays_read=False,
                meaning="COORDINATE_ELIGIBILITY_ONLY_NOT_VALID_CALLS_OR_REPRESENTATIVENESS")


def authenticate(desc, directory):
    require(set(desc) == {"path", "sha256"}, "Exact file descriptor required")
    matches = [p for p in Path(directory).rglob(Path(desc["path"]).name) if p.is_file()]
    require(len(matches) == 1, "Input basename missing or ambiguous: " + desc["path"])
    require(sha256(matches[0]) == desc["sha256"], "Input SHA256 mismatch: " + desc["path"])
    return matches[0]


def load_chunk(path, n, entry, rule, source_n, max_array_mb):
    # Bound uncompressed allocations BEFORE loading. np.load rejects pickles.
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == {x + ".npy" for x in FIELDS}, "Chunk array fields differ")
        total = sum(x.file_size for x in archive.infolist())
        require(total * 4 <= max_array_mb * 2**20, "Chunk exceeds memory admission")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in FIELDS}
    nsites = entry["n_sites"]
    require(type(nsites) is int and nsites > 0, "Positive chunk site count required")
    for key, value in data.items():
        require(value.shape == ((n, nsites) if key in ("dosage", "observed") else (nsites,)),
                "Chunk shape differs: " + key)
    d, mask, pos = data["dosage"], data["observed"], data["position_bp"]
    require(mask.dtype == np.bool_ and d.dtype == np.uint8, "Expected bool mask and uint8 dosage")
    require(np.all(d[mask] <= 2) and np.all(d[~mask] == 255), "Dosage/missing sentinel differs")
    require(np.issubdtype(pos.dtype, np.integer) and np.all(pos > 0) and np.all(np.diff(pos) > 0),
            "Positions must be positive, strictly ordered and unique")
    require(int(pos[0]) == entry["start_bp"] and int(pos[-1]) == entry["end_bp"], "Chunk bounds differ")
    for key in ("ref", "alt", "counted_allele", "allele_count", "allele_number"):
        require(np.issubdtype(data[key].dtype, np.integer), "Integer allele fields required")
    require(np.all(data["ref"] < 4) and np.all(data["ref"] >= 0) and
            np.all(data["alt"] < 4) and np.all(data["alt"] >= 0) and
            np.all(data["ref"] != data["alt"]) and np.isin(data["counted_allele"], [0, 1]).all(),
            "Invalid nucleotide or counted allele")
    ac, an = data["allele_count"], data["allele_number"]
    maximum = Fraction(rule["max_maf"])
    require(np.all(ac >= rule["min_mac"]) and np.all(an > 0) and np.all(an <= 2 * source_n) and
            np.all(ac <= an // 2) and np.all(ac * maximum.denominator <= an * maximum.numerator),
            "Source minor AC/AN/frequency violates inherited rule")
    require(np.isfinite(data["frequency"]).all() and
            np.allclose(data["frequency"], ac / an, rtol=0, atol=1e-12), "Source AC/AN frequency mismatch")
    return data


def validate_sources(paths):
    """Authenticate biological source semantics shared by chunk and chromosome runs."""
    anc = json.loads(paths["ancestry_contract"].read_text())
    event = json.loads(paths["event_manifest"].read_text())
    material = json.loads(paths["material_contract"].read_text())
    require(anc["schema"] == ancestry.SCHEMA and anc["genome_build"] == material["genome_build"] == "GRCh38"
            and anc["coordinate_system"] == "1-based-inclusive", "Ancestry/build/coordinates differ")
    require(anc["population_order"] == list(ancestry.POPULATIONS) and anc["fb_midpoint_rule"] == "round_half_to_even",
            "Ancestry order/midpoint rule differs")
    require(0 <= anc["posterior_sum_tolerance"] <= 1e-3, "Invalid inherited numerical posterior tolerance")
    require(event["schema"] == "r02_genomic_event_chromosome_v1" and
            event["stage"] == "COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING" and
            str(event["chromosome"]) == str(anc["chromosome"]), "Event schema/chromosome differs")
    require(event["source_contract_sha256"] == sha256(paths["material_contract"]), "Material contract differs")
    require(material["schema"] == "r02_structure_materialize_v1" and
            material["rare_allele_convention"] == "SOURCE_MINOR_M02_1", "Source minor convention/schema differs")
    require(material["genotype_policy"] == "PASS_ORIGINALLY_BIALLELIC_COMPLETE_DIPLOID_GT_NO_NEW_DP_GQ",
            "Inherited genotype policy differs")
    require(material["metadata_sha256"]["persons.tsv"] == sha256(paths["persons"]), "Frozen persons table differs")
    with paths["persons"].open() as handle:
        people = list(csv.DictReader(handle, delimiter="\t"))
    samples = [p["sample_id"] for p in people]
    require(samples and len(samples) == len(set(samples)) and
            all(p["role"] in ("FIT", "SELECT", "EVALUATE") for p in people), "Structure roster/roles required")
    require(sample_hash(samples) == event["sample_ids_sha256"] == anc["analytical_samples_sha256"] and
            len(samples) == event["n_people"], "Roster order/hash differs")
    for key in ("msp", "fb", "crosswalk"):
        require(anc[key]["sha256"] == sha256(paths[key]), "Ancestry input binding differs: " + key)
    mapping = {}
    with paths["crosswalk"].open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            require(row["source_id"] not in mapping, "Duplicate crosswalk source")
            mapping[row["source_id"]] = row["sample_id"]
    require(len(mapping) == len(samples) and set(mapping.values()) == set(samples), "Crosswalk not bijective")
    return anc, event, material, samples, mapping


def run(contract_path, expected_sha, inputs_dir, output, scratch):
    started = time.monotonic()
    contract_path, output, scratch = map(Path, (contract_path, output, scratch))
    require(sha256(contract_path) == expected_sha, "K04 contract SHA256 mismatch")
    cfg = json.loads(contract_path.read_text())
    require(cfg["schema"] == SCHEMA and set(cfg["inputs"]) == set(FILES), "K04 schema/input inventory differs")
    require(type(cfg["write_detail"]) is bool, "Explicit detail policy required")
    limits = cfg["resources"]
    require(set(limits) == {"max_array_mb", "max_window_matrix_mb", "max_detail_rows", "min_free_mb"}
            and all(type(v) is int and v > 0 for v in limits.values()), "Positive resource limits required")
    paths = {key: authenticate(cfg["inputs"][key], inputs_dir) for key in FILES}
    require(len({p.name for p in paths.values()}) == len(FILES), "Input basenames must be unique")
    code = [Path(__file__).with_name(name) for name in CODE_FILES]
    before = input_snapshot([contract_path, *paths.values(), *code])
    anc, event, material, samples, mapping = validate_sources(paths)
    entries = [row for row in event["channels"]["rare"] if Path(row["path"]).name == paths["event_chunk"].name]
    require(len(entries) == 1 and entries[0]["sha256"] == sha256(paths["event_chunk"]) and
            entries[0]["bytes"] == paths["event_chunk"].stat().st_size, "Chunk not bound to event manifest")
    entry = entries[0]
    prospective_rows = len(samples) * entry["n_sites"] * 3
    require(not cfg["write_detail"] or prospective_rows <= limits["max_detail_rows"], "Detail row cap; no truncation")
    coverage = coordinate_preflight(paths["event_chunk"], paths["msp"], anc["chromosome"],
                                    entry, limits["max_array_mb"])
    data = load_chunk(paths["event_chunk"], len(samples), entry, material["source_rare_rule"],
                      material["source_n_samples"], limits["max_array_mb"])
    require(scratch.is_dir(), "Existing local scratch directory required")
    ancestry.reserve(scratch, limits["min_free_mb"])
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    counts, ancestry_copies, lower, upper = Counter(), Counter(), Counter(), Counter()
    counts_by_dosage, joint_counts = Counter(), Counter()
    detail = (output / "detail.private.tsv").open("x") if cfg["write_detail"] else None
    writer = csv.writer(detail, delimiter="\t") if detail else None
    if writer:
        writer.writerow(DETAIL)
    try:
        with tempfile.TemporaryDirectory(prefix="k04-ancestry-", dir=scratch) as td:
            windows, hard, posterior, window_receipt = ancestry.prepare_windows(
                paths, anc, samples, mapping, Path(td), limits["max_window_matrix_mb"], limits["min_free_mb"])
            starts = np.array([w[0] for w in windows]); ends = np.array([w[1] for w in windows])
            if not writer:
                counts,counts_by_dosage,joint_counts,ancestry_copies,lower,upper = aggregate_chunk(data,starts,ends,hard)
                ancestry.reserve(output, limits["min_free_mb"])
            for locus, pos in enumerate(data["position_bp"] if writer else []):
                index = locate(pos, starts, ends)
                for person, sample in enumerate(samples):
                    h0, h1 = (-1, -1) if index is None else hard[index, 2*person:2*person+2]
                    obs = bool(data["observed"][person, locus]); dose = int(data["dosage"][person, locus])
                    dose_group = str(dose) if obs else "MISSING_GT"
                    joint_counts[(dose_group, joint_state(dose, obs, int(h0), int(h1)))] += 1
                    for a, label in enumerate(ancestry.POPULATIONS):
                        ad, lo, hi, status = dose_band(dose, obs, int(h0), int(h1), a)
                        counts[(label, status)] += 1
                        # D0 point identification is trivial and must not be
                        # mistaken for resolving the ancestry of a rare copy.
                        counts_by_dosage[(label, dose_group, status)] += 1
                        if ad is not None:
                            ancestry_copies[label] += ad; lower[label] += lo; upper[label] += hi
                        if writer:
                            probs = [float("nan")]*2 if index is None else [posterior[index,2*person+h,a] for h in (0,1)]
                            ref, alt = ("ACGT"[int(data[k][locus])] for k in ("ref", "alt"))
                            row = (sample, str(anc["chromosome"]), int(pos), ref, alt,
                                   alt if data["counted_allele"][locus] else ref, int(obs), dose if obs else None,
                                   label, int(h0) if h0 >= 0 else None, int(h1) if h1 >= 0 else None, ad, lo, hi, status,
                                   *[float(p) if np.isfinite(p) else None for p in probs])
                            writer.writerow(["NA" if v is None else v for v in row])
                if locus % 64 == 0:
                    ancestry.reserve(output, limits["min_free_mb"])
    finally:
        if detail:
            detail.close()
    usable = sum(n for (_, state), n in counts.items()
                 if state in ("POINT_CONDITIONAL", "AMBIGUOUS_CONDITIONAL"))
    status = ("CONDITIONAL_IDENTIFICATION_BANDS_NOT_BIOLOGICAL_VALIDATION" if usable
              else "NO_USABLE_JOINT_GENOTYPE_ANCESTRY_SUPPORT")
    result = dict(schema=SCHEMA, implementation_version=IMPLEMENTATION_VERSION,
        status=status, coordinate_preflight=coverage,
        chromosome=str(anc["chromosome"]), n_people=len(samples), n_loci=entry["n_sites"],
        counts=[dict(ancestry=a, status=s, person_locus_ancestry_rows=n) for (a,s),n in sorted(counts.items())],
        counts_by_observed_dosage=[dict(ancestry=a, observed_minor_dosage=d, status=s,
            person_locus_ancestry_rows=n) for (a,d,s),n in sorted(counts_by_dosage.items())],
        dosage_summary_denominator="ALL_PERSON_LOCUS_ROWS_PER_ANCESTRY; CARRIERS_ARE_OBSERVED_D1_PLUS_D2; JOINT_SUPPORT_REQUIRES_TWO_VALID_HARDCALLS",
        joint_counts=[dict(observed_minor_dosage=d, status=s, person_locus_rows=n)
                      for (d,s),n in sorted(joint_counts.items())],
        joint_counts_denominator="ONE_ROW_PER_PERSON_LOCUS_NOT_TRIPLICATED_BY_ANCESTRY; MISSING_GT_THEN_MISSING_ANCESTRY_TAKE_PRIORITY; ABSENT_CATEGORIES_HAVE_ZERO_COUNT",
        jointly_observed_conditional_ancestry_copies=dict(ancestry_copies),
        marginal_copy_sum_lower=dict(lower), marginal_copy_sum_upper=dict(upper),
        bounds_are="MARGINAL_NOT_SIMULTANEOUS_ASSIGNMENTS_OR_CONFIDENCE_INTERVALS",
        ancestry_is="GNOMIX_ESTIMATED_AFR_EUR_NAM_NO_EAS_CATEGORY",
        posterior_policy="SEPARATE_MARGINALS_NO_INDEPENDENCE_PRODUCT_NO_THRESHOLD",
        denominator="OBSERVED_SOURCE_RARE_CATALOGUE_WITH_TWO_ANCESTRY_HARDCALLS_NOT_GENOMIC_CALLABILITY",
        phase_accuracy="NOT_VALIDATED", genotype_policy=material["genotype_policy"],
        no_rephasing=True, no_imputation=True, write_detail=cfg["write_detail"], windows=window_receipt,
        seconds=time.monotonic()-started, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    verify_unchanged([contract_path, *paths.values(), *code], before)
    manifest = dict(schema=SCHEMA, contract_sha256=expected_sha, sources=before,
                    output_sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file()}, resources=limits)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("contract", "contract-sha256", "inputs-dir", "output-dir", "scratch"):
        p.add_argument("--"+key, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.contract, args.contract_sha256, args.inputs_dir, args.output_dir, args.scratch)))


if __name__ == "__main__":
    main()
