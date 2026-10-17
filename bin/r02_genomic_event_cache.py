#!/usr/bin/env python3
"""Authenticate and stream unpruned FIT loci for individual genotype extraction.

The upstream FIT catalogue spans the mapped chromosome. It is not the pruned
target marker list. Source alleles and FIT frequencies are checked again when
reading the selected genotypes; SELECT/EVALUATE never choose eligibility.
"""
import csv
from pathlib import Path

import r02_structure_materialize as material

require, sha = material.require, material.sha


def authenticate(descriptors, contract, persons, roles, chromosome, map_sha256):
    require(set(descriptors) == {"manifest", "contract", "eligible"}, "Exact FIT eligibility descriptors required")
    for name, item in descriptors.items():
        require(set(item) == {"path", "sha256"} and Path(item["path"]).is_absolute()
                and sha(item["path"]) == item["sha256"], "FIT eligibility file differs: " + name)
    m = material.load_json(descriptors["manifest"]["path"])
    c = material.load_json(descriptors["contract"]["path"])
    require(m["schema"] == "r02_structure_region_support_v1" and
            m["stage"] == "FIT_SUPPORT_MEASURED_NO_BIOLOGICAL_VALIDATION" and
            m["chromosome"] == chromosome and m["genotype_roles_read"] == ["FIT"] and
            m["training_allowed"] is False and m["excludes_unmapped_territory"] is True,
            "FIT eligibility scope differs")
    require(m["contract_sha256"] == descriptors["contract"]["sha256"] and
            m["publish_files"]["fit_eligible.tsv"] == descriptors["eligible"]["sha256"],
            "FIT manifest does not authenticate eligibility")
    fit = [persons[i]["sample_id"] for i in roles["FIT"]]
    require(m["fit_ids_sha256"] == material.evidence.sample_hash(fit) and
            m["n_fit_people"] == len(fit) and
            c["persons"]["sha256"] == contract["metadata_sha256"]["persons.tsv"], "FIT people/roles/order differ")
    require(m["common"] == c["common"] == contract["common"] and
            m["tools"] == c["tools"] == contract["tools"], "FIT filters/tools differ")
    source = contract["sources"][chromosome]
    require(all(c["source"][k] == source[k] for k in ("chromosome", "contig", "length_bp", "genome_build")),
            "FIT source coordinates differ")
    for k in ("common_vcf", "common_index"):
        require(m["source_sha256"][k] == c["source"][k]["sha256"] == source[k]["sha256"], "FIT source content differs")
    require(c["source"]["source_samples_sha256"] == contract["source_samples_sha256"] and
            m["source_sha256"]["map"] == c["map"]["sha256"] == map_sha256, "FIT cohort/map differs")
    return c, sum(int(x["candidate_sites"]) for x in m["supports"])


def rows(path):
    with Path(path).open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["block_id", "chromosome", "position_bp", "variant_id", "ref", "alt",
                                      "position_cm", "fit_called", "fit_an", "fit_alt_af"], "FIT eligibility columns differ")
        yield from reader


def write_targets(path, output, source, blocks, points, expected_count):
    """Check that the cache covers the whole mapped span; never silently subset."""
    spans = sorted((int(b["start_bp"]), int(b["end_bp"])) for b in blocks)
    require(spans and spans[0][0] == points[0][0] and spans[-1][1] == points[-1][0] and
            all(a[1] + 1 == b[0] for a, b in zip(spans, spans[1:])), "FIT cache does not cover complete mapped chromosome")
    block_map = {b["block_id"]: b for b in blocks}
    require(len(block_map) == len(blocks), "Repeated FIT block")
    previous = count = 0
    with Path(output).open("x") as stream:
        for row in rows(path):
            pos = int(row["position_bp"])
            require(row["chromosome"] == source["chromosome"] and pos > previous and
                    row["block_id"] in block_map, "Unsorted/duplicate/wrong FIT eligible locus")
            block = block_map[row["block_id"]]
            require(int(block["start_bp"]) <= pos <= int(block["end_bp"]) and
                    row["variant_id"] == f"{source['chromosome']}:{pos}:{row['ref']}:{row['alt']}", "FIT eligible identity differs")
            stream.write(f"{source['contig']}\t{pos}\n")
            previous = pos
            count += 1
    require(count == expected_count, "FIT eligible catalogue incomplete")
    return count


def check_genotype(row, cached, gt, fit, common):
    require(cached is not None and (row[1], row[2], row[3]) ==
            (cached["position_bp"], cached["ref"], cached["alt"]), "Selected source locus differs from FIT catalogue")
    keep, freq = material.common_eligible(gt, fit, common)
    import numpy as np
    n = int(np.isfinite(gt[fit]).sum())
    require(keep and n == int(cached["fit_called"]) and 2*n == int(cached["fit_an"]) and
            abs(freq - float(cached["fit_alt_af"])) <= 1e-10, "Selected FIT genotypes/frequency differ from catalogue")
    return freq
