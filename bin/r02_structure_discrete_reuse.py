#!/usr/bin/env python3
"""Authenticate/reuse chromosome rare counts for K03; never learn communities.

Only WHOLE_CHROMOSOME is reconstructible from chromosome sufficient statistics.
Outputs are affinity inputs for ProjectionBundle, not a complete frozen-community
bundle and not permission to fit/evaluate K03. Source-cohort ascertainment remains
explicit; byte integrity cannot establish biological independence.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import r02_structure_discrete as discrete
import r02_structure_affinity_io as affinity_io

material = discrete.materialize
evidence = material.evidence
require, sha, load_json = discrete.require, discrete.sha, discrete.load_json
SCHEMA = "r02_structure_discrete_reuse_v1"
CODE_FILES = ("r02_structure_discrete_reuse.py", *affinity_io.CODE_FILES)
ARRAYS = {"samples", "sample_hash", "chromosomes", "kind", "schema", "I", "U", "Q",
          "individual_carrier", "individual_callable", "n_sites"}
MASK = "complete diploid GT; upstream site filters; not new DP/GQ filtering"
ALLELE = "fixed M02.1 RARE_ALLELE; no reorientation in analytical subset"
METHOD = "exact pairwise jointly called Jaccard over ALL source rare catalogue sites"


def code_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def checked(path, expected, limit):
    path = Path(path)
    require(discrete.valid_hash(expected) and path.is_file() and path.stat().st_size <= limit,
            "Missing/oversized authenticated input")
    before = material.signature(path)
    require(sha(path) == expected and material.signature(path) == before, "Input SHA256 changed: " + str(path))
    return path


def pair_indices(persons, roles, policy):
    return affinity_io.pair_indices(persons, roles, policy)


def audit(bundle, request, request_sha256):
    request = checked(request, request_sha256, 2**20)
    r = load_json(request)
    require(set(r) == {"schema", "admission", "exclusion_scope", "same_component_fit_edges",
                       "producer", "chromosomes", "resources", "implementation_sha256"}, "Exact reuse fields required")
    require(r["schema"] == SCHEMA and r["exclusion_scope"] == "WHOLE_CHROMOSOME",
            "Only WHOLE_CHROMOSOME can be reconstructed; chromosome counts lack site positions")
    require(r["same_component_fit_edges"] in {"INCLUDE", "EXCLUDE"}, "Explicit FIT component policy required")
    require(r["implementation_sha256"] == code_hashes(), "Reuse implementation changed")
    limits = r["resources"]
    require(set(limits) == {"max_input_bytes", "max_memory_bytes", "max_pairs", "max_output_bytes", "min_free_bytes"}
            and all(type(v) is int and v > 0 for v in limits.values()), "Explicit positive resource bounds required")
    base = discrete.inspect(bundle, r["admission"])
    source, persons, roles, blocks, _ = material.load_context(bundle)
    require(source["source_rare_rule"] == {"min_mac": 2, "max_maf": "0.01",
            "ascertainment": "FIXED_SOURCE_COHORT_NOT_FIT_RESELECTED"}, "Producer rare rule differs")
    require(set(r["producer"]) == {"path", "sha256"}, "Pinned historical producer required")
    checked(r["producer"]["path"], r["producer"]["sha256"], r["admission"]["max_metadata_bytes"])
    require(isinstance(r["chromosomes"], dict) and set(r["chromosomes"]) == set(source["sources"]),
            "Exactly one evidence descriptor per source chromosome required")
    ids = [p["sample_id"] for p in persons]
    nfit, nquery = len(roles["FIT"]), len(persons)-len(roles["FIT"])
    max_pairs = nfit*(nfit-1)//2 + nfit*nquery
    require(max_pairs <= limits["max_pairs"], "Pair allocation exceeds resource bound")
    inventory, total, max_raw = {}, 0, 0
    for chrom in sorted(r["chromosomes"], key=int):
        d = r["chromosomes"][chrom]
        require(set(d) == {"manifest", "manifest_sha256", "npz", "npz_sha256"}, "Exact chromosome descriptor required")
        manifest = checked(d["manifest"], d["manifest_sha256"], r["admission"]["max_metadata_bytes"])
        m = load_json(manifest)
        require(m["schema"] == evidence.SCHEMA and m["kind"] == "rare" and m["chromosomes"] == [chrom],
                "Expected single-chromosome rare evidence")
        require(m["n_samples"] == len(persons) and m["samples_sha256"] == evidence.sample_hash(ids),
                "Evidence roster/order differs")
        require(m["source_cohort_n"] == source["source_n_samples"] and
                m["source_cohort_sha256"] == source["source_samples_sha256"] and
                m["input_vcf_sha256"] == source["sources"][chrom]["rare_vcf"]["sha256"],
                "Rare source/cohort differs from K02")
        require(m["code_sha256"] == r["producer"]["sha256"] and
                (m["mask"], m["allele"], m["method"]) == (MASK, ALLELE, METHOD), "Producer semantics differ")
        require(m["output_npz_sha256"] == d["npz_sha256"], "Manifest/NPZ digest disagreement")
        p = Path(d["npz"])
        require(p.is_file(), "Missing chromosome NPZ")
        total += p.stat().st_size + manifest.stat().st_size
        require(total <= limits["max_input_bytes"], "Total input byte budget exceeded")
        checked(p, d["npz_sha256"], limits["max_input_bytes"])
        # Inspect the ZIP directory only; this stage does not decompress matrices.
        with zipfile.ZipFile(p) as z:
            entries = z.infolist()
            require(len(entries) == len(ARRAYS) and {e.filename for e in entries} == {k+".npy" for k in ARRAYS},
                    "Exact array inventory required")
            raw = sum(e.file_size for e in entries)
        max_raw = max(max_raw, raw)
        inventory[chrom] = {**d, "bytes": p.stat().st_size, "uncompressed_bytes": raw}
    bound = 5*max_raw + 48*max_pairs + 4*2**20
    require(bound <= limits["max_memory_bytes"], "Conservative decompression/count/pair budget exceeded")
    recipes = material.input_plan.plan_regions(blocks, {c: v["length_bp"] for c, v in source["sources"].items()},
                                               "WHOLE_CHROMOSOME")
    recipes = [{**v, "included_chromosomes": sorted(set(inventory)-{v["target"][0]}, key=int)} for v in recipes]
    require(all(v["included_chromosomes"] for v in recipes), "Exclusion leaves no input chromosome")
    require(sha(request) == request_sha256 and code_hashes() == r["implementation_sha256"], "Request/code changed")
    result = dict(schema=SCHEMA, stage="AUTHENTICATED_BYTES_RECIPE_NOT_K03_EVALUATION",
        request_sha256=request_sha256, implementation_sha256=code_hashes(), inventory=inventory,
        request_path=str(request.resolve()),
        source_metadata_sha256={str((Path(bundle)/k).resolve()): v for k, v in base["input_sha256"].items()},
        persons_content_sha256=material.digest(persons),
        source_contract_sha256=base["source_contract_sha256"], n_people=len(persons),
        role_counts={role: len(v) for role, v in roles.items()}, recipes=recipes,
        same_component_fit_edges=r["same_component_fit_edges"], compressed_input_bytes=total,
        conservative_memory_bound_bytes=bound, resources=limits,
        training_allowed=False, matrices_decoded=False, communities_learned=False,
        public_release=False, ascertainment="FIXED_SOURCE_COHORT_NOT_FIT_RESELECTED",
        ascertainment_independence_certified=False, projection_support_certified=False)
    return result, persons, roles


def validated_counts(record, chrom, ids):
    p = checked(record["npz"], record["npz_sha256"], record["bytes"])
    with np.load(p, allow_pickle=False) as z:
        require(set(z.files) == ARRAYS and z["samples"].tolist() == ids and
                str(z["sample_hash"]) == evidence.sample_hash(ids) and z["chromosomes"].tolist() == [chrom]
                and str(z["kind"]) == "rare" and str(z["schema"]) == evidence.SCHEMA, "NPZ identity differs")
        nsites = z["n_sites"]
        require(nsites.shape == () and nsites.dtype.kind in "iu" and 0 < int(nsites) <= np.iinfo(np.int64).max,
                "Invalid site denominator")
        nsites = int(nsites)
        arrays = [z[k] for k in ("I", "U", "Q")]
        require(all(a.shape == (len(ids), len(ids)) for a in arrays), "Count matrix/roster dimensions differ")
        evidence.validate_rare_counts(*arrays)
        require(np.all(arrays[2] <= nsites), "Joint calls exceed candidate sites")
        carriers, called = z["individual_carrier"], z["individual_callable"]
        require(carriers.dtype.kind in "iu" and called.dtype.kind in "iu" and
                carriers.shape == called.shape == (len(ids),) and np.all(carriers <= called) and
                np.all(called <= nsites) and np.all(carriers >= 0), "Individual denominators invalid")
        require(np.array_equal(np.diag(arrays[0]), carriers) and np.array_equal(np.diag(arrays[1]), carriers)
                and np.array_equal(np.diag(arrays[2]), called), "Pair/individual diagonal mismatch")
    require(sha(p) == record["npz_sha256"], "NPZ changed while decoding")
    return arrays, nsites


def materialize_affinities(admission, persons, roles, block_id, output):
    require(admission["schema"] == SCHEMA and admission["stage"] == "AUTHENTICATED_BYTES_RECIPE_NOT_K03_EVALUATION"
            and admission["implementation_sha256"] == code_hashes(), "Current byte admission required")
    require(material.digest(persons) == admission["persons_content_sha256"] and
            all(np.array_equal(roles[k], v) for k, v in discrete.shared.validate_roles(persons).items()),
            "Materialization persons/roles differ from admission")
    require(sha(admission["request_path"]) == admission["request_sha256"] and
            all(sha(p) == v for p, v in admission["source_metadata_sha256"].items()), "Admission metadata changed")
    recipes = [r for r in admission["recipes"] if r["block_id"] == block_id]
    require(len(recipes) == 1, "Unknown block")
    recipe = recipes[0]
    included = recipe["included_chromosomes"]
    require(included == sorted(set(admission["inventory"])-{recipe["target"][0]}, key=int), "Excluded chromosome entered recipe")
    output = Path(output)
    require(not output.exists(), "Output exists; preserve completed cache")
    output.mkdir(parents=True, mode=0o700)
    limits = admission["resources"]
    require(shutil.disk_usage(output).free >= limits["min_free_bytes"], "Insufficient initial disk reserve")
    keys = pair_indices(persons, roles, admission["same_component_fit_edges"])
    require(sum(len(k[0]) for k in keys) <= limits["max_pairs"], "Pair allocation budget exceeded")
    totals = [np.zeros((3, len(left)), dtype=np.int64) for left, _ in keys]
    nsites = 0
    for chrom in included:
        arrays, count = validated_counts(admission["inventory"][chrom], chrom, [p["sample_id"] for p in persons])
        require(count <= np.iinfo(np.int64).max-nsites, "Site count integer overflow")
        nsites += count
        for aggregate, (left, right) in zip(totals, keys):
            for k, array in enumerate(arrays):
                aggregate[k] += array[left, right].astype(np.int64)
        del arrays
    affinity_io.write_pair_tables(output, keys, totals, nsites, limits)
    material.input_plan.write_table(output/"persons.tsv", persons)
    exclusion = {k: v for k, v in recipe.items() if k != "included_chromosomes"}
    material.write_json(output/"exclusion.json", exclusion)
    result = {**admission, "stage": "COMPLETE_AFFINITY_INPUTS_COMMUNITIES_AND_COMPARATOR_PENDING",
              "block_id": block_id, "exclusion_scope": "WHOLE_CHROMOSOME", "included_chromosomes": included,
              "candidate_sites": nsites, "matrices_decoded": True,
              "missing_edge_policy": "UNKNOWN_NOT_ZERO", "projection_rule": discrete.PROJECTION_RULE,
              "affinity": {"source_measure": "SOURCE_MINOR_JACCARD", "transformation": "IDENTITY_NONNEGATIVE_INPUT",
                           "definition": "Sum of shared carriage I divided by sum of jointly observed carriage union U; U=0 unknown",
                           "code_sha256": sha(__file__), "maximum": 1.0},
              "sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
              "missing_for_projection_bundle": ["communities.private.tsv", "frozen_communities.json", "community_method.json"],
              "n_fit_edges": len(keys[0][0]), "n_query_edges": len(keys[1][0])}
    require(result["implementation_sha256"] == code_hashes(), "Code changed during materialization")
    require(sha(admission["request_path"]) == admission["request_sha256"] and
            all(sha(p) == v for p, v in admission["source_metadata_sha256"].items()), "Metadata changed during materialization")
    material.write_json(output/"affinity_inputs.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--request", required=True, type=Path)
    p.add_argument("--request-sha256", required=True)
    p.add_argument("--stage", required=True, choices=("audit", "materialize"))
    p.add_argument("--block-id")
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    os.umask(0o077)
    require((a.stage == "materialize") == bool(a.block_id), "Block ID only and always required for materialization")
    result, persons, roles = audit(a.bundle, a.request, a.request_sha256)
    if a.stage == "audit":
        require(not a.output_dir.exists(), "Output exists")
        a.output_dir.mkdir(parents=True, mode=0o700)
        material.write_json(a.output_dir/"manifest.json", result)
    else:
        result = materialize_affinities(result, persons, roles, a.block_id, a.output_dir)
    print(json.dumps({k: result[k] for k in ("schema", "stage", "training_allowed")}))


if __name__ == "__main__":
    main()
