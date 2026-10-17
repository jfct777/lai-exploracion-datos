#!/usr/bin/env python3
"""K03 metadata admission, not a clustering or predictive-evaluation engine.

Authenticate the existing K02 preparation contract and describe why its
within-role pair products cannot supply inductive query-to-FIT projection.
The CLI reads no genotype, target tensor, community label, cloud API or model.
The separate sidecar API can project onto authenticated frozen FIT labels; it
cannot learn labels. A successful run NEVER admits real fitting or evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re

import r02_structure_materialize as materialize

shared = materialize.evaluator
require, sha, load_json = shared.require, shared.sha, shared.load_json
SCHEMA = "r02_structure_discrete_admission_v1"
CODE_FILES = ("r02_structure_discrete.py", *materialize.CODE_FILES)
INPUT_FILES = ("materialize.json", *materialize.META)
SCOPES = ("BLOCK_AND_LD_MARGIN", "WHOLE_CHROMOSOME")
CONTRACT_FIELDS = {
    "schema", "evidence_scope", "source_contract_sha256", "implementation_sha256",
    "expected_n_people", "expected_role_counts", "expected_chromosomes",
    "expected_n_targets", "max_metadata_bytes", "projection_route",
}
BLOCKERS = (
    "INDEPENDENT_QUERY_TO_FIT_INPUTS_NOT_SUPPLIED",
    "ADEQUATE_CONTINUOUS_COMPARATOR_NOT_REVIEWED",
    "FIT_ONLY_COMMUNITY_LEARNING_AND_PROJECTION_SUPPORT_POLICY_NOT_REVIEWED",
    "SAME_SUPPORT_CAPACITY_AND_MISSINGNESS_POLICY_NOT_REVIEWED",
    "FROZEN_K03_SELECTION_AND_EVALUATION_CONTRACT_NOT_SUPPLIED",
    "TRAINABLE_K03_CONSUMER_NOT_IMPLEMENTED",
)
PROJECTION_SCHEMA = "r02_structure_discrete_affinities_v1"
PROJECTION_RULE = "OBSERVED_MEAN_PER_FIT_COMMUNITY_EXACT_TIES_NO_FALLBACK"
PROJECTION_FILES = (
    "persons.tsv", "exclusion.json", "fit_affinities.private.tsv",
    "query_affinities.private.tsv", "communities.private.tsv", "frozen_communities.json", "community_method.json",
)
EDGE_COLUMNS = ["left_index", "right_index", "observed", "affinity",
                "joint_callable_sites", "value_denominator_sites", "candidate_sites"]


def ordered_ids_hash(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def observed_community_means(fit_labels, observations):
    """Proposed assignment instrument; neither clustering nor model selection.

    observations maps a FIT roster index to a nonnegative observed affinity.
    An omitted index is unknown. A zero is observed and belongs in the mean.
    Every labeled community is reported, including those without observations.
    No minimum coverage is chosen here; poor coverage remains an explicit rival.
    """
    require(isinstance(fit_labels, dict) and fit_labels, "FIT labels required")
    require(all(type(k) is int and k >= 0 and isinstance(v, str) and v == v.strip()
                and v for k, v in fit_labels.items()), "Invalid FIT index/community label")
    require(set(observations) <= set(fit_labels), "Observed neighbor is not a frozen FIT member")
    groups = {label: [] for label in fit_labels.values() if label != "UNASSIGNED"}
    require(groups, "No assigned FIT community")
    totals = Counter(fit_labels.values())
    ignored = 0
    for index, value in observations.items():
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                "Projection requires finite nonnegative observed affinities")
        label = fit_labels[index]
        if label == "UNASSIGNED":
            ignored += 1
        else:
            groups[label].append(value)
    summary = [{"community": label, "n_fit_members": totals[label],
                "n_observed_members": len(values),
                "observed_fraction": len(values) / totals[label],
                "mean_observed_affinity": math.fsum(values) / len(values) if values else None}
               for label, values in sorted(groups.items())]
    supported = [row for row in summary if row["n_observed_members"]]
    winners = []
    if not supported:
        status = "NO_OBSERVED_COMMUNITY_SUPPORT"
    else:
        highest = max(row["mean_observed_affinity"] for row in supported)
        winners = [row["community"] for row in supported if row["mean_observed_affinity"] == highest]
        status = ("OBSERVED_ZERO_AFFINITY" if highest == 0 else
                  "TIED_COMMUNITIES" if len(winners) > 1 else "ASSIGNED_BY_OBSERVED_MEAN")
    return {"status": status, "community": winners[0] if status == "ASSIGNED_BY_OBSERVED_MEAN" else None,
            "tied_or_maximal_communities": winners, "community_support": summary,
            "observed_unassigned_fit_members_ignored": ignored,
            "adequate_projection_support_certified": False}


class ProjectionBundle:
    """Authenticated streaming sidecar API, never accepted by the K02 reader.

    FIT-only learning is authenticated as upstream provenance, not rerun here.
    Only metadata and edge features are opened; target genotypes are not read.
    Peak edge memory is at most one query's FIT neighbors, not all query pairs.
    """
    def __init__(self, bundle, contract_sha256, source_bundle, source_sha256, *, max_input_bytes, max_metadata_bytes):
        self.bundle = Path(bundle)
        positive_int(max_input_bytes, "projection input bytes")
        positive_int(max_metadata_bytes, "projection metadata bytes")
        require(valid_hash(contract_sha256) and valid_hash(source_sha256), "External projection/source pins required")
        path = self.bundle / "projection.json"
        require(path.is_file() and path.stat().st_size <= max_metadata_bytes, "Projection contract missing/over budget")
        require(sha(path) == contract_sha256, "Projection contract SHA256 mismatch")
        self.contract_hash = contract_sha256
        c = load_json(path)
        require(set(c) == {"schema", "evidence_scope", "source_contract_sha256", "block_id", "exclusion_scope",
                          "sha256", "affinity", "projection_rule", "missing_edge_policy", "same_component_fit_edges"},
                "Exact projection contract fields required")
        require(c["schema"] == PROJECTION_SCHEMA and c["projection_rule"] == PROJECTION_RULE,
                "Unsupported projection schema/rule")
        require(c["evidence_scope"] in {"SYNTHETIC_INSTRUMENT_ONLY", "DNABR_R02"}, "Explicit projection evidence scope required")
        require(c["missing_edge_policy"] == "UNKNOWN_NOT_ZERO", "Missing affinity cannot be a biological zero")
        require(c["same_component_fit_edges"] in {"INCLUDE", "EXCLUDE"}, "Explicit FIT component policy required")
        require(c["source_contract_sha256"] == source_sha256
                and sha(Path(source_bundle) / "materialize.json") == source_sha256, "Projection/source contract mismatch")
        require(set(c["sha256"]) == set(PROJECTION_FILES), "Exact private projection file inventory required")
        files = [self.bundle / name for name in PROJECTION_FILES]
        require(all(p.is_file() for p in files)
                and sum(p.stat().st_size for p in files) + path.stat().st_size <= max_input_bytes,
                "Projection files missing or input byte budget exceeded")
        metadata = [p for p in files if p.name not in {"fit_affinities.private.tsv", "query_affinities.private.tsv"}]
        require(sum(p.stat().st_size for p in metadata) + path.stat().st_size <= max_metadata_bytes,
                "Projection metadata exceeds in-memory byte budget")
        for name, digest in c["sha256"].items():
            require(valid_hash(digest) and sha(self.bundle / name) == digest, "Projection artifact SHA256 mismatch: " + name)
        source_metadata = [Path(source_bundle) / name for name in INPUT_FILES]
        require(all(p.is_file() for p in source_metadata)
                and sum(p.stat().st_size for p in source_metadata) <= max_metadata_bytes,
                "Source metadata missing or input byte budget exceeded")
        source, persons, roles, blocks, _ = materialize.load_context(source_bundle)
        require(source["roles_state"] == "EXPLICITLY_FROZEN", "Projection requires frozen roles")
        require(c["sha256"]["persons.tsv"] == source["metadata_sha256"]["persons.tsv"], "Projection people/roles differ")
        require(c["exclusion_scope"] in SCOPES, "Unknown projection exclusion")
        require(source["exclusion_scope"] == SCOPES[0] and source.get("additional_exclusion_scopes") == [SCOPES[1]],
                "Projection source must retain both exclusion regimes")
        if c["evidence_scope"] == "DNABR_R02":
            require(len(persons) == 2619 and {r: len(roles[r]) for r in shared.ROLES} == {
                "FIT": 1460, "SELECT": 476, "EVALUATE": 683} and len(blocks) == 44
                and Counter(b["chromosome"] for b in blocks) == Counter({str(c): 2 for c in range(1, 23)}),
                "Real K03 projection must retain the frozen R02 people/roles and 44 objectives")
        geometry = materialize.input_plan.plan_regions(blocks, {k: v["length_bp"] for k, v in source["sources"].items()},
                                                       c["exclusion_scope"])
        expected = [row for row in geometry if row["block_id"] == c["block_id"]]
        require(len(expected) == 1 and load_json(self.bundle / "exclusion.json") == expected[0],
                "Projection exclusion differs from authenticated K02 geometry")
        a = c["affinity"]
        require(isinstance(a, dict) and set(a) == {"source_measure", "transformation", "definition", "code_sha256", "maximum"},
                "Explicit affinity transformation and range required")
        require(isinstance(a["source_measure"], str) and a["source_measure"].strip(), "Affinity source measure required")
        require(a["transformation"] in {"IDENTITY_NONNEGATIVE_INPUT", "UPSTREAM_EXPLICIT_NONNEGATIVE_TRANSFORM"}
                and isinstance(a["definition"], str) and len(a["definition"].strip()) >= 20
                and valid_hash(a["code_sha256"]), "Authenticated declared affinity transformation required")
        require(a["source_measure"] != "COMMON_GRM" or a["transformation"] == "UPSTREAM_EXPLICIT_NONNEGATIVE_TRANSFORM",
                "Signed common GRM requires an explicit upstream transformation, never silent truncation")
        require(type(a["maximum"]) in (float, int) and math.isfinite(a["maximum"]) and a["maximum"] > 0,
                "Positive finite affinity upper bound required")
        self.contract, self.persons = c, persons
        self.source_files = {p: sha(p) for p in source_metadata}
        self.fit_indices = {int(i) for i in roles["FIT"]}
        self.query_indices = sorted(int(i) for r in ("SELECT", "EVALUATE") for i in roles[r])
        self.labels = {}
        labels = shared.table(self.bundle / "communities.private.tsv", ["sample_id", "community"])
        mapping = {p["sample_id"]: i for i, p in enumerate(persons)}
        require(len(labels) == len(self.fit_indices), "One frozen label or UNASSIGNED per FIT member required")
        for row in labels:
            i = mapping.get(row["sample_id"])
            require(i in self.fit_indices and i not in self.labels, "Community member outside FIT or duplicated")
            require(row["community"] == row["community"].strip(), "Invalid frozen community label")
            self.labels[i] = row["community"]
        receipt = load_json(self.bundle / "frozen_communities.json")
        require(set(receipt) == {"schema", "status", "fit_ids_sha256", "fit_affinities_sha256", "communities_sha256",
                                "persons_sha256", "exclusion_sha256", "learning_roles", "target_used",
                                "heldout_edges_used", "method_manifest_sha256"}, "Exact frozen-community provenance required")
        require(receipt["schema"] == "r02_fit_communities_frozen_v1"
                and receipt["status"] == "FIT_ONLY_FROZEN_BEFORE_PROJECTION"
                and receipt["learning_roles"] == ["FIT"] and receipt["target_used"] is False
                and receipt["heldout_edges_used"] is False and valid_hash(receipt["method_manifest_sha256"]),
                "Community learning cannot use held-out people/edges or target loci")
        require(receipt["fit_ids_sha256"] == ordered_ids_hash(persons[i]["sample_id"] for i in sorted(self.fit_indices)),
                "Frozen community FIT roster differs")
        for receipt_key, filename in (("fit_affinities_sha256", "fit_affinities.private.tsv"),
                ("communities_sha256", "communities.private.tsv"), ("persons_sha256", "persons.tsv"),
                ("exclusion_sha256", "exclusion.json"), ("method_manifest_sha256", "community_method.json")):
            require(receipt[receipt_key] == c["sha256"][filename], "Frozen community artifact/exclusion binding differs")
        method = load_json(self.bundle / "community_method.json")
        require(set(method) == {"schema", "algorithm", "parameters", "code_sha256", "fit_only", "exclusion_sha256"}
                and method["schema"] == "r02_fit_community_method_v1" and method["fit_only"] is True
                and isinstance(method["algorithm"], str) and method["algorithm"].strip()
                and isinstance(method["parameters"], dict) and valid_hash(method["code_sha256"])
                and method["exclusion_sha256"] == c["sha256"]["exclusion.json"],
                "Frozen FIT method manifest missing or exclusion mismatched")
        # Exhaust both tables before exposing any assignment: invalid late rows
        # must not leave a caller with an apparently successful partial result.
        for name in ("fit_affinities.private.tsv", "query_affinities.private.tsv"):
            for _ in self.edges(name):
                pass
        self.verify()

    def verify(self):
        require(all(sha(path) == digest for path, digest in self.source_files.items()), "K02 metadata changed during projection")
        require(sha(self.bundle / "projection.json") == self.contract_hash, "Projection contract changed during use")
        for name, digest in self.contract["sha256"].items():
            require(sha(self.bundle / name) == digest, "Projection artifact changed during use: " + name)

    def edges(self, name):
        require(name in {"fit_affinities.private.tsv", "query_affinities.private.tsv"}, "Unknown edge table")
        previous = None
        with (self.bundle / name).open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require(reader.fieldnames == EDGE_COLUMNS, "Exact affinity columns required")
            for row in reader:
                require(None not in row and None not in row.values(), "Malformed affinity row")
                i, j = int(row["left_index"]), int(row["right_index"])
                key = (i, j)
                require(previous is None or previous < key, "Affinity keys must be strictly sorted and unique")
                previous = key
                require(j in self.fit_indices, "Affinity reference endpoint is not FIT")
                if name == "fit_affinities.private.tsv":
                    require(i in self.fit_indices and i < j, "FIT edges must be unordered canonical off-diagonal FIT pairs")
                    if self.contract["same_component_fit_edges"] == "EXCLUDE":
                        require(self.persons[i]["component"] != self.persons[j]["component"], "FIT same-component edge forbidden")
                else:
                    require(0 <= i < len(self.persons) and self.persons[i]["role"] in {"SELECT", "EVALUATE"},
                            "Projection edges must be query-to-FIT only")
                joint, denominator, candidate = (int(row[k]) for k in EDGE_COLUMNS[4:])
                require(0 <= denominator <= joint <= candidate, "Invalid support denominators")
                require(row["observed"] in {"0", "1"}, "Observed mask must be 0 or 1")
                observed = row["observed"] == "1"
                if observed:
                    value = float(row["affinity"])
                    require(denominator > 0 and math.isfinite(value) and 0 <= value <= self.contract["affinity"]["maximum"],
                            "Observed affinity outside support/range; negative values cannot be truncated")
                else:
                    require(row["affinity"] == "NA", "Masked affinity must remain NA, not zero")
                    value = None
                yield i, j, value, joint, denominator, candidate

    def project(self):
        """Yield private features; no target, fitting, selection or metric call."""
        self.verify()
        iterator = iter(self.edges("query_affinities.private.tsv"))
        row = next(iterator, None)
        for query in self.query_indices:
            observed, n_rows, n_masked = {}, 0, 0
            denominators = {"joint_callable_sites_sum": 0, "value_denominator_sites_sum": 0, "candidate_sites_sum": 0}
            while row is not None and row[0] == query:
                _, fit, value, joint, denominator, candidate = row
                n_rows += 1
                if value is None:
                    n_masked += 1
                else:
                    observed[fit] = value
                for key, amount in zip(denominators, (joint, denominator, candidate)):
                    denominators[key] += amount
                row = next(iterator, None)
            result = observed_community_means(self.labels, observed)
            yield {"sample_id": self.persons[query]["sample_id"], "role": self.persons[query]["role"],
                   **result, "n_explicit_edge_rows": n_rows, "n_masked_edge_rows": n_masked,
                   "n_omitted_unknown_fit_edges": len(self.fit_indices) - n_rows, **denominators,
                   "denominator_sum_scope": "SUM_OVER_EXPLICIT_EDGES_NOT_UNIQUE_GENOMIC_TERRITORY",
                   "public_release": False, "upstream_fit_only_claim": "AUTHENTICATED_NOT_RECOMPUTED"}
        require(row is None, "Unconsumed projection edge")
        self.verify()


def implementation_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def positive_int(value, label):
    require(type(value) is int and value > 0, f"Positive integer required: {label}")


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_request(request):
    require(set(request) == CONTRACT_FIELDS, "Exact admission fields required; no model parameters")
    require(request["schema"] == SCHEMA, "Unsupported K03 admission schema")
    require(request["evidence_scope"] in {"SYNTHETIC_INSTRUMENT_ONLY", "DNABR_R02"},
            "Explicit synthetic or DNABR scope required")
    require(request["projection_route"] == "AFFINITY_SIDECAR_DESIGN_ONLY_LEARNING_PENDING",
            "Only the reviewed sidecar direction is admitted, not a learning method")
    require(valid_hash(request["source_contract_sha256"]), "Source contract SHA256 required")
    expected = request["implementation_sha256"]
    require(isinstance(expected, dict) and set(expected) == set(CODE_FILES)
            and all(valid_hash(v) for v in expected.values()), "Exact implementation SHA256 inventory required")
    require(expected == implementation_hashes(), "K03 or shared implementation changed")
    for key in ("expected_n_people", "expected_n_targets", "max_metadata_bytes"):
        positive_int(request[key], key)
    counts = request["expected_role_counts"]
    require(isinstance(counts, dict) and set(counts) == set(shared.ROLES), "Exact role counts required")
    for value in counts.values():
        positive_int(value, "role count")
    require(sum(counts.values()) == request["expected_n_people"], "Role counts must sum to people")
    chromosomes = request["expected_chromosomes"]
    require(isinstance(chromosomes, list) and chromosomes
            and all(type(v) is int and 1 <= v <= 22 for v in chromosomes)
            and chromosomes == sorted(set(chromosomes)), "Ordered unique autosomes required")
    require(request["expected_n_targets"] == 2 * len(chromosomes), "Two targets per autosome required")
    if request["evidence_scope"] == "DNABR_R02":
        require(chromosomes == list(range(1, 23)) and request["expected_n_targets"] == 44,
                "Real K03 must retain all 44 objectives over 22 autosomes")
        require(request["expected_n_people"] == 2619 and counts == {
            "FIT": 1460, "SELECT": 476, "EVALUATE": 683}, "Real R02 frozen cohort/roles differ")


def pair_support(persons):
    """Exact combinatorial roster counts; no claim of measured genetic support.

    Query-to-FIT pairs are counted for an affinity route, not materialized.
    They are NOT future target pairs and may never enter the K02 target reader.
    """
    shared.validate_roles(persons)
    rows = []
    n_fit = sum(p["role"] == "FIT" for p in persons)
    for role in shared.ROLES:
        counts = Counter(p["component"] for p in persons if p["role"] == role)
        n = sum(counts.values())
        within_component = sum(k * (k - 1) // 2 for k in counts.values())
        rows.append({
            "role": role, "n_people": n, "n_components": len(counts),
            "within_role_unordered_pairs": n * (n - 1) // 2,
            "within_component_pairs_excluded_by_k02": within_component,
            "within_role_between_component_pairs_in_k02_key_universe":
                (n * n - sum(k * k for k in counts.values())) // 2,
            "query_to_fit_pairs_if_dense_affinity_route": 0 if role == "FIT" else n * n_fit,
        })
    return rows


def inspect(bundle, request):
    """Small authenticated metadata only; no `plan`, `slice`, `fit` or `evaluate`."""
    validate_request(request)
    bundle = Path(bundle)
    paths = [bundle / name for name in INPUT_FILES]
    require(all(path.is_file() for path in paths), "Missing K02 preparation metadata")
    require(sum(path.stat().st_size for path in paths) <= request["max_metadata_bytes"],
            "Metadata byte budget exceeded before parsing")
    observed = {name: sha(bundle / name) for name in INPUT_FILES}
    require(observed["materialize.json"] == request["source_contract_sha256"], "Source contract changed")
    # Existing implementation retains authoritative cohort, allele and interval
    # checks. load_context does not open VCFs, launch PLINK or fit a model.
    source, persons, roles, blocks, _ = materialize.load_context(bundle)
    require(source["roles_state"] == "EXPLICITLY_FROZEN", "K03 requires frozen component roles")
    require(source["exclusion_scope"] == SCOPES[0]
            and source.get("additional_exclusion_scopes") == [SCOPES[1]], "Both R02 exclusion regimes required")
    require(len(persons) == request["expected_n_people"], "Analytical cohort count differs")
    require({r: len(roles[r]) for r in shared.ROLES} == request["expected_role_counts"], "Role counts differ")
    require(source["source_n_samples"] >= len(persons), "Source cohort smaller than analytical cohort")
    require(sorted(map(int, source["sources"])) == request["expected_chromosomes"], "Source autosomes differ")
    require(len(blocks) == request["expected_n_targets"], "Target count differs")
    expected_counts = {str(c): 2 for c in request["expected_chromosomes"]}
    require(dict(Counter(b["chromosome"] for b in blocks)) == expected_counts, "Two targets per chromosome required")
    lengths = {c: src["length_bp"] for c, src in source["sources"].items()}
    regions = {scope: materialize.input_plan.plan_regions(blocks, lengths, scope) for scope in SCOPES}
    for scope, values in regions.items():
        require(all(row["allowed_bp"] > 0 for row in values), f"No predictor territory for {scope}")
    # Recheck metadata and implementation after inspection, before output.
    require(observed == {name: sha(bundle / name) for name in INPUT_FILES}, "Metadata changed during admission")
    require(request["implementation_sha256"] == implementation_hashes(), "Implementation changed during admission")
    support = pair_support(persons)
    projected_pairs = sum(row["query_to_fit_pairs_if_dense_affinity_route"] for row in support)
    return {
        "schema": SCHEMA, "stage": "METADATA_AUTHENTICATED_K03_NOT_READY",
        "evidence_scope": request["evidence_scope"], "training_allowed": False,
        "genotypes_read": False, "target_tensors_read": False, "models_fitted": 0,
        "community_labels_read": False, "scientific_validation": False,
        "source_contract_sha256": request["source_contract_sha256"],
        "input_sha256": observed, "implementation_sha256": request["implementation_sha256"],
        "n_people": len(persons), "n_target_regions": len(blocks), "n_exclusion_regimes": len(SCOPES),
        "n_target_regime_views": len(blocks) * len(SCOPES),
        "scientific_protocol_sha256": source["metadata_sha256"]["protocol.md"],
        "genome_build": source["genome_build"], "rare_allele_convention": source["rare_allele_convention"],
        "unphased_rare_policy": "DIPLOID_NO_HAPLOTYPE_ASSIGNMENT",
        "role_support": support, "regions": regions,
        "affinity_route_dimensions_not_a_method_choice": {
            "missing_query_to_fit_pairs_per_view": projected_pairs,
            "bytes_one_dense_float64_query_to_fit_channel_per_view": projected_pairs * 8,
            "meaning": "ROSTER_DIMENSION_ONLY_NOT_MEASURED_SUPPORT_OR_TOTAL_PEAK_MEMORY",
            "same_component_fit_affinities": "ABSENT_IN_K02_NO_DEFAULT_IMPUTATION",
        },
        "k02_products": "CONTRACT_INSPECTED_COMPLETION_NOT_CHECKED",
        "k02_pair_domain": "WITHIN_ROLE_BETWEEN_COMPONENT_ONLY",
        "projection_route": request["projection_route"], "blocking_requirements": list(BLOCKERS),
        "claims_not_supported": ["DISCRETE_POPULATIONS", "ADEQUATE_CONTINUOUS_BASELINE",
                                 "K03_PREDICTIVE_INCREMENT", "GCS_OR_JOB_COMPLETION"],
        "privacy": {"public_release": False, "individual_identifiers_in_output": False},
    }


def run(bundle, request_path, request_sha256, output):
    os.umask(0o077)
    request_path, output = Path(request_path), Path(output)
    require(valid_hash(request_sha256) and sha(request_path) == request_sha256, "Admission request SHA256 mismatch")
    require(not output.exists(), "Output exists; use completed Nextflow cache, never overwrite")
    request = load_json(request_path)
    result = inspect(Path(bundle), request)
    require(sha(request_path) == request_sha256, "Admission request changed during inspection")
    result["admission_contract_sha256"] = request_sha256
    output.mkdir(parents=True, mode=0o700)
    with (output / "manifest.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--admission-contract", type=Path, required=True)
    parser.add_argument("--admission-contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.bundle, args.admission_contract, args.admission_contract_sha256, args.output_dir)
    print(json.dumps({k: result[k] for k in ("schema", "stage", "training_allowed")}))


if __name__ == "__main__":
    main()
