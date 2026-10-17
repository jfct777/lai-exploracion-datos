#!/usr/bin/env python3
"""Fail-closed K05 paired-comparison consumer.

The consumer compares two FIT-learned partitions on the exact same people and
dyads in SELECT or EVALUATE.  It reports denominators and point contrasts, but
does not manufacture an uncertainty policy or rank the historical 57
partitions as independent evaluations.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, random
from collections import Counter
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def descriptor(item, name):
    require(set(item) == {"path", "sha256"}, f"Exact descriptor required: {name}")
    path = Path(item["path"])
    require(path.is_absolute() and path.is_file() and sha(path) == item["sha256"],
            f"Authenticated input differs: {name}")
    return path


def rows(path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def quantile(values, probability):
    values=sorted(values); position=(len(values)-1)*probability
    lo=int(math.floor(position)); hi=int(math.ceil(position))
    return values[lo] if lo==hi else values[lo]+(values[hi]-values[lo])*(position-lo)


def dependency_bootstrap(contributions, components, policy):
    require(policy.get('method')=='DEPENDENCY_COMPONENT_BOOTSTRAP' and
            policy.get('unit')=='DEPENDENCY_COMPONENT',
            'Only explicit dependency-component bootstrap is implemented')
    repetitions=policy.get('repetitions'); seed=policy.get('seed'); alpha=policy.get('interval_alpha')
    require(type(repetitions) is int and repetitions>=100 and type(seed) is int and
            type(alpha) in (int,float) and 0<alpha<1 and policy.get('source') and policy.get('justification'),
            'Bootstrap repetitions, seed, alpha, source and justification are required')
    grouped={}
    for value,component in zip(contributions,components): grouped.setdefault(component,[]).append(value)
    require(len(contributions) == len(components) and len(grouped) >= 2,
            'Component bootstrap requires at least two independent components; a connected dyad set needs another admitted uncertainty method')
    keys=sorted(grouped); rng=random.Random(seed); estimates=[]
    for _ in range(repetitions):
        sample=[rng.choice(keys) for _ in keys]
        values=[v for key in sample for v in grouped[key]]
        estimates.append(math.fsum(values)/len(values))
    return {'method':policy['method'],'unit':policy['unit'],'dependency_components':len(keys),
            'repetitions':repetitions,'seed':seed,'interval_alpha':alpha,
            'interval':[quantile(estimates,alpha/2),quantile(estimates,1-alpha/2)],
            'bootstrap_mean':math.fsum(estimates)/len(estimates),
            'source':policy['source'],'justification':policy['justification']}


def readiness(contract):
    missing = contract.get("missing_dependencies", [])
    require(all(isinstance(x, str) and x for x in missing), "Invalid missing dependency")
    historical = descriptor(contract["historical_support_audit"], "historical_support_audit")
    audit = json.loads(historical.read_text())
    require(audit["schema"] == "r02_k05_comparable_support_v1" and
            audit["winner_selected"] is False and audit["evaluate_opened"] is False,
            "Historical K05 audit cannot be promoted or ranked")
    return {
        "schema": "r02_k05_paired_comparison_result_v1",
        "status": "BLOCKED_EXPLICIT_DEPENDENCIES" if missing else "READY_FOR_PAIRED_INPUTS",
        "mode": "READINESS",
        "historical_partitions_ranked": False,
        "historical_descriptions_recomputed": False,
        "missing_dependencies": missing,
        "required_next_input": "Two FIT-only partitions with fixed SELECT/EVALUATE projections and one common dyad table",
        "historical_support_audit_sha256": sha(historical),
        "biological_validation": False,
    }


def compare(contract, out):
    require(contract.get("stages") in (["SELECT"], ["EVALUATE"]),
            "One reserved stage must be consumed per immutable run")
    stage = contract["stages"][0]
    require(contract.get("historical_partitions_ranked") is False,
            "Historical partitions cannot be treated as independent evaluations")
    require(contract.get("selection_fixed_before_evaluate") is True or stage == "SELECT",
            "EVALUATE requires selection frozen beforehand")
    policy = contract.get("uncertainty_policy")
    require(isinstance(policy, dict) and policy.get("status") == "ADMITTED" and
            policy.get("unit") in {"PERSON", "DEPENDENCY_COMPONENT"} and
            isinstance(policy.get("method"), str) and policy["method"],
            "An admitted person/component uncertainty policy is required")
    projections = descriptor(contract["projections"], "projections")
    dyads = descriptor(contract["dyads"], "dyads")
    expected_projection = ["sample_id", "stage", "left_label", "right_label", "person_component"]
    pr = list(rows(projections))
    require(pr and list(pr[0]) == expected_projection, "Projection columns differ")
    require(all(r["stage"] == stage and r["sample_id"] and r["person_component"] for r in pr),
            "Projection stage/person dependency differs")
    require(len({r["sample_id"] for r in pr}) == len(pr), "Duplicate projected person")
    require(all(r[name] and r[name].upper() not in {'UNASSIGNED', 'AMBIGUOUS', 'NA', 'NAN', 'NONE'}
                for r in pr for name in ('left_label', 'right_label')),
            'Unassigned or ambiguous projections require an explicit support/uncertainty policy; they cannot form a community')
    by_id = {r["sample_id"]: r for r in pr}
    expected_dyad = ["stage", "sample_a", "sample_b", "outcome", "pair_component"]
    data = list(rows(dyads))
    require(data and list(data[0]) == expected_dyad, "Dyad columns differ")
    seen, contributions, contribution_components, pair_components = set(), [], [], Counter()
    involved = set()
    component_owners = {}
    for row in data:
        require(row["stage"] == stage and row["sample_a"] < row["sample_b"],
                "Dyads must be canonical and from one stage")
        key = row["sample_a"], row["sample_b"]
        require(key not in seen and set(key) <= set(by_id), "Duplicate or unsupported dyad")
        seen.add(key); involved.update(key)
        value = float(row["outcome"])
        require(math.isfinite(value), "Finite dyad outcome required")
        left = by_id[key[0]]["left_label"] == by_id[key[1]]["left_label"]
        right = by_id[key[0]]["right_label"] == by_id[key[1]]["right_label"]
        contributions.append(value * (int(left) - int(right)))
        require(row["pair_component"], "Pair dependency component required")
        # Dyads sharing a person or a recorded person-dependency component
        # cannot be resampled independently. Reject incompatible input rather
        # than silently merging it or changing the admitted bootstrap method.
        for person in key:
            dependency = by_id[person]['person_component']
            owner = component_owners.setdefault(dependency, row['pair_component'])
            require(owner == row['pair_component'],
                    'Dyad bootstrap components split a shared person/dependency component')
        contribution_components.append(row['pair_component']); pair_components[row["pair_component"]] += 1
    require(involved == set(by_id), "Every projected person must enter the common dyad denominator")
    result = {
        "schema": "r02_k05_paired_comparison_result_v1",
        "status": "PAIRED_POINT_CONTRAST_READY_UNCERTAINTY_POLICY_RECORDED",
        "stage": stage,
        "population_people": len(by_id),
        "common_dyads": len(data),
        "person_dependency_components": len({r["person_component"] for r in pr}),
        "pair_dependency_components": len(pair_components),
        "paired_contrast_sum": math.fsum(contributions),
        "paired_contrast_mean_per_dyad": math.fsum(contributions) / len(contributions),
        "contrast_definition": "outcome * (same-community-left - same-community-right) on identical dyads",
        "uncertainty": dependency_bootstrap(contributions,contribution_components,policy),
        "historical_partitions_ranked": False,
        "independent_dyads_claimed": False,
        "winner_selected": False,
        "biological_validation": False,
        "inputs_sha256": {"projections": sha(projections), "dyads": sha(dyads)},
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(contract_path, expected_sha, out):
    require(sha(contract_path) == expected_sha, "Contract changed")
    contract = json.loads(Path(contract_path).read_text())
    require(contract.get("schema") == "r02_k05_paired_comparison_v1", "Unsupported K05 contract")
    if contract.get("mode") == "READINESS":
        result = readiness(contract)
        Path(out).mkdir(parents=True, exist_ok=False)
        (Path(out) / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    require(contract.get("mode") == "COMPARE", "Unsupported K05 mode")
    return compare(contract, Path(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", type=Path, required=True); p.add_argument("--contract-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); print(json.dumps(run(a.contract, a.contract_sha256, a.output_dir)))
