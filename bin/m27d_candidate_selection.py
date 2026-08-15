#!/usr/bin/env python3
"""Turn the M27D configuration results into a candidate list, gates and public counts.

The selection is conservative in a specific sense: a pair counts as related if it
crosses the threshold in *any* preregistered configuration.  The union is not a
statistical estimate, it is a decision rule chosen before the results existed, and its
direction matters.  Taking the union can only shrink the candidate list, so a
configuration that happens to retain more donors cannot be adopted after the fact by
declaring the others noisy.

Three exclusions are applied in order, each with a recorded reason:

1. identity with a baseline donor, which would score the same person twice;
2. kinship with a baseline donor, which would let a sibling stand in for an
   independent evaluation;
3. kinship inside the surviving candidate set, resolved by the same deterministic
   maximal independent set used to build the pass0 training set.

Samples whose metadata row could not be resolved keep their genotypes throughout the
kinship computation and are still excluded from the candidate list.  Their population
is unknown, and a donor of unknown population cannot be placed in an ancestry panel.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from m27d_kinship_graph import (
    adjacency,
    is_interpretable,
    maximal_independent_set,
    read_call_rates,
    read_pairs,
    read_sample_universe,
    stable_hash,
    verify_independent,
)


EXCLUSION_BASELINE_IDENTITY = "BASELINE_IDENTITY"
EXCLUSION_BASELINE_KINSHIP = "BASELINE_KINSHIP"
EXCLUSION_METADATA_UNRESOLVED = "UNMATCHED_SOURCE_METADATA"
EXCLUSION_CANDIDATE_KINSHIP = "CANDIDATE_KINSHIP"
INCLUDED = "INCLUDED"


def read_strata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    unresolved = [row for row in rows if row.get("resolution_method") == "AMBIGUOUS_FAIL_CLOSED"]
    if unresolved:
        raise SystemExit(
            f"Strata table still contains {len(unresolved)} unresolved alias collisions"
        )
    return {row["sample_id"]: row for row in rows}


def is_excluded(row: dict[str, str]) -> bool:
    return str(row.get("Exclude", "")).strip().upper() in {"TRUE", "T", "1", "YES", "Y"}


def candidate_strata(row: dict[str, str]) -> set[str]:
    """Descriptive labels for the report.  None of them takes part in the selection."""
    labels: set[str] = set()
    if str(row.get("Ancestry", "")).strip().lower().startswith("native"):
        labels.add("all_NAM_candidates")
    if str(row.get("Country", "")).strip().lower() == "brazil":
        labels.add("Brazil_metadata")
    return labels


def run(args: argparse.Namespace) -> dict[str, object]:
    contract = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if contract.get("pcrelate", {}).get("king_allowed"):
        raise SystemExit("M27D forbids KING")
    primary_threshold = float(contract["pcrelate"]["primary_phi_threshold"])
    descriptive = [float(value) for value in contract["pcrelate"]["descriptive_phi_thresholds"]]
    expected_configurations = {config["id"] for config in contract["configurations"]}

    strata = read_strata(args.strata)
    call_rate = read_call_rates(args.call_rates)
    universe = read_sample_universe(args.samples)
    baseline_identities = set(read_sample_universe(args.baseline_identities))
    m27c_samples = set(read_sample_universe(args.m27c_samples)) if args.m27c_samples else set()

    observed_configurations = {path.name.split("m27d_pcrelate_")[-1].split("_pairs")[0] for path in args.pairs}
    if observed_configurations != expected_configurations:
        raise SystemExit(
            "Configuration set mismatch: expected "
            f"{sorted(expected_configurations)}, observed {sorted(observed_configurations)}"
        )

    # The union graph is what the conservative rule is made of: one crossing anywhere is
    # enough to call a pair related.  The edges that only some configurations see are
    # the instability the contract asks to surface rather than smooth over.
    edges_by_configuration = {
        path.name.split("m27d_pcrelate_")[-1].split("_pairs")[0]: read_pairs(
            path, primary_threshold
        )
        for path in sorted(args.pairs)
    }
    per_configuration = {name: len(edges) for name, edges in edges_by_configuration.items()}
    union_edges: set[tuple[str, str]] = set().union(*edges_by_configuration.values())
    shared_edges: set[tuple[str, str]] = set.intersection(*edges_by_configuration.values())
    unstable = union_edges - shared_edges

    universe_set = set(universe)
    external = {node for edge in union_edges for node in edge} - universe_set
    if external:
        raise SystemExit(f"{len(external)} kinship identifiers are outside the sample universe")

    graph = adjacency(union_edges)
    decisions: dict[str, str] = {}
    for sample in universe:
        row = strata.get(sample)
        if row is None:
            raise SystemExit("Sample universe contains an identifier absent from the strata table")
        if sample in baseline_identities:
            decisions[sample] = EXCLUSION_BASELINE_IDENTITY
        elif graph.get(sample, set()) & baseline_identities:
            decisions[sample] = EXCLUSION_BASELINE_KINSHIP
        elif not is_interpretable(row):
            decisions[sample] = EXCLUSION_METADATA_UNRESOLVED

    eligible = [sample for sample in universe if sample not in decisions]
    eligible_set = set(eligible)
    internal_edges = {
        edge for edge in union_edges if edge[0] in eligible_set and edge[1] in eligible_set
    }
    selected = maximal_independent_set(eligible, internal_edges, call_rate)
    alternate = maximal_independent_set(eligible, internal_edges, call_rate, descending_hash=True)
    residual = verify_independent(selected, internal_edges)
    if residual:
        raise SystemExit(f"Final candidate set retains {residual} internal kinship edges")

    selected_set = set(selected)
    for sample in eligible:
        decisions[sample] = INCLUDED if sample in selected_set else EXCLUSION_CANDIDATE_KINSHIP

    args.out_private.parent.mkdir(parents=True, exist_ok=True)
    with args.out_private.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "decision",
                "reason",
                "call_rate",
                "n_union_related_partners",
                "in_alternate_order_set",
                "Source",
                "Ancestry",
                "Population",
                "Country",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        alternate_set = set(alternate)
        for sample in sorted(universe, key=stable_hash):
            row = strata[sample]
            decision = decisions[sample]
            writer.writerow(
                {
                    "sample_id": sample,
                    "decision": "INCLUDED" if decision == INCLUDED else "EXCLUDED",
                    "reason": decision,
                    "call_rate": f"{call_rate.get(sample, float('nan')):.6f}",
                    "n_union_related_partners": len(graph.get(sample, ())),
                    "in_alternate_order_set": "TRUE" if sample in alternate_set else "FALSE",
                    "Source": row.get("Source", ""),
                    "Ancestry": row.get("Ancestry", ""),
                    "Population": row.get("Population", ""),
                    "Country": row.get("Country", ""),
                }
            )

    stratum_counts: Counter[str] = Counter()
    for sample in selected:
        for label in candidate_strata(strata[sample]):
            stratum_counts[label] += 1
        if sample in m27c_samples:
            stratum_counts["M27C_128_gVCF"] += 1
            if "Brazil_metadata" in candidate_strata(strata[sample]):
                stratum_counts["intersection_Brazil_M27C"] += 1
    # Every preregistered stratum appears, including the ones that came out empty, so a
    # missing input shows up as a zero rather than as an absent key nobody notices.
    for label in contract["selection"]["report_strata"]:
        stratum_counts.setdefault(label, 0)
    # The same small-cell rule that protects the public TSV has to protect this JSON:
    # publishing an exact count of one in a named stratum identifies that person.
    published_strata = {
        label: (count if count >= args.suppress_below or count == 0 else None)
        for label, count in sorted(stratum_counts.items())
    }
    suppressed_strata = sorted(
        label for label, count in published_strata.items() if count is None
    )

    public_rows = Counter()
    for sample in selected:
        row = strata[sample]
        public_rows[(row.get("Source", ""), row.get("Ancestry", ""), row.get("Country", ""))] += 1
    with args.out_public.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source", "ancestry", "country", "n_candidates"])
        suppressed = 0
        for (source, ancestry, country), count in sorted(public_rows.items()):
            if count < args.suppress_below:
                suppressed += count
                continue
            writer.writerow([source, ancestry, country, count])
        if suppressed:
            writer.writerow(["SUPPRESSED", "SUPPRESSED", "SUPPRESSED", suppressed])

    reason_counts = Counter(decisions.values())

    # Every gate the preregistration names gets a row, including the ones this stage
    # cannot decide.  A gates file that silently omits five of seven reads as "all gates
    # passed" to whoever opens it, which is the opposite of what a fail-closed receipt is
    # for.  NOT_EVALUATED is an honest status; a missing row is not.
    upstream = {}
    for path in args.stage_summaries or []:
        try:
            upstream[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Could not read stage summary {path.name}: {error}")

    def g2a_status() -> tuple[str, str]:
        """Technical integrity of the fits. This one can fail the run."""
        verdicts = {
            name: summary.get("g2a_technical_integrity_status")
            for name, summary in upstream.items()
            if "g2a_technical_integrity_status" in summary
        }
        if not verdicts:
            return "NOT_EVALUATED", "no PCA summary reached the selection stage"
        if any(value == "FAIL" for value in verdicts.values()):
            failed = sorted(k for k, v in verdicts.items() if v == "FAIL")
            return "FAIL", f"the PCA fit is not sound in {failed}"
        if any(value != "PASS" for value in verdicts.values()):
            return "NOT_EVALUATED", "at least one PCA fit could not be checked"
        return "PASS", "every fit is finite, ordered and derived from a single projection"

    def g2b_status() -> tuple[str, str]:
        """Localisation of the axes. REVIEW is a request to look, not a failure.

        This aggregates over every configuration, and that is deliberate rather than a
        leftover of the old single verdict: the certified set is built from the *union* of
        edges across all preregistered configurations, so a certified donor is exposed to
        every prefix, not only to the one its own configuration read.  The per-prefix
        relief belongs to the PCA stage, where a four-component run genuinely does not read
        an eleventh axis and must not be aborted by it.  Here there is no such relief, and
        promising it would be a false comfort.
        """
        flagged: dict[str, list[str]] = {}
        seen = False
        for name, summary in upstream.items():
            by_prefix = summary.get("g2b_status_by_preregistered_n_pcs")
            if not isinstance(by_prefix, dict):
                continue
            seen = True
            for key, entry in by_prefix.items():
                if isinstance(entry, dict) and entry.get("status") == "REVIEW":
                    flagged.setdefault(name, []).append(key)
        if not seen:
            return "NOT_EVALUATED", "no PCA summary reached the selection stage"
        if flagged:
            detail = "; ".join(f"{name}: {', '.join(sorted(keys))}" for name, keys in sorted(flagged.items()))
            return (
                "REVIEW",
                "an axis inside a preregistered prefix is carried by few individuals and "
                f"its cause is unadjudicated ({detail})",
            )
        return "PASS", "every axis inside every preregistered prefix is carried broadly"

    def identity_status() -> tuple[str, str]:
        for summary in upstream.values():
            if summary.get("stage") != "M27D_BASELINE_IDENTITY":
                continue
            if not summary.get("identity_matches_expectation"):
                return "FAIL", "confirmed baseline identities differ from the preregistration"
            if summary.get("unmatched_baseline_donor_blocks_full_kinship_disjointness"):
                return (
                    "PASS_WITH_BLIND_SPOT",
                    "a baseline donor has no panel twin, so no candidate can be tested "
                    "for kinship against that donor",
                )
            return "PASS", "every baseline donor reconciled by dosage concordance"
        return "NOT_EVALUATED", "no baseline identity summary reached the selection stage"

    baseline_identity_verdict, baseline_identity_detail = identity_status()
    g2a_verdict, g2a_detail = g2a_status()
    g2b_verdict, g2b_detail = g2b_status()
    disjoint = not (selected_set & baseline_identities) and not any(
        graph.get(sample, set()) & baseline_identities for sample in selected
    )
    gates: dict[str, tuple[str, str]] = {
        "G0_input_identity_and_build": (
            baseline_identity_verdict,
            baseline_identity_detail,
        ),
        "G1_marker_qc_and_ld_pruning": (
            "NOT_EVALUATED",
            "decided by scientific review of the preparation stage, not by this stage",
        ),
        "G2A_pca_technical_integrity": (g2a_verdict, g2a_detail),
        "G2B_axis_localization": (g2b_verdict, g2b_detail),
        "G2C_ancestry_representativeness": (
            "NOT_EVALUATED",
            "the contract leaves this unadjudicated: it needs the per-population survival "
            "of the training set, which is measured in its own stage",
        ),
        "G3_pcrelate_iteration": (
            "PASS" if len(per_configuration) == len(expected_configurations) else "FAIL",
            f"{len(per_configuration)} of {len(expected_configurations)} configurations produced pairs",
        ),
        "G4_baseline_disjointness": (
            "PASS" if disjoint else "FAIL",
            "no candidate is a baseline donor or a relative of one",
        ),
        "G5_candidate_independence": (
            "PASS" if residual == 0 else "FAIL",
            f"{residual} kinship edges remain inside the final set",
        ),
        "G6_m27c_recomputed_readiness": (
            "NOT_EVALUATED",
            "requires recomputing M27C on the final donors, which is a separate stage",
        ),
        "G_candidate_set_non_empty": (
            "PASS" if selected else "FAIL",
            f"{len(selected)} candidates survived",
        ),
        "G_no_uninterpretable_candidate": (
            "PASS" if all(is_interpretable(strata[s]) for s in selected) else "FAIL",
            "no candidate lacks a resolved metadata row",
        ),
    }
    with args.out_gates.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["gate", "status", "detail"])
        for gate, (status, detail) in sorted(gates.items()):
            writer.writerow([gate, status, detail])

    summary: dict[str, object] = {
        "stage": "M27D_CANDIDATE_SELECTION",
        "king_executed": False,
        "primary_phi_threshold": primary_threshold,
        "descriptive_phi_thresholds": descriptive,
        "edge_rule": contract["pcrelate"]["final_related_edge_rule"],
        "n_universe": len(universe),
        "n_union_related_edges": len(union_edges),
        "n_related_edges_by_configuration": per_configuration,
        "n_edges_not_shared_by_all_configurations": len(unstable),
        "n_excluded_baseline_identity": reason_counts[EXCLUSION_BASELINE_IDENTITY],
        "n_excluded_baseline_kinship": reason_counts[EXCLUSION_BASELINE_KINSHIP],
        "n_excluded_metadata_unresolved": reason_counts[EXCLUSION_METADATA_UNRESOLVED],
        "n_excluded_candidate_kinship": reason_counts[EXCLUSION_CANDIDATE_KINSHIP],
        "n_candidates_selected": len(selected),
        "n_candidates_alternate_order": len(alternate),
        "n_candidates_shared_by_both_orders": len(selected_set & set(alternate)),
        "candidate_counts_by_stratum": published_strata,
        "candidate_strata_suppressed_below": args.suppress_below,
        "candidate_strata_suppressed": suppressed_strata,
        "n_edges_unique_to_one_configuration": {
            name: len(edges - set().union(*(other for key, other in edges_by_configuration.items() if key != name)))
            for name, edges in edges_by_configuration.items()
        },
        "selection_used_ancestry_or_population": False,
        "selection_used_historical_unrelated_flags": False,
        "configuration_chosen_after_seeing_counts": False,
        "gates": {gate: status for gate, (status, _) in sorted(gates.items())},
        "gate_details": {gate: detail for gate, (_, detail) in sorted(gates.items())},
        "sample_ids_emitted_in_public_summary": False,
    }
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failed = [gate for gate, (status, _) in gates.items() if status == "FAIL"]
    # REVIEW does not abort the PCA stage, and that is deliberate: an axis carried by few
    # individuals may be a family, a small differentiated population, an isolate, a
    # technical group or a metadata gap, and stopping a diagnostic on a measurement that
    # cannot tell those apart would be a false NO-GO.  Certifying a donor is a different
    # act, and the contract names it as the thing REVIEW blocks, so it blocks it here.
    blocking_review = contract["pca_axis_contract"]["g2b_axis_localization"]["review_blocks"]
    reviewed = (
        [gate for gate, (status, _) in gates.items() if status == "REVIEW"]
        if "donor_certification" in blocking_review
        else []
    )
    if (failed or reviewed) and not args.report_only:
        reasons = [f"failed: {', '.join(sorted(failed))}"] if failed else []
        if reviewed:
            reasons.append(
                f"unadjudicated review: {', '.join(sorted(reviewed))}. The contract blocks "
                "donor certification while an axis a configuration reads is localised and "
                "its cause has not been named"
            )
        raise SystemExit("M27D candidate gates did not clear: " + "; ".join(reasons))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, nargs="+", required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--call-rates", type=Path, required=True)
    parser.add_argument("--baseline-identities", type=Path, required=True)
    parser.add_argument("--m27c-samples", type=Path, default=None)
    parser.add_argument("--stage-summaries", type=Path, nargs="*", default=None)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--suppress-below", type=int, default=5)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--out-private", type=Path, required=True)
    parser.add_argument("--out-public", type=Path, required=True)
    parser.add_argument("--out-gates", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
