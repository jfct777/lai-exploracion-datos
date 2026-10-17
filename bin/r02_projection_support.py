"""Descriptive influence diagnostics for a frozen FIT community projection.

No threshold selection, prediction scoring, or scientific admission occurs here.
Deleting reference components with labels fixed is NOT out-of-sample validation.
The caller authenticates roster, affinities, labels, and eligible genomic territory.
"""
from collections import Counter, defaultdict
import math

from r02_structure_discrete import observed_community_means

POLICY_ID = "OBSERVED_MEAN_SINGLE_COMPONENT_DELETION_DIAGNOSTIC_V1"


def component_influence(fit_labels, components, observations):
    """Summarize one query, preserving observed zeros and all competitors.

    Inputs use frozen FIT indices. Missing references prevent the diagnostic.
    Memory is O(N + G*C), where N=references, G=components, C=communities.
    No family reweighting or probabilistic independence claim is introduced.
    """
    if set(components) != set(fit_labels) or not all(
        isinstance(g, str) and g.strip() == g and g for g in components.values()
    ):
        raise ValueError("One explicit component per frozen FIT reference required")
    base = observed_community_means(fit_labels, observations)
    result = {
        "policy_id": POLICY_ID, "base": base,
        "biological_validation": False, "adequate_projection_support_certified": False,
        "calibration_status": "NOT_CALIBRATED",
        "uncertainty_scope": "FIXED_LABEL_REFERENCE_COMPONENT_INFLUENCE_NOT_CONFIDENCE_INTERVAL",
        "genomic_coverage_certified": False,
        "stable_under_all_single_component_deletions": False,
        "status": "INCOMPLETE_REFERENCE_SUPPORT",
    }
    groups = defaultdict(list)
    for i, label in fit_labels.items():
        if label != "UNASSIGNED":
            groups[label].append(i)
    support = []
    for label, indices in sorted(groups.items()):
        counts = Counter(components[i] for i in indices)
        n = len(indices)
        support.append({
            "community": label, "n_members": n,
            "n_observed": sum(i in observations for i in indices),
            "n_components": len(counts),
            "max_component_person_share": max(counts.values()) / n,
            "component_concentration_equivalent": n*n / sum(x*x for x in counts.values()),
            "concentration_scope": "DESCRIPTIVE_NOT_EFFECTIVE_INDEPENDENT_SAMPLE_SIZE",
        })
    result["community_support"] = support
    if set(observations) != set(fit_labels):
        return result
    labels = sorted(groups)
    active_components = sorted({components[i] for indices in groups.values() for i in indices})
    result["n_reference_components"] = len(active_components)
    if len(labels) < 2:
        result["status"] = "NO_COMMUNITY_CONTRAST"
        return result
    means = {row["community"]: row["mean_observed_affinity"] for row in base["community_support"]}
    winner = base["community"]
    result["base_margin"] = (means[winner] - max(v for c, v in means.items() if c != winner)
                             if winner is not None else None)
    if winner is None:
        result["status"] = base["status"]
        return result
    # Only affected means are recomputed. Sum remaining raw values with fsum:
    # total-minus-component can lose small surviving signals by cancellation.
    deleted = {}
    for label, indices in groups.items():
        affected = {components[i] for i in indices}
        for g in affected:
            remaining = [observations[i] for i in indices if components[i] != g]
            deleted[g, label] = math.fsum(remaining) / len(remaining) if remaining else None
    counts = Counter()
    margins = []
    for g in active_components:
        scores = {c: deleted.get((g, c), means[c]) for c in labels}
        if any(v is None for v in scores.values()):
            counts["empty_community_cases"] += 1
            continue
        high = max(scores.values())
        winners = [c for c, v in scores.items() if v == high]
        margins.append(scores[winner] - max(v for c, v in scores.items() if c != winner))
        if high == 0:
            counts["zero_winner_cases"] += 1
        elif len(winners) != 1:
            counts["tie_cases"] += 1
        elif winners[0] != winner:
            counts["changed_winner_cases"] += 1
    result["deletion"] = {
        "n_components_tested": len(active_components),
        **{k: counts[k] for k in ("empty_community_cases", "zero_winner_cases", "tie_cases", "changed_winner_cases")},
        "minimum_signed_winner_margin": min(margins) if margins else None,
    }
    stable = not any(counts.values())
    result["stable_under_all_single_component_deletions"] = stable
    result["status"] = ("STABLE_REFERENCE_COMPONENT_DIAGNOSTIC" if stable else
                        "FAMILY_DELETION_UNIDENTIFIABLE" if counts["empty_community_cases"] else
                        "REFERENCE_COMPONENT_SENSITIVE")
    return result
