"""Aggregate LAI log loss over uniquely owned queries before regional means.

Equal weight is assigned to cases, then components, then people.  Reused donors
do not become independent biological replicates through this descriptive score.
"""
from collections import defaultdict
import json
import numpy as np

RULE = "QUERY_SUM_PER_PERSON_THEN_MEAN_COMPONENTS_THEN_MEAN_CASES_V2"


def require(value, message):
    if not value:
        raise ValueError(message)


class QueryScore:
    def __init__(self):
        self.totals = {}
        self.people = {}
        self.owned = defaultdict(set)

    def add(self, case, data, prediction, observed):
        require(bool(case), "scoring requires an authenticated simulation case")
        core = np.asarray(data["query_core"])
        people = tuple(map(str, data["person_id"]))
        groups = tuple(map(str, data["component_id"]))
        require(len(set(people)) == len(people), "duplicate person in a scoring batch")
        require(core.dtype == bool and core.shape == observed.shape
                and np.array_equal(core, np.broadcast_to(core[:1], core.shape))
                and not (observed & ~core).any(), "fixed query ownership required")
        if case in self.people:
            require(self.people[case] == (people, groups),
                    "person/component axes changed between regions of one case")
        else:
            self.people[case] = (people, groups)
        chrom, bp = np.asarray(data["query_chrom"]), np.asarray(data["query_pos_bp"])
        require(chrom.shape == core.shape and bp.shape == core.shape
                and np.array_equal(chrom, np.broadcast_to(chrom[:1], chrom.shape))
                and np.array_equal(bp, np.broadcast_to(bp[:1], bp.shape)),
                "fixed query coordinates required")
        coordinates = list(zip(chrom[0, core[0]].tolist(), bp[0, core[0]].tolist()))
        require(len(set(coordinates)) == len(coordinates)
                and not self.owned[case].intersection(coordinates),
                "query core repeated across scoring regions")
        truth = np.asarray(data["truth_state"])
        probability = np.asarray(prediction)
        require(probability.shape == (*core.shape, 6), "six-state predictions required")
        for index, (person, component) in enumerate(zip(people, groups)):
            mask = observed[index]
            require(mask.any(), "no observed owned query for a scored person")
            states = truth[index, mask]
            require(np.isin(states, np.arange(6)).all(), "invalid observed ancestry state")
            p = probability[index, mask, states]
            require(np.isfinite(p).all() and (p > 0).all() and (p <= 1).all(),
                    "invalid true-state probability")
            key = (case, person, component)
            numerator, denominator = self.totals.get(key, (0.0, 0))
            self.totals[key] = (numerator + float(-np.log(p.astype(np.float64)).sum()),
                                denominator + int(mask.sum()))
        self.owned[case].update(coordinates)

    def result(self):
        errors, components, cases = {}, defaultdict(list), defaultdict(list)
        for (case, person, component), (numerator, denominator) in self.totals.items():
            value = numerator / denominator
            errors[json.dumps([case, person])] = [value]
            components[json.dumps([case, component])].append(value)
        for key, values in components.items():
            case, _ = json.loads(key)
            cases[case].append(sum(values) / len(values))
        require(bool(cases), "no scored LAI cases")
        metric = sum(sum(values) / len(values) for values in cases.values()) / len(cases)
        return metric, errors, dict(components)
