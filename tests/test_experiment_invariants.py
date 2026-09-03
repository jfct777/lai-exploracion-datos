#!/usr/bin/env python3

import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "_experiment_invariants.py"
SPEC = importlib.util.spec_from_file_location("experiment_invariants", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def role_rows():
    rows = []
    for role, person, unit, lineage in (
        ("TRAIN", "p1", "u1", "d1"),
        ("SELECT", "p2", "u2", "d2"),
        ("SCORE", "p3", "u3", "d3"),
    ):
        for haplotype in ("a", "b"):
            rows.append({
                "person_id": person,
                "haplotype_id": f"{person}_{haplotype}",
                "atomic_unit_id": unit,
                "donor_lineage_id": lineage,
                "role": role,
            })
    return rows


def artifact_inventory():
    hashes = {name: str(index) * 64 for index, name in enumerate(
        ("train", "select", "checkpoint", "score"), 1
    )}
    artifacts = [
        {
            "artifact_id": "train_features", "purpose": "input",
            "sha256": hashes["train"], "roles": ["TRAIN"],
            "data_kinds": ["features"], "depends_on": [],
        },
        {
            "artifact_id": "selector", "purpose": "selection",
            "sha256": hashes["select"], "roles": ["SELECT"],
            "data_kinds": ["selection_metric", "select_truth"],
            "depends_on": ["train_features"],
        },
        {
            "artifact_id": "checkpoint", "purpose": "checkpoint",
            "sha256": hashes["checkpoint"], "roles": ["SELECT"],
            "data_kinds": ["model_parameters"],
            "depends_on": ["selector"],
        },
        {
            "artifact_id": "score_truth", "purpose": "score",
            "sha256": hashes["score"], "roles": ["SCORE"],
            "data_kinds": ["local_ancestry_truth"], "depends_on": [],
        },
    ]
    replay = {
        "selector": {
            "without_score": hashes["select"], "with_score": hashes["select"],
        },
        "checkpoint": {
            "without_score": hashes["checkpoint"],
            "with_score": hashes["checkpoint"],
        },
    }
    return artifacts, replay


def signatures():
    fixture = {
        "feature_names": ["F0", "rare_llr", "callability"],
        "event_rate": 0.10,
        "sparsity": 0.90,
        "missingness": 0.02,
        "class_proportions": {"AFR": 0.3, "EUR": 0.5, "NAM": 0.2},
        "component_counts": {"connected": 4, "isolated": 1},
    }
    production = {
        "feature_names": ["F0", "rare_llr", "callability"],
        "event_rate": 0.11,
        "sparsity": 0.89,
        "missingness": 0.021,
        "class_proportions": {"AFR": 0.31, "EUR": 0.49, "NAM": 0.2},
        "component_counts": {"connected": 5, "isolated": 1},
    }
    tolerances = {
        "event_rate_abs": 0.02,
        "sparsity_abs": 0.02,
        "missingness_abs": 0.005,
        "class_proportion_abs": 0.02,
        "class_sum_abs": 1e-12,
        "component_count_abs": 1,
    }
    return fixture, production, tolerances


def null_pair():
    observed = {
        "locus_axis": [
            ("chr22", 10, "a", "g"),
            ("22", 20, "C", "T"),
        ],
        "positions": [0.1, 0.2],
        "masks": [[1, 1], [1, 0]],
        "dosage": [[0, 1], [2, 0]],
        "burden": [1, 2],
        "unit_ids": ["u1", "u2"],
        "ancestry_mapping": {"u1": "AFR", "u2": "NAM"},
        "source_axis": ["p1", "p2"],
    }
    null = copy.deepcopy(observed)
    null["ancestry_mapping"] = {"u1": "NAM", "u2": "AFR"}
    return observed, null


def standard_gates(default=True):
    return {
        name: default
        for _, names in MODULE.STANDARD_CLAIM_REQUIREMENTS
        for name in names
    }


class LocusPartitionTests(unittest.TestCase):
    def setUp(self):
        self.full = [
            ("chr22", 10, "a", "g"),
            ("22", 20, "C", "T"),
            ("22", 30, "G", "A"),
            ("22", 40, "T", "C"),
        ]

    def test_exact_partition_normalizes_real_variant_axis(self):
        result = MODULE.validate_exact_locus_partition(
            self.full, [self.full[0], self.full[2]], [self.full[1], self.full[3]]
        )
        self.assertEqual(result["counts"], {
            "F_full": 4, "F_minus_selected": 2, "selected": 2, "overlap": 0,
        })
        self.assertTrue(result["order_preserved"])
        self.assertEqual(len(result["axis_sha256"]["F_full"]), 64)

    def test_rejects_overlap_duplicates_and_reordering(self):
        with self.assertRaisesRegex(ValueError, "intersects"):
            MODULE.validate_exact_locus_partition(
                self.full, self.full[:3], self.full[2:]
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.validate_exact_locus_partition(
                self.full + [self.full[0]], self.full[:2], self.full[2:]
            )
        with self.assertRaisesRegex(ValueError, "ordered|preserve.*order"):
            MODULE.validate_exact_locus_partition(
                self.full, [self.full[2], self.full[0]], [self.full[1], self.full[3]]
            )

    def test_rejects_position_only_and_non_explicit_alleles(self):
        with self.assertRaisesRegex(ValueError, "CHROM/POS/REF/ALT"):
            MODULE.normalize_locus_axis([(22, 10)])
        with self.assertRaisesRegex(ValueError, "explicit DNA allele"):
            MODULE.normalize_locus_axis([(22, 10, "A", "<DEL>")])


class RoleSeparationTests(unittest.TestCase):
    def test_complete_disjoint_roles_pass(self):
        result = MODULE.validate_role_separation(role_rows())
        self.assertEqual(result["counts"]["TRAIN"]["people"], 1)
        self.assertEqual(result["counts"]["SCORE"]["haplotypes"], 2)
        self.assertEqual(result["no_cross_fields"], [
            "person_id", "atomic_unit_id", "donor_lineage_id",
        ])

    def test_rejects_incomplete_person_and_atomic_unit_crossing(self):
        incomplete = role_rows()[:-1]
        with self.assertRaisesRegex(ValueError, "complete haplotypes"):
            MODULE.validate_role_separation(incomplete)
        crossing = role_rows()
        crossing[2]["atomic_unit_id"] = "u1"
        crossing[3]["atomic_unit_id"] = "u1"
        with self.assertRaisesRegex(ValueError, "atomic_unit_id crosses roles"):
            MODULE.validate_role_separation(crossing)

    def test_rejects_donor_lineage_and_person_role_crossing(self):
        lineage = role_rows()
        lineage[4]["donor_lineage_id"] = "d1"
        lineage[5]["donor_lineage_id"] = "d1"
        with self.assertRaisesRegex(ValueError, "donor_lineage_id crosses roles"):
            MODULE.validate_role_separation(lineage)
        person = role_rows()
        person[2]["person_id"] = "p1"
        person[3]["person_id"] = "p1"
        with self.assertRaisesRegex(ValueError, "inconsistent role|person_id crosses roles"):
            MODULE.validate_role_separation(person)

    def test_atomic_unit_crossing_can_be_explicitly_relaxed(self):
        rows = role_rows()
        rows[2]["atomic_unit_id"] = "u1"
        rows[3]["atomic_unit_id"] = "u1"
        result = MODULE.validate_role_separation(
            rows, no_cross_fields=("person_id", "donor_lineage_id")
        )
        self.assertEqual(result["no_cross_fields"], ["person_id", "donor_lineage_id"])


class SelectionIsolationTests(unittest.TestCase):
    def test_selection_and_checkpoint_are_score_blind(self):
        artifacts, replay = artifact_inventory()
        result = MODULE.validate_selection_isolation(artifacts, replay)
        self.assertEqual(result["selector_artifacts"], ["checkpoint", "selector"])
        self.assertTrue(result["score_invariant_replay"])

    def test_rejects_transitive_truth_dependency(self):
        artifacts, replay = artifact_inventory()
        artifacts[1]["depends_on"].append("score_truth")
        with self.assertRaisesRegex(ValueError, "depends on SCORE"):
            MODULE.validate_selection_isolation(artifacts, replay)

    def test_rejects_score_dependent_selector_hash(self):
        artifacts, replay = artifact_inventory()
        replay["checkpoint"]["with_score"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "changes selector hash"):
            MODULE.validate_selection_isolation(artifacts, replay)

    def test_rejects_untracked_dependency_and_cycle(self):
        artifacts, replay = artifact_inventory()
        artifacts[1]["depends_on"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "missing dependencies"):
            MODULE.validate_selection_isolation(artifacts, replay)
        artifacts, replay = artifact_inventory()
        artifacts[0]["depends_on"] = ["checkpoint"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            MODULE.validate_selection_isolation(artifacts, replay)


class FixtureProductionTests(unittest.TestCase):
    def test_production_matched_fixture_passes_with_explicit_tolerances(self):
        fixture, production, tolerances = signatures()
        result = MODULE.validate_fixture_production_signature(
            fixture, production, tolerances
        )
        self.assertEqual(result["status"], "PASS_FIXTURE_PRODUCTION_SIGNATURE")
        self.assertAlmostEqual(result["absolute_deltas"]["event_rate"], 0.01)

    def test_rejects_unmatched_feature_axis_and_event_rate(self):
        fixture, production, tolerances = signatures()
        production["feature_names"] = ["F0", "rare_llr"]
        with self.assertRaisesRegex(ValueError, "feature axes"):
            MODULE.validate_fixture_production_signature(
                fixture, production, tolerances
            )
        fixture, production, tolerances = signatures()
        production["event_rate"] = 0.2
        with self.assertRaisesRegex(ValueError, "event_rate delta"):
            MODULE.validate_fixture_production_signature(
                fixture, production, tolerances
            )

    def test_rejects_class_or_component_drift_and_implicit_tolerances(self):
        fixture, production, tolerances = signatures()
        production["class_proportions"] = {"AFR": 0.4, "EUR": 0.4, "NAM": 0.2}
        with self.assertRaisesRegex(ValueError, "class proportion delta"):
            MODULE.validate_fixture_production_signature(
                fixture, production, tolerances
            )
        fixture, production, tolerances = signatures()
        production["component_counts"]["connected"] = 8
        with self.assertRaisesRegex(ValueError, "component count delta"):
            MODULE.validate_fixture_production_signature(
                fixture, production, tolerances
            )
        fixture, production, tolerances = signatures()
        tolerances.pop("event_rate_abs")
        with self.assertRaisesRegex(ValueError, "tolerance fields"):
            MODULE.validate_fixture_production_signature(
                fixture, production, tolerances
            )


class NullInvariantTests(unittest.TestCase):
    def test_only_ancestry_mapping_changes(self):
        observed, null = null_pair()
        result = MODULE.validate_null_invariants(observed, null)
        self.assertEqual(result["allowed_changed_fields"], ["ancestry_mapping"])
        self.assertTrue(result["ancestry_class_counts_preserved"])

    def test_rejects_locus_mask_dosage_burden_or_unit_changes(self):
        mutations = {
            "locus_axis": lambda value: value.__setitem__(0, ("22", 11, "A", "G")),
            "masks": lambda value: value[0].__setitem__(0, 0),
            "dosage": lambda value: value[0].__setitem__(0, 1),
            "burden": lambda value: value.__setitem__(0, 2),
            "unit_ids": lambda value: value.__setitem__(0, "u9"),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                observed, null = null_pair()
                mutate(null[field])
                with self.assertRaisesRegex(ValueError, "null changes"):
                    MODULE.validate_null_invariants(observed, null)

    def test_rejects_position_change_or_class_count_change(self):
        observed, null = null_pair()
        null["positions"][0] = 0.11
        with self.assertRaisesRegex(ValueError, "positions"):
            MODULE.validate_null_invariants(observed, null)
        observed, null = null_pair()
        null["ancestry_mapping"] = {"u1": "NAM", "u2": "NAM"}
        with self.assertRaisesRegex(ValueError, "class counts"):
            MODULE.validate_null_invariants(observed, null)

    def test_rejects_identity_null_by_default(self):
        observed, _ = null_pair()
        with self.assertRaisesRegex(ValueError, "unchanged"):
            MODULE.validate_null_invariants(observed, copy.deepcopy(observed))


class ClaimContractTests(unittest.TestCase):
    def test_claim_level_follows_gates_and_hashes_contract(self):
        gates = standard_gates()
        result = MODULE.build_claim_level_contract(
            gates, evidence_scope={"chromosome": "22", "roots": ["R0", "R1", "R2"]}
        )
        self.assertEqual(result["claim_level"], "CONFIRMATORY")
        self.assertEqual(result["status"], "PASS_CONFIRMATORY")
        self.assertEqual(len(result["contract_sha256"]), 64)
        repeated = MODULE.build_claim_level_contract(
            dict(reversed(list(gates.items()))),
            evidence_scope={"roots": ["R0", "R1", "R2"], "chromosome": "22"},
        )
        self.assertEqual(result["contract_sha256"], repeated["contract_sha256"])

    def test_claim_downgrades_instead_of_overclaiming(self):
        gates = standard_gates()
        gates["effect_replicated"] = False
        result = MODULE.build_claim_level_contract(gates)
        self.assertEqual(result["claim_level"], "EXPLORATORY")
        self.assertEqual(result["next_level"], "CONFIRMATORY")
        self.assertEqual(result["blocking_gates"], ["effect_replicated"])

        gates["locus_partition_valid"] = False
        result = MODULE.build_claim_level_contract(gates)
        self.assertEqual(result["claim_level"], "NO_DEFENSIBLE_CLAIM")
        self.assertEqual(result["status"], "STOP_FAILED_INTEGRITY_GATES")

    def test_rejects_non_boolean_missing_or_non_cumulative_gate_contract(self):
        gates = standard_gates()
        gates["power_adequate"] = 1
        with self.assertRaisesRegex(ValueError, "must be booleans"):
            MODULE.build_claim_level_contract(gates)
        gates = standard_gates()
        with self.assertRaisesRegex(ValueError, "not cumulative"):
            MODULE.build_claim_level_contract(
                gates,
                requirements=(("LOW", ("inputs_authenticated",)),
                              ("HIGH", ("schemas_valid",))),
            )
        gates = standard_gates()
        gates["unused_gate"] = True
        with self.assertRaisesRegex(ValueError, "not represented exactly"):
            MODULE.build_claim_level_contract(gates)


if __name__ == "__main__":
    unittest.main()
