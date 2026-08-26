from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m34_adaptive_sweep", ROOT / "bin/m34_adaptive_sweep.py"
)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)
if importlib.util.find_spec("torch") is not None:
    MODEL_SPEC = importlib.util.spec_from_file_location(
        "m34_models_for_sweep", ROOT / "bin/m34_models.py"
    )
    MODELS = importlib.util.module_from_spec(MODEL_SPEC)
    assert MODEL_SPEC.loader is not None
    sys.modules[MODEL_SPEC.name] = MODELS
    MODEL_SPEC.loader.exec_module(MODELS)
else:
    MODELS = None
CONTRACT_PATH = ROOT / "conf/m34_adaptive_sweep_contract.json"


def contract():
    return SUBJECT.validate_contract(SUBJECT.strict_json(CONTRACT_PATH))


def record(
    family, config_id, arm, primary=0.70, guardrail=0.02, *, radius_cM=0.2,
    sweep_stage="triage", maximum_updates=300,
):
    return {
        "family": family,
        "config_id": config_id,
        "seed": 1103,
        "rotation": "R0",
        "arm": arm,
        "radius_cM": radius_cM,
        "sweep_stage": sweep_stage,
        "maximum_updates": maximum_updates,
        "boundary_F1_0.1cM": primary - 0.03,
        "boundary_F1_0.2cM": primary,
        "boundary_F1_0.5cM": primary + 0.03,
        "macro_ancestry_dose_MAE": guardrail,
        "NAM_truth_present_MAE": guardrail,
        "false_transitions_per_cM": guardrail,
        "haplotype_Brier": guardrail,
    }


def paired_records(
    family, config_id, delta, *, radius_cM=0.2, sweep_stage="triage",
    maximum_updates=300,
):
    return [
        record(
            family, config_id, "RD", primary=0.70, radius_cM=radius_cM,
            sweep_stage=sweep_stage, maximum_updates=maximum_updates,
        ),
        record(
            family, config_id, "RE", primary=0.70 + delta, radius_cM=radius_cM,
            sweep_stage=sweep_stage, maximum_updates=maximum_updates,
        ),
    ]


def record_for_task(task, primary):
    return record(
        task["family"], task["config_id"], task["arm"], primary=primary,
        radius_cM=task["radius_cM"], sweep_stage=task["sweep_stage"],
        maximum_updates=task["maximum_updates"],
    )


def append_plan_metrics(payload, plan, deltas_by_family=None, radius_bonus=None):
    deltas_by_family = deltas_by_family or {}
    radius_bonus = radius_bonus or {}
    for task in plan["tasks"]:
        delta = deltas_by_family.get(task["family"], 0.001)
        delta += radius_bonus.get(task["radius_cM"], 0.0)
        primary = 0.70 if task["arm"] == "RD" else 0.70 + delta
        payload["records"].append(record_for_task(task, primary))


def triage_payload(value, strong_families=()):
    records = []
    for family, specification in value["families"].items():
        for config_id in specification["triage_ids"]:
            rank = next(
                config["complexity_rank"]
                for config in specification["configs"]
                if config["id"] == config_id
            )
            delta = 0.001 * rank
            if family in strong_families and rank == 4:
                delta = 0.02
            records.extend(paired_records(family, config_id, delta))
    return {"records": records}


class M34AdaptiveSweepTests(unittest.TestCase):
    def test_contract_has_exact_registry_and_no_fixed_parameter_counts(self):
        value = contract()
        self.assertEqual(tuple(value["families"]), SUBJECT.EXPECTED_FAMILIES)
        for specification in value["families"].values():
            self.assertGreaterEqual(len(specification["triage_ids"]), 3)
            for config in specification["configs"]:
                self.assertEqual(tuple(config["model_spec"]), SUBJECT.MODEL_SPEC_FIELDS)
                self.assertNotIn("parameter_count", config)

    def test_boundary_weighting_is_fit_only_and_frozen(self):
        value = contract()
        self.assertEqual(value["boundary_loss"], {
            "target_transition_weight_share": 0.01,
            "formula": "beta=q*(1-p)/(p*(1-q))-1_fit_truth_only",
            "provenance": "M33_PRE4B",
        })
        broken = copy.deepcopy(value)
        broken["boundary_loss"]["target_transition_weight_share"] = 0.02
        with self.assertRaisesRegex(SUBJECT.ContractError, "boundary loss"):
            SUBJECT.validate_contract(broken)

    @unittest.skipIf(MODELS is None, "requires the pinned PyTorch runtime")
    def test_every_declared_config_constructs_through_the_real_model_registry(self):
        value = contract()
        observed = 0
        for family, specification in value["families"].items():
            for config in specification["configs"]:
                with self.subTest(family=family, config=config["id"]):
                    spec = MODELS.ModelSpec(
                        family=family,
                        channels=13,
                        ancestries=3,
                        **config["model_spec"],
                    )
                    model = MODELS.build_model(spec)
                    self.assertGreater(MODELS.parameter_count(model), 0)
                    observed += 1
        self.assertEqual(observed, 35)

    def test_triage_is_broad_paired_and_uses_one_seed_rotation(self):
        value = contract()
        plan = SUBJECT.triage_plan(value)
        self.assertEqual(plan["task_count"], 7 * 3 * 2)
        self.assertEqual({task["seed"] for task in plan["tasks"]}, {1103})
        self.assertEqual({task["rotation"] for task in plan["tasks"]}, {"R0"})
        self.assertEqual({task["arm"] for task in plan["tasks"]}, {"RD", "RE"})
        self.assertEqual({task["radius_cM"] for task in plan["tasks"]}, {0.2})
        self.assertEqual({task["maximum_updates"] for task in plan["tasks"]}, {300})
        paired = {}
        for task in plan["tasks"]:
            key = (task["family"], task["config_id"], task["seed"], task["rotation"])
            paired.setdefault(key, []).append(task)
        for tasks in paired.values():
            left, right = sorted(tasks, key=lambda row: row["arm"])
            self.assertEqual(
                {key: value for key, value in left.items() if key != "arm"},
                {key: value for key, value in right.items() if key != "arm"},
            )

    def test_local_expansion_is_deterministic_and_stays_inside_space(self):
        value = contract()
        pairs = SUBJECT.load_metric_pairs(value, triage_payload(value))
        first = SUBJECT.expansion_plan(value, pairs)
        second = SUBJECT.expansion_plan(value, pairs)
        self.assertEqual(first, second)
        self.assertEqual(first["task_count"], 7 * 2 * 2)
        self.assertEqual({task["maximum_updates"] for task in first["tasks"]}, {800})
        self.assertEqual({task["radius_cM"] for task in first["tasks"]}, {0.2})
        declared = {
            (family, config["id"])
            for family, specification in value["families"].items()
            for config in specification["configs"]
        }
        self.assertTrue(all((task["family"], task["config_id"]) in declared for task in first["tasks"]))

    def test_expansion_follows_the_best_local_trend(self):
        value = contract()
        payload = triage_payload(value)
        family = "local_linear"
        for row in payload["records"]:
            if row["family"] == family and row["config_id"] == "linear_r0" and row["arm"] == "RE":
                row["boundary_F1_0.1cM"] += 0.05
                row["boundary_F1_0.2cM"] += 0.05
                row["boundary_F1_0.5cM"] += 0.05
        pairs = SUBJECT.load_metric_pairs(value, payload)
        plan = SUBJECT.expansion_plan(value, pairs)
        self.assertEqual(plan["selected_config_ids_by_family"][family], ["linear_r1"])

    def test_no_family_can_be_decided_from_only_two_points(self):
        value = contract()
        payload = triage_payload(value)
        omitted = ("local_linear", "linear_r4")
        payload["records"] = [
            row for row in payload["records"]
            if (row["family"], row["config_id"]) != omitted
        ]
        pairs = SUBJECT.load_metric_pairs(value, payload)
        with self.assertRaisesRegex(SUBJECT.ContractError, "triage pair is missing"):
            SUBJECT.expansion_plan(value, pairs)

    def test_finalists_are_capped_at_two_and_repeat_three_seeds_rotations(self):
        value = contract()
        strong = tuple(value["families"])[0:3]
        payload = triage_payload(value, strong_families=strong)
        triage_pairs = SUBJECT.load_metric_pairs(value, payload)
        expansion = SUBJECT.expansion_plan(value, triage_pairs)
        append_plan_metrics(payload, expansion, {family: 0.01 for family in strong})
        pairs = SUBJECT.load_metric_pairs(value, payload)
        radius_plan = SUBJECT.radius_sensitivity_plan(value, pairs)
        self.assertEqual(radius_plan["family_count"], 2)
        self.assertEqual(radius_plan["task_count"], 2 * 3 * 2)
        self.assertEqual(
            {task["radius_cM"] for task in radius_plan["tasks"]}, {0.05, 0.1, 0.5}
        )
        self.assertEqual(radius_plan["reused_radius_cM_from_local_expansion"], 0.2)
        with self.assertRaisesRegex(SUBJECT.ContractError, "radius sensitivity pair is missing"):
            SUBJECT.finalist_plan(value, pairs)
        append_plan_metrics(payload, radius_plan, radius_bonus={0.1: 0.01})
        pairs = SUBJECT.load_metric_pairs(value, payload)
        plan = SUBJECT.finalist_plan(value, pairs)
        self.assertEqual(plan["finalist_count"], 2)
        self.assertEqual(plan["task_count"], 2 * 3 * 3 * 2)
        self.assertEqual({task["seed"] for task in plan["tasks"]}, {1103, 2207, 3301})
        self.assertEqual({task["rotation"] for task in plan["tasks"]}, {"R0", "R1", "R2"})
        self.assertEqual({task["maximum_updates"] for task in plan["tasks"]}, {2000})
        self.assertEqual({task["radius_cM"] for task in plan["tasks"]}, {0.1})

    def test_guardrail_worsening_blocks_promotion(self):
        value = contract()
        payload = triage_payload(value, strong_families=("local_linear",))
        for row in payload["records"]:
            if row["family"] == "local_linear" and row["arm"] == "RE":
                row["NAM_truth_present_MAE"] += 0.02
        triage_pairs = SUBJECT.load_metric_pairs(value, payload)
        expansion = SUBJECT.expansion_plan(value, triage_pairs)
        for task in expansion["tasks"]:
            row = record_for_task(task, 0.70)
            if task["family"] == "local_linear" and task["arm"] == "RE":
                row["boundary_F1_0.1cM"] += 0.02
                row["boundary_F1_0.2cM"] += 0.02
                row["boundary_F1_0.5cM"] += 0.02
                row["NAM_truth_present_MAE"] += 0.02
            payload["records"].append(row)
        pairs = SUBJECT.load_metric_pairs(value, payload)
        plan = SUBJECT.radius_sensitivity_plan(value, pairs)
        self.assertFalse(
            any(row["family"] == "local_linear" for row in plan["selected_architectures"])
        )

    def test_rank_only_fallback_is_predeclared_capped_and_not_called_promotion(self):
        value = contract()
        payload = triage_payload(value)
        triage_pairs = SUBJECT.load_metric_pairs(value, payload)
        expansion = SUBJECT.expansion_plan(value, triage_pairs)
        append_plan_metrics(payload, expansion, {
            family: 0.003 + index * 0.0001
            for index, family in enumerate(value["families"])
        })
        pairs = SUBJECT.load_metric_pairs(value, payload)
        plan = SUBJECT.radius_sensitivity_plan(value, pairs)
        self.assertEqual(plan["selection_mode"], "exploratory_rank_only_not_promoted")
        self.assertEqual(plan["family_count"], 2)

    def test_architecture_selection_compares_only_equal_medium_budgets(self):
        value = contract()
        payload = triage_payload(value, strong_families=("local_linear",))
        triage_pairs = SUBJECT.load_metric_pairs(value, payload)
        expansion = SUBJECT.expansion_plan(value, triage_pairs)
        self.assertEqual(expansion["task_count"], 28)
        append_plan_metrics(payload, expansion, {"local_linear": -0.01})
        pairs = SUBJECT.load_metric_pairs(value, payload)
        plan = SUBJECT.radius_sensitivity_plan(value, pairs)
        self.assertFalse(
            any(row["family"] == "local_linear" for row in plan["selected_architectures"])
        )
        self.assertEqual({task["maximum_updates"] for task in expansion["tasks"]}, {800})

    def test_undeclared_unpaired_and_nonfinite_metrics_fail_closed(self):
        value = contract()
        base = paired_records("local_linear", "linear_r0", 0.01)
        cases = []
        undeclared = copy.deepcopy(base)
        undeclared[0]["config_id"] = "invented_after_results"
        cases.append((undeclared, "undeclared"))
        cases.append((base[:1], "missing paired"))
        nonfinite = copy.deepcopy(base)
        nonfinite[1]["boundary_F1_0.2cM"] = float("nan")
        cases.append((nonfinite, "finite"))
        for records, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(SUBJECT.ContractError, message):
                SUBJECT.load_metric_pairs(value, {"records": records})


if __name__ == "__main__":
    unittest.main()
