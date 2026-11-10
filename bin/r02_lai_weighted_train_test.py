"""Bounded synthetic tests for GT weighted-plan conversion and training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import r02_lai_donor_experiment as experiment
from r02_lai_donor_context import DonorContextBatch
from r02_lai_donor_head_test import batch as one_query_batch, model
from r02_lai_donor_experiment import load_config, WEIGHTED_EXPERIMENT_SCHEMA
from r02_lai_donor_train import (
    AdamSpecification, DONOR_OBJECTIVE, DonorTrainingBatch, SYNTHETIC_SCOPE,
    TRAINING_SCHEMA, WEIGHTED_DONOR_OBJECTIVE,
    WeightedLogicalBatchPlan, WeightedMicrobatchPlan,
    build_adam, donor_training_identity, donor_weighted_occurrence_loss,
    run_donor_training_call,
)
from r02_lai_fit_query_plan import make_fit_plan
from r02_lai_weighted_train import (
    WEIGHTED_REFERENCE_SCHEMA, load_weighted_fit_plan, sha256,
    validate_weighted_plan_reference,
)


def tiny_context() -> DonorContextBatch:
    return DonorContextBatch(
        target_alt_dosage=torch.tensor([[0.]]),
        reference_alt_dosage=torch.tensor([[0.], [1.], [2.]]),
        target_observed=torch.ones((1, 1), dtype=torch.bool),
        reference_observed=torch.ones((3, 1), dtype=torch.bool),
        ref_alt_alleles=torch.tensor([[0, 1]]),
        rare_catalog=torch.tensor([False]), locus_cm=torch.tensor([0.]),
        locus_chrom=torch.tensor([22]),
        query_cm=torch.tensor([[0., .1, .2]]),
        query_chrom=torch.tensor([[22, 22, 22]]),
        query_valid=torch.ones((1, 3), dtype=torch.bool),
        reference_panel=torch.tensor([0, 1, 2]),
        donor_allowed=torch.ones((1, 3), dtype=torch.bool))


class FixedLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(
            [[.3, -.2, .1, .4, -.1, .2],
             [-.4, .5, .2, -.1, .3, .0],
             [.1, .2, -.3, .0, .6, -.2]], dtype=torch.float64))

    def forward(self, context):
        return SimpleNamespace(logits=self.logits[None, :, :],
                               support=torch.ones((1, 3), dtype=torch.bool))


def training_batch(identity: str, query_indices=(0, 1, 2), query_core=None):
    if query_core is None:
        query_core = torch.ones((1, 3), dtype=torch.bool)
    return DonorTrainingBatch(
        identity, ("p0",), tiny_context(), torch.tensor([[0, 3, 4]]),
        torch.ones((1, 3), dtype=torch.bool), query_core,
        torch.tensor([list(query_indices)]))


class WeightedTrainingTests(unittest.TestCase):
    def assert_nested_equal(self, expected, observed):
        """Compare every tensor and scalar in an optimizer/RNG state."""
        self.assertEqual(type(expected), type(observed))
        if torch.is_tensor(expected):
            torch.testing.assert_close(expected, observed, rtol=0, atol=0)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, observed)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), observed.keys())
            for key in expected:
                self.assert_nested_equal(expected[key], observed[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(expected), len(observed))
            for left, right in zip(expected, observed):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(expected, observed)

    def specification(self, gradient_clip_norm=None):
        return AdamSpecification(
            learning_rate=.001, betas=(.9, .999), epsilon=1e-8,
            weight_decay=0., gradient_clip_norm=gradient_clip_norm, amsgrad=False,
            maximize=False, foreach=False, capturable=False,
            differentiable=False, fused=False)

    def test_full_raw_occurrences_equal_chunked_aggregated_objective_and_gradient(self):
        # Query 0 occurs twice with different coefficients; aggregation must sum
        # 0.1 + 0.3 rather than keep one value or renormalize the microbatch.
        first = WeightedMicrobatchPlan(
            "a" * 64, "r0", 0, 2, ("p0",), ((0, 1),), ((.4, .2),), ((2, 1),))
        second = WeightedMicrobatchPlan(
            "b" * 64, "r1", 2, 3, ("p0",), ((2,),), ((.4,),), ((1,),))
        plan = WeightedLogicalBatchPlan(0, "weighted-u000", (first, second), 4)
        self.assertEqual(plan.occurrence_count, 4)
        chunked = FixedLogits()
        for microbatch, core in ((first, torch.tensor([[1, 1, 0]], dtype=torch.bool)),
                                 (second, torch.tensor([[0, 0, 1]], dtype=torch.bool))):
            result = donor_weighted_occurrence_loss(
                chunked, training_batch(microbatch.identity_sha256,
                                        query_core=core), microbatch)
            result.person_contributions.sum().backward()
        chunked_gradient = chunked.logits.grad.detach().clone()

        direct = FixedLogits()
        nll = -F.log_softmax(direct.logits, dim=-1)[
            torch.tensor([0, 1, 0, 2]), torch.tensor([0, 3, 0, 4])]
        objective = (nll * torch.tensor([.1, .2, .3, .4], dtype=nll.dtype)).sum()
        objective.backward()
        torch.testing.assert_close(chunked_gradient, direct.logits.grad,
                                   rtol=0, atol=1e-15)
        expected_value = float(objective.detach())
        chunked_value = 0.0
        fresh = FixedLogits()
        for microbatch, core in ((first, torch.tensor([[1, 1, 0]], dtype=torch.bool)),
                                 (second, torch.tensor([[0, 0, 1]], dtype=torch.bool))):
            chunked_value += float(donor_weighted_occurrence_loss(
                fresh, training_batch(microbatch.identity_sha256,
                                      query_core=core), microbatch
                ).person_contributions.detach().sum())
        self.assertAlmostEqual(chunked_value, expected_value, places=15)

    def _resume_fixture(self, root: Path):
        batches = []
        microbatches = []
        for index, weight in enumerate((.3, .7)):
            identity = hashlib.sha256(f"batch-{index}".encode()).hexdigest()
            microbatches.append(WeightedMicrobatchPlan(
                identity, f"r{index}", index, index + 1,
                ("p",), ((index,),), ((weight,),), ((1,),)))
            base = one_query_batch()
            batches.append(DonorTrainingBatch(
                identity, ("p",), base, torch.tensor([[index]]),
                torch.ones((1, 1), dtype=torch.bool),
                torch.ones((1, 1), dtype=torch.bool), torch.tensor([[index]])))
        plan = WeightedLogicalBatchPlan(0, "weighted-u000", tuple(microbatches), 2)
        source = root / "source.json"; source.write_text("{}\n")
        policy = root / "policy.json"; policy.write_text("{}\n")
        return plan, tuple(batches), source, policy

    def test_interrupted_resume_is_exact_and_v1_objective_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, batches, source, policy = self._resume_fixture(root)
            specification = self.specification()
            torch.manual_seed(91); direct_model = model()
            direct_optimizer = build_adam(direct_model, specification)
            identity = donor_training_identity(
                task_id="weighted-resume", data_scope=SYNTHETIC_SCOPE,
                model=direct_model, optimizer=direct_optimizer,
                optimizer_specification=specification,
                source_paths={"source": source}, policy_path=policy,
                objective=WEIGHTED_DONOR_OBJECTIVE, seed=91,
                logical_batches=(plan,))
            with self.assertRaisesRegex(ValueError, "objective differs"):
                donor_training_identity(
                    task_id="wrong-v1", data_scope=SYNTHETIC_SCOPE,
                    model=direct_model, optimizer=direct_optimizer,
                    optimizer_specification=specification,
                    source_paths={"source": source}, policy_path=policy,
                    objective=DONOR_OBJECTIVE, seed=91, logical_batches=(plan,))
            random.seed(17); np.random.seed(17); torch.manual_seed(17)
            direct = run_donor_training_call(
                direct_model, direct_optimizer, specification, root / "direct",
                identity, (plan,), batches, max_microbatches_this_call=None)
            direct_python_rng = random.getstate()
            direct_numpy_rng = np.random.get_state()
            direct_rng = torch.get_rng_state().clone()

            torch.manual_seed(91); partial_model = model()
            partial_optimizer = build_adam(partial_model, specification)
            resumed_identity = donor_training_identity(
                task_id="weighted-resume", data_scope=SYNTHETIC_SCOPE,
                model=partial_model, optimizer=partial_optimizer,
                optimizer_specification=specification,
                source_paths={"source": source}, policy_path=policy,
                objective=WEIGHTED_DONOR_OBJECTIVE, seed=91,
                logical_batches=(plan,))
            self.assertEqual(identity, resumed_identity)
            random.seed(17); np.random.seed(17); torch.manual_seed(17)
            partial = run_donor_training_call(
                partial_model, partial_optimizer, specification, root / "resumed",
                resumed_identity, (plan,), batches, max_microbatches_this_call=1)
            self.assertEqual(partial.status, "PARTIAL")

            changed_first = WeightedMicrobatchPlan(
                plan.microbatches[0].identity_sha256, "r0", 0, 1,
                ("p",), ((0,),),
                ((.31,),), ((1,),))
            changed_second = WeightedMicrobatchPlan(
                plan.microbatches[1].identity_sha256, "r1", 1, 2,
                ("p",), ((1,),),
                ((.69,),), ((1,),))
            changed_plan = WeightedLogicalBatchPlan(
                0, plan.logical_batch_id, (changed_first, changed_second), 2)
            torch.manual_seed(91); changed_model = model()
            changed_optimizer = build_adam(changed_model, specification)
            changed_identity = donor_training_identity(
                task_id="weighted-resume", data_scope=SYNTHETIC_SCOPE,
                model=changed_model, optimizer=changed_optimizer,
                optimizer_specification=specification,
                source_paths={"source": source}, policy_path=policy,
                objective=WEIGHTED_DONOR_OBJECTIVE, seed=91,
                logical_batches=(changed_plan,))
            self.assertNotEqual(identity, changed_identity)
            with self.assertRaisesRegex(ValueError, "identity"):
                run_donor_training_call(
                    changed_model, changed_optimizer, specification,
                    root / "resumed", changed_identity, (changed_plan,), batches,
                    max_microbatches_this_call=None)

            # A real process restart has no resident gradients or Adam state.
            del partial_model, partial_optimizer
            torch.manual_seed(91); resumed_model = model()
            resumed_optimizer = build_adam(resumed_model, specification)
            resumed = run_donor_training_call(
                resumed_model, resumed_optimizer, specification, root / "resumed",
                resumed_identity, (plan,), batches, max_microbatches_this_call=None)
            self.assertEqual(direct.history, resumed.history)
            for name, value in direct_model.state_dict().items():
                torch.testing.assert_close(value, resumed_model.state_dict()[name],
                                           rtol=0, atol=0)
            self.assert_nested_equal(direct_optimizer.state_dict(),
                                     resumed_optimizer.state_dict())
            self.assert_nested_equal(direct_python_rng, random.getstate())
            self.assert_nested_equal(direct_numpy_rng, np.random.get_state())
            self.assertTrue(torch.equal(direct_rng, torch.get_rng_state()))

    def test_weighted_clipping_and_adam_step_occur_once_after_all_microbatches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, batches, source, policy = self._resume_fixture(root)
            specification = self.specification(gradient_clip_norm=.05)
            torch.manual_seed(91); trained_model = model()
            optimizer = build_adam(trained_model, specification)
            identity = donor_training_identity(
                task_id="weighted-clipping", data_scope=SYNTHETIC_SCOPE,
                model=trained_model, optimizer=optimizer,
                optimizer_specification=specification,
                source_paths={"source": source}, policy_path=policy,
                objective=WEIGHTED_DONOR_OBJECTIVE, seed=91,
                logical_batches=(plan,))
            wrong_schema = dict(identity); wrong_schema["schema"] = TRAINING_SCHEMA
            with self.assertRaisesRegex(ValueError, "identity differs"):
                run_donor_training_call(
                    trained_model, optimizer, specification, root / "wrong-schema",
                    wrong_schema, (plan,), batches,
                    max_microbatches_this_call=None)
            with patch("torch.nn.utils.clip_grad_norm_",
                       wraps=torch.nn.utils.clip_grad_norm_) as clip, \
                 patch.object(optimizer, "step", wraps=optimizer.step) as step:
                result = run_donor_training_call(
                    trained_model, optimizer, specification, root / "clipped",
                    identity, (plan,), batches, max_microbatches_this_call=None)
            self.assertEqual(result.status, "COMPLETE")
            self.assertEqual(clip.call_count, 1)
            self.assertEqual(step.call_count, 1)


class WeightedPlanConversionTests(unittest.TestCase):
    def prepare(self, root: Path, *, role="FIT", alter_weight=False):
        truth = np.array([[0, 1, 2, 3, 4, 5],
                          [0, 1, 1, 3, 3, 5]], dtype=np.int64)
        observed = np.ones_like(truth, dtype=np.bool_)
        rows, weights, metadata = make_fit_plan(
            truth, observed, role="FIT", seed=0, updates=1,
            blocks_per_update=8, width=3)
        if alter_weight:
            weights = weights.copy(); weights[0] += .01; weights[1] -= .01
        source_spec = root / "input.json"; source_spec.write_text("{}\n")
        overlay = root / "overlay.npz"
        np.savez(overlay, truth_state=truth, truth_observed=observed)
        occurrences = root / "OCCURRENCES.npz"
        np.savez(occurrences, occurrence_rows=rows, loss_coefficient=weights)
        metadata.update(
            role=role, prepared_only=True, trainer_v1_accepts_plan=False,
            source_spec={"path": str(source_spec), "sha256": sha256(source_spec)},
            sources={"query_truth_overlay":
                     {"path": str(overlay), "sha256": sha256(overlay)}},
            row_columns=["update", "block", "person_index", "query_index",
                         "anchor_query_index"], occurrences=len(rows),
            unique_cells=len(set(map(tuple, rows[:, 2:4]))),
            sampled_people=len(np.unique(rows[:, 2])),
            sampled_state_counts=np.bincount(
                truth[rows[:, 2], rows[:, 3]], minlength=6).tolist(),
            weight_sum_per_update=[float(weights.sum())],
            occurrences_sha256=sha256(occurrences))
        plan = root / "PLAN.json"
        plan.write_text(json.dumps(metadata, indent=2) + "\n")
        reference_value = {
            "schema": WEIGHTED_REFERENCE_SCHEMA,
            "plan_json": str(plan), "plan_sha256": sha256(plan),
            "occurrences_npz": str(occurrences),
            "occurrences_sha256": sha256(occurrences)}
        reference = validate_weighted_plan_reference(reference_value, root)
        return reference, reference_value, truth, observed, source_spec, overlay

    def test_weighted_experiment_run_reaches_real_trainer_and_versioned_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference_value, truth, observed, source_spec, overlay = self.prepare(root)
            config_path = root / "weighted-config.json"; config_path.write_text("{}\n")
            built_model = model()
            specification = WeightedTrainingTests().specification()
            report_block = experiment.QueryBlock("fixed-report", 0, 0, 1)
            config = {
                "schema": WEIGHTED_EXPERIMENT_SCHEMA,
                "task_id": "synthetic-weighted-end-to-end", "plan_seed": 0,
                "model_seed": 91, "cpu_threads": 1,
                "model": built_model.configuration,
                "reference_policy": {"expected_reference_people": 3,
                                     "require_every_reference_allowed": True,
                                     "max_output_genotypes": 1000},
                "fit_report": {"label": "FIT_DEVELOPMENT_FIXED_LIST",
                               "region_ids": [report_block.region_id]},
                "weighted_fit_plan": reference_value,
                "max_microbatches_per_call": 1,
            }

            class SyntheticReader:
                def __init__(self, _sources):
                    self.paths = {"query_overlay": overlay}
                    self.query = {
                        "query_valid": observed.copy(),
                        "truth_observed": observed.copy(),
                        "truth_state": truth.copy(),
                        "person_id": np.asarray(["p0", "p1"]),
                    }

                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return False

                def assemble_region(self, ownership, region_id, *, person_indices,
                                    **_arguments):
                    person = person_indices[0]
                    owner = ownership.region_ids.index(region_id)
                    indices = np.flatnonzero(ownership.owner[person] == owner)
                    base = one_query_batch()
                    count = len(indices)
                    context = DonorContextBatch(
                        target_alt_dosage=base.target_alt_dosage,
                        reference_alt_dosage=base.reference_alt_dosage,
                        target_observed=base.target_observed,
                        reference_observed=base.reference_observed,
                        ref_alt_alleles=base.ref_alt_alleles,
                        rare_catalog=base.rare_catalog,
                        locus_cm=base.locus_cm, locus_chrom=base.locus_chrom,
                        query_cm=torch.as_tensor(indices[None, :] / 10,
                                                 dtype=torch.float32),
                        query_chrom=torch.full((1, count), 22, dtype=torch.int64),
                        query_valid=torch.ones((1, count), dtype=torch.bool),
                        reference_panel=base.reference_panel,
                        donor_allowed=base.donor_allowed)
                    targets = SimpleNamespace(
                        truth_state=torch.as_tensor(truth[person, indices][None, :]),
                        truth_observed=torch.ones((1, count), dtype=torch.bool),
                        query_global_index=torch.as_tensor(indices[None, :]),
                    )
                    return SimpleNamespace(
                        batch=context, targets=targets,
                        lineage={"reference_people": 3,
                                 "reference_counts_allowed_by_panel": {"all": 3}})

            def synthetic_evaluate(_model, _reader, _ownership, blocks,
                                   current_config, condition, _prior):
                return {
                    "fixed_list_sha256": experiment.canonical_sha256(
                        [block.receipt() for block in blocks]),
                    "condition": condition,
                    "experiment_configuration_sha256":
                        experiment.canonical_sha256(current_config),
                    "scope": "SYNTHETIC_END_TO_END",
                }

            admission = {"scope": "SYNTHETIC_SOFTWARE_FIXTURE"}
            with patch.object(experiment, "load_config", return_value=config), \
                 patch.object(experiment, "sources_from_spec", return_value=object()), \
                 patch.object(experiment, "require_training_catalogue_coverage",
                              return_value=admission), \
                 patch.object(experiment, "DonorInputReader", SyntheticReader), \
                 patch.object(experiment, "_model", return_value=built_model), \
                 patch.object(experiment, "_adam", return_value=specification), \
                 patch.object(experiment, "parse_region_blocks",
                              return_value=(report_block,)), \
                 patch.object(experiment, "evaluate", side_effect=synthetic_evaluate):
                result = experiment.run(
                    source_spec, config_path, "NO_SOURCE_RARE", root / "output")
            self.assertEqual(
                result["schema"],
                "r02_lai_donor_experiment_result_v2_occurrence_weighted")
            self.assertEqual(result["completed_updates"], 1)
            self.assertEqual(len(result["history_weighted_occurrence_nll"]), 1)
            self.assertIn("unsupported_occurrences_last_call", result)
            conversion = result["weighted_fit_training_conversion"]
            self.assertEqual(conversion["updates"], 1)
            self.assertTrue(conversion["physical_segment_receipts"])
            self.assertEqual(
                json.loads((root / "output/RESULT.json").read_text())["schema"],
                result["schema"])

    def test_converter_authenticates_formula_and_aggregates_heterogeneous_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, reference_value, truth, observed, source, overlay = self.prepare(
                Path(directory))
            loaded = load_weighted_fit_plan(
                reference, person_ids=("p0", "p1"), query_valid=observed,
                truth_observed=observed, truth_state=truth,
                source_spec_path=source, query_truth_overlay_path=overlay)
            plan = loaded.logical_batches[0]
            self.assertEqual(plan.declared_occurrences, 24)
            self.assertLess(plan.unique_scored_queries, plan.declared_occurrences)
            heterogeneous = False
            for microbatch in plan.microbatches:
                for weights, counts in zip(microbatch.loss_coefficients,
                                           microbatch.occurrence_counts):
                    heterogeneous |= any(count > 1 for count in counts)
                    self.assertTrue(all(weight > 0 for weight in weights))
            self.assertTrue(heterogeneous)
            self.assertAlmostEqual(plan.coefficient_sum, 1.0)
            # Global contiguous construction never joins query 5 back to query 0.
            self.assertTrue(all(segment.query_start < segment.query_stop
                                for segment in loaded.physical_segments))
            config = {
                "schema": WEIGHTED_EXPERIMENT_SCHEMA, "task_id": "synthetic",
                "plan_seed": 0, "model_seed": 1, "cpu_threads": 1,
                "paired_conditions": list(experiment.CONDITIONS),
                "model": {
                    "width": 6, "heads": 2, "depth": 1,
                    "feed_forward_multiplier": 2, "locus_radius_cm": .2,
                    "query_radius_cm": .2,
                    "context_execution": {"query_block_size": 2,
                        "max_mask_elements": 100,
                        "checkpoint_blocks": False,
                        "local_mean_mode": "BLOCKED_MASK_V1"},
                    "max_pair_state_elements": 1000,
                    "max_query_state_elements": 1000},
                "optimizer": {"learning_rate": .001, "betas": [.9, .999],
                    "epsilon": 1e-8, "weight_decay": 0.,
                    "gradient_clip_norm": None, "amsgrad": False,
                    "maximize": False, "foreach": False,
                    "capturable": False, "differentiable": False,
                    "fused": False},
                "reference_policy": {"expected_reference_people": 3,
                    "require_every_reference_allowed": True,
                    "max_output_genotypes": 1000},
                "region_plan": {"method":
                    "EXPLICIT_PERSON_CONTIGUOUS_QUERY_BLOCKS",
                    "blocks": [{"region_id": "report", "person_index": 0,
                                "query_start": 0, "query_stop": 1}]},
                "fit_report": {"label": "FIT_DEVELOPMENT_FIXED_LIST",
                               "region_ids": ["report"]},
                "max_microbatches_per_call": 1,
                "catalogue_policy": {"scope":
                    "FIXED_PANEL_AVAILABLE_AXIS_DEVELOPMENT",
                    "input_spec_sha256": "a" * 64,
                    "material_axis_sha256": "b" * 64,
                    "source_rare_total": 2, "material_loci_total": 2,
                    "source_rare_intersect_material": 1},
                "weighted_fit_plan": reference_value,
            }
            config_path = Path(directory) / "weighted-config.json"
            config_path.write_text(json.dumps(config))
            self.assertEqual(load_config(config_path)["weighted_fit_plan"],
                             reference_value)

    def test_select_metadata_and_arbitrary_normalized_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, _, truth, observed, source, overlay = self.prepare(
                root, role="SELECT")
            with self.assertRaisesRegex(ValueError, "scientific metadata"):
                load_weighted_fit_plan(
                    reference, person_ids=("p0", "p1"), query_valid=observed,
                    truth_observed=observed, truth_state=truth,
                    source_spec_path=source, query_truth_overlay_path=overlay)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, _, truth, observed, source, overlay = self.prepare(
                root, alter_weight=True)
            with self.assertRaisesRegex(ValueError, "anchor-state formula"):
                load_weighted_fit_plan(
                    reference, person_ids=("p0", "p1"), query_valid=observed,
                    truth_observed=observed, truth_state=truth,
                    source_spec_path=source, query_truth_overlay_path=overlay)


if __name__ == "__main__":
    unittest.main()
