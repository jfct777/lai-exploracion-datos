import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from r02_context_update_checkpoint import optimizer_start_digest
from r02_lai_donor_context import (
    AutonomousDonorContextLAI, DonorContextBatch, FINAL_HEAD_FACTORIZED_LINEAR,
    FINAL_HEAD_MLP,
)
from r02_lai_donor_experiment import QueryBlock
from r02_lai_donor_train import (
    AdamSpecification, FIT_SCOPE, FROZEN_READOUT_OBJECTIVE,
    FrozenReadoutTrainingBatch, WeightedLogicalBatchPlan, WeightedMicrobatchPlan,
    build_adam, donor_training_identity, run_donor_training_call,
)
from r02_lai_frozen_readout import (
    EXTRACTION_CODE_FILES, PRIOR_CODE_SCHEMA, SEGMENT_SCHEMA, FrozenReadoutModel,
    _evaluate, _import_prior_segment, _load_prior_code_manifest,
    _query_budget_receipt, _restore_segment, atomic_json, atomic_npz,
    canonical_sha256, sha256,
)
from r02_local_attention_blocks import ContextExecution


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def context_model(max_query_state_elements=1000):
    return AutonomousDonorContextLAI(
        width=6, heads=2, depth=1, feed_forward_multiplier=2,
        locus_radius_cm=0.2, query_radius_cm=0.2,
        execution=ContextExecution(query_block_size=2, max_mask_elements=100,
                                   checkpoint_blocks=False),
        max_pair_state_elements=1000,
        max_query_state_elements=max_query_state_elements)


def context_batch():
    return DonorContextBatch(
        target_alt_dosage=torch.tensor([[0., 1.]]),
        reference_alt_dosage=torch.tensor([[0., 1.], [1., 1.], [2., 0.]]),
        target_observed=torch.ones((1, 2), dtype=torch.bool),
        reference_observed=torch.ones((3, 2), dtype=torch.bool),
        ref_alt_alleles=torch.tensor([[0, 1], [2, 3]]),
        rare_catalog=torch.tensor([False, True]),
        locus_cm=torch.tensor([0.0, 0.1]), locus_chrom=torch.tensor([22, 22]),
        query_cm=torch.tensor([[0.05, 0.15]]), query_chrom=torch.tensor([[22, 22]]),
        query_valid=torch.tensor([[True, True]]),
        reference_panel=torch.tensor([0, 1, 2]),
        donor_allowed=torch.ones((1, 3), dtype=torch.bool))


def specification(lr=0.001):
    return AdamSpecification(
        learning_rate=lr, betas=(0.9, 0.999), epsilon=1e-8, weight_decay=0.,
        gradient_clip_norm=None, amsgrad=False, maximize=False, foreach=False,
        capturable=False, differentiable=False, fused=False)


def readout_batch(identity, values, truth=(0, 5), indices=(0, 1), support=(True, False)):
    q = len(truth)
    return FrozenReadoutTrainingBatch(
        identity, ("person",), torch.tensor(values, dtype=torch.float32).reshape(1, q, -1),
        torch.zeros((1, q, 6)), torch.tensor([support]),
        torch.tensor([truth]), torch.ones((1, q), dtype=torch.bool),
        torch.ones((1, q), dtype=torch.bool), torch.ones((1, q), dtype=torch.bool),
        torch.tensor([indices]))


class FrozenReadoutTest(unittest.TestCase):
    def test_extractor_runtime_budget_changes_guard_but_not_output_or_source_identity(self):
        torch.manual_seed(91)
        unrestricted = context_model().eval()
        restricted = context_model(max_query_state_elements=10).eval()
        restricted.load_state_dict(unrestricted.state_dict())
        batch = context_batch()
        original_configuration = restricted.configuration
        original_state = {name: value.detach().clone()
                          for name, value in restricted.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "query-state budget"):
            restricted.readout_representation(batch)
        receipt = _query_budget_receipt(
            restricted, (QueryBlock("fixture", 0, 0, 2),), 3, 36)
        with torch.no_grad():
            expected = unrestricted.readout_representation(batch)
            observed = restricted.readout_representation(batch)
        torch.testing.assert_close(observed.head_input, expected.head_input, atol=0, rtol=0)
        torch.testing.assert_close(observed.support, expected.support, atol=0, rtol=0)
        self.assertEqual(original_configuration["limits"]["max_query_state_elements"], 10)
        self.assertEqual(receipt["effective_max_query_state_elements"], 36)
        self.assertEqual(receipt["maximum_required_elements"], 36)
        for name, value in restricted.state_dict().items():
            torch.testing.assert_close(value, original_state[name], atol=0, rtol=0)

    def test_direct_forward_equals_head_applied_to_exposed_representation(self):
        torch.manual_seed(91)
        model = context_model().eval(); batch = context_batch()
        with torch.no_grad():
            represented = model.readout_representation(batch)
            direct = model(batch)
            cached = model.joint_state_head(represented.head_input)
        torch.testing.assert_close(direct.logits, cached, atol=0, rtol=0)
        torch.testing.assert_close(direct.support, represented.support, atol=0, rtol=0)
        self.assertEqual(represented.head_input.shape, (1, 2, 39))

    def test_identity_and_gelu_start_at_same_prior_and_w1_first_gradient_is_zero(self):
        prior = np.asarray([.1, .2, .1, .2, .1, .3])
        values = np.arange(12, dtype=np.float32).reshape(2, 6) / 10
        identity = digest("initial")
        batch = readout_batch(identity, values)
        plan = WeightedMicrobatchPlan(identity, "segment", 0, 2, ("person",),
                                      ((0, 1),), ((.5, .5),), ((1, 1),))
        outputs = []
        for mode in (FINAL_HEAD_FACTORIZED_LINEAR, FINAL_HEAD_MLP):
            model = FrozenReadoutModel(mode=mode, input_width=6, hidden_width=6,
                                       fit_prior=prior, cache_identity_sha256=digest("cache"))
            output = model(batch).logits
            outputs.append(output.detach())
            from r02_lai_donor_train import donor_weighted_occurrence_loss
            donor_weighted_occurrence_loss(model, batch, plan).person_contributions.sum().backward()
            torch.testing.assert_close(model.input_layer.weight.grad,
                                       torch.zeros_like(model.input_layer.weight.grad),
                                       atol=0, rtol=0)
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        torch.testing.assert_close(outputs[0].softmax(-1)[0, 0],
                                   torch.tensor(prior, dtype=torch.float32), rtol=1e-6, atol=1e-7)

    def test_bias_only_optimizes_six_biases_and_keeps_cached_source_logits(self):
        model = FrozenReadoutModel(
            mode="bias_only", input_width=117, hidden_width=None,
            fit_prior=[1 / 6] * 6, cache_identity_sha256=digest("cache"))
        batch = readout_batch(digest("bias"), np.zeros((2, 117)))
        batch.frozen_source_logits[0, 0] = torch.arange(6)
        torch.testing.assert_close(model(batch).logits[0, 0], torch.arange(6.))
        optimizer = build_adam(model, specification())
        self.assertEqual(sum(parameter.numel() for group in optimizer.param_groups
                             for parameter in group["params"]), 6)

    def test_partial_checkpoint_recovers_head_adam_gradients_and_rng(self):
        ids = (digest("a"), digest("b"))
        micro = (
            WeightedMicrobatchPlan(ids[0], "a", 0, 1, ("person",), ((0,),),
                                   ((.4,),), ((1,),)),
            WeightedMicrobatchPlan(ids[1], "b", 1, 2, ("person",), ((1,),),
                                   ((.6,),), ((1,),)),
        )
        plan = (WeightedLogicalBatchPlan(0, "u000", micro, 2),)
        batches = {
            ids[0]: readout_batch(ids[0], [[.1] * 6], truth=(0,), indices=(0,), support=(True,)),
            ids[1]: readout_batch(ids[1], [[.2] * 6], truth=(5,), indices=(1,), support=(False,)),
        }

        def make():
            model = FrozenReadoutModel(
                mode=FINAL_HEAD_MLP, input_width=6, hidden_width=6,
                fit_prior=[1 / 6] * 6, cache_identity_sha256=digest("cache"))
            opt = build_adam(model, specification())
            identity = donor_training_identity(
                task_id="fixture", data_scope=FIT_SCOPE, model=model, optimizer=opt,
                optimizer_specification=specification(), source_paths={"test": Path(__file__)},
                policy_path=Path(__file__), objective=FROZEN_READOUT_OBJECTIVE,
                seed=7, logical_batches=plan)
            return model, opt, identity

        def source(_model, _step, index):
            return batches[micro[index].identity_sha256]

        with tempfile.TemporaryDirectory() as root:
            full_model, full_opt, full_identity = make()
            full = run_donor_training_call(
                full_model, full_opt, specification(), Path(root) / "full", full_identity,
                plan, source, max_microbatches_this_call=None)
            partial_model, partial_opt, partial_identity = make()
            partial = run_donor_training_call(
                partial_model, partial_opt, specification(), Path(root) / "partial",
                partial_identity, plan, source, max_microbatches_this_call=1)
            self.assertEqual(partial.status, "PARTIAL")
            resumed_model, resumed_opt, resumed_identity = make()
            resumed = run_donor_training_call(
                resumed_model, resumed_opt, specification(), Path(root) / "partial",
                resumed_identity, plan, source, max_microbatches_this_call=None)
            self.assertEqual(full.history, resumed.history)
            self.assertEqual(optimizer_start_digest(full_model, full_opt),
                             optimizer_start_digest(resumed_model, resumed_opt))

    def test_six_state_metrics_use_fixed_complete_fit_mass_and_include_brier_dosage(self):
        counts = np.asarray([1, 2, 1, 2, 1, 3], dtype=np.int64)
        weights = tuple((counts / counts.sum()).tolist())
        identity = digest("six")
        micro = WeightedMicrobatchPlan(
            identity, "all", 0, 6, ("person",), (tuple(range(6)),),
            (weights,), ((1, 1, 1, 1, 1, 1),))
        plans = tuple(WeightedLogicalBatchPlan(i, f"u{i:03d}", (micro,), 6)
                      for i in range(16))
        model = FrozenReadoutModel(
            mode=FINAL_HEAD_FACTORIZED_LINEAR, input_width=6, hidden_width=6,
            fit_prior=counts / counts.sum(), cache_identity_sha256=digest("cache"))
        batch = readout_batch(identity, np.zeros((6, 6)), truth=tuple(range(6)),
                              indices=tuple(range(6)), support=(True,) * 6)

        class Source:
            def __call__(self, *_):
                return batch
        result = _evaluate(model, plans, Source(), counts, trained_updates=16)
        self.assertAlmostEqual(sum(result["observed_weight_per_update_by_state"]), 1.0)
        self.assertEqual(len(result["conditional_nll_fixed_fit_mass_by_state"]), 6)
        self.assertIn("weighted_brier", result)
        self.assertEqual(set(result["weighted_dosage_mae_by_ancestry"]), {"AFR", "EUR", "NAM"})
        self.assertEqual(set(result["weighted_dosage_mse_by_ancestry"]), {"AFR", "EUR", "NAM"})
        self.assertGreater(result["weighted_mean_predicted_probability_NN"], 0)

    def test_complete_segment_cache_is_reused_and_identity_mismatch_is_rejected(self):
        arrays = {
            "head_input": np.zeros((1, 2, 117), dtype=np.float32),
            "frozen_source_logits": np.zeros((1, 2, 6), dtype=np.float32),
            "support": np.asarray([[True, False]]),
            "truth_state": np.asarray([[0, 5]], dtype=np.int64),
            "truth_observed": np.asarray([[True, True]]),
            "query_valid": np.asarray([[True, True]]),
            "query_global_index": np.asarray([[10, 11]], dtype=np.int64),
        }
        identity = {"cache_identity_sha256": digest("cache"),
                    "block": {"region_id": "s", "person_index": 0,
                              "query_start": 10, "query_stop": 12},
                    "person_id": "person"}
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); payload = root / "s.npz"; manifest = root / "s.json"
            atomic_npz(payload, arrays)
            atomic_json(manifest, {"schema": SEGMENT_SCHEMA, "identity": identity,
                        "identity_sha256": canonical_sha256(identity),
                        "payload": payload.name, "payload_sha256": sha256(payload)})
            self.assertTrue(_restore_segment(manifest, identity))
            with self.assertRaisesRegex(ValueError, "differs"):
                _restore_segment(manifest, {**identity, "person_id": "other"})

    def test_prior_segment_import_changes_identity_and_preserves_authenticated_bytes(self):
        arrays = {
            "head_input": np.arange(234, dtype=np.float32).reshape(1, 2, 117),
            "frozen_source_logits": np.arange(12, dtype=np.float32).reshape(1, 2, 6),
            "support": np.asarray([[True, False]]),
            "truth_state": np.asarray([[0, 5]], dtype=np.int64),
            "truth_observed": np.asarray([[True, True]]),
            "query_valid": np.asarray([[True, True]]),
            "query_global_index": np.asarray([[10, 11]], dtype=np.int64),
        }
        block = {"region_id": "s", "person_index": 0,
                 "query_start": 10, "query_stop": 12}
        prior_identity = {"cache_identity_sha256": digest("prior"),
                          "block": block, "person_id": "person"}
        new_identity = {**prior_identity, "cache_identity_sha256": digest("new")}
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); prior_root = root / "prior"; new_root = root / "new"
            prior_root.mkdir(); new_root.mkdir()
            prior_payload = prior_root / "s.npz"; prior_manifest = prior_root / "s.json"
            atomic_npz(prior_payload, arrays)
            atomic_json(prior_manifest, {
                "schema": SEGMENT_SCHEMA, "identity": prior_identity,
                "identity_sha256": canonical_sha256(prior_identity),
                "payload": prior_payload.name, "payload_sha256": sha256(prior_payload),
                "arrays": {name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                           for name, value in arrays.items()},
                "lineage": {"condition": "ALL_AVAILABLE_SOURCE_RARE"},
            })
            prior_hash = sha256(prior_payload)
            receipt = {"path": str(root / "prior-code.json"),
                       "sha256": digest("prior-code"), "schema": PRIOR_CODE_SCHEMA,
                       "code_sha256": {name: digest(name) for name in EXTRACTION_CODE_FILES}}
            destination = new_root / "s.json"
            self.assertTrue(_import_prior_segment(
                prior_manifest, destination, prior_identity, new_identity, receipt))
            self.assertTrue(_restore_segment(destination, new_identity))
            imported = __import__("json").loads(destination.read_text())
            self.assertEqual(imported["reused_from"]["prior_manifest_sha256"],
                             sha256(prior_manifest))
            self.assertTrue(imported["reused_from"]["byte_identical_payload"])
            self.assertEqual(sha256(new_root / "s.npz"), prior_hash)
            self.assertEqual(sha256(prior_payload), prior_hash)

    def test_prior_segment_import_refuses_changed_source_identity_and_corruption(self):
        arrays = {
            "head_input": np.zeros((1, 1, 117), dtype=np.float32),
            "frozen_source_logits": np.zeros((1, 1, 6), dtype=np.float32),
            "support": np.asarray([[True]]), "truth_state": np.asarray([[0]]),
            "truth_observed": np.asarray([[True]]),
            "query_valid": np.asarray([[True]]),
            "query_global_index": np.asarray([[10]], dtype=np.int64),
        }
        identity = {"cache_identity_sha256": digest("condition-and-source"),
                    "block": {"region_id": "s", "person_index": 0,
                              "query_start": 10, "query_stop": 11},
                    "person_id": "person"}
        receipt = {"path": "prior-code.json", "sha256": digest("prior-code"),
                   "schema": PRIOR_CODE_SCHEMA,
                   "code_sha256": {name: digest(name) for name in EXTRACTION_CODE_FILES}}
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); prior = root / "prior"; new = root / "new"
            prior.mkdir(); new.mkdir()
            payload = prior / "s.npz"; manifest = prior / "s.json"
            atomic_npz(payload, arrays)
            atomic_json(manifest, {"schema": SEGMENT_SCHEMA, "identity": identity,
                        "identity_sha256": canonical_sha256(identity),
                        "payload": payload.name, "payload_sha256": sha256(payload)})
            changed = {**identity, "cache_identity_sha256": digest("changed-source")}
            with self.assertRaisesRegex(ValueError, "differs"):
                _import_prior_segment(manifest, new / "changed.json", changed,
                                      changed, receipt)
            payload.write_bytes(payload.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "differs"):
                _import_prior_segment(manifest, new / "corrupt.json", identity,
                                      changed, receipt)

    def test_prior_code_manifest_has_exact_six_names_and_only_extractor_may_change(self):
        current = {name: digest("current-" + name) for name in EXTRACTION_CODE_FILES}
        prior = dict(current); prior["r02_lai_frozen_readout.py"] = digest("old-extractor")
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "PRIOR_CODE.json"
            atomic_json(path, {"schema": PRIOR_CODE_SCHEMA, "code_sha256": prior})
            self.assertEqual(_load_prior_code_manifest(path, current), prior)
            changed_dependency = dict(prior)
            changed_dependency["r02_lai_donor_context.py"] = digest("changed")
            atomic_json(path, {"schema": PRIOR_CODE_SCHEMA,
                               "code_sha256": changed_dependency})
            with self.assertRaisesRegex(ValueError, "non-extractor"):
                _load_prior_code_manifest(path, current)


if __name__ == "__main__":
    unittest.main()
