import sys
import tempfile
import unittest
from pathlib import Path

import torch


BIN = Path(__file__).resolve().parent
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from r02_lai_donor_context import (  # noqa: E402
    AutonomousDonorContextLAI, DonorContextBatch, FINAL_HEAD_FACTORIZED_LINEAR,
    FINAL_HEAD_LINEAR, FINAL_HEAD_MLP,
)
from r02_lai_donor_train import (  # noqa: E402
    AdamSpecification, DONOR_OBJECTIVE, SYNTHETIC_SCOPE, TRAINING_SCHEMA,
    backend_determinism_receipt, build_adam, run_donor_training_call,
)
from r02_local_attention_blocks import ContextExecution  # noqa: E402


def model(**head):
    return AutonomousDonorContextLAI(
        width=6, heads=2, depth=1, feed_forward_multiplier=2,
        locus_radius_cm=0.2, query_radius_cm=0.2,
        execution=ContextExecution(query_block_size=2, max_mask_elements=100,
                                   checkpoint_blocks=False),
        max_pair_state_elements=1000, max_query_state_elements=1000, **head)


def batch():
    return DonorContextBatch(
        target_alt_dosage=torch.tensor([[0., 1.]]),
        reference_alt_dosage=torch.tensor([[0., 1.], [1., 1.], [2., 0.]]),
        target_observed=torch.ones((1, 2), dtype=torch.bool),
        reference_observed=torch.ones((3, 2), dtype=torch.bool),
        ref_alt_alleles=torch.tensor([[0, 1], [2, 3]]),
        rare_catalog=torch.tensor([False, True]),
        locus_cm=torch.tensor([0.0, 0.1]), locus_chrom=torch.tensor([22, 22]),
        query_cm=torch.tensor([[0.05]]), query_chrom=torch.tensor([[22]]),
        query_valid=torch.tensor([[True]]), reference_panel=torch.tensor([0, 1, 2]),
        donor_allowed=torch.ones((1, 3), dtype=torch.bool))


class DonorFinalHeadTest(unittest.TestCase):
    def test_default_linear_preserves_names_rng_configuration_and_forward(self):
        torch.manual_seed(731)
        implicit = model()
        rng_after_implicit = torch.get_rng_state().clone()
        torch.manual_seed(731)
        explicit = model(final_head_mode=FINAL_HEAD_LINEAR,
                         final_head_hidden_width=None)
        self.assertTrue(torch.equal(rng_after_implicit, torch.get_rng_state()))
        self.assertEqual(list(implicit.state_dict()), list(explicit.state_dict()))
        self.assertIn("joint_state_head.weight", implicit.state_dict())
        self.assertIn("joint_state_head.bias", implicit.state_dict())
        self.assertNotIn("final_head", implicit.configuration)
        self.assertEqual(implicit.configuration, explicit.configuration)
        implicit.eval(); explicit.eval()
        left = implicit(batch())
        right = explicit(batch())
        for left_value, right_value in zip(left.__dict__.values(), right.__dict__.values()):
            torch.testing.assert_close(left_value, right_value, atol=0, rtol=0)

    def test_factorized_and_mlp_have_matched_parameter_count(self):
        factorized = model(final_head_mode=FINAL_HEAD_FACTORIZED_LINEAR,
                           final_head_hidden_width=9)
        mlp = model(final_head_mode=FINAL_HEAD_MLP, final_head_hidden_width=9)
        factorized_count = sum(value.numel()
                               for value in factorized.joint_state_head.parameters())
        mlp_count = sum(value.numel() for value in mlp.joint_state_head.parameters())
        self.assertEqual(factorized_count, mlp_count)
        self.assertEqual(mlp.configuration["schema"], "r02_lai_donor_context_config_v2")
        self.assertEqual(factorized.configuration["schema"], "r02_lai_donor_context_config_v2")
        self.assertEqual(list(factorized.joint_state_head.state_dict()),
                         list(mlp.joint_state_head.state_dict()))

    def test_gelu_is_the_only_functional_difference_with_controlled_weights(self):
        factorized = model(final_head_mode=FINAL_HEAD_FACTORIZED_LINEAR,
                           final_head_hidden_width=6)
        mlp = model(final_head_mode=FINAL_HEAD_MLP, final_head_hidden_width=6)
        mlp.joint_state_head.load_state_dict(factorized.joint_state_head.state_dict())
        for head in (factorized.joint_state_head, mlp.joint_state_head):
            with torch.no_grad():
                head.input_layer.weight.zero_(); head.input_layer.bias.zero_()
                head.output_layer.weight.zero_(); head.output_layer.bias.zero_()
                head.input_layer.weight[0, 0] = 1.
                head.output_layer.weight[0, 0] = 1.
        values = torch.zeros((1, 3 * (factorized.width + 7)))
        values[0, 0] = 2.
        linear_value = factorized.joint_state_head(values)[0, 0]
        nonlinear_value = mlp.joint_state_head(values)[0, 0]
        torch.testing.assert_close(linear_value, torch.tensor(2.))
        torch.testing.assert_close(nonlinear_value, torch.nn.functional.gelu(torch.tensor(2.)))
        self.assertFalse(torch.equal(linear_value, nonlinear_value))

    def test_invalid_head_parameters_are_rejected(self):
        invalid = (
            {"final_head_mode": "unknown", "final_head_hidden_width": None},
            {"final_head_mode": FINAL_HEAD_LINEAR, "final_head_hidden_width": 6},
            {"final_head_mode": FINAL_HEAD_FACTORIZED_LINEAR,
             "final_head_hidden_width": 5},
            {"final_head_mode": FINAL_HEAD_FACTORIZED_LINEAR,
             "final_head_hidden_width": True},
            {"final_head_mode": FINAL_HEAD_MLP, "final_head_hidden_width": None},
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                model(**parameters)

    def test_training_checkpoint_identity_rejects_activation_mismatch(self):
        factorized = model(final_head_mode=FINAL_HEAD_FACTORIZED_LINEAR,
                           final_head_hidden_width=6)
        mlp = model(final_head_mode=FINAL_HEAD_MLP, final_head_hidden_width=6)
        specification = AdamSpecification(
            learning_rate=0.001, betas=(0.9, 0.999), epsilon=1e-8,
            weight_decay=0., gradient_clip_norm=None, amsgrad=False,
            maximize=False, foreach=False, capturable=False,
            differentiable=False, fused=False)
        optimizer = build_adam(factorized, specification)
        identity = {
            "schema": TRAINING_SCHEMA,
            "model_configuration": mlp.configuration,
            "optimizer": specification.receipt(),
            "objective": DONOR_OBJECTIVE,
            "backend_determinism": backend_determinism_receipt(),
            "logical_batches": [], "data_scope": SYNTHETIC_SCOPE,
        }
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "training identity differs"):
                run_donor_training_call(
                    factorized, optimizer, specification, Path(root), identity, (),
                    lambda *_: None, max_microbatches_this_call=None)


if __name__ == "__main__":
    unittest.main()
