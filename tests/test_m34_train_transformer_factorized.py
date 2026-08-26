#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m34_models as models
import m34_train_factorized as trainer
import m34_train_transformer_factorized as subject


CONTRACT = ROOT / "conf" / "m34_adaptive_sweep_contract.json"


def packed_rows(count: int = 130) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    baseline = np.full((1, 2, count, 3), 1.0 / 3.0, dtype="<f4")
    packed = {
        "row_ptr": np.arange(count + 1, dtype="<u8"),
        "rare_tokens": np.linspace(0.0, 1.0, count * 5, dtype="<f4").reshape(count, 5),
        "row_sample_index": np.zeros(count, dtype="<u4"),
        "row_marker_index": np.arange(count, dtype="<u4"),
        "F0": baseline,
    }
    labels = np.zeros((1, 2, count), dtype="<i8")
    transitions = np.zeros_like(labels, dtype="|u1")
    return packed, labels, transitions


class TransformerFactorizedBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_capped_batches_are_gap_free_and_respect_the_physical_ceiling(self):
        row_ptr = np.arange(132, dtype="<u8")
        batches = subject.capped_batches(
            trainer.packed_train.plan_row_batches, row_ptr, 2048, 262144, 64,
        )
        self.assertEqual(batches, [(0, 64), (64, 128), (128, 131)])

    def test_cap_tracks_declared_attention_geometry(self):
        contract = trainer.sweep.strict_json(CONTRACT)
        tasks = trainer.sweep.triage_plan(contract)["tasks"]
        observed = {}
        for task in tasks:
            if task["family"] == "transformer_small" and task["arm"] == "RE":
                observed[task["config_id"]] = subject.physical_row_cap(
                    contract, task, 2048,
                )
        self.assertEqual(observed, {
            "transformer_r0": 256,
            "transformer_r2": 170,
            "transformer_r4": 64,
        })

    def test_cap_preserves_loss_gradients_and_one_optimizer_step(self):
        specification = models.ModelSpec(
            family="transformer_small", channels=5, ancestries=3,
            hidden_dim=8, depth=1, dropout=0.0, transformer_heads=1,
            transformer_ff_dim=8, transformer_max_tokens=8, seed=1103,
            zero_init_head=False,
        )
        unrestricted = models.build_model(specification)
        bounded = models.build_model(specification)
        bounded.load_state_dict(unrestricted.state_dict())
        packed, labels, transitions = packed_rows()

        unrestricted.zero_grad(set_to_none=True)
        reference = trainer.weighted_loss(
            unrestricted, packed, labels, transitions, 0.0,
            2048, 262144, torch.device("cpu"), backward=True,
        )
        bounded.zero_grad(set_to_none=True)
        with subject.transformer_batching(64):
            observed = trainer.weighted_loss(
                bounded, packed, labels, transitions, 0.0,
                2048, 262144, torch.device("cpu"), backward=True,
            )
        np.testing.assert_allclose(observed, reference, rtol=1e-6, atol=1e-6)
        for left, right in zip(unrestricted.parameters(), bounded.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)

        first = torch.optim.SGD(unrestricted.parameters(), lr=0.01)
        second = torch.optim.SGD(bounded.parameters(), lr=0.01)
        first.step()
        second.step()
        for left, right in zip(unrestricted.parameters(), bounded.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)

        token, mask, baseline, _target, _weights = (
            trainer.packed_train.dense_batch(
                packed, labels, transitions, 0, len(packed["row_ptr"]) - 1, 0.0,
            )
        )
        unrestricted.eval()
        bounded.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                unrestricted(token, mask, baseline),
                bounded(token, mask, baseline),
                rtol=1e-6, atol=1e-6,
            )

    def test_context_manager_restores_the_shared_planner(self):
        original = trainer.packed_train.plan_row_batches
        with subject.transformer_batching(64):
            self.assertIsNot(trainer.packed_train.plan_row_batches, original)
        self.assertIs(trainer.packed_train.plan_row_batches, original)


if __name__ == "__main__":
    unittest.main()
