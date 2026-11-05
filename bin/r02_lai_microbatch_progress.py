"""Durable gradients inside one regional LAI update; no serialized autograd."""
import torch

from r02_context_checkpoint import _require, identity_digest
from r02_context_update_checkpoint import (
    PartialUpdateStore, capture_parameter_gradients, restore_parameter_gradients,
    optimizer_start_digest,
)


class LaiMicrobatchProgress:
    def __init__(self, root, training_identity, step, entry, model, optimizer,
                 every_people=4):
        _require(type(step) is int and step >= 0, 'invalid LAI update index')
        _require(type(every_people) is int and every_people > 0,
                 'positive checkpoint cadence required')
        source = {k: entry.get(k) for k in ('path', 'sha256', 'receipt_sha256')}
        _require(all(source.values()), 'authenticated LAI entry required')
        identity = dict(schema='r02_lai_person_gradient_progress_v1',
                        training_identity_sha256=identity_digest(training_identity),
                        update_index=step, source=source,
                        model_optimizer_start_sha256=optimizer_start_digest(model, optimizer))
        self.store = PartialUpdateStore(root, identity)
        self.model, self.optimizer = model, optimizer
        self.every_people = every_people

    def resume(self, people):
        _require(type(people) is int and people > 0, 'positive full person count required')
        saved = self.store.restore(self.model, self.optimizer)
        if saved is None:
            return 0, torch.zeros((), dtype=torch.float64), []
        progress, state = saved
        start = progress['next_person_batch_index']
        _require(progress['phase'] == 'LAI_PERSON_MICROBATCH'
                 and state['people'] == people and 0 < start <= people,
                 'LAI person cursor or denominator differs')
        pred, total = state['prediction'], state['loss_sum']
        nonfinite = state['prediction_nan']
        _require(pred.shape[0] == start and nonfinite.shape == pred.shape
                 and nonfinite.dtype == torch.bool
                 and bool(torch.isfinite(pred).all())
                 and total.shape == () and total.dtype == torch.float64
                 and bool(torch.isfinite(total)), 'invalid saved LAI values')
        pred = pred.masked_fill(nonfinite, float('nan'))
        restore_parameter_gradients(self.model, state['gradients'])
        return start, total, [pred]

    def save(self, next_person, total, predictions, people):
        if next_person % self.every_people and next_person != people:
            return
        prediction = torch.cat(predictions, 0)
        _require(0 < next_person <= people and prediction.shape[0] == next_person,
                 'LAI progress does not match completed people')
        _require(not bool(torch.isinf(prediction).any()),
                 'infinite LAI prediction cannot be checkpointed')
        prediction_nan = torch.isnan(prediction)
        finite_prediction = torch.where(prediction_nan, torch.zeros_like(prediction),
                                        prediction)
        self.store.save(self.model, self.optimizer,
                        dict(phase='LAI_PERSON_MICROBATCH',
                             next_person_batch_index=next_person, next_tile_index=0),
                        dict(people=people, loss_sum=total,
                             prediction=finite_prediction,
                             prediction_nan=prediction_nan,
                             gradients=capture_parameter_gradients(self.model)))

    def clear(self):
        """Caller invokes only after the completed Adam checkpoint is durable."""
        self.store.clear()

    def close(self):
        self.store.close()
