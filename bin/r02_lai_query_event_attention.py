"""Query-conditioned EVENT readout, nested in the historical uniform mean.

Reuses the existing exact local attention implementation. This still produces
a finite weighted summary, not a lossless representation or phased haplotypes.
No training, checkpoint migration or scientific admission happens implicitly.
"""
import torch

from r02_lai_common_context import LocalQuerySourceAttention, LOCAL_MEAN_V1
from r02_local_attention_blocks import ContextExecution


QUERY_EVENT_RELATIVE_V1 = "QUERY_EVENT_RELATIVE_V1"
QUERY_EVENT_CONTENT_V1 = "QUERY_EVENT_CONTENT_V1"
LAI_QUERY_EVENT_MODES = frozenset((LOCAL_MEAN_V1, QUERY_EVENT_RELATIVE_V1,
                                  QUERY_EVENT_CONTENT_V1))


class QueryEventAttention(LocalQuerySourceAttention):
    """Attend from a prediction query to its eligible EVENT representations.

    Q receives existing common context, FLARE probabilities and availability.
    K/V receive existing EVENT values. K=0, V=I, output=I and position=0 make
    initial weights uniform; Q remains random so K can learn on the first step.
    The inherited radius/chromosome/observation masks are unchanged.
    """

    def __init__(self, query_features, width, heads, execution, mode):
        if not isinstance(execution, ContextExecution):
            raise ValueError("query-EVENT attention requires bounded execution")
        if mode not in (QUERY_EVENT_RELATIVE_V1, QUERY_EVENT_CONTENT_V1):
            raise ValueError("explicit query-EVENT attention mode required")
        super().__init__(query_features, width, heads, 0., execution,
                         use_relative_position=mode == QUERY_EVENT_RELATIVE_V1)
        self.mode = mode
        self.initialize_as_mean()

    def initialize_as_mean(self):
        """New-module initialization only; never call on a resumed model."""
        with torch.no_grad():
            self.common_key.weight.zero_()
            self.common_key.bias.zero_()
            self.relative_position.weight.zero_()
            self.relative_position.bias.zero_()
            for layer in (self.common_value, self.output):
                layer.weight.copy_(torch.eye(self.width, device=layer.weight.device,
                                              dtype=layer.weight.dtype))
                layer.bias.zero_()


def initialize_from_mean_model(model, state):
    """Import only an exact historical model, requiring a fresh optimizer.

    The caller authenticates source files and records the architecture change.
    A partially compatible checkpoint is rejected rather than silently loaded.
    Normal resume uses strict load_state_dict instead of this initializer.
    """
    if not isinstance(getattr(model, "query_event_attention", None), QueryEventAttention):
        raise ValueError("target must use query-EVENT attention")
    expected = model.state_dict()
    added = {key for key in expected if key.startswith("query_event_attention.")}
    if set(state) != set(expected) - added:
        raise ValueError("source does not exactly match the historical mean model")
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError("source state shape/dtype differs: " + key)
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("source state is nonfinite: " + key)
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != added or result.unexpected_keys:
        raise ValueError("unexpected checkpoint migration")
    model.query_event_attention.initialize_as_mean()
    return {"added_state_keys": sorted(added), "optimizer_state_imported": False,
            "semantics": "WEIGHT_INITIALIZATION_NOT_TRAINING_RESUME"}
