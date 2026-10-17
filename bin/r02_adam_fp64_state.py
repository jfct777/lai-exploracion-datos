#!/usr/bin/env python3
"""Adam for FP32 parameters with FP64 first/second-moment state.

This is a bounded numerical repair candidate for R02 neural training. It keeps
the Adam equation, learning rate, betas, epsilon and coupled weight decay while
moving both accumulators and the denominator calculation to float64. It does
not clip gradients, rescale the objective, or accept a nonfinite prior state.
"""
from __future__ import annotations

import copy
import math

import torch


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class AdamFP64State(torch.optim.Optimizer):
    """Adam with FP32 parameters/gradients and strictly finite FP64 state."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, *, amsgrad=False, foreach=False,
                 maximize=False, capturable=False, differentiable=False,
                 fused=False, decoupled_weight_decay=False):
        _require(not amsgrad, "amsgrad is unsupported")
        _require(foreach is False, "foreach must be explicitly false")
        _require(not maximize, "maximize is unsupported")
        _require(not capturable, "capturable is unsupported")
        _require(not differentiable, "differentiable is unsupported")
        _require(not fused, "fused is unsupported")
        _require(not decoupled_weight_decay, "decoupled weight decay is unsupported")
        _require(isinstance(lr, (int, float)) and math.isfinite(lr) and lr >= 0.0,
                 "lr must be a finite nonnegative scalar")
        _require(isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0.0,
                 "eps must be a finite positive scalar")
        _require(isinstance(weight_decay, (int, float)) and
                 math.isfinite(weight_decay) and weight_decay >= 0.0,
                 "weight_decay must be a finite nonnegative scalar")
        _require(len(betas) == 2 and all(isinstance(value, (int, float)) and
                 math.isfinite(value) and 0.0 <= value < 1.0 for value in betas),
                 "betas must be two finite values in [0, 1)")
        defaults = dict(lr=float(lr), betas=(float(betas[0]), float(betas[1])),
                        eps=float(eps), weight_decay=float(weight_decay),
                        amsgrad=False, foreach=False, maximize=False,
                        capturable=False, differentiable=False, fused=False,
                        decoupled_weight_decay=False)
        super().__init__(params, defaults)
        self._validate_groups()

    def _validate_groups(self):
        for group in self.param_groups:
            _require(group.get("amsgrad", False) is False, "amsgrad is unsupported")
            _require(group.get("foreach", False) is False,
                     "foreach must be explicitly false")
            for flag in ("maximize", "capturable", "differentiable", "fused",
                         "decoupled_weight_decay"):
                _require(group.get(flag, False) is False, f"{flag} is unsupported")
            lr, eps, decay = group["lr"], group["eps"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            _require(all(isinstance(value, (int, float)) and math.isfinite(value)
                         for value in (lr, eps, decay, beta1, beta2)),
                     "Adam hyperparameters must be finite scalar numbers")
            _require(lr >= 0.0 and eps > 0.0 and decay >= 0.0 and
                     0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0,
                     "Adam hyperparameters are outside their admitted ranges")
            for parameter in group["params"]:
                _require(parameter.dtype == torch.float32,
                         "AdamFP64State requires FP32 parameters")
                _require(not parameter.is_complex(), "complex parameters are unsupported")

    @staticmethod
    def _validate_saved_state(state_dict):
        _require(isinstance(state_dict, dict) and isinstance(state_dict.get("state"), dict)
                 and isinstance(state_dict.get("param_groups"), list),
                 "invalid optimizer state dictionary")
        for parameter_state in state_dict["state"].values():
            _require(set(parameter_state) == {"step", "exp_avg", "exp_avg_sq"},
                     "optimizer parameter state has unsupported fields")
            step = parameter_state["step"]
            if torch.is_tensor(step):
                _require(step.numel() == 1 and torch.isfinite(step).all().item(),
                         "optimizer step is nonfinite or nonscalar")
                step = step.item()
            _require(isinstance(step, (int, float)) and float(step).is_integer() and step >= 0,
                     "optimizer step must be a nonnegative integer")
            for name in ("exp_avg", "exp_avg_sq"):
                value = parameter_state[name]
                _require(torch.is_tensor(value) and value.dtype == torch.float64,
                         f"{name} must be a float64 tensor")
                _require(torch.isfinite(value).all().item(), f"{name} is nonfinite")
            _require((parameter_state["exp_avg_sq"] >= 0).all().item(),
                     "exp_avg_sq contains a negative value")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._validate_groups()
        for group in self.param_groups:
            lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                _require(not gradient.is_sparse, "sparse gradients are unsupported")
                _require(gradient.dtype == torch.float32,
                         "AdamFP64State requires FP32 gradients")
                _require(torch.isfinite(parameter).all().item(), "parameter is nonfinite")
                _require(torch.isfinite(gradient).all().item(), "gradient is nonfinite")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter, dtype=torch.float64, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, dtype=torch.float64, memory_format=torch.preserve_format)
                _require(set(state) == {"step", "exp_avg", "exp_avg_sq"},
                         "optimizer parameter state has unsupported fields")
                first, second = state["exp_avg"], state["exp_avg_sq"]
                _require(first.dtype == torch.float64 and second.dtype == torch.float64,
                         "Adam accumulators must remain float64")
                _require(torch.isfinite(first).all().item() and
                         torch.isfinite(second).all().item() and (second >= 0).all().item(),
                         "Adam accumulator is invalid")
                state["step"] = int(state["step"]) + 1
                step = state["step"]
                gradient64 = gradient.to(dtype=torch.float64)
                if weight_decay != 0.0:
                    gradient64 = gradient64.add(
                        parameter.detach().to(dtype=torch.float64), alpha=weight_decay)
                first.mul_(beta1).add_(gradient64, alpha=1.0 - beta1)
                second.mul_(beta2).addcmul_(gradient64, gradient64, value=1.0 - beta2)
                _require(torch.isfinite(first).all().item() and
                         torch.isfinite(second).all().item(),
                         "Adam accumulator became nonfinite")
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denominator = second.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update64 = first.div(denominator)
                _require(torch.isfinite(update64).all().item(), "Adam update is nonfinite")
                updated64 = parameter.detach().to(dtype=torch.float64).add(
                    update64, alpha=-(lr / bias_correction1))
                parameter.copy_(updated64.to(dtype=torch.float32))
                _require(torch.isfinite(parameter).all().item(),
                         "parameter became nonfinite")
        return loss

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict casts floating state to parameter dtype.
        # Preserve validated FP64 moments explicitly after loading group options.
        source = copy.deepcopy(state_dict)
        self._validate_saved_state(source)
        saved_groups = source["param_groups"]
        _require(len(saved_groups) == len(self.param_groups),
                 "optimizer state has a different number of parameter groups")
        saved_by_parameter = []
        for saved_group, live_group in zip(saved_groups, self.param_groups):
            saved_ids = saved_group.get("params", [])
            live_parameters = live_group["params"]
            _require(len(saved_ids) == len(live_parameters),
                     "optimizer state has a different parameter count")
            saved_by_parameter.extend(zip(saved_ids, live_parameters))
        super().load_state_dict(source)
        self._validate_groups()
        for saved_id, parameter in saved_by_parameter:
            if saved_id not in source["state"]:
                continue
            saved = source["state"][saved_id]
            _require(saved["exp_avg"].shape == parameter.shape and
                     saved["exp_avg_sq"].shape == parameter.shape,
                     "optimizer state shape differs from parameter")
            step = saved["step"].item() if torch.is_tensor(saved["step"]) else saved["step"]
            self.state[parameter] = {
                "step": int(step),
                "exp_avg": saved["exp_avg"].detach().to(
                    device=parameter.device, dtype=torch.float64).clone(),
                "exp_avg_sq": saved["exp_avg_sq"].detach().to(
                    device=parameter.device, dtype=torch.float64).clone(),
            }
