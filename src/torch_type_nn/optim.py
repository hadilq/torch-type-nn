"""Adam with one step counter per Or.

type-nn gives each Or its own Adam bias correction: an Or born late starts
at t = 1, so its first steps are corrected exactly like those of an Or that
existed from the start. Its moments live in the layer (``AndOr.adam_*``) so
structural edits carry them along.

After every step the assembly index is projected back to ``a >= 1`` and
free slots stay the identity Or.

Parameters that do not belong to an :class:`~torch_type_nn.layer.AndOr`
(say, a ``nn.Linear`` head in the same model) get ordinary Adam, so one
``TypeAdam`` can drive a mixed model.
"""

from __future__ import annotations

import torch
from torch import nn

from .layer import AndOr

__all__ = ["TypeAdam"]


class TypeAdam(torch.optim.Optimizer):
    """Adam whose step count is per Or, for models that contain :class:`AndOr` layers.

    Unlike stock optimizers, ``TypeAdam`` takes the *module*, not a parameter
    list: structural edits replace parameter tensors, and the optimizer
    re-reads them from the module on every step.

    Args:
        module: the model.
        lr: learning rate.
        betas, eps: as in :class:`torch.optim.Adam`.
    """

    def __init__(self, module: nn.Module, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        if lr < 0:
            raise ValueError(f"invalid lr {lr}")
        self.module = module
        super().__init__(list(module.parameters()), {"lr": lr, "betas": betas, "eps": eps})

    def _sync(self) -> None:
        self.param_groups[0]["params"] = list(self.module.parameters())

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._sync()
        super().zero_grad(set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._sync()
        group = self.param_groups[0]
        lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
        owned = set()
        for layer in self.module.modules():
            if isinstance(layer, AndOr):
                _step_layer(layer, lr, b1, b2, eps)
                owned.update(id(p) for p in layer.parameters(recurse=False))
        for p in group["params"]:
            if id(p) in owned or p.grad is None:
                continue
            st = self.state[p]
            if not st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
            st["step"] += 1
            t = st["step"]
            st["exp_avg"].mul_(b1).add_(p.grad, alpha=1 - b1)
            st["exp_avg_sq"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            c1, c2 = 1.0 / (1.0 - b1 ** t), 1.0 / (1.0 - b2 ** t)
            p.sub_(lr * (st["exp_avg"] * c1) / ((st["exp_avg_sq"] * c2).sqrt() + eps))
        return loss


def _step_layer(layer: AndOr, lr: float, b1: float, b2: float, eps: float) -> None:
    live = layer.mask
    layer.adam_step.add_(live.long())
    t = layer.adam_step.clamp(min=1).to(layer.weight.dtype)
    c1 = 1.0 / (1.0 - b1 ** t)                              # (m, R)
    c2 = 1.0 / (1.0 - b2 ** t)
    livef = live.to(layer.weight.dtype)
    for p, m, v, cc1, cc2, lf in (
        (layer.weight, layer.adam_m_weight, layer.adam_v_weight,
         c1.unsqueeze(-1), c2.unsqueeze(-1), livef.unsqueeze(-1)),
        (layer.bias, layer.adam_m_bias, layer.adam_v_bias, c1, c2, livef),
        (layer.assembly, layer.adam_m_assembly, layer.adam_v_assembly, c1, c2, livef),
    ):
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        g = g * lf
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        p.sub_(lf * lr * (m * cc1) / ((v * cc2).sqrt() + eps))
    layer.assembly.clamp_(min=1.0)                          # at least one copy
