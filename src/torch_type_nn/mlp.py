"""An MLP whose width and depth can change while it trains.

``ScalableMLP`` is ``Linear -> ReLU -> ... -> Linear`` with an optional
type-nn readout ``F(u) = sign(u) ln(1 + |u|)`` on the output (so it can be
compared with the type-nn board's MLP baseline). Its layers are
:class:`TrackedLinear`: an ``nn.Linear`` that can add or drop output units and
input columns, reports each parameter replacement as an
:class:`~torch_type_nn.edits.Edit`, and collects the input statistics the
scaler needs. Grow and prune it with
:class:`~torch_type_nn.scaling.StructureScaler`, which wraps it in
:class:`~torch_type_nn.adapters.mlp.MLPAdapter`.

The identities it offers (Net2Net, Chen et al. 2016):

* width: a new hidden unit read with an all-zero column is exact;
* depth: after a ReLU the activations are >= 0, so ``ReLU(I a + 0) = a``;
  an identity layer is exact in any gap between hidden layers (but not in
  front of the first layer, whose input may be negative).
"""

from __future__ import annotations

import math
import re

import torch
from torch import nn

from .edits import Edit, grow_index, keep_index
from .functional import readout
from .layer import uniform

__all__ = ["TrackedLinear", "ScalableMLP"]


class TrackedLinear(nn.Linear):
    """``nn.Linear`` with structural edits, edit reporting and input statistics."""

    def __init__(self, in_features: int, out_features: int, *,
                 generator: torch.Generator | None = None, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, device=device, dtype=dtype)
        self.probe = False
        self.born = 0
        self.track_stats = False
        self.edit_sink: list | None = None
        f = {"device": device, "dtype": self.weight.dtype}
        self.register_buffer("unit_probe", torch.zeros(out_features, dtype=torch.bool,
                                                       device=device))
        self.register_buffer("unit_born", torch.zeros(out_features, dtype=torch.long,
                                                      device=device))
        self.register_buffer("stat_sx", torch.zeros(in_features, **f), persistent=False)
        self.register_buffer("stat_gin", torch.zeros((), **f), persistent=False)
        self.register_buffer("stat_ns", torch.zeros((), **f), persistent=False)
        if generator is not None:
            self.reset_from(generator)

    @torch.no_grad()
    def reset_from(self, generator: torch.Generator) -> None:
        """The ``nn.Linear`` law U(+-1/sqrt(n)), drawn from ``generator``."""
        k = 1.0 / math.sqrt(max(self.in_features, 1))
        self.weight.copy_(uniform(self.weight.shape, k, generator, self.weight))
        self.bias.copy_(uniform(self.bias.shape, k, generator, self.bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.track_stats and self.training and torch.is_grad_enabled():
            x = self._observe(x)
        return super().forward(x)

    def _observe(self, x: torch.Tensor) -> torch.Tensor:
        n = self.in_features
        with torch.no_grad():
            xf = x.detach().reshape(-1, n)
            self.stat_sx.add_(xf.sum(0))
            self.stat_ns.add_(xf.shape[0])
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        gin = self.stat_gin

        def hook(g: torch.Tensor) -> None:
            with torch.no_grad():
                gin.add_(g.reshape(-1, n).abs().mean(1).sum())

        x.register_hook(hook)
        return x

    @torch.no_grad()
    def reset_stats(self) -> None:
        for t in (self.stat_sx, self.stat_gin, self.stat_ns):
            t.zero_()

    def num_params(self) -> int:
        return self.out_features * (self.in_features + 1)

    def extra_repr(self) -> str:
        return super().extra_repr() + (", probe" if self.probe else "")

    # -------------------------------------------------------- structure edits

    def _set(self, name: str, value: torch.Tensor, dim: int, index: torch.Tensor) -> None:
        if name in self._parameters:
            old = self._parameters[name]
            new = nn.Parameter(value.contiguous())
            self._parameters[name] = new
            if self.edit_sink is not None:
                self.edit_sink.append(Edit(old, new, dim, index))
        else:
            self._buffers[name] = value.contiguous()

    @torch.no_grad()
    def add_out(self, weight: torch.Tensor, bias: float, *, probe: bool = False,
                born: int = 0) -> int:
        """Append an output unit; returns its index."""
        m = self.out_features
        idx = grow_index(m, self.weight.device)
        self._set("weight", torch.cat([self.weight.detach(), weight.reshape(1, -1)], 0), 0, idx)
        self._set("bias", torch.cat([self.bias.detach(), self.bias.new_full((1,), bias)]), 0, idx)
        self._set("unit_probe", torch.cat([self.unit_probe, self.unit_probe.new_full((1,), probe)]),
                  0, idx)
        self._set("unit_born", torch.cat([self.unit_born, self.unit_born.new_full((1,), born)]),
                  0, idx)
        self.out_features = m + 1
        return m

    @torch.no_grad()
    def drop_out(self, k: int) -> None:
        keep = _without(self.out_features, k, self.weight.device)
        for name in ("weight", "bias", "unit_probe", "unit_born"):
            self._set(name, getattr(self, name).detach().index_select(0, keep), 0,
                      keep_index(keep))
        self.out_features -= 1

    @torch.no_grad()
    def add_in(self) -> None:
        """A new input column of zeros: the function is unchanged."""
        idx = grow_index(self.in_features, self.weight.device)
        w = self.weight.detach()
        self._set("weight", torch.cat([w, torch.zeros_like(w[:, :1])], 1), 1, idx)
        self.in_features += 1
        self._resize_stats()

    @torch.no_grad()
    def drop_in(self, j: int) -> None:
        keep = _without(self.in_features, j, self.weight.device)
        self._set("weight", self.weight.detach().index_select(1, keep), 1, keep_index(keep))
        self.in_features -= 1
        self._resize_stats()

    def _resize_stats(self) -> None:
        self._buffers["stat_sx"] = self.weight.new_zeros(self.in_features)
        self.reset_stats()

    # ------------------------------------------------------------ persistence

    def get_extra_state(self):
        return {"probe": self.probe, "born": self.born}

    def set_extra_state(self, state) -> None:
        self.probe = bool(state.get("probe", False))
        self.born = int(state.get("born", 0))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        w = state_dict.get(prefix + "weight")
        if w is not None and w.shape != self.weight.shape:
            m, n = w.shape
            for name, t in [*self._parameters.items(), *self._buffers.items()]:
                src = state_dict.get(prefix + name)
                if src is not None:
                    new = torch.empty(src.shape, dtype=t.dtype, device=t.device)
                    if name in self._parameters:
                        self._parameters[name] = nn.Parameter(new)
                    else:
                        self._buffers[name] = new
            self.in_features, self.out_features = n, m
            self._resize_stats()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class ScalableMLP(nn.Module):
    """``Linear -> ReLU -> ... -> Linear [-> F]`` whose width and depth can change.

    Args:
        in_features, out_features: task widths (fixed for life).
        hidden: widths of the hidden layers at birth; default one hidden layer
            of ``max(2, out_features)`` units: small, since the scaler grows it.
        readout: apply ``F(u) = sign(u) ln(1 + |u|)`` to the output, as the
            type-nn board's MLP baseline does.
        seed: seeds ``self.generator`` (initialisation and structural draws).
    """

    def __init__(self, in_features: int, out_features: int, hidden: list[int] | None = None,
                 *, readout: bool = True, seed: int = 1, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.readout = readout
        self.generator = torch.Generator().manual_seed(int(seed))
        hidden = [max(2, self.out_features)] if hidden is None else list(hidden)
        widths = [self.in_features, *hidden, self.out_features]
        self.linears = nn.ModuleList(
            TrackedLinear(a, b, generator=self.generator, device=device, dtype=dtype)
            for a, b in zip(widths[:-1], widths[1:], strict=True))

    @property
    def depth(self) -> int:
        return len(self.linears)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = len(self.linears) - 1
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < last:
                x = torch.relu(x)
        return readout(x) if self.readout else x

    def num_params(self) -> int:
        return sum(l.num_params() for l in self.linears)

    def structure(self) -> list[int]:
        """Hidden widths."""
        return [l.out_features for l in self.linears[:-1]]

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"hidden={self.structure()}, params={self.num_params()}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_generator_state"] = state.pop("generator").get_state()
        return state

    def __setstate__(self, state):
        gen_state = state.pop("_generator_state", None)
        self.__dict__.update(state)
        self.generator = torch.Generator()
        if gen_state is not None:
            self.generator.set_state(gen_state)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Accepts a checkpoint of any structure."""
        idx = {int(m.group(1)) for k in state_dict
               if (m := re.match(r"linears\.(\d+)\.weight$", k))}
        D = max(idx) + 1 if idx else 0
        ref = self.linears[0].weight
        while len(self.linears) < D:
            self.linears.append(TrackedLinear(1, 1, device=ref.device, dtype=ref.dtype))
        while len(self.linears) > D:
            del self.linears[len(self.linears) - 1]
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def _without(n: int, i: int, device) -> torch.Tensor:
    return torch.cat([torch.arange(i, device=device), torch.arange(i + 1, n, device=device)])
