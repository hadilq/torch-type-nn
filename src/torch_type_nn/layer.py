"""The type-nn layer as an ``nn.Module`` whose structure can change.

Storage is dense and padded so a whole layer is a few tensor ops:

    weight    (m, R, n)   w of every Or slot
    bias      (m, R)      b
    assembly  (m, R)      a >= 1
    mask      (m, R)      True where the slot holds a live Or

``R`` is the largest degree (Ors per And) in the layer. A free slot holds
the identity Or (w = 0, b = 1, a = 1), which is exactly 1, so padding never
changes the output (see :mod:`torch_type_nn.functional`).

Structural edits (add/drop an Or, a unit, an input column) replace the
parameter tensors. The layer therefore carries its own per-Or Adam state
(``adam_*`` buffers): every edit moves the optimizer state with the
parameters, which is what :class:`torch_type_nn.optim.TypeAdam` uses. A
static layer also works with any stock ``torch.optim`` optimizer; the
``adam_*`` buffers then simply stay zero.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .edits import Edit, grow_index, keep_index
from .functional import and_or, readout

__all__ = ["AndOr", "uniform"]

# name -> fill value of a free slot
_OR_PARAMS = {"bias": 1.0, "assembly": 1.0}
_OR_BUFFERS = {
    "mask": False,
    "or_probe": False,
    "or_born": 0,
    "adam_step": 0,
    "adam_m_bias": 0.0,
    "adam_v_bias": 0.0,
    "adam_m_assembly": 0.0,
    "adam_v_assembly": 0.0,
}
_ORN_BUFFERS = ("adam_m_weight", "adam_v_weight")      # (m, R, n), fill 0
_UNIT_BUFFERS = {"unit_probe": False, "unit_born": 0}  # (m,)
_STAT_BUFFERS = ("stat_sx", "stat_sxx", "stat_su", "stat_suu", "stat_sxu")  # (n,)


class AndOr(nn.Module):
    r"""One type-nn layer: ``out_features`` Ands, each a product of Ors over the input.

    .. math::
        z_k = \mathrm{sign}(A_k)\ln(1+|A_k|),\quad
        A_k = \prod_r (w_{kr}\cdot x + b_{kr})^{a_{kr}}

    Args:
        in_features: input width ``n``.
        out_features: number of units (Ands) ``m``.
        ors_per_unit: initial degree; every initial Or is drawn with the
            ``nn.Linear`` law ``U(+-1/sqrt(n))`` and ``a = 1``.
        generator: optional ``torch.Generator`` for the initial draw.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ors_per_unit: int = 1,
        *,
        generator: torch.Generator | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        n, m, R = int(in_features), int(out_features), int(ors_per_unit)
        self.in_features, self.out_features = n, m
        self.probe = False      # True while this layer is the depth probe
        self.born = 0           # training step at birth
        self.track_stats = False
        # When a list, every parameter replacement is appended to it as an
        # :class:`~torch_type_nn.edits.Edit`, so a stock optimizer can follow.
        self.edit_sink: list | None = None
        f = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.zeros(m, R, n, **f))
        self.bias = nn.Parameter(torch.ones(m, R, **f))
        self.assembly = nn.Parameter(torch.ones(m, R, **f))
        for name, fill in _OR_BUFFERS.items():
            self.register_buffer(name, _full((m, R), fill, dtype, device))
        for name in _ORN_BUFFERS:
            self.register_buffer(name, torch.zeros(m, R, n, **f))
        for name, fill in _UNIT_BUFFERS.items():
            self.register_buffer(name, _full((m,), fill, dtype, device))
        for name in _STAT_BUFFERS:
            self.register_buffer(name, torch.zeros(n, **f), persistent=False)
        self.register_buffer("stat_gin", torch.zeros((), **f), persistent=False)
        self.register_buffer("stat_ns", torch.zeros((), **f), persistent=False)
        self.mask.fill_(True)
        self.reset_parameters(generator)

    # ------------------------------------------------------------------ basics

    @property
    def degree(self) -> int:
        """Slots per unit (R)."""
        return self.weight.shape[1]

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        """Every live Or gets the ``nn.Linear`` law ``U(+-1/sqrt(n))``, a = 1."""
        with torch.no_grad():
            k = 1.0 / math.sqrt(max(self.in_features, 1))
            w = uniform(self.weight.shape, k, generator, self.weight)
            b = uniform(self.bias.shape, k, generator, self.bias)
            live = self.mask
            self.weight.copy_(torch.where(live.unsqueeze(-1), w, torch.zeros_like(w)))
            self.bias.copy_(torch.where(live, b, torch.ones_like(b)))
            self.assembly.fill_(1.0)
            self.reset_adam_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.track_stats and self.training and torch.is_grad_enabled():
            x = self._observe(x)
        return and_or(x, self.weight, self.bias, self.assembly, self.mask)

    def num_ors(self) -> torch.Tensor:
        """Live Ors per unit, shape (m,)."""
        return self.mask.sum(1)

    def num_params(self) -> int:
        """Every learnable scalar of every live Or: w, b and a."""
        return int(self.mask.sum()) * (self.in_features + 2)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"ors={int(self.mask.sum())}, degree<={self.degree}"
                + (", probe" if self.probe else ""))

    @torch.no_grad()
    def project_(self) -> None:
        """Enforce the invariants: a >= 1, and every free slot is the identity Or."""
        self.assembly.clamp_(min=1.0)
        free = ~self.mask
        self.weight.masked_fill_(free.unsqueeze(-1), 0.0)
        self.bias.masked_fill_(free, 1.0)
        self.assembly.masked_fill_(free, 1.0)

    @torch.no_grad()
    def reset_adam_state(self, unit: int | None = None, slot: int | None = None) -> None:
        """Zero the per-Or Adam state (everywhere, or of one Or)."""
        idx = (slice(None), slice(None)) if unit is None else (unit, slot)
        for name in ("adam_step", "adam_m_bias", "adam_v_bias",
                     "adam_m_assembly", "adam_v_assembly", *_ORN_BUFFERS):
            getattr(self, name)[idx] = 0

    # ------------------------------------------------------ input statistics

    @torch.no_grad()
    def reset_stats(self) -> None:
        """Clear the epoch statistics of the layer input."""
        for name in (*_STAT_BUFFERS, "stat_gin", "stat_ns"):
            getattr(self, name).zero_()

    def _observe(self, x: torch.Tensor) -> torch.Tensor:
        # Statistics of x and of u = F(x) (what a type-identity layer would
        # emit), used to fold a depth edit; and the mean |dL/dx|, used to
        # pick the gap where depth is wanted.
        n = self.in_features
        with torch.no_grad():
            xf = x.detach().reshape(-1, n)
            u = readout(xf)
            self.stat_sx.add_(xf.sum(0))
            self.stat_sxx.add_((xf * xf).sum(0))
            self.stat_su.add_(u.sum(0))
            self.stat_suu.add_((u * u).sum(0))
            self.stat_sxu.add_((xf * u).sum(0))
            self.stat_ns.add_(xf.shape[0])
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        if n:
            gin = self.stat_gin

            def hook(g: torch.Tensor) -> None:
                with torch.no_grad():
                    gin.add_(g.reshape(-1, n).abs().mean(1).sum())

            x.register_hook(hook)
        return x

    # -------------------------------------------------------- structure edits

    def _set(self, name: str, value: torch.Tensor, dim: int | None = None,
             index: torch.Tensor | None = None) -> None:
        if name in self._parameters:
            old = self._parameters[name]
            new = nn.Parameter(value.contiguous())
            self._parameters[name] = new
            if self.edit_sink is not None and index is not None:
                d = dim if dim >= 0 else old.dim() + dim
                self.edit_sink.append(Edit(old, new, d, index))
        else:
            self._buffers[name] = value.contiguous()

    def _or_tensors(self):
        """(name, fill) of every tensor indexed by (unit, slot[, input])."""
        yield "weight", 0.0
        for name, fill in _OR_PARAMS.items():
            yield name, fill
        for name, fill in _OR_BUFFERS.items():
            yield name, fill
        for name in _ORN_BUFFERS:
            yield name, 0.0

    def _unit_tensors(self):
        yield from self._or_tensors()
        yield from _UNIT_BUFFERS.items()

    @torch.no_grad()
    def _grow_slots(self) -> None:
        for name, fill in self._or_tensors():
            t = getattr(self, name)
            pad = _full((t.shape[0], 1, *t.shape[2:]), fill, t.dtype, t.device)
            self._set(name, torch.cat([t.detach(), pad], 1), 1,
                      grow_index(t.shape[1], t.device))

    @torch.no_grad()
    def add_or(self, unit: int, weight: torch.Tensor | None = None, bias: float = 1.0,
               assembly: float = 1.0, *, probe: bool = False, born: int = 0) -> int:
        """Put a live Or in a free slot of ``unit`` (growing R if needed); returns the slot."""
        free = (~self.mask[unit]).nonzero()
        if len(free) == 0:
            self._grow_slots()
            r = self.degree - 1
        else:
            r = int(free[0])
        self.weight[unit, r] = 0.0 if weight is None else weight
        self.bias[unit, r] = bias
        self.assembly[unit, r] = assembly
        self.mask[unit, r] = True
        self.or_probe[unit, r] = probe
        self.or_born[unit, r] = born
        self.reset_adam_state(unit, r)
        return r

    @torch.no_grad()
    def drop_or(self, unit: int, slot: int) -> None:
        """Free a slot: it becomes the identity Or again."""
        self.weight[unit, slot] = 0.0
        self.bias[unit, slot] = 1.0
        self.assembly[unit, slot] = 1.0
        self.mask[unit, slot] = False
        self.or_probe[unit, slot] = False
        self.or_born[unit, slot] = 0
        self.reset_adam_state(unit, slot)

    @torch.no_grad()
    def compact(self) -> None:
        """Drop slot columns that are free in every unit."""
        keep = self.mask.any(0)
        if bool(keep.all()):
            return
        idx = keep.nonzero().flatten()
        for name, _ in self._or_tensors():
            self._set(name, getattr(self, name).detach().index_select(1, idx), 1,
                      keep_index(idx))

    @torch.no_grad()
    def add_unit(self) -> int:
        """Append a unit with no Ors (it outputs ln 2 until it gets one); returns its index."""
        for name, fill in self._unit_tensors():
            t = getattr(self, name)
            pad = _full((1, *t.shape[1:]), fill, t.dtype, t.device)
            self._set(name, torch.cat([t.detach(), pad], 0), 0,
                      grow_index(t.shape[0], t.device))
        self.out_features += 1
        return self.out_features - 1

    @torch.no_grad()
    def drop_unit(self, unit: int) -> None:
        keep = _without(self.out_features, unit, self.weight.device)
        for name, _ in self._unit_tensors():
            self._set(name, getattr(self, name).detach().index_select(0, keep), 0,
                      keep_index(keep))
        self.out_features -= 1

    @torch.no_grad()
    def add_input(self) -> None:
        """New incoming coordinate. Every Or meets it with weight exactly 0,
        so the layer computes the same function (Coq: widen_preserves)."""
        for name in ("weight", *_ORN_BUFFERS):
            t = getattr(self, name)
            self._set(name, torch.cat([t.detach(), torch.zeros_like(t[..., :1])], -1), -1,
                      grow_index(t.shape[-1], t.device))
        self.in_features += 1
        self._resize_stats()

    @torch.no_grad()
    def drop_input(self, j: int) -> None:
        keep = _without(self.in_features, j, self.weight.device)
        for name in ("weight", *_ORN_BUFFERS):
            self._set(name, getattr(self, name).detach().index_select(-1, keep), -1,
                      keep_index(keep))
        self.in_features -= 1
        self._resize_stats()

    def _resize_stats(self) -> None:
        for name in _STAT_BUFFERS:
            self._buffers[name] = self.weight.new_zeros(self.in_features)
        self.reset_stats()

    # ------------------------------------------------------------ persistence

    def get_extra_state(self):
        return {"probe": self.probe, "born": self.born}

    def set_extra_state(self, state) -> None:
        self.probe = bool(state.get("probe", False))
        self.born = int(state.get("born", 0))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A checkpoint may hold a different structure: take its shapes first.
        w = state_dict.get(prefix + "weight")
        if w is not None and w.shape != self.weight.shape:
            m, R, n = w.shape
            with torch.no_grad():
                for name, t in [*self._parameters.items(), *self._buffers.items()]:
                    src = state_dict.get(prefix + name)
                    if src is not None:
                        self._set(name, torch.empty(src.shape, dtype=t.dtype, device=t.device))
            self.in_features, self.out_features = n, m
            self._resize_stats()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def uniform(shape, bound: float, generator: torch.Generator | None,
            like: torch.Tensor) -> torch.Tensor:
    """U(-bound, bound) drawn from ``generator`` (on its own device), then moved to ``like``."""
    dev = generator.device if generator is not None else like.device
    t = torch.empty(shape, dtype=like.dtype, device=dev)
    t.uniform_(-bound, bound, generator=generator)
    return t.to(like.device)


def _full(shape, fill, dtype, device) -> torch.Tensor:
    if isinstance(fill, bool):
        return torch.full(shape, fill, dtype=torch.bool, device=device)
    if isinstance(fill, int):
        return torch.full(shape, fill, dtype=torch.long, device=device)
    return torch.full(shape, fill, dtype=dtype, device=device)


def _without(n: int, i: int, device) -> torch.Tensor:
    return torch.cat([torch.arange(i, device=device), torch.arange(i + 1, n, device=device)])
