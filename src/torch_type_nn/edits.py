"""Keep a stock ``torch.optim`` optimizer in step with structural edits.

PyTorch optimizers key their state by the ``Parameter`` object, and
structural edits replace parameters with tensors of a new shape. An
:class:`Edit` records one such replacement together with an *index map*
along one dimension: ``index[j]`` is the position in the old tensor that
slice ``j`` of the new tensor came from, or ``-1`` for a new slice.

:func:`follow_structure` applies a list of edits to an optimizer. Every state
tensor with the old parameter's shape (Adam's ``exp_avg``/``exp_avg_sq``,
SGD's ``momentum_buffer``, ...) is remapped with the same index map, new
slices starting at 0. Scalar state (Adam's ``step``) is kept as is, so a new
slice inherits the tensor's step count; :class:`~torch_type_nn.optim.TypeAdam`
avoids that with per-Or step counts. Parameters that appeared or disappeared
without an edit (a new or removed layer) are added to or dropped from the
optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["Edit", "follow_structure", "grow_index", "keep_index"]


@dataclass
class Edit:
    """``old`` was replaced by ``new``; along ``dim``, ``new[j]`` came from ``old[index[j]]``."""

    old: nn.Parameter
    new: nn.Parameter
    dim: int
    index: torch.Tensor


def grow_index(n: int, device=None) -> torch.Tensor:
    """Index map of ``n`` kept slices followed by one new slice."""
    return torch.cat([torch.arange(n, device=device), torch.full((1,), -1, device=device)])


def keep_index(keep: torch.Tensor) -> torch.Tensor:
    """Index map of a selection (``index_select`` with ``keep``)."""
    return keep.clone()


def remap_tensor(t: torch.Tensor, new_shape: torch.Size, dim: int,
                 index: torch.Tensor) -> torch.Tensor:
    """``t`` rearranged by the index map, new slices zero."""
    out = t.new_zeros(new_shape)
    index = index.to(t.device)
    src = index >= 0
    if bool(src.any()):
        pos = src.nonzero().flatten()
        out.index_copy_(dim, pos, t.index_select(dim, index[src]))
    return out


@torch.no_grad()
def follow_structure(optimizer: torch.optim.Optimizer, model: nn.Module,
                     edits: list[Edit]) -> None:
    """Bring ``optimizer`` in line with ``model`` after structural edits.

    1. Each edit, in order: the old parameter is replaced by the new one in
       its param group, and its state is remapped along the edit's index map.
    2. Parameters of ``model`` the optimizer does not know get added to the
       first param group, with fresh state.
    3. Parameters the optimizer holds that are no longer in ``model`` are
       dropped with their state.
    """
    for e in edits:
        for group in optimizer.param_groups:
            for i, p in enumerate(group["params"]):
                if p is e.old:
                    group["params"][i] = e.new
        st = optimizer.state.pop(e.old, None)
        if st:
            optimizer.state[e.new] = {
                k: (remap_tensor(v, e.new.shape, e.dim, e.index)
                    if isinstance(v, torch.Tensor) and v.shape == e.old.shape and v.dim() > 0
                    else v)
                for k, v in st.items()}
    live = list(model.parameters())
    live_ids = {id(p) for p in live}
    held = set()
    for group in optimizer.param_groups:
        kept = []
        for p in group["params"]:
            if id(p) in live_ids and id(p) not in held:
                kept.append(p)
                held.add(id(p))
            else:
                optimizer.state.pop(p, None)
        group["params"] = kept
    missing = [p for p in live if id(p) not in held]
    if missing:
        optimizer.param_groups[0]["params"].extend(missing)
