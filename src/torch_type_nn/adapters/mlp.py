"""The MLP adapter: :class:`~torch_type_nn.mlp.ScalableMLP` as a :class:`Scalable`.

Addresses:

* ``Item(WIDTH, (i, k))``: hidden unit ``k`` of linear ``i`` (its row in
  linear ``i`` and its column in the *consumer*, the next linear that is not
  the depth probe); sites are the non-probe hidden linears ``i``. When the
  depth probe sits between producer and consumer, the unit passes through
  it on an identity entry (weight exactly 1), which keeps the probe square
  and the edit exact, so a depth probe never blocks width growth.
* ``Item(DEPTH, (i,))``: linear ``i`` (with its ReLU); sites are gaps ``g``
  in front of linear ``g``, counted in the stack without the depth probe.
  Only gaps ``g >= 1`` qualify: they follow a ReLU, so the identity layer is
  exact there.
* No degree axis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..layer import uniform
from ..mlp import ScalableMLP, TrackedLinear
from ..protocol import DEPTH, WIDTH, EditContext, Item, Trial

__all__ = ["MLPAdapter"]


class MLPAdapter:
    """:class:`ScalableMLP` under the :class:`~torch_type_nn.protocol.Scalable` protocol."""

    def __init__(self, model: ScalableMLP) -> None:
        self.model = model
        self._sink: list | None = None
        self._tracking = False

    @property
    def L(self):
        return self.model.linears

    # ------------------------------------------------------------ bookkeeping

    def num_params(self) -> int:
        return self.model.num_params()

    def set_tracking(self, on: bool) -> None:
        self._tracking = on
        for lin in self.L:
            lin.track_stats = on

    def reset_stats(self) -> None:
        for lin in self.L:
            lin.reset_stats()

    def set_edit_sink(self, sink: list | None) -> None:
        self._sink = sink
        for lin in self.L:
            lin.edit_sink = sink

    def finalize(self) -> None:
        pass

    # ----------------------------------------------------------------- probes

    def _probe_layer(self) -> int | None:
        for i, lin in enumerate(self.L):
            if lin.probe:
                return i
        return None

    @staticmethod
    def _probe_unit(lin: TrackedLinear) -> int | None:
        k = lin.unit_probe.nonzero()
        return int(k[0]) if len(k) else None

    def _consumer(self, i: int) -> TrackedLinear:
        L = self.L
        return L[i + 2] if L[i + 1].probe else L[i + 1]

    def probe_sites(self, axis: str) -> Sequence:
        if axis == WIDTH:
            L = self.L
            return [i for i in range(len(L) - 1) if not L[i].probe]
        return []

    def probe_at(self, axis: str, site=None) -> Item | None:
        if axis == WIDTH:
            k = self._probe_unit(self.L[site])
            return None if k is None else Item(WIDTH, (site, k))
        if axis == DEPTH:
            p = self._probe_layer()
            return None if p is None else Item(DEPTH, (p,))
        return None

    def add_probe(self, axis: str, site, ctx: EditContext) -> None:
        if axis == WIDTH:
            self._add_width_probe(site, ctx)
        elif axis == DEPTH:
            self._insert_depth_probe(site, ctx)

    def promote(self, item: Item) -> None:
        if item.axis == WIDTH:
            i, k = item.address
            self.L[i].unit_probe[k] = False
        else:
            self.L[item.address[0]].probe = False

    def born(self, item: Item) -> int:
        if item.axis == WIDTH:
            i, k = item.address
            return int(self.L[i].unit_born[k])
        return self.L[item.address[0]].born

    def displacement(self, item: Item) -> float:
        if item.axis == WIDTH:
            i, k = item.address
            col = self._consumer(i).weight.detach()[:, k]
            return math.sqrt(float((col * col).mean())) if col.numel() else 0.0
        lin = self.L[item.address[0]]
        if lin.in_features != lin.out_features:
            return math.inf
        W, b = lin.weight.detach(), lin.bias.detach()
        eye = torch.eye(lin.in_features, dtype=W.dtype, device=W.device)
        s = float(((W - eye) ** 2).sum() + (b * b).sum())
        return math.sqrt(s / lin.num_params())

    def best_site(self, axis: str, moving: Item | None = None):
        """Loudest gap by mean |dL/dx| at its input, among gaps after a ReLU."""
        probe = None if moving is None else moving.address[0]

        def loud(lin: TrackedLinear) -> float:
            ns = float(lin.stat_ns)
            return float(lin.stat_gin) / ns if ns else 0.0

        best_g, best, g = 1, -1.0, 0
        for i, lin in enumerate(self.L):
            if i == probe:
                continue
            if g >= 1:
                v = loud(lin)
                if probe is not None and i == probe + 1:
                    v = max(v, loud(self.L[probe]))
                if v > best:
                    best, best_g = v, g
            g += 1
        return best_g

    def site_of(self, item: Item):
        return item.address[0]

    def remove_probe(self, item: Item) -> None:
        i = item.address[0]
        del self.L[i]
        self.L[i].reset_stats()

    # ------------------------------------------------------ structural edits

    def _add_width_probe(self, i: int, ctx: EditContext) -> None:
        """Linear i grows a trained hidden unit; the consumer reads it with zeros
        (through an identity entry of the depth probe, if one is in between)."""
        p, c = self.L[i], self._consumer(i)
        bound = 1.0 / math.sqrt(max(p.in_features, 1))
        w = uniform((p.in_features,), bound, ctx.generator, p.weight)
        b = float(uniform((), bound, ctx.generator, p.bias))
        k = p.add_out(w, b, probe=True, born=ctx.step)
        if self.L[i + 1].probe:
            self._pass_through(self.L[i + 1], k)
        c.add_in()

    @torch.no_grad()
    def _pass_through(self, P: TrackedLinear, k: int) -> None:
        """Square identity layer P grows input and output k with weight 1."""
        P.add_in()
        e = P.weight.new_zeros(P.in_features)
        e[k] = 1.0
        P.add_out(e, 0.0)

    @torch.no_grad()
    def _drop_coordinate(self, i: int, k: int) -> None:
        """Drop hidden unit k of linear i; linear i+1 keeps the mean of what it
        read (b += w E[a_k])."""
        p, c = self.L[i], self._consumer(i)
        ns = float(c.stat_ns)
        if ns > 0:
            c.bias += c.weight[:, k] * (c.stat_sx[k] / ns)
        if self.L[i + 1].probe:
            self.L[i + 1].drop_in(k)
            self.L[i + 1].drop_out(k)
        p.drop_out(k)
        c.drop_in(k)

    @torch.no_grad()
    def _insert_depth_probe(self, g: int, ctx: EditContext) -> None:
        """ReLU(I a + 0) = a for a >= 0: an identity layer in front of linear g.
        It carries every unit of the junction, width probes included."""
        L = self.L
        c = L[g]
        d = c.in_features
        l = TrackedLinear(d, d, device=c.weight.device, dtype=c.weight.dtype)
        w = uniform((d, d), ctx.noise, ctx.generator, l.weight)
        w += torch.eye(d, dtype=w.dtype, device=w.device)
        l.weight.copy_(w)
        l.bias.copy_(uniform((d,), ctx.noise, ctx.generator, l.bias))
        l.probe, l.born = True, ctx.step
        l.track_stats, l.edit_sink = self._tracking, self._sink
        c.reset_stats()
        L.insert(g, l)

    # -------------------------------------------------- measurement / pruning

    def ablate(self, item: Item):
        if item.axis == WIDTH:
            i, k = item.address
            c = self._consumer(i)
            with torch.no_grad():
                s = c.weight[:, k].clone()
                c.weight[:, k] = 0.0

            def undo() -> None:
                with torch.no_grad():
                    c.weight[:, k] = s
            return undo
        i = item.address[0]
        lin = self.L[i]
        if i == 0 or lin.in_features != lin.out_features:
            return None                  # no exact identity in front of layer 0
        with torch.no_grad():
            W, b = lin.weight.clone(), lin.bias.clone()
            lin.weight.copy_(torch.eye(lin.in_features, dtype=W.dtype, device=W.device))
            lin.bias.zero_()

        def undo_layer() -> None:
            with torch.no_grad():
                lin.weight.copy_(W)
                lin.bias.copy_(b)
        return undo_layer

    def cost(self, item: Item) -> int:
        if item.axis == WIDTH:
            i, _ = item.address
            cost = (self.L[i].in_features + 1) + self._consumer(i).out_features
            if self.L[i + 1].probe:
                cost += self.L[i + 1].in_features + 1 + self.L[i + 1].out_features
            return cost
        return self.L[item.address[0]].num_params()

    def trial_remove(self, item: Item) -> Trial | None:
        i = item.address[0]
        L = self.L
        lin = L[i]
        if i == 0 or i >= len(L) - 1 or lin.in_features != lin.out_features:
            return None
        del L[i]

        def undo() -> None:
            L.insert(i, lin)

        return Trial(undo=undo, commit=L[i].reset_stats)

    def removal_order(self, axis: str) -> Sequence[Item]:
        if axis == DEPTH:     # hidden-to-hidden layers only
            return [Item(DEPTH, (i,)) for i in range(1, len(self.L) - 1)]
        return []

    def prune_candidates(self) -> Sequence[Item]:
        return [Item(WIDTH, (i, k)) for i in range(len(self.L) - 1) if not self.L[i].probe
                for k in range(self.L[i].out_features)]

    def can_remove(self, item: Item, removed: Sequence[Item]) -> bool:
        units = {it.address for it in removed}
        i, k = item.address
        if (i, k) in units:
            return False
        left = self.L[i].out_features - sum(1 for a in units if a[0] == i)
        return left > 1

    def params_without(self, removed: Sequence[Item]) -> int:
        units = {it.address for it in removed}
        gone = [0] * len(self.L)
        for i, _ in units:
            gone[i] += 1
            if self.L[i + 1].probe:     # the pass-through shrinks with it
                gone[i + 1] += 1
        total, n_in = 0, self.L[0].in_features
        for i, lin in enumerate(self.L):
            n_out = lin.out_features - gone[i]
            total += n_out * (n_in + 1)
            n_in = n_out
        return total

    def commit_removals(self, removed: Sequence[Item]) -> dict[str, int]:
        units = {it.address for it in removed}
        c = {"or_add": 0, "or_drop": 0}
        for i in reversed(range(len(self.L) - 1)):
            lin = self.L[i]
            if lin.probe:
                continue
            for k in reversed(range(lin.out_features)):
                was_probe = bool(lin.unit_probe[k])
                if (i, k) in units:
                    self._drop_coordinate(i, k)
                    if not was_probe:
                        c["or_drop"] += 1
                elif was_probe:
                    lin.unit_probe[k] = False
                    c["or_add"] += 1
        return c
